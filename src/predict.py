"""
Prediction Agent — T04: Previsão de Localização de Ônibus
Responsabilidade: processar arquivos de teste e gerar previsões de ETA e posição.

Tarefa 1 (T1): dado {id, ordem, linha, latitude, longitude} → prever datahora (epoch ms)
Tarefa 2 (T2): dado {id, ordem, linha, datahora} → prever (latitude, longitude)

Design: processamento em lote por arquivo de query
  - 1 query SQL para obter último ping de todos os veículos simultaneamente
  - KD-trees pré-construídas por trajeto (uma por sentido por linha)
  - Speed profile carregado em dict em memória
"""
from __future__ import annotations

import json
import logging
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import requests
import urllib3
from pyproj import Transformer
from scipy.spatial import cKDTree

from src.ingest import DB_PATH
from src.utils import parse_lat_lon, TARGET_LINES

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
BRT_OFFSET_H    = -3        # UTC-3 para Brasília
JANELA_HIST_H   = 4         # horas de histórico GPS a buscar antes de ts_query
GEOCERCA_M      = 200       # distância máxima para snap válido ao trajeto (rotas têm grid 150m)
DEFAULT_V_KMH   = 20.0      # velocidade default quando sem perfil
SUBMIT_URL      = "https://barra.cos.ufrj.br:443/datamining/rpc/avalia"
ALUNO           = "Joao Pedro Barbosa Martins"
SENHA           = "CPS833_T04"

DATA_RAW    = Path("data/raw")
RESULTS_DIR = Path("results")

_utm_tf = Transformer.from_crs("EPSG:4326", "EPSG:31983", always_xy=True)


# ── Utilitários ───────────────────────────────────────────────────────────────

def _epoch_ms_to_dt(ms: int | str) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


def _dt_to_epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _to_utm(lat_arr: np.ndarray, lon_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = _utm_tf.transform(lon_arr, lat_arr)
    return x, y


def _ensure_ts_utc(ts) -> datetime:
    """Garante que o timestamp é datetime UTC."""
    if hasattr(ts, 'to_pydatetime'):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


# ── Cache de rotas e speed profile ───────────────────────────────────────────

class RouteCache:
    """Carrega e mantém em memória os trajetos canônicos e speed profiles."""

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self._con = con
        self._routes: dict[tuple, pd.DataFrame] = {}
        self._trees:  dict[tuple, tuple] = {}   # (linha, sentido) → (cKDTree, route_df)
        self._speeds: dict[tuple, float] = {}   # (linha, sentido, seg, hora) → v_p50

    def get_route(self, linha: str, sentido: int) -> Optional[pd.DataFrame]:
        key = (linha, sentido)
        if key not in self._routes:
            df = self._con.execute(
                "SELECT seq, lat, lon, dist_acum_m FROM route_canonical "
                "WHERE linha=? AND sentido=? ORDER BY seq",
                [str(linha), int(sentido)],
            ).fetchdf()
            self._routes[key] = df if not df.empty else None
        return self._routes[key]

    def get_tree(self, linha: str, sentido: int) -> Optional[tuple]:
        key = (linha, sentido)
        if key not in self._trees:
            route_df = self.get_route(linha, sentido)
            if route_df is None:
                self._trees[key] = None
            else:
                rx, ry = _to_utm(route_df["lat"].to_numpy(), route_df["lon"].to_numpy())
                tree = cKDTree(np.column_stack([rx, ry]))
                self._trees[key] = (tree, rx, ry, route_df)
        return self._trees[key]

    def preload_speeds(self, linhas: set[str]) -> None:
        """Carrega speed_profile das linhas necessárias para o arquivo."""
        rows = self._con.execute(
            "SELECT linha, sentido, seg_km, hora, velocidade_p50 FROM speed_profile "
            "WHERE linha IN (SELECT unnest(?) as l)",
            [list(linhas)],
        ).fetchall()
        for linha, sentido, seg, hora, v in rows:
            self._speeds[(str(linha), int(sentido), int(seg), int(hora))] = float(v) if v else DEFAULT_V_KMH

    def get_speed(self, linha: str, sentido: int, dist_acum_m: float, hora: int) -> float:
        seg = int(dist_acum_m // 500)
        v = self._speeds.get((str(linha), int(sentido), seg, int(hora)))
        if v is not None:
            return v
        # Vizinhança ±1 hora
        for dh in (0, 1, -1):
            h2 = (hora + dh) % 24
            v = self._speeds.get((str(linha), int(sentido), seg, h2))
            if v is not None:
                return v
        return DEFAULT_V_KMH


def _snap_single(lat: float, lon: float, tree_data: tuple) -> tuple[float, float, float, float]:
    """Snap de um ponto ao trajeto. Retorna (lat_snap, lon_snap, dist_acum_m, dist_m)."""
    tree, rx, ry, route_df = tree_data
    px, py = _to_utm(np.array([lat]), np.array([lon]))
    dist_utm, idx = tree.query([px[0], py[0]])
    return (
        float(route_df["lat"].iloc[idx]),
        float(route_df["lon"].iloc[idx]),
        float(route_df["dist_acum_m"].iloc[idx]),
        float(dist_utm),
    )


def _route_lookup_dist(dist_acum_m: float, route_df: pd.DataFrame) -> tuple[float, float]:
    """Interpolação linear: dist_acum_m → (lat, lon) no trajeto."""
    dist_arr = route_df["dist_acum_m"].to_numpy()
    lat_arr  = route_df["lat"].to_numpy()
    lon_arr  = route_df["lon"].to_numpy()
    dist_acum_m = float(np.clip(dist_acum_m, dist_arr[0], dist_arr[-1]))
    idx = np.searchsorted(dist_arr, dist_acum_m, side="right")
    if idx == 0:
        return float(lat_arr[0]), float(lon_arr[0])
    if idx >= len(dist_arr):
        return float(lat_arr[-1]), float(lon_arr[-1])
    d0, d1 = dist_arr[idx - 1], dist_arr[idx]
    t = (dist_acum_m - d0) / (d1 - d0) if d1 > d0 else 0.0
    return (
        float(lat_arr[idx - 1] + t * (lat_arr[idx] - lat_arr[idx - 1])),
        float(lon_arr[idx - 1] + t * (lon_arr[idx] - lon_arr[idx - 1])),
    )


# ── Batch: último ping por veículo ────────────────────────────────────────────

def _batch_get_last_pings(
    ts_before: datetime,
    janela_h: float,
    con: duckdb.DuckDBPyConnection,
) -> dict[tuple[str, str], dict]:
    """
    Retorna último ping por (ordem, linha) em uma única query SQL.
    Usa tabela gps (raw) para capturar horas fora da janela 08-23 BRT.
    """
    ts_cutoff = ts_before - timedelta(hours=janela_h)
    df = con.execute(
        """
        SELECT ordem, linha, lat, lon, ts_servidor
        FROM (
            SELECT ordem, linha, lat, lon, ts_servidor,
                   ROW_NUMBER() OVER (PARTITION BY ordem, linha ORDER BY ts_servidor DESC) AS rn
            FROM gps
            WHERE ts_servidor < ? AND ts_servidor > ?
        ) sub
        WHERE rn = 1
        """,
        [ts_before, ts_cutoff],
    ).fetchdf()

    result: dict[tuple[str, str], dict] = {}
    for _, row in df.iterrows():
        key = (str(row["ordem"]), str(row["linha"]))
        result[key] = {
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "ts":  _ensure_ts_utc(row["ts_servidor"]),
        }
    return result


# ── Predição T1: posição → timestamp ─────────────────────────────────────────

def _predict_eta_single(
    rec: dict,
    ts_query: datetime,
    last_pings: dict,
    cache: RouteCache,
) -> int:
    """T1: retorna datahora em epoch ms."""
    linha = str(rec.get("linha", ""))
    ordem = str(rec.get("ordem", ""))
    lat_tgt = parse_lat_lon(rec["latitude"])
    lon_tgt = parse_lat_lon(rec["longitude"])

    # Snap da posição alvo ao melhor sentido
    best_sentido = 1
    best_dist_acum = 0.0
    best_dist_route = float("inf")

    for sentido in (1, 2):
        tree_data = cache.get_tree(linha, sentido)
        if tree_data is None:
            continue
        _, _, d_acum, d_route = _snap_single(lat_tgt, lon_tgt, tree_data)
        if d_route < best_dist_route:
            best_dist_route = d_route
            best_sentido    = sentido
            best_dist_acum  = d_acum

    if best_dist_route > GEOCERCA_M:
        return _dt_to_epoch_ms(ts_query + timedelta(minutes=1))

    # Último ping do veículo
    last = last_pings.get((ordem, linha))
    if last is None:
        return _dt_to_epoch_ms(ts_query + timedelta(minutes=1))

    tree_data = cache.get_tree(linha, best_sentido)
    if tree_data is None:
        return _dt_to_epoch_ms(ts_query + timedelta(minutes=1))

    _, _, dist_last, _ = _snap_single(last["lat"], last["lon"], tree_data)
    ts_last = last["ts"]
    hora_brt = (ts_last.hour + BRT_OFFSET_H) % 24

    dist_travel = best_dist_acum - dist_last
    if abs(dist_travel) < 1.0:
        # Veículo já está lá
        return _dt_to_epoch_ms(ts_last)

    # Velocidade média entre as duas posições
    dist_mid = (dist_last + best_dist_acum) / 2
    v_kmh = cache.get_speed(linha, best_sentido, dist_mid, hora_brt)
    v_mps = max(v_kmh / 3.6, 0.5)
    eta_s = abs(dist_travel) / v_mps

    ts_pred = ts_last + timedelta(seconds=eta_s)
    # Clamp
    ts_pred = max(ts_pred, ts_last + timedelta(seconds=1))
    ts_pred = min(ts_pred, ts_query + timedelta(hours=2))
    return _dt_to_epoch_ms(ts_pred)


# ── Predição T2: timestamp → posição ─────────────────────────────────────────

def _predict_position_single(
    rec: dict,
    last_pings: dict,
    cache: RouteCache,
) -> tuple[float, float] | None:
    """T2: retorna (lat, lon)."""
    linha = str(rec.get("linha", ""))
    ordem = str(rec.get("ordem", ""))
    ts_ms = rec.get("datahora")
    if ts_ms is None:
        return None

    ts_target = _epoch_ms_to_dt(ts_ms)
    last = last_pings.get((ordem, linha))

    if last is None:
        route_df = cache.get_route(linha, 1)
        if route_df is None:
            return None
        return float(route_df.iloc[0]["lat"]), float(route_df.iloc[0]["lon"])

    # Snap da última posição ao melhor sentido
    best_sentido = 1
    best_dist_last = 0.0
    best_dist_route = float("inf")

    for sentido in (1, 2):
        tree_data = cache.get_tree(linha, sentido)
        if tree_data is None:
            continue
        _, _, d_acum, d_route = _snap_single(last["lat"], last["lon"], tree_data)
        if d_route < best_dist_route:
            best_dist_route = d_route
            best_sentido    = sentido
            best_dist_last  = d_acum

    if best_dist_route > GEOCERCA_M:
        return float(last["lat"]), float(last["lon"])

    ts_last = last["ts"]
    dt_s = (ts_target - ts_last).total_seconds()
    hora_brt = (ts_last.hour + BRT_OFFSET_H) % 24

    v_kmh = cache.get_speed(linha, best_sentido, best_dist_last, hora_brt)
    v_mps = max(v_kmh / 3.6, 0.5)
    dist_target = best_dist_last + (v_mps * max(dt_s, 0.0))

    route_df = cache.get_route(linha, best_sentido)
    if route_df is None:
        return float(last["lat"]), float(last["lon"])

    return _route_lookup_dist(dist_target, route_df)


# ── Pipeline de processamento de arquivo de teste ─────────────────────────────

def process_test_file(
    zip_path: Path,
    query_filename: str,
    db_path: Path = DB_PATH,
) -> list[list]:
    """
    Processa um arquivo de query em lote.
    Retorna lista [[id, val, ...], ...].
    T1: [id, datahora_ms]
    T2: [id, lat, lon]
    """
    con = duckdb.connect(str(db_path), read_only=True)

    with zipfile.ZipFile(zip_path) as z:
        records = json.loads(z.read(query_filename))

    if not records:
        con.close()
        return []

    # Extrair data e hora da query do nome do arquivo
    stem = Path(query_filename).stem
    clean = stem.replace("treino-", "").replace("teste-", "")
    date_part, hora_str = clean.rsplit("_", 1)
    hora_int = int(hora_str)
    dt_date  = datetime.strptime(date_part, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    ts_query = dt_date + timedelta(hours=hora_int - BRT_OFFSET_H)

    has_lat = "latitude" in records[0]
    has_dh  = "datahora" in records[0]
    task    = "T1" if (has_lat and not has_dh) else "T2"

    logger.info(
        "Processando %s (%d registros, tarefa=%s, ts_query=%s)",
        query_filename, len(records), task, ts_query.isoformat(),
    )

    # Preload: rotas + speed profiles para todas as linhas necessárias
    cache = RouteCache(con)
    linhas_needed = {str(r.get("linha", "")) for r in records}
    cache.preload_speeds(linhas_needed)

    # Batch: último ping por veículo (1 query para todos)
    last_pings = _batch_get_last_pings(ts_query, JANELA_HIST_H, con)

    # Processar registros
    previsoes = []
    n_fallback = 0

    for rec in records:
        rid   = rec["id"]
        linha = str(rec.get("linha", ""))

        if linha not in TARGET_LINES:
            n_fallback += 1
            if task == "T1":
                previsoes.append([rid, _dt_to_epoch_ms(ts_query)])
            else:
                previsoes.append([rid, -22.9, -43.2])
            continue

        try:
            if task == "T1":
                ts_ms = _predict_eta_single(rec, ts_query, last_pings, cache)
                previsoes.append([rid, ts_ms])
            else:
                result = _predict_position_single(rec, last_pings, cache)
                if result is None:
                    n_fallback += 1
                    previsoes.append([rid, -22.9, -43.2])
                else:
                    previsoes.append([rid, result[0], result[1]])
        except Exception as exc:
            logger.warning("Erro rec %d (linha %s): %s", rid, linha, exc)
            n_fallback += 1
            if task == "T1":
                previsoes.append([rid, _dt_to_epoch_ms(ts_query)])
            else:
                previsoes.append([rid, -22.9, -43.2])

    con.close()
    pct_fb = 100 * n_fallback / max(len(previsoes), 1)
    logger.info(
        "Concluído: %d previsões, %d fallbacks (%.1f%%)",
        len(previsoes), n_fallback, pct_fb,
    )
    return previsoes


def process_all_test_zips(
    test_dir: Path,
    db_path: Path = DB_PATH,
    glob: str = "*.zip",
) -> tuple[list[list], list[tuple[Path, str, list[list]]]]:
    """
    Processa todos os ZIPs de teste em test_dir.
    Retorna (all_previsoes, per_file_results) onde per_file_results é uma lista de
    (zip_path, query_filename, previsoes) para validação por arquivo.
    """
    all_previsoes: list[list] = []
    per_file: list[tuple[Path, str, list[list]]] = []

    for zip_path in sorted(test_dir.glob(glob)):
        with zipfile.ZipFile(zip_path) as z:
            query_files = [
                f for f in z.namelist()
                if (f.split("/")[-1].startswith("treino-") or
                    f.split("/")[-1].startswith("teste-"))
                and f.endswith(".json")
                and "__MACOSX" not in f
            ]
        for qf in sorted(query_files):
            prevs = process_test_file(zip_path, qf, db_path)
            all_previsoes.extend(prevs)
            per_file.append((zip_path, qf, prevs))

    logger.info("Total de previsões geradas: %d", len(all_previsoes))
    return all_previsoes, per_file


# ── Submissão via API ─────────────────────────────────────────────────────────

def build_resposta(previsoes: list[list]) -> dict:
    return {
        "aluno": ALUNO,
        "datahora": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "previsoes": previsoes,
        "senha": SENHA,
    }


def submit(resposta: dict, endpoint: str = SUBMIT_URL, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.post(endpoint, json=resposta, verify=False, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning("Tentativa %d falhou (%s) — aguardando %ds...", attempt+1, exc, wait)
                time.sleep(wait)
            else:
                raise


def save_resposta(resposta: dict, version: int = 1) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts_str = datetime.now().strftime("%Y-%m-%d")
    path = RESULTS_DIR / f"resposta-{ts_str}-v{version}.json"
    path.write_text(json.dumps(resposta, ensure_ascii=False, indent=2))
    logger.info("Resposta salva em %s (%d previsões)", path, len(resposta.get("previsoes", [])))
    return path


# ── Validação local com gabarito ──────────────────────────────────────────────

def validate_per_file(zip_path: Path, query_filename: str, previsoes: list[list]) -> dict:
    """
    Compara previsões de UM arquivo de query contra seu resposta correspondente.
    Evita colisão de IDs entre arquivos diferentes do mesmo ZIP.
    """
    pred_dict = {p[0]: p[1:] for p in previsoes}
    results: dict[str, list] = {"T1_mae_s": [], "T2_haversine_m": []}

    # treino-DATE_HH.json → resposta-DATE_HH.json
    stem = Path(query_filename).stem.replace("treino-", "").replace("teste-", "")
    resp_name = f"resposta-{stem}.json"
    dir_part = "/".join(query_filename.split("/")[:-1])
    resp_path = f"{dir_part}/{resp_name}" if dir_part else resp_name

    with zipfile.ZipFile(zip_path) as z:
        if resp_path not in z.namelist():
            return {}
        gabarito = json.loads(z.read(resp_path))
        for item in gabarito:
            rid = item["id"]
            if rid not in pred_dict:
                continue
            if "datahora" in item:
                ts_true = int(item["datahora"]) / 1000
                ts_pred = pred_dict[rid][0] / 1000
                results["T1_mae_s"].append(abs(ts_pred - ts_true))
            elif "latitude" in item:
                if len(pred_dict[rid]) < 2:
                    continue
                from geopy.distance import geodesic
                lat_t = parse_lat_lon(item["latitude"])
                lon_t = parse_lat_lon(item["longitude"])
                lat_p, lon_p = float(pred_dict[rid][0]), float(pred_dict[rid][1])
                d = geodesic((lat_t, lon_t), (lat_p, lon_p)).meters
                results["T2_haversine_m"].append(d)

    summary: dict = {}
    if results["T1_mae_s"]:
        arr = np.array(results["T1_mae_s"])
        summary.update(T1_n=len(arr), T1_mae_s=float(arr.mean()),
                       T1_mae_min=float(arr.mean()/60), T1_p50_s=float(np.median(arr)))
    if results["T2_haversine_m"]:
        arr = np.array(results["T2_haversine_m"])
        summary.update(T2_n=len(arr), T2_mae_m=float(arr.mean()), T2_p50_m=float(np.median(arr)))
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"

    if mode == "validate":
        logger.info("=== Modo: validação (data/raw/teste/) ===")
        all_prevs, per_file = process_all_test_zips(DATA_RAW / "teste")
        resposta  = build_resposta(all_prevs)
        save_resposta(resposta, version=1)

        print("\n=== Métricas de validação (por arquivo) ===")
        t1_all, t2_all = [], []
        for zip_path, qf, prevs in per_file:
            metrics = validate_per_file(zip_path, qf, prevs)
            if metrics:
                lbl = f"{zip_path.name}/{Path(qf).name}"
                print(f"\n{lbl}:")
                for k, v in metrics.items():
                    print(f"  {k}: {v:.2f}")
                if "T1_mae_s" in metrics:
                    t1_all.extend([metrics["T1_mae_s"]] * int(metrics.get("T1_n", 0)))
                if "T2_mae_m" in metrics:
                    t2_all.extend([metrics["T2_mae_m"]] * int(metrics.get("T2_n", 0)))

        if t1_all:
            print(f"\n=== GLOBAL T1 MAE: {np.mean(t1_all):.1f}s ({np.mean(t1_all)/60:.1f} min) ===")
        if t2_all:
            print(f"=== GLOBAL T2 MAE: {np.mean(t2_all):.1f}m ===")

    elif mode == "final":
        logger.info("=== Modo: final (data/raw/teste-final/) ===")
        all_prevs, per_file = process_all_test_zips(DATA_RAW / "teste-final")

        # Salva resposta completa para backup
        resposta_all = build_resposta(all_prevs)
        save_resposta(resposta_all, version=1)

        print(f"\n=== Submetendo {len(per_file)} arquivos de teste à API ===")
        all_scores: list[dict] = []
        for zip_path, qf, prevs in per_file:
            # datahora = timestamp do arquivo de teste, ex: "2024-05-16 08:00:00"
            stem  = Path(qf).stem.replace("teste-", "").replace("treino-", "")
            dh_str = stem.rsplit("_", 1)
            datahora_api = f"{dh_str[0]} {dh_str[1]}:00:00"
            resposta = {
                "aluno": ALUNO,
                "datahora": datahora_api,
                "previsoes": prevs,
                "senha": SENHA,
            }
            try:
                result = submit(resposta)
                result["arquivo"] = Path(qf).name
                all_scores.append(result)
                print(f"  {Path(qf).name}: RMSE={result.get('rmse', 'N/A'):.2f}  "
                      f"ids_testados={result.get('ids testados', '?')}")
            except Exception as exc:
                logger.error("Falha ao submeter %s: %s", qf, exc)
                all_scores.append({"arquivo": Path(qf).name, "erro": str(exc)})

        print(f"\n=== Scores completos ===")
        print(json.dumps(all_scores, indent=2, ensure_ascii=False))

    else:
        print("Uso: python -m src.predict [validate|final]")
        sys.exit(1)

    print("\nCONCLUÍDO")
