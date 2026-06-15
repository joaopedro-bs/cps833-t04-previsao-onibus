"""
Speed Model Agent — T04: Previsão de Localização de Ônibus
Responsabilidade: calcular perfil de velocidade por segmento × hora e fator de tráfego.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.ingest import DB_PATH
from src.routes import snap_to_route

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
SEGMENT_M       = 500      # comprimento de cada segmento ao longo do trajeto (metros)
MAX_DELTA_S     = 300      # ignorar pares com Δt > 5 min (veículo pode ter parado)
MIN_DELTA_S     = 5        # ignorar pares com Δt < 5 s (provável duplicata)
SPEED_MIN_KMH   = 0.5      # velocidade mínima válida calculada (km/h)
SPEED_MAX_KMH   = 100.0    # velocidade máxima válida calculada (km/h)
MIN_SAMPLES_SEG = 3        # mínimo de amostras para incluir célula no perfil
TRAFFIC_JANELA_H= 2.0      # janela de observação de tráfego recente (horas)
TRAFFIC_MIN_OBS = 3        # mínimo de observações recentes para aplicar fator
TRAFFIC_CLAMP   = (0.3, 2.0)  # limites do fator de ajuste (evitar explosão)
DEFAULT_SPEED_KMH = 20.0   # velocidade default quando sem perfil (urbana RJ)

MAX_SAMPLE_ROWS = 200_000  # amostra máxima de registros por linha para o profile

# ── Utilitários ───────────────────────────────────────────────────────────────

def _seg_idx(dist_acum_m: float) -> int:
    """Retorna o índice do segmento de 500m correspondente à distância acumulada."""
    return int(dist_acum_m // SEGMENT_M)


def _compute_speeds_for_line(
    linha: str, sentido: int, con: duckdb.DuckDBPyConnection
) -> pd.DataFrame:
    """
    Calcula velocidade real (Δdist_acum_m / Δts_servidor) entre pings
    consecutivos do mesmo veículo para uma linha/sentido.

    Retorna DataFrame com colunas: [seg_500m, hora, velocidade_kmh]
    """
    # Carregar rota
    route_df = con.execute(
        "SELECT seq, lat, lon, dist_acum_m FROM route_canonical WHERE linha=? AND sentido=? ORDER BY seq",
        [str(linha), int(sentido)],
    ).fetchdf()
    if route_df.empty:
        return pd.DataFrame()

    # Buscar registros GPS limpos amostrados (SAMPLE não aceita parâmetro ? — interpolar direto)
    df = con.execute(
        f"""
        SELECT ordem, ts_servidor, lat, lon
        FROM (SELECT * FROM gps_clean WHERE linha = ?)
        USING SAMPLE {MAX_SAMPLE_ROWS} ROWS
        ORDER BY ordem, ts_servidor
        """,
        [str(linha)],
    ).fetchdf()

    if df.empty or len(df) < 10:
        return pd.DataFrame()

    # Snap de todos os pontos ao trajeto canônico (vetorizado com KD-Tree)
    from scipy.spatial import cKDTree
    from pyproj import Transformer

    _t = Transformer.from_crs("EPSG:4326", "EPSG:31983", always_xy=True)
    rx, ry = _t.transform(route_df["lon"].to_numpy(), route_df["lat"].to_numpy())
    px, py = _t.transform(df["lon"].to_numpy(), df["lat"].to_numpy())

    tree = cKDTree(np.column_stack([rx, ry]))
    dists_to_route, idxs = tree.query(np.column_stack([px, py]))

    # Filtrar pontos muito fora da rota (> 30m)
    mask_on_route = dists_to_route <= 30.0
    df = df[mask_on_route].copy()
    df["dist_acum_m"] = route_df["dist_acum_m"].iloc[idxs[mask_on_route]].to_numpy()

    if df.empty:
        return pd.DataFrame()

    # Calcular Δt e Δdist entre pings consecutivos do mesmo veículo
    df = df.sort_values(["ordem", "ts_servidor"]).reset_index(drop=True)
    ts_s = df["ts_servidor"].values.astype("datetime64[s]").astype("int64")
    ordem_arr = df["ordem"].to_numpy()
    dist_arr  = df["dist_acum_m"].to_numpy()

    records = []
    for k in range(1, len(df)):
        if ordem_arr[k] != ordem_arr[k - 1]:
            continue
        delta_t = int(ts_s[k]) - int(ts_s[k - 1])
        if delta_t < MIN_DELTA_S or delta_t > MAX_DELTA_S:
            continue
        delta_d = abs(dist_arr[k] - dist_arr[k - 1])
        speed_mps = delta_d / delta_t
        speed_kmh = speed_mps * 3.6
        if not (SPEED_MIN_KMH <= speed_kmh <= SPEED_MAX_KMH):
            continue

        # Hora do dia em Brasília
        ts_dt = datetime.fromtimestamp(int(ts_s[k - 1]), tz=timezone.utc)
        hora_brt = (ts_dt.hour - 3) % 24  # UTC-3

        seg = _seg_idx(dist_arr[k - 1])
        records.append({"seg_500m": seg, "hora": hora_brt, "velocidade_kmh": speed_kmh})

    return pd.DataFrame(records)


def build_speed_profile(
    linha: str,
    sentido: int,
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Agrega velocidades reais em percentis por (segmento × hora).
    Retorna DataFrame com colunas: [linha, sentido, seg_500m, hora, v_p25, v_p50, v_p75, v_p85, n].
    """
    df_speeds = _compute_speeds_for_line(linha, sentido, con)
    if df_speeds.empty:
        logger.warning("Linha %s sentido %d: sem amostras de velocidade.", linha, sentido)
        return pd.DataFrame()

    agg = (
        df_speeds.groupby(["seg_500m", "hora"])["velocidade_kmh"]
        .agg(
            n="count",
            v_p25=lambda x: float(np.percentile(x, 25)),
            v_p50=lambda x: float(np.percentile(x, 50)),
            v_p75=lambda x: float(np.percentile(x, 75)),
            v_p85=lambda x: float(np.percentile(x, 85)),
        )
        .reset_index()
    )
    agg = agg[agg["n"] >= MIN_SAMPLES_SEG].copy()
    agg["linha"]   = str(linha)
    agg["sentido"] = int(sentido)

    return agg[["linha", "sentido", "seg_500m", "hora", "v_p25", "v_p50", "v_p75", "v_p85", "n"]]


def save_speed_profile(profile_df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> int:
    """Insere perfil na tabela speed_profile (após limpar registros anteriores da linha/sentido)."""
    if profile_df.empty:
        return 0
    linha   = str(profile_df["linha"].iloc[0])
    sentido = int(profile_df["sentido"].iloc[0])

    con.execute(
        "DELETE FROM speed_profile WHERE linha=? AND sentido=?",
        [linha, sentido],
    )
    con.executemany(
        """
        INSERT INTO speed_profile
            (linha, sentido, seg_km, hora, velocidade_p50, velocidade_p85, n_amostras)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (linha, sentido, int(r["seg_500m"]), int(r["hora"]),
             float(r["v_p50"]), float(r["v_p85"]), int(r["n"]))
            for _, r in profile_df.iterrows()
        ],
    )
    return len(profile_df)


def build_all_speed_profiles(db_path: Path = DB_PATH) -> dict[str, str]:
    """
    Constrói e salva perfis de velocidade para todas as linhas com route_canonical.
    Retorna dict chave="<linha>_s<sentido>" → status.
    """
    con = duckdb.connect(str(db_path))
    _ensure_speed_profile_schema(con)

    linhas = [r[0] for r in con.execute(
        "SELECT DISTINCT linha FROM route_canonical ORDER BY linha"
    ).fetchall()]

    status: dict[str, str] = {}
    for linha in linhas:
        for sentido in (1, 2):
            key = f"{linha}_s{sentido}"
            try:
                profile = build_speed_profile(linha, sentido, con)
                if profile.empty:
                    status[key] = "sem_dados"
                    continue
                n = save_speed_profile(profile, con)
                total_obs = int(profile["n"].sum())
                logger.info(
                    "Linha %s sentido %d: %d células (seg×hora), %d observações.",
                    linha, sentido, n, total_obs,
                )
                status[key] = "ok"
            except Exception as exc:
                logger.error("Linha %s sentido %d: erro — %s", linha, sentido, exc)
                status[key] = "erro"

    con.close()
    return status


def _ensure_speed_profile_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Garante que colunas extras (v_p25, v_p75) existam na tabela speed_profile."""
    for col, typ in [("velocidade_p25", "DOUBLE"), ("velocidade_p75", "DOUBLE")]:
        try:
            con.execute(f"ALTER TABLE speed_profile ADD COLUMN {col} {typ}")
        except Exception:
            pass  # já existe


# ── Estimativa de ETA ─────────────────────────────────────────────────────────

def get_speed_estimate(
    linha: str,
    sentido: int,
    dist_acum_m: float,
    hora: int,
    con: duckdb.DuckDBPyConnection,
) -> float:
    """
    Retorna velocidade estimada (km/h) para um ponto no trajeto na hora dada.
    Busca o segmento mais próximo com dados; fallback = DEFAULT_SPEED_KMH.
    """
    seg = _seg_idx(dist_acum_m)

    # Buscar exato; se não tiver, expandir para ± 2 segmentos e ± 1 hora
    row = con.execute(
        """
        SELECT velocidade_p50
        FROM speed_profile
        WHERE linha=? AND sentido=? AND seg_km=? AND hora=?
        LIMIT 1
        """,
        [str(linha), int(sentido), seg, int(hora)],
    ).fetchone()

    if row and row[0] is not None:
        return float(row[0])

    # Vizinhança (± 2 seg, ± 1 hora)
    hora_min = max(0, int(hora) - 1)
    hora_max = min(23, int(hora) + 1)
    row = con.execute(
        """
        SELECT AVG(velocidade_p50)
        FROM speed_profile
        WHERE linha=? AND sentido=?
          AND seg_km BETWEEN ? AND ?
          AND hora BETWEEN ? AND ?
        """,
        [str(linha), int(sentido), max(0, seg - 2), seg + 2, hora_min, hora_max],
    ).fetchone()

    if row and row[0] is not None:
        return float(row[0])

    return DEFAULT_SPEED_KMH


def compute_traffic_factor(
    linha: str,
    sentido: int,
    dist_acum_m: float,
    ts_now: datetime,
    con: duckdb.DuckDBPyConnection,
    db_path: Path = DB_PATH,
    janela_h: float = TRAFFIC_JANELA_H,
) -> float:
    """
    Estima fator de congestionamento comparando velocidade recente com histórico.

    Busca observações reais das últimas `janela_h` horas no segmento atual.
    Retorna fator = v_recente_p50 / v_historica_p50.
    Se n_recente < TRAFFIC_MIN_OBS → retorna 1.0 (sem ajuste).
    Clamp: [0.3, 2.0].
    """
    seg = _seg_idx(dist_acum_m)
    hora = (ts_now.hour - 3) % 24  # UTC-3

    # Velocidade histórica para este seg/hora
    v_hist = get_speed_estimate(linha, sentido, dist_acum_m, hora, con)

    # Velocidade recente (últimas janela_h horas) — requer conexão R/W para query de dados
    ts_cutoff_ms = int((ts_now.timestamp() - janela_h * 3600) * 1000)
    ts_now_ms    = int(ts_now.timestamp() * 1000)

    # Usar db_path separado para não conflitar com a conexão principal
    con2 = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con2.execute(
            """
            SELECT COUNT(*), AVG(velocidade_raw)
            FROM gps_clean
            WHERE linha=?
              AND epoch(ts_servidor)*1000 BETWEEN ? AND ?
              AND velocidade_raw BETWEEN 1 AND 120
            """,
            [str(linha), ts_cutoff_ms, ts_now_ms],
        ).fetchone()
    finally:
        con2.close()

    n_recente = int(row[0]) if row and row[0] else 0
    if n_recente < TRAFFIC_MIN_OBS:
        return 1.0

    v_recente = float(row[1])
    if v_hist <= 0:
        return 1.0

    fator = v_recente / v_hist
    return float(np.clip(fator, *TRAFFIC_CLAMP))


def estimate_eta_seconds(
    linha: str,
    sentido: int,
    dist_atual_m: float,
    dist_alvo_m: float,
    ts_now: datetime,
    con: duckdb.DuckDBPyConnection,
    db_path: Path = DB_PATH,
) -> float:
    """
    Estima tempo (segundos) para percorrer de dist_atual_m até dist_alvo_m.

    Integra o speed_profile segmento a segmento, aplicando fator de tráfego recente.
    Retorna número de segundos estimados (≥ 0).
    """
    if dist_alvo_m <= dist_atual_m:
        return 0.0

    traffic_factor = compute_traffic_factor(
        linha, sentido, dist_atual_m, ts_now, con, db_path
    )

    total_segundos = 0.0
    pos = dist_atual_m
    hora = (ts_now.hour - 3) % 24

    while pos < dist_alvo_m:
        # Próxima fronteira de segmento
        seg_end = (_seg_idx(pos) + 1) * SEGMENT_M
        trecho = min(seg_end, dist_alvo_m) - pos

        v_kmh = get_speed_estimate(linha, sentido, pos, hora, con)
        v_ajust = max(v_kmh * traffic_factor, 1.0)  # mínimo 1 km/h
        v_mps = v_ajust / 3.6

        total_segundos += trecho / v_mps
        pos = seg_end

        # Avançar hora se necessário (ETA pode cruzar hora cheia)
        hora = (hora + int(total_segundos // 3600)) % 24

    return total_segundos


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("=== Speed Model Agent — build_all_speed_profiles ===")
    results = build_all_speed_profiles()

    ok    = [k for k, v in results.items() if v == "ok"]
    sem   = [k for k, v in results.items() if v == "sem_dados"]
    erros = [k for k, v in results.items() if v == "erro"]

    print(f"\n=== STATUS ===")
    print(f"ok: {len(ok)} | sem_dados: {len(sem)} | erro: {len(erros)}")
    if sem:
        print("sem_dados:", sorted(sem))
    if erros:
        print("erros:", sorted(erros))

    # Verificar tabela
    con = duckdb.connect(str(DB_PATH), read_only=True)
    total_rows = con.execute("SELECT COUNT(*) FROM speed_profile").fetchone()[0]
    total_obs  = con.execute("SELECT SUM(n_amostras) FROM speed_profile").fetchone()[0]
    linhas_ok  = con.execute("SELECT COUNT(DISTINCT linha) FROM speed_profile").fetchone()[0]
    low_n      = con.execute(
        "SELECT linha, sentido, COUNT(*), SUM(n_amostras) FROM speed_profile "
        "GROUP BY linha, sentido HAVING SUM(n_amostras) < 50 ORDER BY 4"
    ).fetchall()
    con.close()

    print(f"\nLinhas no speed_profile: {linhas_ok}")
    print(f"Células (seg×hora):      {total_rows:,}")
    print(f"Observações totais:      {total_obs:,}")
    if low_n:
        print("\nLinhas com < 50 amostras totais (alerta):")
        for row in low_n:
            print(f"  linha={row[0]} s{row[1]}: {row[2]} células, {row[3]} obs")

    print("\nCONCLUÍDO")
