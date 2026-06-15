"""
Direction & Terminal Agent — T04: Previsão de Localização de Ônibus
Responsabilidade: detectar terminais, inicializar e atualizar estado de veículos.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from geopy.distance import geodesic as geodesic_dist
from pyproj import Transformer
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from src.ingest import DB_PATH
from src.routes import snap_to_route

logger = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────────────────────
UTM_CRS = "EPSG:31983"          # UTM-23S (metros, projeção plana para DBSCAN)
WGS84   = "EPSG:4326"

TERMINAL_MIN_PAUSA_S   = 10 * 60   # pausa mínima para considerar terminal (10 min)
TERMINAL_MAX_PAUSA_S   = 30 * 60   # pausa máxima razoável antes de garagem (30 min)
TERMINAL_PING_GAP_S    = 300       # gap máximo entre pings dentro de um episódio
TERMINAL_DBSCAN_EPS_M  = 150       # eps DBSCAN para detectar cluster de terminal
TERMINAL_DBSCAN_MIN    = 5         # min_samples DBSCAN
TERMINAL_RAIO_M        = 100       # raio mínimo de chegada ao terminal
SNAP_GEOCERCA_M        = 30        # distância máxima ao trajeto para snap válido

_utm_transformer = Transformer.from_crs(WGS84, UTM_CRS, always_xy=True)


def _to_utm(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = _utm_transformer.transform(lon, lat)
    return x, y


# ── Estado do veículo ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VehicleState:
    linha: str
    sentido: int                          # 1=ida, 2=volta; 0=FORA_DE_ROTA
    dist_acum_m: float                    # posição no trajeto canônico (metros)
    ts_last_update: datetime
    em_terminal: bool
    ts_chegada_terminal: Optional[datetime]


# ── Helpers de acesso ao banco ────────────────────────────────────────────────

def _load_route(
    linha: str, sentido: int, con: duckdb.DuckDBPyConnection
) -> pd.DataFrame | None:
    """Carrega trajeto canônico da tabela route_canonical."""
    df = con.execute(
        """
        SELECT seq, lat, lon, dist_acum_m
        FROM route_canonical
        WHERE linha = ? AND sentido = ?
        ORDER BY seq
        """,
        [str(linha), int(sentido)],
    ).fetchdf()
    return df if not df.empty else None


def _load_terminals(
    linha: str, sentido: int, con: duckdb.DuckDBPyConnection
) -> list[dict]:
    """Carrega terminais ordenados por seq (início → fim do sentido)."""
    rows = con.execute(
        """
        SELECT lat, lon, raio_m, seq
        FROM terminals
        WHERE linha = ? AND sentido = ?
        ORDER BY seq
        """,
        [str(linha), int(sentido)],
    ).fetchdf()
    return rows.to_dict("records") if not rows.empty else []


def _ensure_terminals_seq_column(con: duckdb.DuckDBPyConnection) -> None:
    """Adiciona coluna seq à tabela terminals se ainda não existir."""
    try:
        con.execute("ALTER TABLE terminals ADD COLUMN seq INTEGER DEFAULT 0")
    except Exception:
        pass  # já existe


# ── Detecção de terminais ─────────────────────────────────────────────────────

def _find_long_stops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica pings de velocidade=0 que fazem parte de episódios de parada
    com duração >= TERMINAL_MIN_PAUSA_S.

    Agrupa pings consecutivos do mesmo veículo (gap < TERMINAL_PING_GAP_S)
    em episódios e filtra pelos de duração suficiente.
    """
    resultado = []
    for _, grp in df.groupby("ordem", sort=False):
        grp = grp.sort_values("ts_servidor").reset_index(drop=True)

        # Converter timestamps para segundos inteiros (seguro para qualquer precisão)
        ts_s = grp["ts_servidor"].values.astype("datetime64[s]").astype("int64")

        ep_ids = np.zeros(len(grp), dtype=int)
        ep_id = 0
        for k in range(1, len(grp)):
            if ts_s[k] - ts_s[k - 1] > TERMINAL_PING_GAP_S:
                ep_id += 1
            ep_ids[k] = ep_id
        grp["_ep"] = ep_ids

        # Duração por episódio
        ep_stats = grp.groupby("_ep")["ts_servidor"].agg(["min", "max"])
        ep_stats["dur_s"] = (ep_stats["max"] - ep_stats["min"]).dt.total_seconds()
        ep_stats = ep_stats[ep_stats["dur_s"] >= TERMINAL_MIN_PAUSA_S]

        qualificados = grp[grp["_ep"].isin(ep_stats.index)]
        if not qualificados.empty:
            resultado.append(qualificados)

    if not resultado:
        return pd.DataFrame(columns=["lat", "lon"])
    return pd.concat(resultado, ignore_index=True)


def _terminal_fallback(
    linha: str, sentido: int, route_df: pd.DataFrame, con: duckdb.DuckDBPyConnection
) -> list[dict]:
    """Fallback: primeiro e último ponto do trajeto canônico como terminais."""
    terminals = [
        {"lat": float(route_df.iloc[0]["lat"]),  "lon": float(route_df.iloc[0]["lon"]),
         "raio_m": float(TERMINAL_RAIO_M), "seq": 0},
        {"lat": float(route_df.iloc[-1]["lat"]), "lon": float(route_df.iloc[-1]["lon"]),
         "raio_m": float(TERMINAL_RAIO_M), "seq": 1},
    ]
    for t in terminals:
        con.execute(
            "INSERT INTO terminals (linha, sentido, lat, lon, raio_m, seq) VALUES (?,?,?,?,?,?)",
            [str(linha), int(sentido), t["lat"], t["lon"], t["raio_m"], t["seq"]],
        )
    logger.info("Linha %s sentido %d: terminais por fallback (extremos do trajeto).", linha, sentido)
    return terminals


def detect_terminals(
    linha: str,
    sentido: int,
    db_path: Path = DB_PATH,
) -> list[dict]:
    """
    Detecta os 2 terminais de uma linha/sentido.

    Estratégia:
    1. Busca todos os pontos com velocidade_raw=0 para a linha.
    2. Agrupa pings consecutivos por veículo em episódios de parada.
    3. Mantém apenas episódios com duração >= 10 min.
    4. Aplica DBSCAN (eps=150m em UTM-23S) nos pontos qualificados.
    5. Seleciona os 2 clusters mais distantes entre si como terminais.
    6. Ordena pelo dist_acum_m no trajeto canônico (início → fim do sentido).
    7. Salva na tabela terminals com seq=0 (início) e seq=1 (fim).

    Retorna lista de dicts: [{"lat", "lon", "raio_m", "seq"}, ...]
    """
    con = duckdb.connect(str(db_path))
    _ensure_terminals_seq_column(con)

    # Apaga terminais anteriores desta linha/sentido
    con.execute(
        "DELETE FROM terminals WHERE linha = ? AND sentido = ?",
        [str(linha), int(sentido)],
    )

    route_df = _load_route(linha, sentido, con)
    if route_df is None:
        logger.warning("Linha %s sentido %d: sem trajeto canônico.", linha, sentido)
        con.close()
        return []

    # Amostrar até 30k pontos para evitar OOM no DBSCAN (linha pode ter 500k+ parados)
    df_parado = con.execute(
        """
        SELECT ordem, ts_servidor, lat, lon
        FROM (SELECT * FROM gps_clean WHERE linha = ? AND velocidade_raw = 0)
        USING SAMPLE 30000 ROWS
        ORDER BY ordem, ts_servidor
        """,
        [str(linha)],
    ).fetchdf()

    if df_parado.empty:
        logger.warning("Linha %s: sem pontos de parada.", linha)
        result = _terminal_fallback(linha, sentido, route_df, con)
        con.close()
        return result

    pts_long = _find_long_stops(df_parado)
    if len(pts_long) < TERMINAL_DBSCAN_MIN * 2:
        logger.warning("Linha %s: poucos stop-episodes (%d pts) → fallback.", linha, len(pts_long))
        result = _terminal_fallback(linha, sentido, route_df, con)
        con.close()
        return result

    lat_arr = pts_long["lat"].to_numpy(dtype=float)
    lon_arr = pts_long["lon"].to_numpy(dtype=float)
    x, y = _to_utm(lat_arr, lon_arr)
    coords_utm = np.column_stack([x, y])

    labels = DBSCAN(
        eps=TERMINAL_DBSCAN_EPS_M, min_samples=TERMINAL_DBSCAN_MIN
    ).fit_predict(coords_utm)

    valid_labels = [lbl for lbl in set(labels) if lbl != -1]
    if len(valid_labels) < 2:
        logger.warning(
            "Linha %s: DBSCAN retornou %d cluster(s) → fallback.", linha, len(valid_labels)
        )
        result = _terminal_fallback(linha, sentido, route_df, con)
        con.close()
        return result

    # Centróides + raio p95 por cluster
    centroids = []
    for lbl in valid_labels:
        mask = labels == lbl
        cx, cy = x[mask].mean(), y[mask].mean()
        clat, clon = lat_arr[mask].mean(), lon_arr[mask].mean()
        dists = np.sqrt((x[mask] - cx) ** 2 + (y[mask] - cy) ** 2)
        raio = float(max(np.percentile(dists, 95), TERMINAL_RAIO_M))
        centroids.append({"cx": cx, "cy": cy, "lat": clat, "lon": clon, "raio_m": raio})

    # Dois clusters mais distantes entre si
    cent_coords = np.array([[c["cx"], c["cy"]] for c in centroids])
    dist_matrix = cdist(cent_coords, cent_coords)
    idx_i, idx_j = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
    selected = [centroids[int(idx_i)], centroids[int(idx_j)]]
    sep_km = dist_matrix[idx_i, idx_j] / 1000

    # Ordenar por dist_acum_m no trajeto canônico (seq=0 = início do sentido)
    d0 = snap_to_route(selected[0]["lat"], selected[0]["lon"], route_df)[2]
    d1 = snap_to_route(selected[1]["lat"], selected[1]["lon"], route_df)[2]
    if d1 < d0:
        selected = [selected[1], selected[0]]

    for seq, term in enumerate(selected):
        con.execute(
            "INSERT INTO terminals (linha, sentido, lat, lon, raio_m, seq) VALUES (?,?,?,?,?,?)",
            [str(linha), int(sentido), term["lat"], term["lon"], term["raio_m"], seq],
        )

    con.close()
    logger.info(
        "Linha %s sentido %d: 2 terminais detectados (separação: %.1f km).",
        linha, sentido, sep_km,
    )
    return [{"lat": t["lat"], "lon": t["lon"], "raio_m": t["raio_m"], "seq": i}
            for i, t in enumerate(selected)]


def detect_all_terminals(db_path: Path = DB_PATH) -> dict[str, str]:
    """
    Detecta terminais para todas as linhas com route_canonical.
    Retorna dict chave="<linha>_s<sentido>" → status.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    linhas = [r[0] for r in con.execute(
        "SELECT DISTINCT linha FROM route_canonical ORDER BY linha"
    ).fetchall()]
    con.close()

    status: dict[str, str] = {}
    for linha in linhas:
        for sentido in (1, 2):
            key = f"{linha}_s{sentido}"
            try:
                terms = detect_terminals(linha, sentido, db_path)
                status[key] = "ok" if terms else "vazio"
            except Exception as exc:
                logger.error("Linha %s sentido %d: erro — %s", linha, sentido, exc)
                status[key] = "erro"

    return status


# ── Estado de veículo ─────────────────────────────────────────────────────────

def _snap_best_sentido(
    linha: str, lat: float, lon: float, con: duckdb.DuckDBPyConnection
) -> tuple[int, float, float]:
    """
    Projeta a posição nos dois sentidos e retorna (melhor_sentido, dist_acum_m, dist_ao_trajeto_m).
    """
    best_sentido = 1
    best_dist_acum = 0.0
    best_dist_route = float("inf")

    for sentido in (1, 2):
        route_df = _load_route(linha, sentido, con)
        if route_df is None:
            continue
        lat_s, lon_s, dist_acum = snap_to_route(lat, lon, route_df)
        dist_route = geodesic_dist((lat, lon), (lat_s, lon_s)).meters
        if dist_route < best_dist_route:
            best_dist_route = dist_route
            best_sentido = sentido
            best_dist_acum = dist_acum

    return best_sentido, best_dist_acum, best_dist_route


def initialize_vehicle_state(
    ordem: str,
    linha: str,
    ts: datetime,
    lat: float,
    lon: float,
    con: duckdb.DuckDBPyConnection,
) -> VehicleState:
    """
    Cria VehicleState inicial projetando a posição GPS no trajeto canônico.
    Testa ambos os sentidos e escolhe o com menor distância ao trajeto.
    Retorna sentido=0 (FORA_DE_ROTA) se distância > SNAP_GEOCERCA_M.
    """
    sentido, dist_acum, dist_route = _snap_best_sentido(linha, lat, lon, con)

    if dist_route > SNAP_GEOCERCA_M:
        logger.debug("Veículo %s linha %s: FORA_DE_ROTA (%.0fm ao trajeto).", ordem, linha, dist_route)
        return VehicleState(
            linha=linha,
            sentido=0,
            dist_acum_m=0.0,
            ts_last_update=ts,
            em_terminal=False,
            ts_chegada_terminal=None,
        )

    return VehicleState(
        linha=linha,
        sentido=sentido,
        dist_acum_m=dist_acum,
        ts_last_update=ts,
        em_terminal=False,
        ts_chegada_terminal=None,
    )


def update_vehicle_state(
    state: VehicleState,
    lat: float,
    lon: float,
    ts: datetime,
    con: duckdb.DuckDBPyConnection,
) -> VehicleState:
    """
    Atualiza VehicleState com nova observação GPS.
    Detecta chegada ao terminal e inverte sentido após pausa >= 10 min.
    Retorna novo estado imutável.
    """
    # Se FORA_DE_ROTA, tentar re-projetar
    if state.sentido == 0:
        sentido, dist_acum, dist_route = _snap_best_sentido(state.linha, lat, lon, con)
        if dist_route > SNAP_GEOCERCA_M:
            return replace(state, ts_last_update=ts)
        return VehicleState(
            linha=state.linha,
            sentido=sentido,
            dist_acum_m=dist_acum,
            ts_last_update=ts,
            em_terminal=False,
            ts_chegada_terminal=None,
        )

    route_df = _load_route(state.linha, state.sentido, con)
    if route_df is None:
        return replace(state, ts_last_update=ts)

    lat_s, lon_s, dist_acum = snap_to_route(lat, lon, route_df)
    dist_route = geodesic_dist((lat, lon), (lat_s, lon_s)).meters

    # Verificar proximidade ao terminal de chegada ANTES da geocerca.
    # Terminais são loops de retorno que ficam tipicamente fora do trajeto canônico.
    terminals = _load_terminals(state.linha, state.sentido, con)
    em_terminal_agora = False
    if terminals:
        # Terminal de chegada: seq=1 (fim do sentido atual)
        term_chegada = next((t for t in terminals if t["seq"] == 1), terminals[-1])
        dist_term = geodesic_dist(
            (lat, lon), (term_chegada["lat"], term_chegada["lon"])
        ).meters
        em_terminal_agora = dist_term <= term_chegada["raio_m"]

    # Se off-route e não no terminal → FORA_DE_ROTA
    if dist_route > SNAP_GEOCERCA_M and not em_terminal_agora:
        return replace(state, sentido=0, ts_last_update=ts)

    if em_terminal_agora and not state.em_terminal:
        # Entrando no terminal agora
        return replace(
            state,
            dist_acum_m=dist_acum,
            ts_last_update=ts,
            em_terminal=True,
            ts_chegada_terminal=ts,
        )

    if em_terminal_agora and state.em_terminal and state.ts_chegada_terminal is not None:
        # Permanece no terminal — checar se pausa >= 10 min
        pausa_s = (ts - state.ts_chegada_terminal).total_seconds()
        if pausa_s >= TERMINAL_MIN_PAUSA_S:
            novo_sentido = 2 if state.sentido == 1 else 1
            logger.info(
                "Linha %s: inversão de sentido %d→%d após %.1f min no terminal.",
                state.linha, state.sentido, novo_sentido, pausa_s / 60,
            )
            return VehicleState(
                linha=state.linha,
                sentido=novo_sentido,
                dist_acum_m=dist_acum,
                ts_last_update=ts,
                em_terminal=False,
                ts_chegada_terminal=None,
            )
        return replace(state, dist_acum_m=dist_acum, ts_last_update=ts)

    # Fora do terminal ou saiu sem pausa suficiente
    return replace(
        state,
        dist_acum_m=dist_acum,
        ts_last_update=ts,
        em_terminal=False,
        ts_chegada_terminal=None,
    )


# ── Entry point de teste ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    LINHA_TESTE = "483"   # Linha bem representada nos dados
    logger.info("=== Teste direction.py — Linha %s ===", LINHA_TESTE)

    # 1. Detectar terminais
    logger.info("-- Detectando terminais sentido 1 --")
    terms1 = detect_terminals(LINHA_TESTE, 1)
    logger.info("Terminais sentido 1: %s", terms1)

    logger.info("-- Detectando terminais sentido 2 --")
    terms2 = detect_terminals(LINHA_TESTE, 2)
    logger.info("Terminais sentido 2: %s", terms2)

    # 2. Simular inversão de sentido ao chegar no terminal
    if terms1:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        term_fim_s1 = terms1[-1]    # terminal de fim do sentido 1

        # Estado: veículo no meio do trajeto, sentido 1
        route_df = _load_route(LINHA_TESTE, 1, con)
        mid_idx = len(route_df) // 2
        lat_meio = float(route_df.iloc[mid_idx]["lat"])
        lon_meio = float(route_df.iloc[mid_idx]["lon"])
        ts0 = datetime(2024, 5, 15, 10, 0, 0)

        state = initialize_vehicle_state("VH001", LINHA_TESTE, ts0, lat_meio, lon_meio, con)
        logger.info("Estado inicial: sentido=%d, dist_acum=%.0f m", state.sentido, state.dist_acum_m)

        # Simular chegada ao terminal de fim (sentido 1)
        lat_term = term_fim_s1["lat"]
        lon_term = term_fim_s1["lon"]

        ts1 = datetime(2024, 5, 15, 10, 30, 0)
        state = update_vehicle_state(state, lat_term, lon_term, ts1, con)
        logger.info("Ao chegar no terminal: sentido=%d, em_terminal=%s", state.sentido, state.em_terminal)

        # Simular 11 minutos depois ainda no terminal
        ts2 = datetime(2024, 5, 15, 10, 41, 0)
        state = update_vehicle_state(state, lat_term, lon_term, ts2, con)
        logger.info("Após 11 min no terminal: sentido=%d, em_terminal=%s", state.sentido, state.em_terminal)

        con.close()

        if state.sentido == 2:
            logger.info("SUCESSO: sentido invertido de 1 → 2 após pausa no terminal.")
        else:
            logger.warning("ATENÇÃO: sentido não inverteu. Verificar raio do terminal.")
    else:
        logger.warning("Sem terminais detectados para linha %s — verificar dados.", LINHA_TESTE)
