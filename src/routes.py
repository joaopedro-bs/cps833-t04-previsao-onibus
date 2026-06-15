"""
Route Modeling Agent — T04: Previsão de Localização de Ônibus
Responsabilidade: extrair trajeto canônico por linha/sentido e salvar em route_canonical.

Pipeline por linha/sentido (AGENTS.md):
  1. Buscar pontos GPS limpos no DuckDB
  2. Segmentar viagens por pausa > 5 min (detecção de terminal)
  3. Separar ida (sentido 1) e volta (sentido 2) via direção predominante por viagem
  4. Aplicar DBSCAN espacial (eps=30m UTM-23S, min_samples=5)
  5. Ordenar clusters por timestamp mediano
  6. Suavizar com rolling average (janela 5)
  7. Calcular dist_acum_m via geodesic WGS84
  8. Salvar em route_canonical no DuckDB
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from sklearn.cluster import DBSCAN

from src.ingest import DB_PATH
from src.utils import TARGET_LINES

logger = logging.getLogger(__name__)

# Constantes — SPEC.md + AGENTS.md
UTM_CRS          = "EPSG:31983"   # UTM-23S (Sul do Brasil)
WGS84_CRS        = "EPSG:4326"
DBSCAN_EPS_M     = 30             # raio em metros para DBSCAN
DBSCAN_MIN_SAMP  = 5              # mínimo de amostras por cluster
MIN_POINTS       = 500            # mínimo de pontos limpos para gerar trajeto
SMOOTH_WINDOW    = 5              # janela do rolling average
TRIP_GAP_S       = 300            # pausa > 5 min = nova viagem (detecção de terminal)
MAX_SAMPLE_PTS   = 80_000         # teto de pontos carregados do DB por linha/sentido

# Transformadores UTM-23S ↔ WGS84
_to_utm   = Transformer.from_crs(WGS84_CRS, UTM_CRS, always_xy=True)
_from_utm = Transformer.from_crs(UTM_CRS, WGS84_CRS, always_xy=True)


def _to_utm_coords(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = _to_utm.transform(lon, lat)   # always_xy: (lon, lat) → (easting, northing)
    return x, y


def _from_utm_coords(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = _from_utm.transform(x, y)
    return lat, lon


def _segment_trips(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segmenta trajetórias de cada veículo em viagens separadas por pausas > TRIP_GAP_S.
    Adiciona coluna 'trip_id' (único por veículo × viagem).
    """
    df = df.sort_values(["ordem", "ts_servidor"]).copy()
    df["ts_epoch"] = df["ts_servidor"].astype(np.int64) // 1_000_000_000
    gap = df.groupby("ordem")["ts_epoch"].diff().fillna(0) > TRIP_GAP_S
    df["trip_id"] = df["ordem"] + "_" + gap.groupby(df["ordem"]).cumsum().astype(str)
    return df


def _assign_sentidos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Atribui sentido (1 ou 2) a cada ponto baseado na direção predominante da viagem.

    Sentido 1 = direção predominante dos viagens (maior variação de lat ou lon).
    Sentido 2 = direção oposta.
    """
    # ΔLat por viagem (último - primeiro ponto)
    trip_dlat = df.groupby("trip_id").apply(
        lambda g: g["lat"].iloc[-1] - g["lat"].iloc[0] if len(g) > 1 else 0.0
    )
    trip_dlon = df.groupby("trip_id").apply(
        lambda g: g["lon"].iloc[-1] - g["lon"].iloc[0] if len(g) > 1 else 0.0
    )

    # Usar ΔLat se variação é dominante, senão ΔLon
    use_lat = trip_dlat.abs().median() >= trip_dlon.abs().median()

    if use_lat:
        trip_dir = trip_dlat
    else:
        trip_dir = trip_dlon

    # Sentido 1 = direção mais frequente (moda de sinal)
    median_dir = trip_dir.median()
    # Mapeia: viagem alinhada com direção predominante → sentido 1
    trip_sentido = (trip_dir * (1 if median_dir >= 0 else -1) >= 0).map({True: 1, False: 2})
    df["sentido"] = df["trip_id"].map(trip_sentido).fillna(1).astype(int)
    return df


def _dbscan_filter(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Aplica DBSCAN em UTM-23S (metros) e retorna máscara de pontos core/border.
    Retorna índices dos pontos pertencentes a algum cluster (não ruído).
    """
    x, y = _to_utm_coords(lat, lon)
    coords_utm = np.column_stack([x, y])
    labels = DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_SAMP, n_jobs=-1).fit_predict(coords_utm)
    return labels >= 0   # True = pertence a cluster


GRID_CELL_M = 150   # tamanho da célula do grid espacial (metros)


def _grid_route_centroids(
    lat: np.ndarray, lon: np.ndarray, ts_epoch: np.ndarray,
    cell_m: int = GRID_CELL_M, reverse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrai centroides de rota via grid espacial regular (células de `cell_m` metros em UTM-23S).

    Para cada célula ocupada:
      - centróide = média das coordenadas UTM dos pontos naquela célula
      - ts_mediano = mediana dos timestamps (usado para ordenação do trajeto)

    Retorna (lat_centroids, lon_centroids) ordenados por ts_mediano.
    """
    x, y = _to_utm_coords(lat, lon)

    # Primeiro remover outliers via DBSCAN (eps=30m — conforme SPEC)
    coords = np.column_stack([x, y])
    labels_dbscan = DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_SAMP, n_jobs=-1).fit_predict(coords)
    mask_ok = labels_dbscan >= 0
    if mask_ok.sum() < 50:
        return lat, lon     # muito poucos pontos válidos — devolver bruto

    x, y, ts_epoch = x[mask_ok], y[mask_ok], ts_epoch[mask_ok]

    # Grid espacial
    xi = (x // cell_m).astype(int)
    yi = (y // cell_m).astype(int)
    cell_id = xi.astype(np.int64) * 10_000_000 + yi.astype(np.int64)

    # Agregar por célula
    df_grid = pd.DataFrame({"xi": xi, "yi": yi, "cell": cell_id, "ts": ts_epoch})
    agg = df_grid.groupby("cell").agg(
        xi_mean=("xi", "mean"), yi_mean=("yi", "mean"), ts_med=("ts", "median"), n=("ts", "count")
    ).reset_index()
    agg = agg[agg["n"] >= 2]   # descarta células visitadas apenas uma vez

    if len(agg) < 5:
        return lat, lon

    # Centralizar células de volta para UTM reais
    cx = agg["xi_mean"].to_numpy() * cell_m + cell_m / 2
    cy = agg["yi_mean"].to_numpy() * cell_m + cell_m / 2

    # Ordenar por nearest-neighbor greedy a partir de um extremo do eixo PCA
    # (timestamp médio não funciona: múltiplos ônibus distribuídos ao longo do trajeto)
    n = len(cx)
    coords_c = np.column_stack([cx, cy])
    center_c = coords_c.mean(axis=0)
    cov_c = np.cov((coords_c - center_c).T)
    eigvals, eigvecs = np.linalg.eigh(cov_c)
    principal = eigvecs[:, eigvals.argmax()]
    proj = (coords_c - center_c) @ principal
    start = int(np.argmax(proj) if reverse else np.argmin(proj))

    visited = np.zeros(n, dtype=bool)
    order_list = [start]
    visited[start] = True
    for _ in range(n - 1):
        curr = order_list[-1]
        dists = (cx - cx[curr]) ** 2 + (cy - cy[curr]) ** 2
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        order_list.append(nxt)
        visited[nxt] = True

    cx, cy = cx[np.array(order_list)], cy[np.array(order_list)]
    clat, clon = _from_utm_coords(cx, cy)
    return clat, clon


def _smooth_route(lat: np.ndarray, lon: np.ndarray, window: int = SMOOTH_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Suaviza sequência de pontos com rolling average."""
    lat_s = pd.Series(lat).rolling(window, center=True, min_periods=1).mean().to_numpy()
    lon_s = pd.Series(lon).rolling(window, center=True, min_periods=1).mean().to_numpy()
    return lat_s, lon_s


def _cumulative_distance(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Calcula distância acumulada em metros via geodesic WGS84."""
    dist = [0.0]
    for i in range(1, len(lat)):
        d = geodesic((lat[i-1], lon[i-1]), (lat[i], lon[i])).meters
        dist.append(dist[-1] + d)
    return np.array(dist)


def extract_canonical_route(
    linha: str,
    sentido: int,
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame | None:
    """
    Extrai e retorna o trajeto canônico de uma linha/sentido como DataFrame.

    Colunas: [seq, lat, lon, dist_acum_m]
    Retorna None se a linha tiver pontos insuficientes.
    """
    # 1. Carregar pontos do DuckDB (subquery garante sample APÓS filtro por linha)
    rows = con.execute(f"""
        SELECT lat, lon, ts_servidor, ordem
        FROM (SELECT lat, lon, ts_servidor, ordem FROM gps_clean WHERE linha = ?)
        USING SAMPLE {MAX_SAMPLE_PTS} ROWS
    """, [linha]).fetchall()

    if len(rows) < MIN_POINTS:
        logger.warning("Linha %s: apenas %d pontos — pulando (mínimo: %d).", linha, len(rows), MIN_POINTS)
        return None

    df = pd.DataFrame(rows, columns=["lat", "lon", "ts_servidor", "ordem"])

    # 2. Segmentar viagens
    df = _segment_trips(df)

    # 3. Separar sentidos
    df = _assign_sentidos(df)

    # 4. Filtrar pelo sentido desejado
    df_s = df[df["sentido"] == sentido].copy()
    if len(df_s) < MIN_POINTS:
        logger.warning(
            "Linha %s sentido %d: apenas %d pontos após separação — pulando.",
            linha, sentido, len(df_s)
        )
        return None

    # 5. Grid espacial → centroides ordenados por timestamp mediano
    lat_arr = df_s["lat"].to_numpy()
    lon_arr = df_s["lon"].to_numpy()
    ts_arr  = df_s["ts_epoch"].to_numpy()

    clat, clon = _grid_route_centroids(lat_arr, lon_arr, ts_arr, reverse=(sentido == 2))
    if len(clat) < 10:
        logger.warning("Linha %s sentido %d: apenas %d waypoints — pulando.", linha, sentido, len(clat))
        return None

    # 6. Suavizar
    clat, clon = _smooth_route(clat, clon)

    # 7. Distância acumulada
    dist_acum = _cumulative_distance(clat, clon)

    route_df = pd.DataFrame({
        "linha":       linha,
        "sentido":     sentido,
        "seq":         np.arange(len(clat)),
        "lat":         clat,
        "lon":         clon,
        "dist_acum_m": dist_acum,
    })
    return route_df


def save_route(route_df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> None:
    """Insere trajeto canônico no DuckDB (remove existente primeiro)."""
    linha   = str(route_df["linha"].iloc[0])
    sentido = int(route_df["sentido"].iloc[0])
    con.execute("DELETE FROM route_canonical WHERE linha = ? AND sentido = ?", [linha, sentido])
    con.executemany(
        "INSERT INTO route_canonical (linha, sentido, seq, lat, lon, dist_acum_m) VALUES (?,?,?,?,?,?)",
        route_df[["linha", "sentido", "seq", "lat", "lon", "dist_acum_m"]].values.tolist(),
    )


def snap_to_route(
    lat: float, lon: float, route_df: pd.DataFrame
) -> tuple[float, float, float]:
    """
    Projeta um ponto GPS no ponto mais próximo do trajeto canônico.
    Usa KD-Tree em UTM-23S para eficiência.
    Retorna (lat_snap, lon_snap, dist_acum_m).
    """
    rx, ry = _to_utm_coords(route_df["lat"].to_numpy(), route_df["lon"].to_numpy())
    px, py = _to_utm_coords(np.array([lat]), np.array([lon]))

    tree = cKDTree(np.column_stack([rx, ry]))
    _, idx = tree.query([px[0], py[0]])

    return (
        route_df["lat"].iloc[idx],
        route_df["lon"].iloc[idx],
        route_df["dist_acum_m"].iloc[idx],
    )


def process_all_lines(db_path: Path = DB_PATH) -> dict[str, str]:
    """
    Extrai trajetos canônicos para todas as 50 linhas alvo (ambos sentidos).
    Retorna dict linha → status ('ok' | 'sem_dados' | 'poucos_pontos' | 'erro').
    """
    con = duckdb.connect(str(db_path))
    linhas_db = {r[0] for r in con.execute("SELECT DISTINCT linha FROM gps_clean").fetchall()}

    status: dict[str, str] = {}
    ausentes = sorted(TARGET_LINES - linhas_db)
    for l in ausentes:
        logger.warning("Linha %s: ausente no conjunto de treino — ignorando.", l)
        status[l] = "sem_dados"

    linhas_ok = sorted(TARGET_LINES & linhas_db)
    logger.info("Processando %d linhas...", len(linhas_ok))

    for linha in linhas_ok:
        try:
            ok_sentidos = 0
            for sentido in (1, 2):
                route_df = extract_canonical_route(linha, sentido, con)
                if route_df is not None:
                    save_route(route_df, con)
                    ok_sentidos += 1
                    logger.info(
                        "Linha %s sentido %d: %d pontos, %.1f km",
                        linha, sentido, len(route_df),
                        route_df["dist_acum_m"].iloc[-1] / 1000,
                    )
            status[linha] = "ok" if ok_sentidos > 0 else "poucos_pontos"
        except Exception as exc:
            logger.error("Linha %s: erro — %s", linha, exc, exc_info=True)
            status[linha] = "erro"

    con.close()
    return status


def build_route_map(linha: str, db_path: Path = DB_PATH) -> object:
    """
    Gera mapa folium com trajeto canônico da linha (ida=azul, volta=vermelho).
    Salva em figures/mapa_rota_{linha}.html e retorna o objeto folium.Map.
    """
    import folium

    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute(
        "SELECT sentido, lat, lon FROM route_canonical WHERE linha = ? ORDER BY sentido, seq",
        [linha]
    ).fetchall()
    con.close()

    if not rows:
        logger.warning("Linha %s: sem trajeto canônico para mapear.", linha)
        return None

    df = pd.DataFrame(rows, columns=["sentido", "lat", "lon"])
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")
    colors = {1: "blue", 2: "red"}
    labels = {1: f"Linha {linha} — Ida", 2: f"Linha {linha} — Volta"}

    for sentido, grp in df.groupby("sentido"):
        coords = list(zip(grp["lat"], grp["lon"]))
        folium.PolyLine(
            coords, color=colors[sentido], weight=4, opacity=0.8,
            tooltip=labels[sentido]
        ).add_to(m)
        if coords:
            folium.Marker(coords[0],  icon=folium.Icon(color=colors[sentido], icon="play"),
                          tooltip=f"{labels[sentido]} — início").add_to(m)
            folium.Marker(coords[-1], icon=folium.Icon(color=colors[sentido], icon="stop"),
                          tooltip=f"{labels[sentido]} — fim").add_to(m)

    path = Path("figures") / f"mapa_rota_{linha}.html"
    m.save(str(path))
    logger.info("Mapa salvo: %s", path)
    return m


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    linhas_arg = sys.argv[1:] if len(sys.argv) > 1 else None

    con = duckdb.connect(str(DB_PATH))
    results: dict[str, str] = {}

    target = linhas_arg if linhas_arg else sorted(TARGET_LINES)
    linhas_db = {r[0] for r in con.execute("SELECT DISTINCT linha FROM gps_clean").fetchall()}

    for linha in target:
        if linha not in linhas_db:
            logger.warning("Linha %s: ausente no treino — pulando.", linha)
            results[linha] = "sem_dados"
            continue
        try:
            ok = 0
            for sentido in (1, 2):
                rd = extract_canonical_route(linha, sentido, con)
                if rd is not None:
                    save_route(rd, con)
                    ok += 1
            results[linha] = "ok" if ok > 0 else "poucos_pontos"
        except Exception as e:
            logger.error("Linha %s: %s", linha, e, exc_info=True)
            results[linha] = "erro"

    con.close()

    # Gerar mapas para as linhas processadas com sucesso
    for linha, st in results.items():
        if st == "ok":
            build_route_map(linha)

    print("\n=== Resultado ===")
    for st in ("ok", "poucos_pontos", "sem_dados", "erro"):
        linhas_st = [l for l, s in results.items() if s == st]
        if linhas_st:
            print(f"  {st}: {linhas_st}")
