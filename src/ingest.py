"""
Data Ingestion Agent — T04: Previsão de Localização de Ônibus
Responsabilidade: descompactar ZIPs, normalizar registros GPS, inserir no DuckDB.

Contrato:
  - normalize_record() converte lat/lon com vírgula e epoch ms
  - load_zip() é idempotente: rodar duas vezes não duplica registros
  - Descartados são logados por motivo (nunca silenciosamente)
"""
from __future__ import annotations

import json
import logging
import zipfile
from collections import defaultdict
from pathlib import Path

import duckdb

from src.utils import (
    LAT_MAX, LAT_MIN, LON_MAX, LON_MIN,
    TARGET_LINES,
    epoch_ms_to_datetime,
    is_in_bbox,
    parse_lat_lon,
)

logger = logging.getLogger(__name__)

DB_PATH = Path("data/processed/bus.duckdb")

# Limites de velocidade (km/h)
SPEED_MIN = 0
SPEED_MAX = 120


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Cria tabelas conforme SPEC.md §6 (idempotente)."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS gps (
            ordem           VARCHAR,
            linha           VARCHAR,
            lat             DOUBLE,
            lon             DOUBLE,
            ts_servidor     TIMESTAMP WITH TIME ZONE,
            ts_onibus       TIMESTAMP WITH TIME ZONE,
            velocidade_raw  INTEGER,
            arquivo_origem  VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS route_canonical (
            linha        VARCHAR,
            sentido      INTEGER,
            seq          INTEGER,
            lat          DOUBLE,
            lon          DOUBLE,
            dist_acum_m  DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS terminals (
            linha    VARCHAR,
            sentido  INTEGER,
            lat      DOUBLE,
            lon      DOUBLE,
            raio_m   DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS speed_profile (
            linha          VARCHAR,
            sentido        INTEGER,
            seg_km         INTEGER,
            hora           INTEGER,
            velocidade_p50 DOUBLE,
            velocidade_p85 DOUBLE,
            n_amostras     INTEGER
        )
    """)
    # Índices analíticos recomendados em SPEC.md §3
    for ddl in [
        "CREATE INDEX IF NOT EXISTS idx_gps_linha_ts ON gps (linha, ts_servidor)",
        "CREATE INDEX IF NOT EXISTS idx_gps_ordem_ts ON gps (ordem, ts_servidor)",
        "CREATE INDEX IF NOT EXISTS idx_route_linha  ON route_canonical (linha, sentido, seq)",
    ]:
        try:
            con.execute(ddl)
        except duckdb.CatalogException:
            pass  # índice já existe


def normalize_record(raw: dict) -> dict | None:
    """
    Normaliza um registro GPS bruto.

    Retorna None (com motivo logado externamente) se o registro for inválido.
    Motivos de descarte:
      - lat/lon fora do bounding box do Rio
      - timestamp server inválido (≤ 0 ou ausente)
      - velocidade fora de [0, 120] km/h
      - linha ausente na lista das 50 linhas alvo
    """
    try:
        lat = parse_lat_lon(raw["latitude"])
        lon = parse_lat_lon(raw["longitude"])
    except (KeyError, ValueError):
        return None

    try:
        ts_ms = int(raw["datahoraservidor"])
        if ts_ms <= 0:
            return None
        ts_servidor = epoch_ms_to_datetime(ts_ms)
    except (KeyError, ValueError, TypeError):
        return None

    try:
        ts_onibus = epoch_ms_to_datetime(int(raw["datahora"]))
    except (KeyError, ValueError, TypeError):
        ts_onibus = None

    try:
        velocidade = int(raw.get("velocidade", 0))
    except (ValueError, TypeError):
        velocidade = 0

    linha = str(raw.get("linha", "")).strip()
    ordem = str(raw.get("ordem", "")).strip()

    return {
        "ordem": ordem,
        "linha": linha,
        "lat": lat,
        "lon": lon,
        "ts_servidor": ts_servidor,
        "ts_onibus": ts_onibus,
        "velocidade_raw": velocidade,
    }


def _classify_discard(raw: dict, normalized: dict | None) -> str | None:
    """Retorna motivo de descarte ou None se o registro deve ser inserido."""
    if normalized is None:
        return "parse_error"

    if not is_in_bbox(normalized["lat"], normalized["lon"]):
        return "fora_bbox"

    if normalized["linha"] not in TARGET_LINES:
        return "linha_nao_alvo"

    v = normalized["velocidade_raw"]
    if not (SPEED_MIN <= v <= SPEED_MAX):
        return "velocidade_invalida"

    return None


def _load_json_records(
    z: zipfile.ZipFile,
    fname: str,
    arquivo_origem: str,
) -> tuple[list[tuple], dict[str, int]]:
    """
    Lê um único JSON dentro do ZIP e retorna (registros_validos, contagem_descartados).
    registros_validos é lista de tuplas prontas para INSERT no DuckDB.
    """
    raw_bytes = z.read(fname)
    if not raw_bytes.strip():
        return [], {"vazio": 1}

    try:
        records = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        logger.warning("JSON inválido em %s/%s: %s", arquivo_origem, fname, exc)
        return [], {"json_invalido": 1}

    if not isinstance(records, list):
        return [], {"formato_inesperado": 1}

    valid_rows: list[tuple] = []
    discards: dict[str, int] = defaultdict(int)

    for raw in records:
        norm = normalize_record(raw)
        motivo = _classify_discard(raw, norm)
        if motivo:
            discards[motivo] += 1
        else:
            valid_rows.append((
                norm["ordem"],
                norm["linha"],
                norm["lat"],
                norm["lon"],
                norm["ts_servidor"],
                norm["ts_onibus"],
                norm["velocidade_raw"],
                arquivo_origem,
            ))

    return valid_rows, dict(discards)


def load_zip(zip_path: Path, db_path: Path = DB_PATH) -> int:
    """
    Carrega um arquivo ZIP de GPS para o DuckDB.

    Idempotente: registros do mesmo arquivo_origem já presentes são ignorados.
    Retorna número de registros inseridos nesta chamada.
    """
    zip_path = Path(zip_path)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    arquivo_origem = zip_path.name

    con = duckdb.connect(str(db_path))
    create_schema(con)

    # Idempotência: checar se este arquivo já foi carregado
    existing = con.execute(
        "SELECT COUNT(*) FROM gps WHERE arquivo_origem = ?", [arquivo_origem]
    ).fetchone()[0]
    if existing > 0:
        logger.info(
            "ZIP %s já carregado (%d registros). Pulando.", arquivo_origem, existing
        )
        con.close()
        return 0

    total_inserted = 0
    total_discards: dict[str, int] = defaultdict(int)

    with zipfile.ZipFile(zip_path) as z:
        json_files = [
            f for f in z.namelist()
            if f.endswith(".json")
            and not f.split("/")[-1].startswith(("treino-", "resposta-", "teste-", "._"))
            and "__MACOSX" not in f
        ]

        if not json_files:
            logger.warning("Nenhum JSON de GPS encontrado em %s", zip_path)
            con.close()
            return 0

        for fname in sorted(json_files):
            rows, discards = _load_json_records(z, fname, arquivo_origem)

            if rows:
                con.executemany(
                    """
                    INSERT INTO gps
                        (ordem, linha, lat, lon, ts_servidor, ts_onibus,
                         velocidade_raw, arquivo_origem)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                total_inserted += len(rows)

            for motivo, n in discards.items():
                total_discards[motivo] += n

    con.close()

    if total_discards:
        discard_summary = ", ".join(
            f"{motivo}={n}" for motivo, n in sorted(total_discards.items())
        )
        logger.info(
            "ZIP %s: %d inseridos | descartados: %s",
            arquivo_origem, total_inserted, discard_summary,
        )
    else:
        logger.info("ZIP %s: %d inseridos | 0 descartados", arquivo_origem, total_inserted)

    return total_inserted


def load_all_zips(data_dir: Path, db_path: Path = DB_PATH, glob: str = "**/*.zip") -> int:
    """Carrega todos os ZIPs de uma pasta (e subpastas). Retorna total inserido."""
    data_dir = Path(data_dir)
    zips = sorted(data_dir.glob(glob))
    logger.info("Encontrados %d ZIPs em %s", len(zips), data_dir)

    total = 0
    for z in zips:
        total += load_zip(z, db_path)

    logger.info("Ingestão completa: %d registros inseridos no total.", total)
    return total


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if len(sys.argv) < 2:
        print("Uso: python -m src.ingest <caminho_zip_ou_pasta> [db_path]")
        sys.exit(1)

    target = Path(sys.argv[1])
    db = Path(sys.argv[2]) if len(sys.argv) > 2 else DB_PATH

    if target.is_dir():
        n = load_all_zips(target, db)
    else:
        n = load_zip(target, db)

    # Resumo pós-ingestão
    con = duckdb.connect(str(db))
    total_db = con.execute("SELECT COUNT(*) FROM gps").fetchone()[0]
    by_origin = con.execute(
        "SELECT arquivo_origem, COUNT(*) as n FROM gps GROUP BY arquivo_origem ORDER BY arquivo_origem"
    ).fetchall()
    con.close()

    print(f"\n=== Resumo ===")
    print(f"Inseridos nesta chamada : {n}")
    print(f"Total no banco          : {total_db}")
    print(f"\nRegistros por arquivo:")
    for origem, cnt in by_origin:
        print(f"  {origem}: {cnt:,}")
