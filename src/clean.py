"""
EDA & Visualization Agent — T04: Previsão de Localização de Ônibus
Responsabilidade: aplicar filtros de qualidade e gerar tabela gps_clean no DuckDB.
"""
from __future__ import annotations

import logging

import duckdb

from src.ingest import DB_PATH

logger = logging.getLogger(__name__)

# Filtros documentados em SPEC.md §2.2
HORA_MIN = 8
HORA_MAX = 23
SPEED_MAX = 120
LAT_MIN, LAT_MAX = -23.5, -22.5
LON_MIN, LON_MAX = -43.9, -43.0


def build_gps_clean(db_path=DB_PATH, force: bool = False) -> int:
    """
    Cria tabela gps_clean aplicando filtros de qualidade em ordem.
    Retorna número de registros na tabela limpa.

    Filtros (em ordem de aplicação — SPEC.md §2.2):
      1. Janela operacional: hora 08–23 (horário de Brasília)
      2. Velocidade ≤ 120 km/h
      3. Bounding box RJ: lat [-23.5, -22.5], lon [-43.9, -43.0]
    """
    con = duckdb.connect(str(db_path))

    exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='gps_clean'"
    ).fetchone()[0]

    if exists and not force:
        n = con.execute("SELECT COUNT(*) FROM gps_clean").fetchone()[0]
        logger.info("gps_clean já existe com %d registros. Use force=True para recriar.", n)
        con.close()
        return n

    logger.info("Criando tabela gps_clean...")
    con.execute("DROP TABLE IF EXISTS gps_clean")

    con.execute(f"""
        CREATE TABLE gps_clean AS
        SELECT *
        FROM gps
        WHERE EXTRACT(HOUR FROM ts_servidor AT TIME ZONE 'America/Sao_Paulo')
              BETWEEN {HORA_MIN} AND {HORA_MAX}
          AND velocidade_raw <= {SPEED_MAX}
          AND lat BETWEEN {LAT_MIN} AND {LAT_MAX}
          AND lon BETWEEN {LON_MIN} AND {LON_MAX}
    """)

    # Índices para queries de modelagem
    for ddl in [
        "CREATE INDEX IF NOT EXISTS idx_clean_linha_ts ON gps_clean (linha, ts_servidor)",
        "CREATE INDEX IF NOT EXISTS idx_clean_ordem_ts ON gps_clean (ordem, ts_servidor)",
    ]:
        try:
            con.execute(ddl)
        except duckdb.CatalogException:
            pass

    n_raw = con.execute("SELECT COUNT(*) FROM gps").fetchone()[0]
    n_clean = con.execute("SELECT COUNT(*) FROM gps_clean").fetchone()[0]
    con.close()

    pct_retido = n_clean / n_raw * 100
    pct_descartado = 100 - pct_retido
    logger.info(
        "gps_clean criada: %d registros (%.1f%% retidos, %.1f%% descartados).",
        n_clean, pct_retido, pct_descartado,
    )
    return n_clean


def quality_report(db_path=DB_PATH) -> dict:
    """Retorna métricas de qualidade da tabela gps para exibição no EDA."""
    con = duckdb.connect(str(db_path), read_only=True)
    row = con.execute("""
        SELECT
            COUNT(*)                                                                AS n_total,
            SUM(CASE WHEN velocidade_raw = 0 THEN 1 ELSE 0 END)                    AS n_parado,
            SUM(CASE WHEN velocidade_raw > 120 THEN 1 ELSE 0 END)                  AS n_vel_invalida,
            SUM(CASE WHEN EXTRACT(HOUR FROM ts_servidor
                         AT TIME ZONE 'America/Sao_Paulo') NOT BETWEEN 8 AND 23
                     THEN 1 ELSE 0 END)                                             AS n_fora_janela,
            SUM(CASE WHEN ts_onibus IS NULL THEN 1 ELSE 0 END)                     AS n_sem_ts_onibus,
            ROUND(AVG(
                CASE WHEN ts_onibus IS NOT NULL
                      AND epoch(ts_servidor) - epoch(ts_onibus) BETWEEN 0 AND 3600
                     THEN epoch(ts_servidor) - epoch(ts_onibus)
                END
            ), 1)                                                                   AS lag_medio_s
        FROM gps
    """).fetchone()
    con.close()

    keys = ["n_total", "n_parado", "n_vel_invalida", "n_fora_janela", "n_sem_ts_onibus", "lag_medio_s"]
    return dict(zip(keys, row))


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    report = quality_report()
    print("\n=== Relatório de Qualidade (gps raw) ===")
    for k, v in report.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")

    n = build_gps_clean(force=True)
    print(f"\ngps_clean: {n:,} registros")
