"""Utility functions shared across all T04 modules."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from geopy.distance import geodesic

logger = logging.getLogger(__name__)

# Bounding box do Rio de Janeiro (WGS84)
LAT_MIN, LAT_MAX = -23.5, -22.5
LON_MIN, LON_MAX = -43.9, -43.0

# Linhas alvo (50 linhas da especificação)
TARGET_LINES = {
    "483", "864", "639", "3", "309", "774", "629", "371", "397", "100",
    "838", "315", "624", "388", "918", "665", "328", "497", "878", "355",
    "138", "606", "457", "550", "803", "917", "638", "2336", "399", "298",
    "867", "553", "565", "422", "756", "186012003", "292", "554", "634",
    "232", "415", "2803", "324", "852", "557", "759", "343", "779", "905", "108",
}


def parse_lat_lon(s: str | float | int) -> float:
    """Converte '-22,84753' → -22.84753. Aceita float/int passthrough."""
    if isinstance(s, (int, float)):
        return float(s)
    return float(str(s).replace(",", "."))


def epoch_ms_to_datetime(ms: int | str) -> datetime:
    """Converte epoch em milissegundos → datetime UTC."""
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distância em metros entre dois pontos WGS84."""
    return geodesic((lat1, lon1), (lat2, lon2)).meters


def is_in_bbox(lat: float, lon: float) -> bool:
    """Verifica se o ponto está dentro do bounding box do Rio de Janeiro."""
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX
