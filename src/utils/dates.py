"""
dates.py — Gestion des dates et des maturités.

Conventions utilisées :
  - Toutes les maturités sont en années (Act/365).
  - Les dates d'expiry Deribit sont au format "DMMMYY" (ex: "30MAY25").
  - Le temps à maturité est calculé à partir de datetime.utcnow() sauf indication contraire.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.utils.constants import DAYS_PER_YEAR


# ─────────────────────────────────────────────────────────────────────────────
# Parsing des dates d'expiry Deribit
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_deribit_expiry(expiry_str: str) -> datetime:
    """
    Parse une date d'expiry Deribit au format "DDMMMYY" → datetime UTC.

    Exemple : "30MAY25" → datetime(2025, 5, 30, 8, 0, 0, tzinfo=timezone.utc)

    Sur Deribit, les options expirent à 08:00 UTC.
    """
    m = re.fullmatch(r"(\d{1,2})([A-Z]{3})(\d{2})", expiry_str.strip().upper())
    if m is None:
        raise ValueError(f"Format d'expiry non reconnu: '{expiry_str}'")
    day = int(m.group(1))
    month = _MONTH_MAP[m.group(2)]
    year = 2000 + int(m.group(3))
    return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)


def time_to_maturity(
    expiry: datetime,
    reference: Optional[datetime] = None,
    day_count: int = DAYS_PER_YEAR,
) -> float:
    """
    Calcule la maturité résiduelle en années (Act/365).

    Parameters
    ----------
    expiry      : date d'expiry (datetime avec tzinfo)
    reference   : date de référence (défaut = maintenant UTC)
    day_count   : nombre de jours dans l'année (365 par défaut)

    Returns
    -------
    T en années (float ≥ 0)
    """
    if reference is None:
        reference = datetime.now(tz=timezone.utc)
    delta = expiry - reference
    return max(delta.total_seconds() / (day_count * 86400.0), 0.0)


def shift_maturity(T: float, weeks: float) -> float:
    """
    Réduit la maturité T (en années) d'un certain nombre de semaines.
    Utilisé pour le scénario de stress à 1 semaine.
    """
    return max(T - weeks / 52.0, 0.0)


def now_utc() -> datetime:
    """Retourne l'instant courant en UTC."""
    return datetime.now(tz=timezone.utc)


def expiry_from_timestamp_ms(ts_ms: int) -> datetime:
    """Convertit un timestamp Deribit (ms depuis epoch) en datetime UTC."""
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
