"""
loaders.py — Chargement et structuration des données brutes Deribit.

Transforme les réponses brutes de l'API en DataFrames propres et typés.
Conventions :
  - Prix en USD (les prix BTC-quotés sont convertis via le spot)
  - Maturité T en années (Act/365)
  - Moneyness log-forward k = ln(K / F_T)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.utils.dates import parse_deribit_expiry, time_to_maturity, now_utc

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing des noms d'instruments Deribit
# ─────────────────────────────────────────────────────────────────────────────

_OPTION_PATTERN = re.compile(
    r"^(?P<ccy>[A-Z]+)-(?P<expiry>\d{1,2}[A-Z]{3}\d{2})-(?P<strike>\d+)-(?P<type>[CP])$"
)


def parse_instrument_name(name: str) -> dict[str, Any] | None:
    """
    Parse un nom d'instrument Deribit de type 'BTC-30MAY25-50000-C'.

    Returns None si le format est invalide.
    """
    m = _OPTION_PATTERN.match(name.strip().upper())
    if m is None:
        return None
    expiry_dt = parse_deribit_expiry(m.group("expiry"))
    return {
        "instrument_name": name,
        "currency": m.group("ccy"),
        "expiry_str": m.group("expiry"),
        "expiry_dt": expiry_dt,
        "strike": float(m.group("strike")),
        "option_type": m.group("type"),  # "C" ou "P"
    }


# ─────────────────────────────────────────────────────────────────────────────
# Construction du DataFrame options
# ─────────────────────────────────────────────────────────────────────────────

def build_options_dataframe(
    raw_options: list[dict],
    spot: float,
    futures_data: list[dict] | None = None,
    reference_dt: datetime | None = None,
) -> pd.DataFrame:
    """
    Construit un DataFrame structuré à partir des données brutes de l'API.

    Colonnes produites :
      instrument_name, currency, expiry_str, expiry_dt, strike,
      option_type, T, bid, ask, mid, mark_price,
      underlying_price, forward_price, log_moneyness, moneyness

    Parameters
    ----------
    raw_options   : liste de dicts issus de get_book_summary_by_currency
    spot          : prix spot de l'index (USD)
    futures_data  : résumés des futures pour inférer les forward prices
    reference_dt  : instant de référence pour calculer T (défaut = maintenant)
    """
    if reference_dt is None:
        reference_dt = now_utc()

    # Construire un mapping expiry → prix forward à partir des futures
    forward_map: dict[str, float] = _build_forward_map(futures_data or [], spot)

    rows = []
    for item in raw_options:
        name = item.get("instrument_name", "")
        parsed = parse_instrument_name(name)
        if parsed is None:
            logger.debug("Nom d'instrument ignoré : %s", name)
            continue

        T = time_to_maturity(parsed["expiry_dt"], reference_dt)
        expiry_str = parsed["expiry_str"]

        # Prix en BTC (fraction du sous-jacent) → USD
        # Sur Deribit, bid_price/ask_price/mark_price pour les options BTC
        # sont exprimés en BTC (ex: 0.0617 BTC). On convertit en USD via underlying_price.
        # NOTE : l'API retourne "bid_price"/"ask_price", PAS "best_bid_price"/"best_ask_price".
        bid_raw = _to_float(item.get("bid_price", 0.0))
        ask_raw = _to_float(item.get("ask_price", 0.0))
        mark_raw = _to_float(item.get("mark_price", 0.0))
        und_raw = _to_float(item.get("underlying_price", spot))
        mark_iv_raw = _to_float(item.get("mark_iv", 0.0))           # IV Deribit (%)
        und_index = item.get("underlying_index", "index_price")      # nom du future ou 'index_price'

        # Conversion BTC → USD (underlying_price est en USD/BTC)
        underlying = und_raw if und_raw > 0 else spot
        bid = bid_raw * underlying if bid_raw > 0 else 0.0
        ask = ask_raw * underlying if ask_raw > 0 else 0.0
        mark = mark_raw * underlying

        # mid = (bid+ask)/2 si two-sided, sinon mid_price API, sinon mark_price
        api_mid_raw = _to_float(item.get("mid_price", 0.0))
        api_mid = api_mid_raw * underlying if api_mid_raw > 0 else 0.0
        if bid > 0 and ask > 0:
            mid = 0.5 * (bid + ask)
        elif api_mid > 0:
            mid = api_mid
        else:
            mid = mark

        # Forward price :
        #   - Si underlying_index != 'index_price' → underlying_price EST le forward du future Deribit
        #   - Sinon → utilise le future coté (forward_map) ou le spot comme fallback
        if und_index != "index_price" and und_raw > 0:
            forward = und_raw   # forward officiel Deribit pour cette maturité
        else:
            forward = forward_map.get(expiry_str, und_raw if und_raw > 0 else spot)

        log_moneyness = np.log(parsed["strike"] / forward) if forward > 0 else np.nan

        rows.append({
            **parsed,
            "T": T,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "mark_price": mark,
            "mark_iv": mark_iv_raw,                        # IV Deribit (% annualisé)
            "underlying_price": und_raw if und_raw > 0 else spot,
            "underlying_index": und_index,
            "spot": spot,
            "forward_price": forward,
            "log_moneyness": log_moneyness,
            "moneyness": parsed["strike"] / forward if forward > 0 else np.nan,
            "open_interest": _to_float(item.get("open_interest", 0.0)),
            "volume": _to_float(item.get("volume", 0.0)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("Aucun instrument option parsé correctement.")
        return df

    df = df.sort_values(["expiry_dt", "option_type", "strike"]).reset_index(drop=True)
    logger.info("Options DataFrame : %d lignes, %d maturités", len(df), df["expiry_str"].nunique())
    return df


def build_futures_dataframe(
    raw_futures: list[dict],
    spot: float,
    reference_dt: datetime | None = None,
) -> pd.DataFrame:
    """
    Construit un DataFrame des futures actifs.

    Colonnes : instrument_name, expiry_str, expiry_dt, T, mid, bid, ask, mark_price
    """
    if reference_dt is None:
        reference_dt = now_utc()

    rows = []
    for item in raw_futures:
        name = item.get("instrument_name", "")
        # Format futures : "BTC-30MAY25" ou "BTC-PERPETUAL"
        if "PERPETUAL" in name:
            continue  # On ignore le perpetual
        parts = name.split("-")
        if len(parts) < 2:
            continue
        try:
            expiry_dt = parse_deribit_expiry(parts[1])
        except Exception:
            continue

        T = time_to_maturity(expiry_dt, reference_dt)
        bid = _to_float(item.get("bid_price", 0.0))
        ask = _to_float(item.get("ask_price", 0.0))
        mark = _to_float(item.get("mark_price", 0.0))
        mid = 0.5 * (bid + ask) if (bid > 0 and ask > 0) else mark

        rows.append({
            "instrument_name": name,
            "expiry_str": parts[1],
            "expiry_dt": expiry_dt,
            "T": T,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "mark_price": mark,
            "spot": spot,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty:
        df = df.sort_values("T").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires internes
# ─────────────────────────────────────────────────────────────────────────────

def _build_forward_map(futures_data: list[dict], spot: float) -> dict[str, float]:
    """
    Construit un dictionnaire expiry_str → forward_price à partir des futures.
    Si aucun future n'est disponible pour une maturité, on utilise le spot.
    """
    forward_map: dict[str, float] = {}
    for item in futures_data:
        name = item.get("instrument_name", "")
        if "PERPETUAL" in name:
            continue
        parts = name.split("-")
        if len(parts) < 2:
            continue
        expiry_str = parts[1]
        mark = _to_float(item.get("mark_price", 0.0))
        bid_f  = _to_float(item.get("bid_price", 0.0))
        ask_f  = _to_float(item.get("ask_price", 0.0))
        mid  = 0.5*(bid_f+ask_f) if (bid_f>0 and ask_f>0) else mark
        if mid > 0:
            forward_map[expiry_str] = mid
    return forward_map


def _to_float(value: Any, default: float = 0.0) -> float:
    """Conversion sûre vers float."""
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default
