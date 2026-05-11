"""
put_call_parity.py — Extraction des taux sans risque implicites via la parité call-put.

Méthode :
  La parité call-put pour une option européenne sans dividende sur un sous-jacent
  coté via un forward F_T est :

      C - P = (F_T - K) · e^{-r·T}

  D'où :
      r(T) = -ln[(C - P) / (F_T - K)] / T      pour F_T ≠ K

  Pratiquement :
    - On agrège sur plusieurs strikes ATM-proches pour robustesse.
    - On utilise la médiane comme estimateur robuste par maturité.
    - On élimine les paires avec F_T ≈ K (dénominateur instable).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.transforms import get_call_put_pairs

logger = logging.getLogger(__name__)

# Seuil minimal |F - K| pour éviter la division par zéro (en % du forward)
_MIN_FMK_PCT = 0.005   # 0.5%
_MAX_IV_RATE = 0.50    # 50% comme borne haute pour écarter les anomalies


def extract_implied_rates(
    df_clean: pd.DataFrame,
    min_fmk_pct: float = _MIN_FMK_PCT,
    aggregation: str = "median",
) -> pd.DataFrame:
    """
    Extrait les taux sans risque implicites par maturité.

    Parameters
    ----------
    df_clean      : DataFrame options nettoyé (colonnes standard)
    min_fmk_pct   : seuil minimum |F-K|/F pour inclure une paire
    aggregation   : "median" (robuste) ou "mean"

    Returns
    -------
    DataFrame trié par T avec colonnes :
      expiry_str, T, rate, n_pairs_used, rate_std
    """
    pairs = get_call_put_pairs(df_clean)
    if pairs.empty:
        logger.warning("Aucune paire call-put trouvée.")
        return pd.DataFrame(columns=["expiry_str", "T", "rate", "n_pairs_used", "rate_std"])

    # Calcul du taux pour chaque paire
    F = pairs["forward_price"]
    K = pairs["strike"]
    T = pairs["T"]
    C = pairs["mid_c"]
    P = pairs["mid_p"]

    # Filtre : |F - K| / F suffisant et T > 0
    fmk = F - K
    valid = (
        (T > 0)
        & (np.abs(fmk) / F > min_fmk_pct)
        & (C > 0) & (P > 0)
    )
    pairs_valid = pairs[valid].copy()

    if pairs_valid.empty:
        logger.warning("Aucune paire valide pour l'extraction des taux.")
        return pd.DataFrame(columns=["expiry_str", "T", "rate", "n_pairs_used", "rate_std"])

    F_v = pairs_valid["forward_price"]
    K_v = pairs_valid["strike"]
    T_v = pairs_valid["T"]
    C_v = pairs_valid["mid_c"]
    P_v = pairs_valid["mid_p"]

    # r = -ln[(C - P) / (F - K)] / T
    ratio = (C_v - P_v) / (F_v - K_v)

    # On élimine les ratios aberrants (> 0 requis pour ln)
    ok = (ratio > 0) & np.isfinite(ratio)
    pairs_valid = pairs_valid[ok].copy()
    ratio_ok = ratio[ok]
    T_ok = T_v[ok]

    rate_raw = -np.log(ratio_ok) / T_ok

    # Bornes réalistes [-20%, +50%]
    ok2 = (rate_raw > -0.20) & (rate_raw < _MAX_IV_RATE)
    pairs_valid = pairs_valid[ok2].copy()
    rate_raw = rate_raw[ok2]

    pairs_valid = pairs_valid.copy()
    pairs_valid["implied_rate"] = rate_raw.values

    # Agrégation par maturité
    agg_fn = np.median if aggregation == "median" else np.mean
    records = []
    for (expiry_str, T_val), grp in pairs_valid.groupby(["expiry_str", "T"]):
        rates = grp["implied_rate"].values
        records.append({
            "expiry_str": expiry_str,
            "T": float(T_val),
            "rate": float(agg_fn(rates)),
            "rate_std": float(np.std(rates)) if len(rates) > 1 else 0.0,
            "n_pairs_used": len(rates),
        })

    df_rates = pd.DataFrame(records).sort_values("T").reset_index(drop=True)
    logger.info(
        "Taux implicites extraits : %d maturités (médiane=%.4f)",
        len(df_rates),
        df_rates["rate"].median() if not df_rates.empty else 0.0,
    )
    return df_rates
