"""
validation.py — Validation des contraintes minimales sur les données d'options.

Chaque règle est une fonction retournant un masque booléen pandas.
Le rapport de nettoyage est un dict ordonné : raison → nb de lignes rejetées.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Règles individuelles (renvoient un masque "invalide")
# ─────────────────────────────────────────────────────────────────────────────

def mask_invalid_bid_ask(df: pd.DataFrame) -> pd.Series:
    """bid > ask ou bid < 0."""
    return (df["bid"] > df["ask"]) | (df["bid"] < 0)


def mask_zero_price(df: pd.DataFrame) -> pd.Series:
    """mid price nul ou négatif."""
    return df["mid"] <= 0


def mask_zero_strike(df: pd.DataFrame) -> pd.Series:
    """Strike nul ou négatif."""
    return df["strike"] <= 0


def mask_negative_maturity(df: pd.DataFrame) -> pd.Series:
    """Maturité négative ou nulle."""
    return df["T"] <= 0


def mask_short_maturity(df: pd.DataFrame, min_T: float) -> pd.Series:
    """Maturité trop courte (< min_T ans)."""
    return df["T"] < min_T


def mask_long_maturity(df: pd.DataFrame, max_T: float) -> pd.Series:
    """Maturité trop longue (> max_T ans)."""
    return df["T"] > max_T


def mask_wide_spread(df: pd.DataFrame, max_spread_pct: float) -> pd.Series:
    """
    Spread bid-ask trop large.

    Critère :
      - Si bid ET ask > 0 : (ask - bid) / mid > max_spread_pct → rejet
      - Si bid=0 OU ask=0 mais mark_price > 0 : accepté (cotation mark uniquement,
        situation normale sur Deribit quand il n'y a pas de market maker actif)
      - Si mid = 0 : rejet (aucun prix disponible)
    """
    has_two_sided = (df["bid"] > 0) & (df["ask"] > 0)
    spread_pct = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)
    wide = spread_pct > max_spread_pct
    no_price = df["mid"] <= 0
    # Rejet uniquement si : (cotation two-sided mais spread trop large) OU (aucun prix du tout)
    return (has_two_sided & wide.fillna(True)) | no_price


def mask_extreme_moneyness(
    df: pd.DataFrame, min_moneyness: float, max_moneyness: float
) -> pd.Series:
    """K/F hors des bornes [min_moneyness, max_moneyness]."""
    m = df["moneyness"]
    return (m < min_moneyness) | (m > max_moneyness) | m.isna()


def mask_negative_intrinsic(df: pd.DataFrame) -> pd.Series:
    """
    Détecte les options en dessous de leur valeur intrinsèque actualisée.

    On utilise une borne inférieure conservative : e^{-r*T} * max(0, F-K)
    avec r ≈ 5% (estimation grossière). Pour les options européennes crypto,
    la borne réelle est encore plus basse, donc on ne rejette que les cas
    clairement aberrants (en dessous de 90% de la valeur intrinsèque actualisée).
    """
    import numpy as np
    r_approx = 0.05
    F = df["forward_price"]
    K = df["strike"]
    T = df["T"].clip(lower=1e-4)
    mid = df["mid"]
    discount = np.exp(-r_approx * T)
    intrinsic = np.where(
        df["option_type"] == "C",
        np.maximum(0.0, F - K),
        np.maximum(0.0, K - F),
    )
    return mid < discount * intrinsic * 0.90  # tolérance de 10%


# ─────────────────────────────────────────────────────────────────────────────
# Rapport de nettoyage complet
# ─────────────────────────────────────────────────────────────────────────────

def compute_spread_pct(df: pd.DataFrame) -> pd.Series:
    """Calcule (ask - bid) / mid pour chaque ligne."""
    return (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)


def run_validation_report(
    df: pd.DataFrame,
    max_spread_pct: float = 0.25,
    min_T: float = 1 / 365.0,
    max_T: float = 3.0,
    min_moneyness: float = 0.50,
    max_moneyness: float = 2.00,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Applique toutes les règles de validation et retourne :
      - le DataFrame filtré (lignes valides)
      - un rapport dict[raison → nb_rejetées]
    """
    n_initial = len(df)
    report: dict[str, int] = {"initial": n_initial}

    # On ajoute la colonne spread_pct pour visualisation
    df = df.copy()
    df["spread_pct"] = compute_spread_pct(df)

    # Masque global des invalides (union progressive pour comptage individuel)
    invalid = pd.Series(False, index=df.index)

    def _apply(mask: pd.Series, label: str) -> None:
        new_invalid = mask & ~invalid
        report[label] = int(new_invalid.sum())
        invalid.__ior__(mask)

    _apply(mask_invalid_bid_ask(df),                              "bid > ask ou bid < 0")
    _apply(mask_zero_price(df),                                   "prix nul ou négatif")
    _apply(mask_zero_strike(df),                                  "strike ≤ 0")
    _apply(mask_negative_maturity(df),                            "maturité ≤ 0")
    _apply(mask_short_maturity(df, min_T),                        f"maturité < {min_T:.4f}a")
    _apply(mask_long_maturity(df, max_T),                         f"maturité > {max_T}a")
    _apply(mask_wide_spread(df, max_spread_pct),                  f"spread > {max_spread_pct*100:.0f}% du mid")
    _apply(mask_extreme_moneyness(df, min_moneyness, max_moneyness), f"moneyness hors [{min_moneyness}, {max_moneyness}]")
    _apply(mask_negative_intrinsic(df),                           "prix < valeur intrinsèque")

    df_clean = df[~invalid].reset_index(drop=True)
    report["retenues"] = len(df_clean)
    report["total_rejetées"] = n_initial - len(df_clean)

    logger.info(
        "Validation : %d initiales → %d retenues (%d rejetées)",
        n_initial, len(df_clean), report["total_rejetées"],
    )
    return df_clean, report
