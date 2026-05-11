"""
implied_vol.py — Extraction de la volatilité implicite point par point.

Deux méthodes implémentées conformément au sujet :
  1. Newton-Raphson  : convergence rapide (~5 itérations) quand le vega est grand
  2. Dichotomie      : robuste, toujours convergente si l'intervalle est bien défini

Logique de sélection :
  - On tente Newton-Raphson en premier.
  - Si Newton ne converge pas (vega trop faible, divergence, hors bornes),
    on bascule sur la dichotomie.
  - La volatilité implicite est bornée dans [iv_lower, iv_upper].

Convention de prix : formule de Black (1976) avec forward F_T.
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import pandas as pd

from config.settings import ImpliedVolConfig, CONFIG
from src.pricing.black_scholes import black_price, black_vega, black_d1_d2
from src.utils.math_utils import newton_raphson, bisection
from src.utils.constants import EPS

logger = logging.getLogger(__name__)


class IVMethod(str, Enum):
    NEWTON = "newton"
    BISECTION = "bisection"
    FAILED = "failed"


# ─────────────────────────────────────────────────────────────────────────────
# Extraction scalaire
# ─────────────────────────────────────────────────────────────────────────────

def implied_vol(
    market_price: float,
    F: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    cfg: ImpliedVolConfig | None = None,
) -> tuple[float, IVMethod, int]:
    """
    Extrait la volatilité implicite d'une option via Newton-Raphson puis dichotomie.

    Parameters
    ----------
    market_price : prix de marché observé (mid ou mark)
    F, K, T, r   : paramètres Black
    option_type  : "C" ou "P"
    cfg          : configuration (défaut = CONFIG.implied_vol)

    Returns
    -------
    (iv, method_used, iterations)
    """
    cfg = cfg or CONFIG.implied_vol

    if T <= 0 or market_price <= 0 or F <= 0 or K <= 0:
        return np.nan, IVMethod.FAILED, 0

    lo, hi = cfg.iv_lower, cfg.iv_upper

    def f(sigma: float) -> float:
        return black_price(F, K, T, r, sigma, option_type) - market_price

    def df(sigma: float) -> float:
        return black_vega(F, K, T, r, sigma)

    # ── Newton-Raphson ────────────────────────────────────────────────────────
    # Point de départ : 0.5 ou Brenner-Subrahmanyam approximation
    sigma0 = _brenner_subrahmanyam(market_price, F, T)
    sigma0 = max(lo, min(sigma0, hi))

    iv_nr, iters_nr, conv_nr = newton_raphson(
        f=f,
        df=df,
        x0=sigma0,
        tol=cfg.newton_tol,
        max_iter=cfg.newton_max_iter,
        x_min=lo,
        x_max=hi,
    )

    if conv_nr and lo <= iv_nr <= hi:
        return iv_nr, IVMethod.NEWTON, iters_nr

    # ── Dichotomie (fallback) ─────────────────────────────────────────────────
    try:
        # Vérification que f(lo) et f(hi) sont de signes opposés
        f_lo = f(lo)
        f_hi = f(hi)
        if f_lo * f_hi > 0:
            # Cherche manuellement une borne haute qui fonctionne
            for h in [2.0, 5.0, 10.0, 15.0, hi]:
                f_h = f(h)
                if f_lo * f_h <= 0:
                    f_hi = f_h
                    hi = h
                    break
            else:
                return np.nan, IVMethod.FAILED, 0

        iv_bis, iters_bis, conv_bis = bisection(
            f=f,
            a=lo,
            b=hi,
            tol=cfg.bisect_tol,
            max_iter=cfg.bisect_max_iter,
        )
        if conv_bis and lo <= iv_bis <= hi:
            return iv_bis, IVMethod.BISECTION, iters_bis

    except ValueError as e:
        logger.debug("Dichotomie échouée : %s", e)

    return np.nan, IVMethod.FAILED, 0


def _brenner_subrahmanyam(market_price: float, F: float, T: float) -> float:
    """
    Approximation de Brenner-Subrahmanyam (1988) pour ATM :
    σ ≈ √(2π/T) · (C/F)

    Utilisée comme point de départ de Newton.
    """
    if F <= 0 or T <= 0:
        return 0.5
    import math
    return math.sqrt(2 * math.pi / T) * market_price / F


# ─────────────────────────────────────────────────────────────────────────────
# Extraction vectorisée (sur DataFrame)
# ─────────────────────────────────────────────────────────────────────────────

def compute_implied_vols(
    df: pd.DataFrame,
    price_col: str = "mid",
    cfg: ImpliedVolConfig | None = None,
) -> pd.DataFrame:
    """
    Calcule la volatilité implicite pour chaque ligne du DataFrame.

    Colonnes requises : mid (ou autre), F (forward_price), strike, T,
                        option_type, + une colonne 'rate' pour r(T).

    Colonnes ajoutées : iv, iv_method, iv_iters, bs_price, iv_error_abs
    """
    cfg = cfg or CONFIG.implied_vol
    df = df.copy()

    ivs, methods, iters_list, bs_prices = [], [], [], []

    for _, row in df.iterrows():
        price = float(row.get(price_col, 0.0))
        F = float(row.get("forward_price", 0.0))
        K = float(row.get("strike", 0.0))
        T = float(row.get("T", 0.0))
        r = float(row.get("rate", 0.0))
        ot = str(row.get("option_type", "C"))

        iv, method, it = implied_vol(price, F, K, T, r, ot, cfg)
        ivs.append(iv)
        methods.append(method.value)
        iters_list.append(it)

        # Prix reconstitué via BS
        if np.isfinite(iv):
            bs = black_price(F, K, T, r, iv, ot)
        else:
            bs = np.nan
        bs_prices.append(bs)

    df["iv"] = ivs
    df["iv_method"] = methods
    df["iv_iters"] = iters_list
    df["bs_price"] = bs_prices
    df["iv_error_abs"] = np.abs(df[price_col] - df["bs_price"])

    n_ok = df["iv"].notna().sum()
    n_total = len(df)
    logger.info(
        "Volatilité implicite : %d/%d converties (%.1f%%) | Newton: %d | Bisect: %d",
        n_ok, n_total, 100 * n_ok / max(n_total, 1),
        (df["iv_method"] == IVMethod.NEWTON.value).sum(),
        (df["iv_method"] == IVMethod.BISECTION.value).sum(),
    )
    return df
