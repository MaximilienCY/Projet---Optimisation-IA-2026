"""
black_scholes.py — Formule de Black-Scholes pour options européennes.

Convention utilisée :
  - Le sous-jacent est un forward F_T (prix futures Deribit).
  - La formule de Black (1976) est utilisée :

        d₁ = [ln(F/K) + (σ²/2)·T] / (σ·√T)
        d₂ = d₁ - σ·√T

        C = e^{-rT} · [F·N(d₁) - K·N(d₂)]
        P = e^{-rT} · [K·N(-d₂) - F·N(-d₁)]

  - Cette formule est équivalente à Black-Scholes avec q = r (cost-of-carry = 0),
    ce qui est naturel pour les futures/crypto.
  - Le taux r provient de la courbe Nelson-Siegel.
"""

from __future__ import annotations

import math

import numpy as np

from src.utils.math_utils import standard_normal_cdf as N, standard_normal_pdf as n
from src.utils.constants import EPS


# ─────────────────────────────────────────────────────────────────────────────
# Calcul des d₁, d₂
# ─────────────────────────────────────────────────────────────────────────────

def black_d1_d2(
    F: float, K: float, T: float, sigma: float
) -> tuple[float, float]:
    """
    Calcule (d₁, d₂) de la formule de Black.

    Parameters
    ----------
    F     : forward price
    K     : strike
    T     : maturité (années)
    sigma : volatilité implicite annualisée

    Returns
    -------
    (d1, d2)
    """
    sqrt_T = math.sqrt(max(T, EPS))
    sigma_sqrt_T = sigma * sqrt_T
    if sigma_sqrt_T < EPS:
        # Limit case: deep ITM/OTM, d1 → ±∞
        if F > K:
            return 1e10, 1e10
        elif F < K:
            return -1e10, -1e10
        else:
            return 0.0, 0.0
    d1 = (math.log(F / max(K, EPS)) + 0.5 * sigma ** 2 * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    return d1, d2


# ─────────────────────────────────────────────────────────────────────────────
# Prix call / put
# ─────────────────────────────────────────────────────────────────────────────

def black_call(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Prix d'un call européen par la formule de Black (1976).

    Parameters
    ----------
    F     : forward price (ou futures price)
    K     : strike
    T     : maturité résiduelle (années)
    r     : taux sans risque continu (Nelson-Siegel)
    sigma : volatilité implicite annualisée

    Returns
    -------
    Prix du call en USD
    """
    if T <= 0:
        return max(F - K, 0.0)
    d1, d2 = black_d1_d2(F, K, T, sigma)
    discount = math.exp(-r * T)
    return discount * (F * N(d1) - K * N(d2))


def black_put(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Prix d'un put européen par la formule de Black (1976).
    """
    if T <= 0:
        return max(K - F, 0.0)
    d1, d2 = black_d1_d2(F, K, T, sigma)
    discount = math.exp(-r * T)
    return discount * (K * N(-d2) - F * N(-d1))


def black_price(
    F: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """
    Dispatch call/put.

    Parameters
    ----------
    option_type : "C" ou "P"
    """
    if option_type.upper() == "C":
        return black_call(F, K, T, r, sigma)
    elif option_type.upper() == "P":
        return black_put(F, K, T, r, sigma)
    else:
        raise ValueError(f"option_type doit être 'C' ou 'P', reçu: '{option_type}'")


# ─────────────────────────────────────────────────────────────────────────────
# Vega (identique pour call et put)
# ─────────────────────────────────────────────────────────────────────────────

def black_vega(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Vega = ∂Price/∂σ = e^{-rT} · F · n(d₁) · √T

    Identique pour call et put.
    """
    if T <= 0 or sigma < EPS:
        return 0.0
    d1, _ = black_d1_d2(F, K, T, sigma)
    return math.exp(-r * T) * F * n(d1) * math.sqrt(T)


# ─────────────────────────────────────────────────────────────────────────────
# Versions vectorisées
# ─────────────────────────────────────────────────────────────────────────────

def black_price_vec(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    option_type: np.ndarray,  # array de "C" ou "P"
) -> np.ndarray:
    """Version vectorisée de black_price."""
    return np.array([
        black_price(float(f), float(k), float(t), float(ri), float(s), str(ot))
        for f, k, t, ri, s, ot in zip(F, K, T, r, sigma, option_type)
    ])
