"""
greeks.py — Calcul des grecques (Black 1976).

Formules :
  Delta_C  = e^{-rT} · N(d₁)
  Delta_P  = -e^{-rT} · N(-d₁)       [= e^{-rT} · (N(d₁) - 1)]
  Gamma    = e^{-rT} · n(d₁) / (F · σ · √T)
  Vega     = e^{-rT} · F · n(d₁) · √T    [= ∂Price/∂σ]
  Theta_C  = e^{-rT} · [-F·n(d₁)·σ/(2√T) + r·(F·N(d₁) - K·N(d₂))] - r·C
  Rho_C    = -T · C                        (sensibilité au taux pour Black)

Note sur les unités :
  - Delta : $/$ (sans unité si F et Price en USD)
  - Gamma : ∂²Price/∂F²   [1/USD]
  - Vega  : ∂Price/∂σ     [$/annualisé, donc $/1]
  - Theta : ∂Price/∂T     [$/an → diviser par 365 pour avoir $/jour si souhaité]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.pricing.black_scholes import black_d1_d2, black_price
from src.utils.math_utils import standard_normal_cdf as N, standard_normal_pdf as n
from src.utils.constants import EPS


# ─────────────────────────────────────────────────────────────────────────────
# Structure de données pour les grecques
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Greeks:
    """Conteneur des grecques d'une option."""
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "Delta": self.delta,
            "Gamma": self.gamma,
            "Vega": self.vega,
            "Theta": self.theta,
            "Rho": self.rho,
        }

    def __add__(self, other: "Greeks") -> "Greeks":
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
            rho=self.rho + other.rho,
        )

    def __mul__(self, scalar: float) -> "Greeks":
        return Greeks(
            delta=self.delta * scalar,
            gamma=self.gamma * scalar,
            vega=self.vega * scalar,
            theta=self.theta * scalar,
            rho=self.rho * scalar,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Calcul des grecques
# ─────────────────────────────────────────────────────────────────────────────

def compute_greeks(
    F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> Greeks:
    """
    Calcule les grecques d'une option européenne (formule de Black 1976).

    Parameters
    ----------
    F, K, T, r, sigma : paramètres Black
    option_type       : "C" ou "P"

    Returns
    -------
    Greeks
    """
    if T <= 0 or sigma < EPS or F <= 0 or K <= 0:
        return _expiry_greeks(F, K, option_type)

    d1, d2 = black_d1_d2(F, K, T, sigma)
    sqrt_T = math.sqrt(T)
    disc = math.exp(-r * T)
    n_d1 = n(d1)

    if option_type.upper() == "C":
        delta = disc * N(d1)
        theta = disc * (
            -F * n_d1 * sigma / (2 * sqrt_T)
            - r * K * N(d2)
            + r * F * N(d1)
        )
        # Theta usuel pour Black : ∂C/∂T (attention signe convention)
        # En pratique Theta = ∂C/∂t < 0 où t est le temps actuel (time decay)
        # On conserve ∂C/∂T > 0 (maturité résiduelle), à convertir si besoin
    else:  # Put
        delta = -disc * N(-d1)
        theta = disc * (
            -F * n_d1 * sigma / (2 * sqrt_T)
            + r * K * N(-d2)
            - r * F * N(-d1)
        )

    gamma = disc * n_d1 / (F * sigma * sqrt_T)
    vega = disc * F * n_d1 * sqrt_T
    rho = -T * black_price(F, K, T, r, sigma, option_type)   # sensibilité à r

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def _expiry_greeks(F: float, K: float, option_type: str) -> Greeks:
    """Grecques à maturité (T=0)."""
    if option_type.upper() == "C":
        delta = 1.0 if F > K else 0.0
    else:
        delta = -1.0 if F < K else 0.0
    return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Calcul vectorisé sur DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def compute_greeks_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les colonnes delta, gamma, vega, theta, rho au DataFrame.

    Colonnes requises : forward_price, strike, T, rate, iv, option_type
    """
    df = df.copy()

    deltas, gammas, vegas, thetas, rhos = [], [], [], [], []

    for _, row in df.iterrows():
        iv = float(row.get("iv", np.nan))
        if not np.isfinite(iv):
            deltas.append(np.nan)
            gammas.append(np.nan)
            vegas.append(np.nan)
            thetas.append(np.nan)
            rhos.append(np.nan)
            continue

        g = compute_greeks(
            F=float(row.get("forward_price", 0.0)),
            K=float(row.get("strike", 0.0)),
            T=float(row.get("T", 0.0)),
            r=float(row.get("rate", 0.0)),
            sigma=iv,
            option_type=str(row.get("option_type", "C")),
        )
        deltas.append(g.delta)
        gammas.append(g.gamma)
        vegas.append(g.vega)
        thetas.append(g.theta)
        rhos.append(g.rho)

    df["delta"] = deltas
    df["gamma"] = gammas
    df["vega"] = vegas
    df["theta"] = thetas
    df["rho"] = rhos
    return df
