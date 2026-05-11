"""
arbitrage_checks.py — Vérifications de non-arbitrage sur la surface de volatilité.

Deux types d'arbitrage :
  1. Calendar spread : w(k, T₁) ≤ w(k, T₂) pour T₁ ≤ T₂ à k fixé.
  2. Butterfly : g(k, t) ≥ 0 pour tout (k, t), où g est la densité implicite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.volatility.ssvi import SSVIParams, ssvi_total_variance


# ─────────────────────────────────────────────────────────────────────────────
# Calendar spread
# ─────────────────────────────────────────────────────────────────────────────

def check_calendar_spread_discrete(
    df_iv: pd.DataFrame,
    k_tol: float = 0.02,
) -> pd.DataFrame:
    """
    Vérifie le no-calendar-spread sur les données discrètes.

    Pour chaque paire de maturités (T₁ < T₂) et chaque strike voisin,
    vérifie σ²(k,T₁)·T₁ ≤ σ²(k,T₂)·T₂.

    Returns
    -------
    DataFrame des violations détectées (peut être vide si aucune violation).
    """
    mats = sorted(df_iv["T"].unique())
    violations = []

    for i in range(len(mats) - 1):
        T1, T2 = mats[i], mats[i + 1]
        sub1 = df_iv[df_iv["T"] == T1][["log_moneyness", "total_variance"]].dropna()
        sub2 = df_iv[df_iv["T"] == T2][["log_moneyness", "total_variance"]].dropna()

        for _, row1 in sub1.iterrows():
            k = row1["log_moneyness"]
            w1 = row1["total_variance"]
            # Trouve les points proches en k dans T2
            near = sub2[np.abs(sub2["log_moneyness"] - k) < k_tol]
            if near.empty:
                continue
            w2 = near["total_variance"].mean()
            if w1 > w2 + 1e-6:
                violations.append({
                    "k": k, "T1": T1, "T2": T2,
                    "w_T1": w1, "w_T2": w2,
                    "violation": w1 - w2,
                })

    return pd.DataFrame(violations)


# ─────────────────────────────────────────────────────────────────────────────
# Butterfly (densité de probabilité)
# ─────────────────────────────────────────────────────────────────────────────

def ssvi_density(k: np.ndarray, t: float, params: SSVIParams) -> np.ndarray:
    """
    Calcule la densité implicite g(k) = (1 - k/2w · ∂w/∂k)² - 1/4(1/w + 1/4)(∂w/∂k)² + 1/2·∂²w/∂k²
    (formule de Dupire via SVI, Gatheral 2011)

    Pour la vérification, on utilise une différence finie centrale.
    """
    dk = 1e-4
    w = ssvi_total_variance(k, np.full_like(k, t), params)
    w_p = ssvi_total_variance(k + dk, np.full_like(k, t), params)
    w_m = ssvi_total_variance(k - dk, np.full_like(k, t), params)

    dw = (w_p - w_m) / (2 * dk)
    d2w = (w_p - 2 * w + w_m) / dk ** 2

    safe_w = np.where(w < 1e-10, 1e-10, w)
    term1 = (1.0 - k * dw / (2.0 * safe_w)) ** 2
    term2 = 0.25 * (0.25 + 1.0 / safe_w) * dw ** 2
    term3 = 0.5 * d2w

    return term1 - term2 + term3


def check_butterfly_density(
    params: SSVIParams,
    k_grid: np.ndarray | None = None,
    t_grid: np.ndarray | None = None,
) -> dict[str, float | bool]:
    """
    Vérifie la positivité de la densité (condition butterfly).

    Returns dict avec 'min_density', 'n_violations', 'butterfly_ok'.
    """
    if k_grid is None:
        k_grid = np.linspace(-1.5, 1.5, 200)
    if t_grid is None:
        t_grid = np.linspace(0.1, 2.0, 10)

    min_g = np.inf
    n_violations = 0
    for t in t_grid:
        g = ssvi_density(k_grid, t, params)
        min_g = min(min_g, float(np.min(g)))
        n_violations += int(np.sum(g < -1e-6))

    return {
        "min_density": float(min_g),
        "n_violations": n_violations,
        "butterfly_ok": min_g >= -1e-6,
    }
