"""
payoff.py — Fonctions de payoff génériques et visualisation.
"""

from __future__ import annotations

import numpy as np


def call_payoff(S_T: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(S_T - K, 0.0)


def put_payoff(S_T: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - S_T, 0.0)


def call_spread_payoff(S_T: np.ndarray, K1: float, K2: float) -> np.ndarray:
    """Long call K1, short call K2 (K1 < K2)."""
    return call_payoff(S_T, K1) - call_payoff(S_T, K2)


def put_spread_payoff(S_T: np.ndarray, K1: float, K2: float) -> np.ndarray:
    """Long put K2, short put K1 (K1 < K2)."""
    return put_payoff(S_T, K2) - put_payoff(S_T, K1)


def payoff_grid(
    K1: float,
    K2: float,
    spot: float,
    product_type: str = "call_spread",
    n_points: int = 500,
    range_pct: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère une grille de spots et les payoffs correspondants.

    Returns
    -------
    (spots, payoffs)
    """
    lo = spot * (1 - range_pct)
    hi = spot * (1 + range_pct)
    spots = np.linspace(lo, hi, n_points)

    if product_type == "call_spread":
        payoffs = call_spread_payoff(spots, K1, K2)
    elif product_type == "put_spread":
        payoffs = put_spread_payoff(spots, K1, K2)
    elif product_type == "call":
        payoffs = call_payoff(spots, K1)
    elif product_type == "put":
        payoffs = put_payoff(spots, K1)
    else:
        raise ValueError(f"product_type inconnu : {product_type}")

    return spots, payoffs
