"""
math_utils.py — Utilitaires mathématiques fondamentaux.

Fonctions pures, sans dépendances au reste du projet.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.stats import norm

from src.utils.constants import EPS, INV_SQRT_2PI


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions de distribution normale
# ─────────────────────────────────────────────────────────────────────────────

def standard_normal_cdf(x: float) -> float:
    """CDF de la loi normale standard N(0,1)."""
    return float(norm.cdf(x))


def standard_normal_pdf(x: float) -> float:
    """PDF de la loi normale standard n(0,1)."""
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)


def standard_normal_cdf_vec(x: np.ndarray) -> np.ndarray:
    """Version vectorisée de la CDF normale standard."""
    return norm.cdf(x)


def standard_normal_pdf_vec(x: np.ndarray) -> np.ndarray:
    """Version vectorisée de la PDF normale standard."""
    return norm.pdf(x)


# ─────────────────────────────────────────────────────────────────────────────
# Analyse numérique
# ─────────────────────────────────────────────────────────────────────────────

def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> tuple[float, int, bool]:
    """
    Méthode de bisection (dichotomie) pour trouver x tel que f(x) = 0.

    Précondition : f(a) et f(b) sont de signes opposés.

    Returns
    -------
    (root, iterations, converged)
    """
    fa = f(a)
    fb = f(b)

    if abs(fa) < tol:
        return a, 0, True
    if abs(fb) < tol:
        return b, 0, True

    if fa * fb > 0:
        raise ValueError(
            f"bisection: f(a)={fa:.6g} et f(b)={fb:.6g} doivent être de signes opposés."
        )

    for i in range(max_iter):
        mid = 0.5 * (a + b)
        fmid = f(mid)
        if abs(fmid) < tol or (b - a) < tol:
            return mid, i + 1, True
        if fa * fmid < 0:
            b, fb = mid, fmid
        else:
            a, fa = mid, fmid

    return 0.5 * (a + b), max_iter, False


def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-8,
    max_iter: int = 100,
    x_min: float | None = None,
    x_max: float | None = None,
) -> tuple[float, int, bool]:
    """
    Méthode de Newton-Raphson pour trouver x tel que f(x) = 0.

    Si df est trop petit ou si x sort des bornes, on renvoie converged=False.

    Returns
    -------
    (root, iterations, converged)
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x, i + 1, True
        dfx = df(x)
        if abs(dfx) < EPS:
            return x, i + 1, False
        x_new = x - fx / dfx
        # Clamping éventuel
        if x_min is not None:
            x_new = max(x_new, x_min)
        if x_max is not None:
            x_new = min(x_new, x_max)
        if abs(x_new - x) < tol:
            return x_new, i + 1, True
        x = x_new
    return x, max_iter, False


def safe_log(x: float, floor: float = EPS) -> float:
    """Logarithme naturel protégé contre log(0)."""
    return math.log(max(x, floor))


def safe_sqrt(x: float) -> float:
    """Racine carrée protégée contre les valeurs négatives."""
    return math.sqrt(max(x, 0.0))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient de détermination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < EPS:
        return 1.0
    return float(1.0 - ss_res / ss_tot)
