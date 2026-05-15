"""
nelson_siegel.py — Modèle Nelson-Siegel pour le lissage de la courbe des taux.

Modèle de Nelson & Siegel (1987) :
    r(T) = β₀ + β₁ · [(1 - e^{-λT}) / (λT)]
               + β₂ · [(1 - e^{-λT}) / (λT) - e^{-λT}]

Paramètres :
    β₀  : niveau à long terme (taux long)
    β₁  : pente (différence long terme - court terme)
    β₂  : courbure (bosse)
    λ   : paramètre de décroissance (position du maximum de courbure à T = 1/λ)

Calibration par moindres carrés non-linéaires (scipy.optimize.minimize)
avec multi-start pour robustesse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize, differential_evolution

from config.settings import NelsonSiegelConfig, CONFIG
from src.utils.math_utils import rmse, r_squared

logger = logging.getLogger(__name__)

EPS = 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Modèle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NelsonSiegelParams:
    """Paramètres calibrés du modèle Nelson-Siegel."""
    beta0: float = 0.02
    beta1: float = -0.01
    beta2: float = 0.01
    lambda_: float = 1.0

    def to_dict(self) -> dict:
        return {
            "beta0": self.beta0,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "lambda_": self.lambda_,
        }

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "NelsonSiegelParams":
        return cls(beta0=arr[0], beta1=arr[1], beta2=arr[2], lambda_=arr[3])

    def to_array(self) -> np.ndarray:
        return np.array([self.beta0, self.beta1, self.beta2, self.lambda_])


def ns_factor(T: np.ndarray, lambda_: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule les deux facteurs Nelson-Siegel.

    Returns
    -------
    (factor1, factor2) où :
      factor1 = (1 - e^{-λT}) / (λT)
      factor2 = factor1 - e^{-λT}
    """
    lT = lambda_ * T
    # Éviter la division par zéro pour T → 0 (limite = 1)
    safe_lT = np.where(np.abs(lT) < EPS, EPS, lT)
    exp_neg = np.exp(-lT)
    f1 = (1.0 - exp_neg) / safe_lT
    f2 = f1 - exp_neg
    return f1, f2


def nelson_siegel_rate(T: np.ndarray, params: NelsonSiegelParams) -> np.ndarray:
    """
    Calcule le taux zéro-coupon Nelson-Siegel pour les maturités T (années).

    r(T) = β₀ + β₁·f₁(T,λ) + β₂·f₂(T,λ)
    """
    f1, f2 = ns_factor(np.asarray(T, dtype=float), params.lambda_)
    return params.beta0 + params.beta1 * f1 + params.beta2 * f2


def nelson_siegel_rate_scalar(T: float, params: NelsonSiegelParams) -> float:
    """Version scalaire de nelson_siegel_rate."""
    return float(nelson_siegel_rate(np.array([T]), params)[0])


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def _objective(
    params_arr: np.ndarray,
    T: np.ndarray,
    rates: np.ndarray,
    weights: np.ndarray | None,
) -> float:
    """Somme des carrés pondérée (objectif de calibration)."""
    p = NelsonSiegelParams.from_array(params_arr)
    r_hat = nelson_siegel_rate(T, p)
    residuals = rates - r_hat
    if weights is not None:
        return float(np.sum(weights * residuals ** 2))
    return float(np.sum(residuals ** 2))


def calibrate_nelson_siegel(
    maturities: np.ndarray,
    rates: np.ndarray,
    weights: np.ndarray | None = None,
    cfg: NelsonSiegelConfig | None = None,
) -> tuple[NelsonSiegelParams, dict]:
    """
    Calibre les paramètres Nelson-Siegel sur les taux empiriques.

    Stratégie :
      1. Évolution différentielle (recherche globale, robuste)
      2. Raffinement L-BFGS-B depuis la meilleure solution

    Les maturités T < cfg.min_fit_T (défaut 5j) sont exclues du fit car les
    taux courts sont numériquement instables (division par T très petit dans
    la parité call-put). Ils sont conservés dans df_rates pour traçage mais
    ignorés par l'optimiseur.

    Parameters
    ----------
    maturities : tableau des maturités en années
    rates      : taux observés correspondants
    weights    : poids optionnels (ex: 1/std²)
    cfg        : configuration (défaut = CONFIG.nelson_siegel)

    Returns
    -------
    (params, fit_metrics) avec fit_metrics = {rmse, r2, n_points, n_excluded_short}
    """
    cfg = cfg or CONFIG.nelson_siegel

    T_all = np.asarray(maturities, dtype=float)
    R_all = np.asarray(rates, dtype=float)

    # Exclure les maturités très courtes du fit (taux instables)
    fit_mask = T_all >= cfg.min_fit_T
    n_excluded = int((~fit_mask).sum())
    T = T_all[fit_mask]
    R = R_all[fit_mask]
    W = weights[fit_mask] if weights is not None else None

    if n_excluded > 0:
        logger.info(
            "Nelson-Siegel : %d point(s) T < %.4fa exclus du fit (taux courts instables).",
            n_excluded, cfg.min_fit_T,
        )

    if len(T) < 2:
        logger.warning("Trop peu de points pour calibrer Nelson-Siegel (%d).", len(T))
        return NelsonSiegelParams(), {"rmse": np.nan, "r2": np.nan, "n_points": len(T),
                                     "n_excluded_short": n_excluded}

    # Bornes : [β₀, β₁, β₂, λ]
    bounds = [
        cfg.beta0_bounds,
        cfg.beta1_bounds,
        cfg.beta2_bounds,
        cfg.lambda_bounds,
    ]

    best_result = None
    best_val = np.inf

    # Évolution différentielle (robuste aux optima locaux)
    try:
        de_result = differential_evolution(
            func=lambda x: _objective(x, T, R, W),
            bounds=bounds,
            seed=42,
            maxiter=cfg.max_iter // 10,
            tol=cfg.tol,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=True,
            workers=1,
        )
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_result = de_result
    except Exception as e:
        logger.warning("Évolution différentielle échouée : %s", e)

    # Raffinement multi-start L-BFGS-B
    rng = np.random.default_rng(0)
    for _ in range(cfg.n_starts):
        x0 = np.array([
            rng.uniform(*cfg.beta0_bounds),
            rng.uniform(*cfg.beta1_bounds),
            rng.uniform(*cfg.beta2_bounds),
            rng.uniform(*cfg.lambda_bounds),
        ])
        try:
            res = minimize(
                fun=lambda x: _objective(x, T, R, W),
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": cfg.max_iter, "ftol": cfg.tol},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_result = res
        except Exception:
            pass

    if best_result is None:
        logger.error("Calibration Nelson-Siegel échouée.")
        return NelsonSiegelParams(), {"rmse": np.nan, "r2": np.nan, "n_points": len(T),
                                     "n_excluded_short": n_excluded}

    params = NelsonSiegelParams.from_array(best_result.x)
    r_hat = nelson_siegel_rate(T, params)

    metrics = {
        "rmse": rmse(R, r_hat),
        "r2": r_squared(R, r_hat),
        "n_points": len(T),
        "n_excluded_short": n_excluded,
        "optimizer_success": bool(best_result.success) if hasattr(best_result, "success") else True,
    }
    logger.info(
        "Nelson-Siegel calibré : β₀=%.4f β₁=%.4f β₂=%.4f λ=%.4f | RMSE=%.6f R²=%.4f",
        params.beta0, params.beta1, params.beta2, params.lambda_,
        metrics["rmse"], metrics["r2"],
    )
    return params, metrics
