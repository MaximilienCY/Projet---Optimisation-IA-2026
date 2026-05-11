"""
calibration_objectives.py — Fonctions objectif et pipeline de calibration SSVI.

Stratégie en deux temps (comme demandé dans le sujet) :

  Étape 1 — Terme de structure ATM :
    On extrait les variances totales ATM θ̂_t = σ²_ATM(t) · t pour chaque
    maturité t. On ajuste θ(t) = ν_∞·t + (ν₀-ν_∞)/κ · (1-e^{-κt})
    par moindres carrés.

  Étape 2 — Smile (ρ, η, λ) conditionnel à (κ, ν₀, ν_∞) :
    On minimise l'erreur quadratique entre w_SSVI(k, t) et
    w̃_market(k, t) = σ²_market(k, t) · t sur l'ensemble des points.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize, differential_evolution

from config.settings import SSVIConfig, CONFIG
from src.volatility.ssvi import (
    SSVIParams, ssvi_theta, ssvi_total_variance, ssvi_implied_vol,
    check_butterfly_arbitrage,
)
from src.utils.math_utils import rmse, r_squared

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 : calibration du terme de structure ATM
# ─────────────────────────────────────────────────────────────────────────────

def objective_atm(
    params_arr: np.ndarray,
    T_atm: np.ndarray,
    theta_atm: np.ndarray,
) -> float:
    """MSE entre θ(t) modèle et θ̂ empiriques."""
    kappa, nu0, nu_inf = params_arr
    theta_model = ssvi_theta(T_atm, kappa, nu0, nu_inf)
    return float(np.mean((theta_model - theta_atm) ** 2))


def calibrate_atm_term_structure(
    T_atm: np.ndarray,
    theta_atm: np.ndarray,
    cfg: SSVIConfig | None = None,
) -> tuple[float, float, float, dict]:
    """
    Calibre (κ, ν₀, ν_∞) sur les variances totales ATM empiriques.

    Parameters
    ----------
    T_atm    : maturités des points ATM (en années)
    theta_atm: θ̂ = σ²_ATM · T empiriques

    Returns
    -------
    (kappa, nu0, nu_inf, metrics)
    """
    cfg = cfg or CONFIG.ssvi
    bounds = [cfg.kappa_bounds, cfg.nu0_bounds, cfg.nu_inf_bounds]

    best_result = None
    best_val = np.inf

    # Évolution différentielle
    try:
        de_res = differential_evolution(
            func=lambda x: objective_atm(x, T_atm, theta_atm),
            bounds=bounds,
            seed=42,
            maxiter=500,
            tol=1e-10,
            polish=True,
            workers=1,
        )
        if de_res.fun < best_val:
            best_val = de_res.fun
            best_result = de_res
    except Exception as e:
        logger.warning("DE step1 : %s", e)

    # Multi-start L-BFGS-B
    rng = np.random.default_rng(1)
    for _ in range(cfg.n_starts):
        x0 = np.array([
            rng.uniform(*cfg.kappa_bounds),
            rng.uniform(*cfg.nu0_bounds),
            rng.uniform(*cfg.nu_inf_bounds),
        ])
        try:
            res = minimize(
                lambda x: objective_atm(x, T_atm, theta_atm),
                x0=x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": cfg.max_iter, "ftol": cfg.tol},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_result = res
        except Exception:
            pass

    if best_result is None:
        logger.error("Calibration ATM échouée.")
        return 1.0, float(np.mean(theta_atm / T_atm)), float(np.mean(theta_atm / T_atm)), {}

    kappa, nu0, nu_inf = best_result.x
    theta_model = ssvi_theta(T_atm, kappa, nu0, nu_inf)
    metrics = {
        "rmse_theta": rmse(theta_atm, theta_model),
        "r2_theta": r_squared(theta_atm, theta_model),
        "n_atm_points": len(T_atm),
    }
    logger.info("ATM calibré : κ=%.4f ν₀=%.4f ν_∞=%.4f | RMSE=%.6f", kappa, nu0, nu_inf, metrics["rmse_theta"])
    return float(kappa), float(nu0), float(nu_inf), metrics


# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 : calibration du smile (ρ, η, λ)
# ─────────────────────────────────────────────────────────────────────────────

def objective_smile(
    params_arr: np.ndarray,
    k_all: np.ndarray,
    t_all: np.ndarray,
    w_market: np.ndarray,
    kappa: float,
    nu0: float,
    nu_inf: float,
    penalty_weight: float = 1000.0,
) -> float:
    """
    MSE entre w_SSVI et w_market + pénalité si contrainte butterfly violée.
    """
    rho, eta, lambda_ = params_arr
    params = SSVIParams(kappa=kappa, nu0=nu0, nu_inf=nu_inf, rho=rho, eta=eta, lambda_=lambda_)

    w_model = ssvi_total_variance(k_all, t_all, params)
    mse = float(np.mean((w_model - w_market) ** 2))

    # Pénalité pour non-respect de la contrainte butterfly
    penalty = 0.0
    violation = eta * (1.0 + abs(rho)) - 4.0
    if violation > 0:
        penalty = penalty_weight * violation ** 2

    return mse + penalty


def calibrate_smile(
    k_all: np.ndarray,
    t_all: np.ndarray,
    w_market: np.ndarray,
    kappa: float,
    nu0: float,
    nu_inf: float,
    cfg: SSVIConfig | None = None,
) -> tuple[float, float, float, dict]:
    """
    Calibre (ρ, η, λ) conditionnellement à (κ, ν₀, ν_∞).

    Returns
    -------
    (rho, eta, lambda_, metrics)
    """
    cfg = cfg or CONFIG.ssvi
    bounds = [cfg.rho_bounds, cfg.eta_bounds, cfg.lambda_bounds]

    best_result = None
    best_val = np.inf

    obj = lambda x: objective_smile(x, k_all, t_all, w_market, kappa, nu0, nu_inf)

    # Évolution différentielle
    try:
        de_res = differential_evolution(
            func=obj, bounds=bounds, seed=42,
            maxiter=1000, tol=cfg.tol, polish=True, workers=1,
        )
        if de_res.fun < best_val:
            best_val = de_res.fun
            best_result = de_res
    except Exception as e:
        logger.warning("DE step2 : %s", e)

    # Multi-start
    rng = np.random.default_rng(2)
    for _ in range(cfg.n_starts):
        x0 = np.array([
            rng.uniform(*cfg.rho_bounds),
            rng.uniform(*cfg.eta_bounds),
            rng.uniform(*cfg.lambda_bounds),
        ])
        try:
            res = minimize(obj, x0=x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": cfg.max_iter, "ftol": cfg.tol})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res
        except Exception:
            pass

    if best_result is None:
        logger.error("Calibration smile échouée.")
        return -0.3, 1.0, 0.3, {}

    rho, eta, lambda_ = best_result.x
    params = SSVIParams(kappa=kappa, nu0=nu0, nu_inf=nu_inf, rho=rho, eta=eta, lambda_=lambda_)
    w_model = ssvi_total_variance(k_all, t_all, params)

    # Convertir en IV pour les métriques
    safe_t = np.where(t_all < 1e-10, 1e-10, t_all)
    iv_model = np.sqrt(np.maximum(w_model / safe_t, 0.0))
    iv_market = np.sqrt(np.maximum(w_market / safe_t, 0.0))

    metrics = {
        "rmse_iv": rmse(iv_market, iv_model),
        "r2_iv": r_squared(iv_market, iv_model),
        "rmse_w": rmse(w_market, w_model),
        "n_points": len(k_all),
        "butterfly_ok": check_butterfly_arbitrage(params, np.linspace(0.05, 3.0, 50)),
    }
    logger.info(
        "Smile calibré : ρ=%.4f η=%.4f λ=%.4f | RMSE_IV=%.4f R²=%.4f",
        rho, eta, lambda_, metrics["rmse_iv"], metrics["r2_iv"],
    )
    return float(rho), float(eta), float(lambda_), metrics


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline complet de calibration SSVI
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_ssvi(
    df_iv: "pd.DataFrame",
    cfg: SSVIConfig | None = None,
    atm_band: float = 0.05,
) -> tuple[SSVIParams, dict]:
    """
    Calibration SSVI complète en deux étapes.

    Parameters
    ----------
    df_iv    : DataFrame avec colonnes [T, log_moneyness, iv, total_variance]
    cfg      : configuration SSVI
    atm_band : bande ATM pour l'étape 1 (|k| < atm_band)

    Returns
    -------
    (SSVIParams, calibration_report)
    """
    import pandas as pd

    cfg = cfg or CONFIG.ssvi

    df = df_iv.dropna(subset=["T", "log_moneyness", "iv"]).copy()
    df = df[df["iv"] > 0].copy()
    df["total_variance"] = df["iv"] ** 2 * df["T"]

    if df.empty:
        raise ValueError("Aucun point de vol implicite valide pour calibrer SSVI.")

    # ── Étape 1 : terme de structure ATM ─────────────────────────────────────
    df_atm = df[df["log_moneyness"].abs() < atm_band].copy()
    if df_atm.empty:
        # Fallback : utilise les 3 points les plus proches de ATM par maturité
        df_atm = (
            df.assign(abs_k=df["log_moneyness"].abs())
            .sort_values(["T", "abs_k"])
            .groupby("T")
            .head(3)
            .reset_index(drop=True)
        )
        logger.warning("Aucun point ATM strict, utilise les 3 plus proches par maturité.")

    # Variance totale ATM par maturité (médiane pour robustesse)
    atm_agg = df_atm.groupby("T")["total_variance"].median().reset_index()
    T_atm = atm_agg["T"].values
    theta_atm = atm_agg["total_variance"].values

    kappa, nu0, nu_inf, metrics_atm = calibrate_atm_term_structure(T_atm, theta_atm, cfg)

    # ── Étape 2 : smile ────────────────────────────────────────────────────────
    k_all = df["log_moneyness"].values
    t_all = df["T"].values
    w_market = df["total_variance"].values

    rho, eta, lambda_, metrics_smile = calibrate_smile(
        k_all, t_all, w_market, kappa, nu0, nu_inf, cfg
    )

    params = SSVIParams(kappa=kappa, nu0=nu0, nu_inf=nu_inf, rho=rho, eta=eta, lambda_=lambda_)

    report = {
        "step1": metrics_atm,
        "step2": metrics_smile,
        "params": params.to_dict(),
    }
    return params, report
