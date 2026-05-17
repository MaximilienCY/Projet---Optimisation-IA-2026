"""
optimizer.py — Optimisation Delta-Gamma-Vega neutre.

Formulation mathématique :
─────────────────────────────────────────────────────────────────────────────
Variables de décision : q = (q₁, ..., qₙ) ∈ ℝⁿ  (quantités des instruments)

On vend 1 unité du produit. Ses grecques sont (Δₚ, Γₚ, 𝒱ₚ).
Le portefeuille de couverture doit satisfaire :

    Σᵢ qᵢ·Δᵢ = Δₚ          (neutralisation Delta)
    Σᵢ qᵢ·Γᵢ = Γₚ          (neutralisation Gamma)
    Σᵢ qᵢ·𝒱ᵢ = 𝒱ₚ          (neutralisation Vega)

On minimise la norme L2 des positions (robustesse + régularisation) :
    min_q  ‖q‖² + α · coût_de_couverture(q)

sujet à :
    |q_i| ≤ q_max      (borne sur les positions)

Résolution : scipy.optimize.minimize avec méthode "SLSQP".
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

from config.settings import HedgeConfig, CONFIG
from src.hedge.portfolio import HedgeInstrument, HedgePortfolio
from src.pricing.greeks import Greeks

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Construction du problème d'optimisation
# ─────────────────────────────────────────────────────────────────────────────

def build_greeks_matrix(instruments: list[HedgeInstrument]) -> np.ndarray:
    """
    Construit la matrice G (3 × n) des grecques des instruments.
    Lignes : Delta, Gamma, Vega.
    Colonnes : instruments.
    """
    n = len(instruments)
    G = np.zeros((3, n))
    for j, inst in enumerate(instruments):
        G[0, j] = inst.greeks.delta
        G[1, j] = inst.greeks.gamma
        G[2, j] = inst.greeks.vega
    return G


def optimize_hedge(
    product_greeks: Greeks,
    instruments: list[HedgeInstrument],
    cfg: HedgeConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Résout le problème d'optimisation Delta-Gamma-Vega neutre.

    On cherche q tel que G·q ≈ g_product, en minimisant ‖q‖² + régularisation.

    Parameters
    ----------
    product_greeks : grecques du produit vendu
    instruments    : liste des instruments de couverture avec leurs grecques
    cfg            : configuration

    Returns
    -------
    (quantities, optimization_report)
    """
    cfg = cfg or CONFIG.hedge
    n = len(instruments)

    if n == 0:
        logger.warning("Aucun instrument de couverture disponible.")
        return np.array([]), {"status": "no_instruments"}

    G = build_greeks_matrix(instruments)

    # Filtrer les colonnes NaN/inf (grecques invalides restantes)
    valid_cols = np.all(np.isfinite(G), axis=0)
    if not np.all(valid_cols):
        logger.warning(
            "%d instrument(s) avec grecques non-finies écartés de l'optimisation.",
            (~valid_cols).sum(),
        )
        G = G[:, valid_cols]
        instruments = [inst for inst, ok in zip(instruments, valid_cols) if ok]
        n = len(instruments)
        if n == 0:
            return np.array([]), {"status": "all_instruments_invalid"}

    # Cible : neutraliser les grecques du produit vendu
    target = np.array([
        product_greeks.delta,
        product_greeks.gamma,
        product_greeks.vega,
    ])

    # Prix des instruments (pour le terme de coût)
    prices = np.array([i.mid_price for i in instruments])
    prices_norm = prices / (np.max(np.abs(prices)) + 1e-10)

    reg = cfg.regularization

    def objective(q: np.ndarray) -> float:
        """Somme des carrés des q + pénalité de coût."""
        greek_error = G @ q - target
        return (
            np.dot(q, q) * reg
            + np.dot(greek_error, greek_error)
            + cfg.liquidity_weight * 1e-6 * np.abs(prices_norm @ q)
        )

    def gradient(q: np.ndarray) -> np.ndarray:
        greek_error = G @ q - target
        return (
            2.0 * reg * q
            + 2.0 * G.T @ greek_error
            + cfg.liquidity_weight * 1e-6 * prices_norm * np.sign(q + 1e-15)
        )

    # Contraintes égalité : G·q = target
    # Le Jacobien de (G·q - target) par rapport à q est G (shape 3×n)
    constraints = {
        "type": "eq",
        "fun": lambda q: G @ q - target,
        "jac": lambda q: G,
    }

    bounds = Bounds(
        lb=-cfg.max_position_size * np.ones(n),
        ub=cfg.max_position_size * np.ones(n),
    )

    # Point de départ : solution minimum-norme via pseudo-inverse
    # lstsq(G, target) → q₀ tel que G@q₀ ≈ target avec ‖q₀‖ minimal
    try:
        q0, _, _, _ = np.linalg.lstsq(G, target, rcond=None)
        q0 = np.clip(q0, -cfg.max_position_size, cfg.max_position_size)
    except Exception:
        q0 = np.zeros(n)

    result = minimize(
        fun=objective,
        jac=gradient,
        x0=q0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-10, "eps": 1e-8},
    )

    if not result.success:
        # Fallback : moindres carrés (pseudo-inverse) — robuste même si G est mal conditionné
        logger.warning("SLSQP n'a pas convergé (%s). Fallback pseudo-inverse.", result.message)
        try:
            q_ls, _, _, _ = np.linalg.lstsq(G, target, rcond=None)
        except np.linalg.LinAlgError:
            # SVD ne converge pas → utiliser la pseudo-inverse Moore-Penrose
            logger.warning("lstsq SVD échoué, bascule sur pinv.")
            q_ls = np.linalg.pinv(G) @ target
        q_ls = np.clip(q_ls, -cfg.max_position_size, cfg.max_position_size)
        quantities = q_ls
        obj_val = float(np.dot(q_ls, q_ls))
    else:
        quantities = result.x
        obj_val = float(result.fun)

    # Calcul des résidus
    achieved = G @ quantities
    residuals = achieved - target
    report = {
        "status": result.message if hasattr(result, "message") else "ok",
        "converged": bool(result.success),
        "achieved_delta": float(achieved[0]),
        "achieved_gamma": float(achieved[1]),
        "achieved_vega": float(achieved[2]),
        "target_delta": float(target[0]),
        "target_gamma": float(target[1]),
        "target_vega": float(target[2]),
        "residual_delta": float(residuals[0]),
        "residual_gamma": float(residuals[1]),
        "residual_vega": float(residuals[2]),
        "objective_value": obj_val,
    }
    logger.info(
        "Optimisation hedge : Δ_résidu=%.6f Γ_résidu=%.6f 𝒱_résidu=%.6f",
        residuals[0], residuals[1], residuals[2],
    )
    return quantities, report


def build_hedge_portfolio(
    product_greeks: Greeks,
    instruments: list[HedgeInstrument],
    cfg: HedgeConfig | None = None,
) -> HedgePortfolio:
    """
    Construit le portefeuille de couverture optimal.

    Returns
    -------
    HedgePortfolio avec les quantités optimales renseignées.
    """
    quantities, report = optimize_hedge(product_greeks, instruments, cfg)
    portfolio = HedgePortfolio(product_greeks=product_greeks, instruments=instruments)

    for i, q in enumerate(quantities):
        instruments[i].quantity = float(q)

    logger.info("Portefeuille de couverture : %d instruments", len(instruments))
    portfolio._opt_report = report  # type: ignore[attr-defined]
    return portfolio
