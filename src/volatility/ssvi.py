"""
ssvi.py — Modèle SSVI (Surface SVI) de Gatheral & Jacquier (2014).

Paramétrage utilisé :
─────────────────────────────────────────────────────────────────────────────
Variance totale implicite :
    w(k, t) = (θ_t / 2) · {1 + ρ · φ(θ_t) · k + √[(φ(θ_t)·k + ρ)² + (1-ρ²)]}

Terme de structure ATM (type Heston) :
    θ_t = ν_∞ · t + (ν₀ - ν_∞)/κ · (1 - e^{-κt})

    → θ_t/t → ν₀  quand t → 0  (variance ATM à court terme)
    → θ_t/t → ν_∞ quand t → ∞  (variance ATM à long terme)

Paramètre de skew (power-law) :
    φ(θ) = η / [θ^λ · (1 + θ)^{1-λ}]

Paramètres :
    Étape 1 : κ, ν₀, ν_∞ (terme de structure ATM)
    Étape 2 : ρ, η, λ     (forme du smile)

Conditions de non-arbitrage (Gatheral & Jacquier 2014) :
    1. θ_t ≥ 0 pour tout t ≥ 0  → κ, ν₀, ν_∞ > 0
    2. 0 < w(k,t) < ∞
    3. Butterfly : η · (1 + |ρ|) ≤ 4  (condition suffisante simplifiée)
    4. Calendar spread : θ_t croissant en t (assuré par la forme fonctionnelle)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.utils.constants import EPS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Paramètres SSVI
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SSVIParams:
    """Paramètres du modèle SSVI calibrés."""
    kappa: float = 1.0
    nu0: float = 0.04
    nu_inf: float = 0.04
    rho: float = -0.3
    eta: float = 1.0
    lambda_: float = 0.3

    def to_dict(self) -> dict:
        return {
            "kappa": self.kappa,
            "nu0": self.nu0,
            "nu_inf": self.nu_inf,
            "rho": self.rho,
            "eta": self.eta,
            "lambda_": self.lambda_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SSVIParams":
        return cls(**{k: float(v) for k, v in d.items()})


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions SSVI
# ─────────────────────────────────────────────────────────────────────────────

def ssvi_theta(t: float | np.ndarray, kappa: float, nu0: float, nu_inf: float) -> float | np.ndarray:
    """
    Variance totale ATM : θ(t) = ν_∞·t + (ν₀ - ν_∞)/κ · (1 - e^{-κt})
    """
    t = np.asarray(t, dtype=float)
    if abs(kappa) < EPS:
        return nu0 * t  # limite κ → 0
    return nu_inf * t + (nu0 - nu_inf) / kappa * (1.0 - np.exp(-kappa * t))


def ssvi_phi(theta: float | np.ndarray, eta: float, lambda_: float) -> float | np.ndarray:
    """
    Paramètre de skew power-law :
    φ(θ) = η / [θ^λ · (1 + θ)^{1-λ}]
    """
    theta = np.asarray(theta, dtype=float)
    safe_theta = np.where(theta < EPS, EPS, theta)
    return eta / (safe_theta ** lambda_ * (1.0 + safe_theta) ** (1.0 - lambda_))


def ssvi_total_variance(
    k: float | np.ndarray,
    t: float | np.ndarray,
    params: SSVIParams,
) -> float | np.ndarray:
    """
    Variance totale SSVI : w(k, t).

    Parameters
    ----------
    k : log-forward moneyness ln(K/F_T) — scalaire ou tableau
    t : maturité résiduelle (années) — scalaire ou tableau (même shape que k)

    Returns
    -------
    w : variance totale ≥ 0
    """
    k = np.asarray(k, dtype=float)
    t = np.asarray(t, dtype=float)

    theta = ssvi_theta(t, params.kappa, params.nu0, params.nu_inf)
    phi = ssvi_phi(theta, params.eta, params.lambda_)

    rho = params.rho
    x = phi * k + rho
    w = 0.5 * theta * (1.0 + rho * phi * k + np.sqrt(x ** 2 + 1.0 - rho ** 2))
    return np.maximum(w, 0.0)


def ssvi_implied_vol(
    k: float | np.ndarray,
    t: float | np.ndarray,
    params: SSVIParams,
) -> float | np.ndarray:
    """
    Volatilité implicite SSVI : σ(k, t) = √[w(k, t) / t]
    """
    k = np.asarray(k, dtype=float)
    t = np.asarray(t, dtype=float)
    safe_t = np.where(t < EPS, EPS, t)
    w = ssvi_total_variance(k, t, params)
    return np.sqrt(np.maximum(w / safe_t, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Vérification des contraintes de non-arbitrage
# ─────────────────────────────────────────────────────────────────────────────

def check_butterfly_arbitrage(params: SSVIParams, t_grid: np.ndarray) -> bool:
    """
    Vérifie la condition de non-arbitrage butterfly (condition suffisante) :
    η · (1 + |ρ|) ≤ 4  (Gatheral & Jacquier, Prop. 4.2 pour power-law SSVI)

    Retourne True si la condition est satisfaite.
    """
    return params.eta * (1.0 + abs(params.rho)) <= 4.0


def check_calendar_spread_arbitrage(params: SSVIParams, t_grid: np.ndarray) -> bool:
    """
    Vérifie que θ(t) est croissante (pas d'arbitrage calendaire).
    Pour notre paramétrage, cela équivaut à ν_∞ ≥ 0 (toujours vrai avec bornes > 0).
    """
    theta_vals = ssvi_theta(t_grid, params.kappa, params.nu0, params.nu_inf)
    return bool(np.all(np.diff(theta_vals) >= 0))


def check_no_arbitrage(params: SSVIParams, t_grid: np.ndarray | None = None) -> dict[str, bool]:
    """
    Vérifie les contraintes de non-arbitrage SSVI.

    Returns
    -------
    dict avec les clés 'butterfly_ok', 'calendar_ok', 'all_ok'
    """
    if t_grid is None:
        t_grid = np.linspace(0.05, 3.0, 50)

    butterfly_ok = check_butterfly_arbitrage(params, t_grid)
    calendar_ok = check_calendar_spread_arbitrage(params, t_grid)
    return {
        "butterfly_ok": butterfly_ok,
        "calendar_ok": calendar_ok,
        "all_ok": butterfly_ok and calendar_ok,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Génération de la surface
# ─────────────────────────────────────────────────────────────────────────────

def build_ssvi_surface(
    params: SSVIParams,
    k_grid: np.ndarray | None = None,
    t_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit une grille de surface de volatilité implicite SSVI.

    Returns
    -------
    (K_mesh, T_mesh, IV_mesh) : grilles 2D pour tracé
    """
    if k_grid is None:
        k_grid = np.linspace(-1.0, 1.0, 100)
    if t_grid is None:
        t_grid = np.array([0.083, 0.25, 0.5, 1.0, 2.0])

    K_mesh, T_mesh = np.meshgrid(k_grid, t_grid)
    IV_mesh = ssvi_implied_vol(K_mesh, T_mesh, params)
    return K_mesh, T_mesh, IV_mesh
