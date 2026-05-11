"""
settings.py — Configuration centralisée du projet.

Tous les paramètres configurables sont ici. Aucune constante magique
ne doit exister dans le reste du code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

# ─────────────────────────────────────────────
# Répertoires
# ─────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_HERE)

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DATA_EXPORTS_DIR = os.path.join(BASE_DIR, "data", "exports")

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_EXPORTS_DIR):
    os.makedirs(_d, exist_ok=True)


# ─────────────────────────────────────────────
# API Deribit
# ─────────────────────────────────────────────
@dataclass
class DeribitConfig:
    """Paramètres d'accès à l'API publique Deribit."""

    base_url: str = "https://www.deribit.com/api/v2"
    currency: str = "BTC"         # "BTC" ou "ETH"
    timeout: int = 30             # secondes
    max_retries: int = 3
    retry_delay: float = 1.0      # secondes entre tentatives


# ─────────────────────────────────────────────
# Nettoyage des données
# ─────────────────────────────────────────────
@dataclass
class DataCleaningConfig:
    """Seuils de filtrage des options."""

    max_spread_pct: float = 0.25      # Exclure si (ask-bid)/mid > seuil
    min_time_to_maturity: float = 1 / 365.0   # ≥ 1 jour
    max_time_to_maturity: float = 3.0          # ≤ 3 ans
    min_strike: float = 1.0
    min_mid_price: float = 1e-6       # Prix positif
    min_vega_iv: float = 1e-6         # Vega minimum pour IV

    # Filtres OTM/ITM extrêmes (moneyness = K/S ou K/F)
    min_moneyness: float = 0.50       # K/F >= 0.50 (deep ITM exclu)
    max_moneyness: float = 2.00       # K/F <= 2.00 (deep OTM exclu)


# ─────────────────────────────────────────────
# Calibration Nelson-Siegel
# ─────────────────────────────────────────────
@dataclass
class NelsonSiegelConfig:
    """Paramètres de calibration Nelson-Siegel."""

    lambda_bounds: Tuple[float, float] = (0.05, 10.0)
    beta0_bounds: Tuple[float, float] = (-0.20, 0.50)
    beta1_bounds: Tuple[float, float] = (-0.50, 0.50)
    beta2_bounds: Tuple[float, float] = (-0.50, 0.50)
    max_iter: int = 10_000
    tol: float = 1e-10
    n_starts: int = 20    # Multi-start pour robustesse


# ─────────────────────────────────────────────
# Black-Scholes / Vol implicite
# ─────────────────────────────────────────────
@dataclass
class ImpliedVolConfig:
    """Paramètres d'extraction de la volatilité implicite."""

    iv_lower: float = 1e-4      # Borne basse (0.01%)
    iv_upper: float = 20.0      # Borne haute (2000%)
    newton_max_iter: int = 100
    newton_tol: float = 1e-8
    bisect_max_iter: int = 200
    bisect_tol: float = 1e-8


# ─────────────────────────────────────────────
# Calibration SSVI
# ─────────────────────────────────────────────
@dataclass
class SSVIConfig:
    """
    Paramètres de calibration SSVI (Gatheral & Jacquier 2014).

    Paramétrage utilisé :
      w(k, t) = (θ_t/2) * {1 + ρ·φ(θ_t)·k + √[(φ(θ_t)·k + ρ)² + (1-ρ²)]}
      θ_t = ν_∞·t + (ν_0 - ν_∞)/κ · (1 - e^{-κt})   [variance totale ATM]
      φ(θ) = η / (θ^λ · (1+θ)^{1-λ})                  [paramètre de skew]
    """

    # Étape 1 : terme de structure ATM
    kappa_bounds: Tuple[float, float] = (0.01, 10.0)
    nu0_bounds: Tuple[float, float] = (1e-4, 5.0)
    nu_inf_bounds: Tuple[float, float] = (1e-4, 5.0)

    # Étape 2 : smile (conditionnelle à θ_t)
    rho_bounds: Tuple[float, float] = (-0.999, 0.999)
    eta_bounds: Tuple[float, float] = (1e-4, 10.0)
    lambda_bounds: Tuple[float, float] = (1e-4, 0.5)   # λ ∈ ]0, 1/2]

    max_iter: int = 5_000
    tol: float = 1e-10
    n_starts: int = 30


# ─────────────────────────────────────────────
# Produit dérivé
# ─────────────────────────────────────────────
@dataclass
class ProductConfig:
    """Paramètres du produit dérivé à pricer."""

    # Call spread (achat call K1, vente call K2, K1 < K2)
    product_type: str = "call_spread"
    # Les strikes seront exprimés relativement au spot (ex: K1=0.95*F, K2=1.05*F)
    k1_moneyness: float = 0.95    # K1 = k1_moneyness * F_T
    k2_moneyness: float = 1.05    # K2 = k2_moneyness * F_T
    maturity_label: str = "auto"  # "auto" = la maturité médiane disponible


# ─────────────────────────────────────────────
# Couverture Delta-Gamma-Vega
# ─────────────────────────────────────────────
@dataclass
class HedgeConfig:
    """Paramètres du problème d'optimisation de couverture."""

    max_hedge_instruments: int = 10    # Nb max d'instruments de couverture
    max_position_size: float = 50.0   # Borne sur |q_i|
    use_futures: bool = True          # Inclure les futures
    liquidity_weight: float = 1.0     # Pénalisation sur illiquidité
    regularization: float = 1e-4     # Régularisation L2 sur positions


# ─────────────────────────────────────────────
# Scénario de stress
# ─────────────────────────────────────────────
@dataclass
class StressConfig:
    """Paramètres du scénario de stress à 1 semaine."""

    spot_shock_pct: float = 0.10      # Choc spot +10% relatif
    vol_shock_abs: float = -0.10      # Choc vol -10% absolu (en unités de σ)
    horizon_weeks: float = 1.0        # Horizon : 1 semaine


# ─────────────────────────────────────────────
# Configuration globale
# ─────────────────────────────────────────────
@dataclass
class AppConfig:
    """Configuration globale de l'application."""

    deribit: DeribitConfig = field(default_factory=DeribitConfig)
    cleaning: DataCleaningConfig = field(default_factory=DataCleaningConfig)
    nelson_siegel: NelsonSiegelConfig = field(default_factory=NelsonSiegelConfig)
    implied_vol: ImpliedVolConfig = field(default_factory=ImpliedVolConfig)
    ssvi: SSVIConfig = field(default_factory=SSVIConfig)
    product: ProductConfig = field(default_factory=ProductConfig)
    hedge: HedgeConfig = field(default_factory=HedgeConfig)
    stress: StressConfig = field(default_factory=StressConfig)


# Instance partagée (modifiable depuis l'app Streamlit)
CONFIG = AppConfig()
