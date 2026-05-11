"""
pnl.py — Calcul du P&L sous scénario de stress à 1 semaine.

Scénario imposé par le sujet :
  - Spot : +10% en relatif   →  F_new = F * 1.10
  - Volatilité : -10% en absolu  →  σ_new(k, t) = σ_SSVI(k, t) - 0.10
  - Temps : - 1 semaine   →  T_new = T - 1/52

Convention "baisse de 10% en absolu" :
  Si σ_SSVI(k, T) = 0.60 (60%), alors σ_new = 0.50 (50%).
  Cette convention est cohérente avec les risk managers de dérivés
  qui mesurent les chocs de vol en points de pourcentage (niveau absolu).
  Elle est différente d'un choc relatif de 10% qui donnerait σ * 0.90.

P&L décomposition :
  P&L_total = P&L_produit + P&L_couverture
    P&L_produit  = - (V_product_new - V_product_old)   [on a vendu]
    P&L_hedge    = + (V_hedge_new   - V_hedge_old)      [on a acheté]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import StressConfig, CONFIG
from src.pricing.black_scholes import black_price
from src.pricing.greeks import compute_greeks
from src.volatility.ssvi import SSVIParams, ssvi_implied_vol
from src.products.structures import CallSpread
from src.hedge.portfolio import HedgePortfolio, HedgeInstrument
from src.utils.dates import shift_maturity

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Application du choc
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StressScenario:
    """Décrit le choc appliqué."""
    spot_shock_pct: float
    vol_shock_abs: float
    horizon_weeks: float

    def apply_to_forward(self, F: float) -> float:
        return F * (1.0 + self.spot_shock_pct)

    def apply_to_vol(self, iv: float) -> float:
        return max(iv + self.vol_shock_abs, 0.001)   # vol plancher à 0.1%

    def apply_to_maturity(self, T: float) -> float:
        return shift_maturity(T, self.horizon_weeks)


def reprice_call_spread(
    product: CallSpread,
    F_new: float,
    r: float,
    ssvi_params: SSVIParams,
    vol_shock_abs: float,
    T_new: float | None = None,
) -> tuple[float, float]:
    """
    Reprice un call spread après choc.

    Le choc de vol s'applique en soustrayant vol_shock_abs (en absolu) de la
    volatilité implicite SSVI à chaque strike.

    Returns
    -------
    (price_new, price_old)
    """
    T_use = T_new if T_new is not None else product.T

    # Prix ancien (avant choc)
    price_old = product.price(F=F_new / (1.0 + 0.10),  # on revient au F original
                               r=r, ssvi_params=ssvi_params)
    # En pratique on stocke déjà le prix initial, donc on le recalcule ici
    # pour la cohérence (cf. app.py)

    # Prix nouveau (après choc)
    k1 = np.log(product.K1 / F_new)
    k2 = np.log(product.K2 / F_new)
    iv1_base = float(ssvi_implied_vol(k1, T_use, ssvi_params))
    iv2_base = float(ssvi_implied_vol(k2, T_use, ssvi_params))
    iv1_stressed = max(iv1_base + vol_shock_abs, 0.001)
    iv2_stressed = max(iv2_base + vol_shock_abs, 0.001)

    c1_new = black_price(F_new, product.K1, T_use, r, iv1_stressed, "C")
    c2_new = black_price(F_new, product.K2, T_use, r, iv2_stressed, "C")
    price_new = c1_new - c2_new

    return price_new, price_old


def reprice_instrument(
    inst: HedgeInstrument,
    F_new: float,
    ssvi_params: SSVIParams,
    vol_shock_abs: float,
    T_new: float,
) -> float:
    """Reprice un instrument de couverture après choc."""
    if inst.option_type == "F":
        # Future : prix = F_new * e^{-r*T_new} ≈ F_new
        return F_new

    k = np.log(inst.strike / F_new)
    iv_base = float(ssvi_implied_vol(k, T_new, ssvi_params))
    iv_stressed = max(iv_base + vol_shock_abs, 0.001)
    return black_price(F_new, inst.strike, T_new, inst.rate, iv_stressed, inst.option_type)


# ─────────────────────────────────────────────────────────────────────────────
# Calcul du P&L complet
# ─────────────────────────────────────────────────────────────────────────────

def compute_stress_pnl(
    product: CallSpread,
    product_price_initial: float,
    portfolio: HedgePortfolio,
    hedge_instruments_initial_prices: list[float],
    F_initial: float,
    r: float,
    ssvi_params: SSVIParams,
    cfg: StressConfig | None = None,
) -> dict:
    """
    Calcule le P&L sous scénario de stress.

    Parameters
    ----------
    product                       : CallSpread initial
    product_price_initial         : prix initial du produit (avant choc)
    portfolio                     : HedgePortfolio avec quantités
    hedge_instruments_initial_prices : prix initiaux des instruments de couverture
    F_initial                     : forward price initial
    r                             : taux sans risque
    ssvi_params                   : paramètres SSVI courants
    cfg                           : configuration du stress

    Returns
    -------
    dict complet avec P&L décomposé
    """
    cfg = cfg or CONFIG.stress
    scenario = StressScenario(
        spot_shock_pct=cfg.spot_shock_pct,
        vol_shock_abs=cfg.vol_shock_abs,
        horizon_weeks=cfg.horizon_weeks,
    )

    F_new = scenario.apply_to_forward(F_initial)
    T_new = scenario.apply_to_maturity(product.T)
    vol_shock = scenario.vol_shock_abs

    # ── P&L Produit vendu ────────────────────────────────────────────────────
    # On a vendu le produit : on perd si sa valeur monte, on gagne si elle baisse.
    # Prix du produit après choc
    k1_new = np.log(product.K1 / F_new)
    k2_new = np.log(product.K2 / F_new)
    iv1_new_base = float(ssvi_implied_vol(k1_new, T_new, ssvi_params))
    iv2_new_base = float(ssvi_implied_vol(k2_new, T_new, ssvi_params))
    iv1_stressed = max(iv1_new_base + vol_shock, 0.001)
    iv2_stressed = max(iv2_new_base + vol_shock, 0.001)

    c1_new = black_price(F_new, product.K1, T_new, r, iv1_stressed, "C")
    c2_new = black_price(F_new, product.K2, T_new, r, iv2_stressed, "C")
    product_price_new = c1_new - c2_new

    # P&L = -(V_new - V_old) car position SHORT sur le produit
    pnl_product = -(product_price_new - product_price_initial)

    # ── P&L Couverture ────────────────────────────────────────────────────────
    pnl_hedge = 0.0
    hedge_details = []

    for inst, price_init in zip(portfolio.instruments, hedge_instruments_initial_prices):
        price_new = reprice_instrument(inst, F_new, ssvi_params, vol_shock, T_new)
        inst_pnl = inst.quantity * (price_new - price_init)
        pnl_hedge += inst_pnl
        hedge_details.append({
            "instrument": inst.instrument_name,
            "quantity": inst.quantity,
            "price_initial": price_init,
            "price_new": price_new,
            "pnl": inst_pnl,
        })

    pnl_total = pnl_product + pnl_hedge

    # ── Résumé ────────────────────────────────────────────────────────────────
    result = {
        # Paramètres du scénario
        "F_initial": F_initial,
        "F_new": F_new,
        "T_initial": product.T,
        "T_new": T_new,
        "vol_shock_abs": vol_shock,
        "spot_shock_pct": cfg.spot_shock_pct,

        # Produit
        "product_price_initial": product_price_initial,
        "product_price_new": product_price_new,
        "pnl_product": pnl_product,

        # Couverture
        "pnl_hedge": pnl_hedge,
        "hedge_details": pd.DataFrame(hedge_details),

        # Total
        "pnl_total": pnl_total,

        # Vols utilisées
        "iv1_initial": float(ssvi_implied_vol(np.log(product.K1 / F_initial), product.T, ssvi_params)),
        "iv2_initial": float(ssvi_implied_vol(np.log(product.K2 / F_initial), product.T, ssvi_params)),
        "iv1_stressed": iv1_stressed,
        "iv2_stressed": iv2_stressed,
    }

    logger.info(
        "Stress P&L : produit=%.4f couverture=%.4f total=%.4f",
        pnl_product, pnl_hedge, pnl_total,
    )
    return result


def build_pnl_waterfall_data(stress_result: dict) -> dict:
    """Prépare les données pour le graphique waterfall du P&L."""
    return {
        "P&L Produit vendu": stress_result["pnl_product"],
        "P&L Couverture": stress_result["pnl_hedge"],
        "P&L Total": stress_result["pnl_total"],
    }
