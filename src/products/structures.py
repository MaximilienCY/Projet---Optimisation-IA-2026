"""
structures.py — Structures de produits dérivés.

Produit choisi : Bull Call Spread (achat call K₁, vente call K₂, K₁ < K₂).

Justification du choix :
  - Un call spread est plus réaliste qu'un call nu : coût réduit, risque borné.
  - Il permet de tester les deux strikes du modèle SSVI.
  - Sa couverture Delta-Gamma-Vega est plus riche qu'une option simple.
  - C'est un produit standard très utilisé en crypto trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from src.pricing.black_scholes import black_price
from src.pricing.greeks import compute_greeks, Greeks
from src.volatility.ssvi import SSVIParams, ssvi_implied_vol


class ProductType(str, Enum):
    CALL = "call"
    PUT = "put"
    CALL_SPREAD = "call_spread"
    PUT_SPREAD = "put_spread"


@dataclass
class CallSpread:
    """
    Bull Call Spread : long call K₁ + short call K₂ (K₁ < K₂).

    Prix = C(K₁) - C(K₂)
    Payoff à maturité = max(S-K₁, 0) - max(S-K₂, 0)
                      = min(max(S-K₁, 0), K₂-K₁)
    """
    K1: float   # strike bas (long call)
    K2: float   # strike haut (short call)
    T: float    # maturité (années)

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """Payoff à maturité pour un vecteur de prix spot S_T."""
        S_T = np.asarray(S_T, dtype=float)
        return np.maximum(S_T - self.K1, 0.0) - np.maximum(S_T - self.K2, 0.0)

    def price(
        self,
        F: float,
        r: float,
        ssvi_params: SSVIParams,
    ) -> float:
        """Prix du call spread via la surface SSVI."""
        k1 = np.log(self.K1 / F)
        k2 = np.log(self.K2 / F)
        iv1 = float(ssvi_implied_vol(k1, self.T, ssvi_params))
        iv2 = float(ssvi_implied_vol(k2, self.T, ssvi_params))
        c1 = black_price(F, self.K1, self.T, r, iv1, "C")
        c2 = black_price(F, self.K2, self.T, r, iv2, "C")
        return c1 - c2

    def greeks(
        self,
        F: float,
        r: float,
        ssvi_params: SSVIParams,
    ) -> Greeks:
        """Grecques du call spread (additivité)."""
        k1 = np.log(self.K1 / F)
        k2 = np.log(self.K2 / F)
        iv1 = float(ssvi_implied_vol(k1, self.T, ssvi_params))
        iv2 = float(ssvi_implied_vol(k2, self.T, ssvi_params))
        g1 = compute_greeks(F, self.K1, self.T, r, iv1, "C")
        g2 = compute_greeks(F, self.K2, self.T, r, iv2, "C")
        return g1 + (g2 * -1.0)  # long K1, short K2

    def ivols(self, F: float, ssvi_params: SSVIParams) -> tuple[float, float]:
        """Retourne (iv1, iv2) des deux jambes."""
        k1 = np.log(self.K1 / F)
        k2 = np.log(self.K2 / F)
        iv1 = float(ssvi_implied_vol(k1, self.T, ssvi_params))
        iv2 = float(ssvi_implied_vol(k2, self.T, ssvi_params))
        return iv1, iv2


@dataclass
class PutSpread:
    """
    Bear Put Spread : long put K₂ + short put K₁ (K₁ < K₂).

    Prix = P(K₂) - P(K₁)
    """
    K1: float
    K2: float
    T: float

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        S_T = np.asarray(S_T, dtype=float)
        return np.maximum(self.K2 - S_T, 0.0) - np.maximum(self.K1 - S_T, 0.0)

    def price(self, F: float, r: float, ssvi_params: SSVIParams) -> float:
        k1 = np.log(self.K1 / F)
        k2 = np.log(self.K2 / F)
        iv1 = float(ssvi_implied_vol(k1, self.T, ssvi_params))
        iv2 = float(ssvi_implied_vol(k2, self.T, ssvi_params))
        p2 = black_price(F, self.K2, self.T, r, iv2, "P")
        p1 = black_price(F, self.K1, self.T, r, iv1, "P")
        return p2 - p1

    def greeks(self, F: float, r: float, ssvi_params: SSVIParams) -> Greeks:
        k1 = np.log(self.K1 / F)
        k2 = np.log(self.K2 / F)
        iv1 = float(ssvi_implied_vol(k1, self.T, ssvi_params))
        iv2 = float(ssvi_implied_vol(k2, self.T, ssvi_params))
        g2 = compute_greeks(F, self.K2, self.T, r, iv2, "P")
        g1 = compute_greeks(F, self.K1, self.T, r, iv1, "P")
        return g2 + (g1 * -1.0)
