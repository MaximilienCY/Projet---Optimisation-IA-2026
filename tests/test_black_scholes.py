"""
test_black_scholes.py — Tests unitaires pour la formule de Black (1976).
"""

import math
import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pricing.black_scholes import (
    black_call, black_put, black_price, black_vega, black_d1_d2
)


class TestBlackScholes:
    """Tests de la formule de Black 1976."""

    def test_call_atm_positive(self):
        """Un call ATM doit avoir un prix positif."""
        price = black_call(F=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert price > 0

    def test_put_atm_positive(self):
        """Un put ATM doit avoir un prix positif."""
        price = black_put(F=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert price > 0

    def test_put_call_parity(self):
        """C - P = e^{-rT}(F - K) (parité call-put de Black)."""
        F, K, T, r, sigma = 100.0, 95.0, 0.5, 0.03, 0.25
        C = black_call(F, K, T, r, sigma)
        P = black_put(F, K, T, r, sigma)
        expected = math.exp(-r * T) * (F - K)
        assert abs((C - P) - expected) < 1e-10

    def test_call_deep_itm(self):
        """Call deep ITM ≈ e^{-rT} * (F - K)."""
        F, K, T, r, sigma = 200.0, 50.0, 1.0, 0.05, 0.20
        C = black_call(F, K, T, r, sigma)
        lower = math.exp(-r * T) * (F - K)
        assert C >= lower - 1e-6

    def test_call_deep_otm_near_zero(self):
        """Call deep OTM ≈ 0."""
        price = black_call(F=100, K=500, T=0.1, r=0.05, sigma=0.20)
        assert price < 0.01

    def test_call_intrinsic_at_expiry(self):
        """À maturité (T=0), call = max(F-K, 0)."""
        assert abs(black_call(100, 90, 0, 0.05, 0.20) - 10.0) < 1e-9
        assert abs(black_call(100, 110, 0, 0.05, 0.20) - 0.0) < 1e-9

    def test_put_intrinsic_at_expiry(self):
        """À maturité (T=0), put = max(K-F, 0)."""
        assert abs(black_put(100, 110, 0, 0.05, 0.20) - 10.0) < 1e-9
        assert abs(black_put(100, 90, 0, 0.05, 0.20) - 0.0) < 1e-9

    def test_vega_positive(self):
        """Vega doit être positif pour une option standard."""
        v = black_vega(F=100, K=100, T=1.0, r=0.05, sigma=0.30)
        assert v > 0

    def test_vega_at_expiry_zero(self):
        """Vega = 0 à maturité."""
        v = black_vega(F=100, K=100, T=0.0, r=0.05, sigma=0.30)
        assert v == 0.0

    def test_black_price_dispatch(self):
        """black_price dispatch call/put correctement."""
        F, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        assert abs(black_price(F, K, T, r, sigma, "C") - black_call(F, K, T, r, sigma)) < 1e-12
        assert abs(black_price(F, K, T, r, sigma, "P") - black_put(F, K, T, r, sigma)) < 1e-12

    def test_invalid_option_type_raises(self):
        """Type d'option invalide lève une ValueError."""
        with pytest.raises(ValueError):
            black_price(100, 100, 1.0, 0.05, 0.20, "X")

    def test_price_increases_with_vol(self):
        """Le prix augmente avec la volatilité (toutes choses égales)."""
        F, K, T, r = 100.0, 100.0, 1.0, 0.05
        p1 = black_call(F, K, T, r, 0.10)
        p2 = black_call(F, K, T, r, 0.30)
        assert p2 > p1

    def test_call_spread_positive(self):
        """Bull call spread (K1 < K2) a un prix positif."""
        F, T, r, sigma = 100.0, 1.0, 0.05, 0.25
        c1 = black_call(F, 95.0, T, r, sigma)
        c2 = black_call(F, 105.0, T, r, sigma)
        assert c1 - c2 > 0

    def test_d1_d2_relationship(self):
        """d2 = d1 - σ√T."""
        d1, d2 = black_d1_d2(100, 100, 1.0, 0.20)
        assert abs(d1 - d2 - 0.20) < 1e-10

    @pytest.mark.parametrize("sigma", [0.01, 0.10, 0.50, 1.50, 5.0])
    def test_call_non_negative_various_vol(self, sigma):
        """Le prix d'un call est toujours ≥ 0."""
        price = black_call(F=100, K=100, T=1.0, r=0.05, sigma=sigma)
        assert price >= 0
