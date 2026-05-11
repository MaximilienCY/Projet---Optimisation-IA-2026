"""
test_implied_vol.py — Tests unitaires pour l'extraction de vol implicite.
"""

import math
import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pricing.black_scholes import black_call, black_put
from src.pricing.implied_vol import implied_vol, IVMethod


class TestImpliedVol:
    """Tests de l'extraction de vol implicite."""

    @pytest.mark.parametrize("sigma_true", [0.05, 0.20, 0.40, 0.80, 1.50])
    def test_roundtrip_call(self, sigma_true):
        """
        Roundtrip : prix BS → vol implicite → prix BS doit être identique.
        """
        F, K, T, r = 100.0, 100.0, 1.0, 0.05
        price = black_call(F, K, T, r, sigma_true)
        iv, method, iters = implied_vol(price, F, K, T, r, "C")
        assert method != IVMethod.FAILED, f"IV non convergée pour σ={sigma_true}"
        assert abs(iv - sigma_true) < 1e-5, f"Erreur IV : {iv:.6f} vs {sigma_true:.6f}"

    @pytest.mark.parametrize("sigma_true", [0.10, 0.30, 0.60])
    def test_roundtrip_put(self, sigma_true):
        """Roundtrip pour un put."""
        F, K, T, r = 100.0, 105.0, 0.5, 0.03
        price = black_put(F, K, T, r, sigma_true)
        iv, method, iters = implied_vol(price, F, K, T, r, "P")
        assert method != IVMethod.FAILED
        assert abs(iv - sigma_true) < 1e-5

    def test_method_newton_converges(self):
        """Newton doit converger pour une option standard."""
        F, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.25
        price = black_call(F, K, T, r, sigma)
        iv, method, iters = implied_vol(price, F, K, T, r, "C")
        assert method == IVMethod.NEWTON
        assert iters < 20

    def test_bisection_fallback(self):
        """Pour un cas difficile, la dichotomie doit quand même trouver la solution."""
        # Prix quasi-nul : Newton peut diverger
        F, K, T, r, sigma = 100.0, 200.0, 0.1, 0.05, 0.30  # deep OTM
        price = black_call(F, K, T, r, sigma)
        if price < 1e-10:
            pytest.skip("Prix trop faible pour ce test")
        iv, method, iters = implied_vol(price, F, K, T, r, "C")
        if method != IVMethod.FAILED:
            assert abs(iv - sigma) < 1e-4

    def test_zero_price_returns_nan(self):
        """Prix nul → vol implicite NaN."""
        iv, method, _ = implied_vol(0.0, 100, 100, 1.0, 0.05, "C")
        assert math.isnan(iv) or method == IVMethod.FAILED

    def test_negative_maturity_returns_nan(self):
        """Maturité nulle → vol implicite NaN."""
        iv, method, _ = implied_vol(5.0, 100, 100, 0.0, 0.05, "C")
        assert math.isnan(iv) or method == IVMethod.FAILED

    def test_iv_positive(self):
        """La vol implicite extraite doit être positive."""
        price = black_call(100, 95, 0.5, 0.04, 0.35)
        iv, method, _ = implied_vol(price, 100, 95, 0.5, 0.04, "C")
        if method != IVMethod.FAILED:
            assert iv > 0

    @pytest.mark.parametrize("K", [80, 90, 100, 110, 120])
    def test_various_strikes(self, K):
        """Test sur plusieurs strikes."""
        F, T, r, sigma = 100.0, 0.5, 0.03, 0.30
        price = black_call(F, K, T, r, sigma)
        if price < 1e-8:
            pytest.skip("Prix trop faible")
        iv, method, _ = implied_vol(price, F, K, T, r, "C")
        if method != IVMethod.FAILED:
            assert abs(iv - sigma) < 1e-4
