"""
test_nelson_siegel.py — Tests unitaires pour Nelson-Siegel.
"""

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rates.nelson_siegel import (
    nelson_siegel_rate, NelsonSiegelParams, calibrate_nelson_siegel, ns_factor
)


class TestNelsonSiegel:
    """Tests du modèle Nelson-Siegel."""

    def test_flat_curve(self):
        """β₁=β₂=0 → courbe plate à β₀."""
        params = NelsonSiegelParams(beta0=0.03, beta1=0.0, beta2=0.0, lambda_=1.0)
        T = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
        rates = nelson_siegel_rate(T, params)
        assert np.allclose(rates, 0.03, atol=1e-10)

    def test_rate_at_infinity_is_beta0(self):
        """r(T→∞) → β₀ car les facteurs f1 et f2 → 0."""
        params = NelsonSiegelParams(beta0=0.04, beta1=-0.02, beta2=0.01, lambda_=0.5)
        T_large = np.array([50.0, 100.0])
        rates = nelson_siegel_rate(T_large, params)
        assert np.allclose(rates, 0.04, atol=1e-3)

    def test_rate_at_zero_is_beta0_plus_beta1(self):
        """r(T→0) → β₀ + β₁ car f1→1 et f2→0."""
        params = NelsonSiegelParams(beta0=0.04, beta1=-0.02, beta2=0.005, lambda_=1.0)
        T_small = np.array([0.001, 0.0001])
        rates = nelson_siegel_rate(T_small, params)
        expected = params.beta0 + params.beta1
        assert np.allclose(rates, expected, atol=1e-3)

    def test_ns_factor_short_limit(self):
        """f1(T≈0) ≈ 1."""
        f1, f2 = ns_factor(np.array([1e-6]), lambda_=1.0)
        assert abs(f1[0] - 1.0) < 1e-4
        assert abs(f2[0]) < 1e-4

    def test_ns_factor_long_limit(self):
        """f1(T→∞) → 0, f2(T→∞) → 0."""
        f1, f2 = ns_factor(np.array([1000.0]), lambda_=1.0)
        assert f1[0] < 0.01
        assert f2[0] < 0.01

    def test_calibration_flat_curve(self):
        """Calibration sur courbe plate : β₀ ≈ taux, β₁ ≈ β₂ ≈ 0."""
        T = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
        rates = np.full_like(T, 0.02)
        params, metrics = calibrate_nelson_siegel(T, rates)
        assert abs(params.beta0 - 0.02) < 1e-4
        assert abs(params.beta1) < 1e-3
        assert metrics["rmse"] < 1e-4

    def test_calibration_upward_slope(self):
        """Calibration sur courbe croissante."""
        T = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        rates = 0.01 + 0.005 * T  # courbe linéaire croissante
        params, metrics = calibrate_nelson_siegel(T, rates)
        assert metrics["r2"] > 0.95

    def test_calibration_roundtrip(self):
        """Générer des taux depuis paramètres connus puis récalibrer."""
        true_params = NelsonSiegelParams(beta0=0.03, beta1=-0.01, beta2=0.02, lambda_=1.5)
        T = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
        rates_true = nelson_siegel_rate(T, true_params)
        # Ajoute un bruit léger
        rng = np.random.default_rng(42)
        rates_noisy = rates_true + rng.normal(0, 0.0001, len(T))
        params_est, metrics = calibrate_nelson_siegel(T, rates_noisy)
        # Le RMSE doit être très faible
        assert metrics["rmse"] < 0.005

    def test_params_to_dict_and_back(self):
        """Conversion to_dict / from_array."""
        p = NelsonSiegelParams(beta0=0.03, beta1=-0.01, beta2=0.02, lambda_=1.5)
        arr = p.to_array()
        p2 = NelsonSiegelParams.from_array(arr)
        assert abs(p.beta0 - p2.beta0) < 1e-12
        assert abs(p.lambda_ - p2.lambda_) < 1e-12
