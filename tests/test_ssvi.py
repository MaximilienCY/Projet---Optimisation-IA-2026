"""
test_ssvi.py — Tests unitaires pour le modèle SSVI.
"""

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.volatility.ssvi import (
    SSVIParams, ssvi_theta, ssvi_phi, ssvi_total_variance, ssvi_implied_vol,
    check_no_arbitrage, build_ssvi_surface,
)
from src.volatility.arbitrage_checks import check_butterfly_density


class TestSSVI:
    """Tests du modèle SSVI."""

    @pytest.fixture
    def default_params(self) -> SSVIParams:
        return SSVIParams(kappa=1.0, nu0=0.04, nu_inf=0.04, rho=-0.3, eta=1.0, lambda_=0.3)

    def test_theta_positive(self, default_params):
        """θ(t) doit être positif pour t > 0."""
        T = np.array([0.1, 0.5, 1.0, 2.0])
        theta = ssvi_theta(T, default_params.kappa, default_params.nu0, default_params.nu_inf)
        assert np.all(theta > 0)

    def test_theta_short_limit(self):
        """θ(t)/t → ν₀ quand t → 0."""
        kappa, nu0, nu_inf = 1.0, 0.04, 0.06
        T_small = 1e-6
        theta = ssvi_theta(T_small, kappa, nu0, nu_inf)
        ratio = theta / T_small
        assert abs(ratio - nu0) < 0.01

    def test_theta_long_limit(self):
        """θ(t)/t → ν_∞ quand t → ∞."""
        kappa, nu0, nu_inf = 2.0, 0.04, 0.06
        T_large = 100.0
        theta = ssvi_theta(T_large, kappa, nu0, nu_inf)
        ratio = theta / T_large
        assert abs(ratio - nu_inf) < 0.001

    def test_phi_positive(self, default_params):
        """φ(θ) > 0."""
        theta = np.array([0.01, 0.05, 0.10, 0.50])
        phi = ssvi_phi(theta, default_params.eta, default_params.lambda_)
        assert np.all(phi > 0)

    def test_total_variance_positive(self, default_params):
        """w(k, t) ≥ 0 pour tout k ∈ [-2, 2] et t > 0."""
        k = np.linspace(-2, 2, 50)
        T = np.linspace(0.1, 2.0, 10)
        K_m, T_m = np.meshgrid(k, T)
        w = ssvi_total_variance(K_m.ravel(), T_m.ravel(), default_params)
        assert np.all(w >= 0)

    def test_atm_variance_matches_theta(self, default_params):
        """À k=0, w(0, t) = θ(t) (variance totale ATM = θ).

        Démonstration : w(0,t) = θ/2 · {1 + ρ·φ·0 + √[(φ·0+ρ)² + (1-ρ²)]}
                               = θ/2 · {1 + √[ρ² + 1 - ρ²]}
                               = θ/2 · {1 + 1} = θ
        """
        T = 1.0
        theta = ssvi_theta(T, default_params.kappa, default_params.nu0, default_params.nu_inf)
        expected = theta  # w(k=0, t) = θ exactement
        actual = float(ssvi_total_variance(0.0, T, default_params))
        assert abs(actual - expected) < 1e-10

    def test_implied_vol_positive(self, default_params):
        """La vol implicite SSVI est positive."""
        k = np.linspace(-1.5, 1.5, 30)
        T = 1.0
        iv = ssvi_implied_vol(k, T, default_params)
        assert np.all(iv > 0)

    def test_no_arbitrage_default_params(self, default_params):
        """Les paramètres par défaut satisfont les conditions de non-arbitrage."""
        result = check_no_arbitrage(default_params)
        assert result["butterfly_ok"], "Condition butterfly violée pour les paramètres par défaut"

    def test_build_surface_shape(self, default_params):
        """La surface construite a la bonne forme."""
        k_grid = np.linspace(-1, 1, 20)
        t_grid = np.array([0.25, 0.5, 1.0])
        K_m, T_m, IV_m = build_ssvi_surface(default_params, k_grid, t_grid)
        assert IV_m.shape == (3, 20)

    def test_vol_smile_positive_skew_for_negative_rho(self):
        """ρ < 0 → skew négatif (calls OTM ont moins de vol que puts OTM)."""
        params = SSVIParams(kappa=1.0, nu0=0.04, nu_inf=0.04, rho=-0.5, eta=1.5, lambda_=0.3)
        T = 1.0
        iv_otm_call = float(ssvi_implied_vol(0.3, T, params))   # OTM call
        iv_otm_put = float(ssvi_implied_vol(-0.3, T, params))   # OTM put
        # Avec ρ < 0, le smile est incliné vers la gauche : puts OTM > calls OTM
        assert iv_otm_put > iv_otm_call

    def test_butterfly_violation_detected(self):
        """Paramètres violant butterfly doivent être détectés."""
        # η*(1+|ρ|) > 4 → violation butterfly
        params_bad = SSVIParams(kappa=1.0, nu0=0.04, nu_inf=0.04, rho=0.9, eta=3.0, lambda_=0.3)
        result = check_no_arbitrage(params_bad)
        assert not result["butterfly_ok"]

    def test_density_check(self, default_params):
        """La densité implicite SSVI est globalement positive."""
        result = check_butterfly_density(
            default_params,
            k_grid=np.linspace(-1.0, 1.0, 50),
            t_grid=np.linspace(0.25, 2.0, 5),
        )
        assert result["butterfly_ok"], f"min_density={result['min_density']:.4f}"
