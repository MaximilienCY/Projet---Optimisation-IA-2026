"""
01_⚙️_Paramètres.py — Configuration globale de l'application.

Permet de modifier tous les paramètres configurables et les sauvegarde
dans st.session_state.config pour être accessible par les autres pages.
"""

import json
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (
    AppConfig, DeribitConfig, DataCleaningConfig,
    NelsonSiegelConfig, ImpliedVolConfig, SSVIConfig,
    HedgeConfig, StressConfig, ProductConfig,
)

st.title("⚙️ Paramètres Globaux")
st.markdown(
    "Configurez tous les paramètres du projet. "
    "Les modifications sont prises en compte immédiatement dans toutes les pages."
)

# ─── Initialisation ───────────────────────────────────────────────────────────
if "config" not in st.session_state:
    st.session_state.config = AppConfig()

cfg: AppConfig = st.session_state.config

# ─── Deribit ──────────────────────────────────────────────────────────────────
st.subheader("🔗 API Deribit")
col1, col2 = st.columns(2)
with col1:
    currency = st.selectbox("Sous-jacent", ["BTC", "ETH"], index=0,
                            help="BTC : liquidité maximale sur Deribit")
    cfg.deribit.currency = currency
with col2:
    timeout = st.number_input("Timeout API (s)", 5, 120, cfg.deribit.timeout)
    cfg.deribit.timeout = timeout

st.info(
    "**Choix du sous-jacent : BTC**\n"
    "BTC présente la profondeur de carnet et le volume d'options les plus importants "
    "sur Deribit (~70% du volume total). Les maturités disponibles vont de quelques "
    "jours à 3 ans, avec plusieurs milliers de strikes actifs."
)

# ─── Nettoyage ────────────────────────────────────────────────────────────────
st.subheader("🧹 Nettoyage des Données")
col1, col2, col3 = st.columns(3)
with col1:
    spread_pct = st.slider(
        "Spread max (% du mid)", 5, 100, int(cfg.cleaning.max_spread_pct * 100), 5,
        help="Exclure les options avec (ask-bid)/mid > seuil"
    )
    cfg.cleaning.max_spread_pct = spread_pct / 100.0

with col2:
    min_T = st.number_input(
        "Maturité min (jours)", 1, 30, int(cfg.cleaning.min_time_to_maturity * 365),
        help="Exclure les options expirant dans moins de N jours"
    )
    cfg.cleaning.min_time_to_maturity = min_T / 365.0

with col3:
    max_T = st.number_input(
        "Maturité max (années)", 0.5, 5.0, cfg.cleaning.max_time_to_maturity, 0.5,
        help="Exclure les options de très longue maturité (peu liquides)"
    )
    cfg.cleaning.max_time_to_maturity = max_T

col4, col5 = st.columns(2)
with col4:
    min_m = st.number_input(
        "Moneyness min (K/F)", 0.1, 0.9, cfg.cleaning.min_moneyness, 0.05,
        help="Exclure les options deep ITM (moneyness < seuil)"
    )
    cfg.cleaning.min_moneyness = min_m
with col5:
    max_m = st.number_input(
        "Moneyness max (K/F)", 1.1, 5.0, cfg.cleaning.max_moneyness, 0.05,
        help="Exclure les options deep OTM (moneyness > seuil)"
    )
    cfg.cleaning.max_moneyness = max_m

# ─── Nelson-Siegel ────────────────────────────────────────────────────────────
st.subheader("📈 Nelson-Siegel")
col1, col2 = st.columns(2)
with col1:
    ns_lambda_min = st.number_input("λ min", 0.01, 1.0, cfg.nelson_siegel.lambda_bounds[0], 0.01)
    ns_lambda_max = st.number_input("λ max", 1.0, 20.0, cfg.nelson_siegel.lambda_bounds[1], 0.5)
    cfg.nelson_siegel.lambda_bounds = (ns_lambda_min, ns_lambda_max)
with col2:
    ns_starts = st.number_input("Multi-start (n)", 5, 100, cfg.nelson_siegel.n_starts,
                                help="Nombre de points de départ pour l'optimisation")
    cfg.nelson_siegel.n_starts = ns_starts

# ─── Vol Implicite ────────────────────────────────────────────────────────────
st.subheader("🌊 Volatilité Implicite")
col1, col2 = st.columns(2)
with col1:
    iv_lo = st.number_input("IV min (fraction)", 1e-5, 0.05, cfg.implied_vol.iv_lower, format="%.5f")
    cfg.implied_vol.iv_lower = iv_lo
with col2:
    iv_hi = st.number_input("IV max (fraction)", 2.0, 30.0, cfg.implied_vol.iv_upper)
    cfg.implied_vol.iv_upper = iv_hi

# ─── SSVI ─────────────────────────────────────────────────────────────────────
st.subheader("🎯 Calibration SSVI")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Bornes κ**")
    kappa_min = st.number_input("κ min", 0.001, 1.0, cfg.ssvi.kappa_bounds[0])
    kappa_max = st.number_input("κ max", 1.0, 20.0, cfg.ssvi.kappa_bounds[1])
    cfg.ssvi.kappa_bounds = (kappa_min, kappa_max)
with col2:
    st.markdown("**Bornes ρ**")
    rho_min = st.number_input("ρ min", -0.9999, 0.0, cfg.ssvi.rho_bounds[0])
    rho_max = st.number_input("ρ max", 0.0, 0.9999, cfg.ssvi.rho_bounds[1])
    cfg.ssvi.rho_bounds = (rho_min, rho_max)
with col3:
    st.markdown("**Bornes η, λ**")
    eta_max = st.number_input("η max", 1.0, 20.0, cfg.ssvi.eta_bounds[1])
    cfg.ssvi.eta_bounds = (cfg.ssvi.eta_bounds[0], eta_max)
    lam_max = st.number_input("λ_SSVI max", 0.1, 0.5, cfg.ssvi.lambda_bounds[1])
    cfg.ssvi.lambda_bounds = (cfg.ssvi.lambda_bounds[0], lam_max)

ssvi_starts = st.number_input("SSVI multi-start (n)", 5, 100, cfg.ssvi.n_starts)
cfg.ssvi.n_starts = ssvi_starts

# ─── Produit ──────────────────────────────────────────────────────────────────
st.subheader("💎 Produit Dérivé")
col1, col2, col3 = st.columns(3)
with col1:
    prod_type = st.selectbox("Type de produit", ["call_spread", "put_spread"],
                             index=0 if cfg.product.product_type == "call_spread" else 1)
    cfg.product.product_type = prod_type
with col2:
    k1_m = st.number_input("Strike K₁ (×F)", 0.70, 1.0, cfg.product.k1_moneyness, 0.01,
                           help="K₁ = k1_moneyness × F_T (jambe longue)")
    cfg.product.k1_moneyness = k1_m
with col3:
    k2_m = st.number_input("Strike K₂ (×F)", 1.0, 1.50, cfg.product.k2_moneyness, 0.01,
                           help="K₂ = k2_moneyness × F_T (jambe courte)")
    cfg.product.k2_moneyness = k2_m

# ─── Couverture ────────────────────────────────────────────────────────────────
st.subheader("🛡️ Couverture")
col1, col2, col3 = st.columns(3)
with col1:
    max_inst = st.number_input("Nb max instruments", 3, 50, cfg.hedge.max_hedge_instruments)
    cfg.hedge.max_hedge_instruments = max_inst
with col2:
    max_pos = st.number_input("Borne |q_i| max", 1.0, 200.0, cfg.hedge.max_position_size)
    cfg.hedge.max_position_size = max_pos
with col3:
    use_fut = st.checkbox("Inclure futures", value=cfg.hedge.use_futures)
    cfg.hedge.use_futures = use_fut

# ─── Stress ────────────────────────────────────────────────────────────────────
st.subheader("💥 Scénario de Stress")
col1, col2, col3 = st.columns(3)
with col1:
    spot_shock = st.number_input(
        "Choc spot (relatif)", 0.0, 0.50, cfg.stress.spot_shock_pct, 0.01,
        format="%.2f",
        help="+10% = 0.10"
    )
    cfg.stress.spot_shock_pct = spot_shock
with col2:
    vol_shock = st.number_input(
        "Choc vol (absolu)", -1.0, 0.0, cfg.stress.vol_shock_abs, 0.01,
        format="%.2f",
        help="-10% = -0.10 (en unités de σ)"
    )
    cfg.stress.vol_shock_abs = vol_shock
with col3:
    horizon = st.number_input("Horizon (semaines)", 0.5, 4.0, cfg.stress.horizon_weeks, 0.5)
    cfg.stress.horizon_weeks = horizon

# ─── Sauvegarde ───────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("💾 Sauvegarder la configuration"):
        st.session_state.config = cfg
        st.success("Configuration sauvegardée dans la session.")

with col2:
    cfg_dict = {
        "deribit": {"currency": cfg.deribit.currency, "timeout": cfg.deribit.timeout},
        "cleaning": {"max_spread_pct": cfg.cleaning.max_spread_pct,
                     "min_T": cfg.cleaning.min_time_to_maturity,
                     "max_T": cfg.cleaning.max_time_to_maturity},
        "product": {"type": cfg.product.product_type,
                    "k1": cfg.product.k1_moneyness, "k2": cfg.product.k2_moneyness},
        "stress": {"spot_shock": cfg.stress.spot_shock_pct,
                   "vol_shock": cfg.stress.vol_shock_abs,
                   "horizon_weeks": cfg.stress.horizon_weeks},
    }
    st.download_button(
        "📥 Exporter config (JSON)",
        data=json.dumps(cfg_dict, indent=2),
        file_name="config.json",
        mime="application/json",
    )
