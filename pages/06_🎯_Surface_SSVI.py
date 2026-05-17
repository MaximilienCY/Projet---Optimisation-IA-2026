"""
06_🎯_Surface_SSVI.py — Calibration SSVI et visualisation de la surface.
"""

import json

import numpy as np
import pandas as pd
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import AppConfig, DATA_EXPORTS_DIR
from src.volatility.ssvi import SSVIParams, ssvi_implied_vol, build_ssvi_surface
from src.volatility.calibration_objectives import calibrate_ssvi
from src.volatility.arbitrage_checks import check_calendar_spread_discrete, check_butterfly_density
from src.utils.io import save_json
from src.utils.plotting import (
    plot_iv_smile, plot_iv_surface, plot_iv_heatmap, plot_iv_errors
)

st.title("🎯 Surface SSVI — Calibration et Visualisation")
st.markdown("**Modèle SSVI** (Gatheral & Jacquier, 2014) :")
st.latex(
    r"w(k,t) = \frac{\theta_t}{2}"
    r"\left\{1 + \rho\,\phi(\theta_t)\,k"
    r"+ \sqrt{[\phi(\theta_t)\,k + \rho]^2 + (1-\rho^2)}\right\}"
)
st.markdown(
    r"avec $\theta_t = \nu_\infty t + \frac{\nu_0 - \nu_\infty}{\kappa}(1 - e^{-\kappa t})$"
    r" et $\phi(\theta) = \eta\,/\,[\theta^\lambda(1+\theta)^{1-\lambda}]$"
)

if "config" not in st.session_state:
    st.session_state.config = AppConfig()
cfg: AppConfig = st.session_state.config

if "df_iv" not in st.session_state:
    st.warning("⚠️ Calculez d'abord les vol implicites (page 5).")
    st.stop()

df_iv: pd.DataFrame = st.session_state.df_iv
df_iv_valid = df_iv[df_iv["iv"].notna() & (df_iv["iv"] > 0)].copy()

# ─── Calibration ──────────────────────────────────────────────────────────────
st.subheader("1️⃣ Calibration en Deux Étapes")

with st.expander("ℹ️ Détail de la méthode", expanded=False):
    st.markdown(
        r"""
        **Étape 1** — Terme de structure ATM :
        On extrait la variance totale ATM $\hat{\theta}_t = \sigma^2_{ATM}(t) \cdot t$
        pour chaque maturité disponible. On ajuste $(\kappa, \nu_0, \nu_\infty)$.

        **Étape 2** — Smile (conditionnelle à l'étape 1) :
        On minimise $\sum_{i,j}(w_{SSVI}(k_{ij}, t_i) - \hat{w}_{ij})^2$
        par rapport à $(\rho, \eta, \lambda)$ avec pénalité butterfly.
        """
    )

atm_band = st.slider("Bande ATM pour l'étape 1 (|k| < bande)", 0.01, 0.20, 0.05, 0.01)

if st.button("🎯 Calibrer SSVI", type="primary"):
    with st.spinner("Calibration SSVI en cours (peut prendre 30-60s)..."):
        try:
            ssvi_params, ssvi_report = calibrate_ssvi(
                df_iv=df_iv_valid,
                cfg=cfg.ssvi,
                atm_band=atm_band,
            )
            st.session_state.ssvi_params = ssvi_params
            st.session_state.ssvi_report = ssvi_report
            st.success("✅ Calibration SSVI terminée !")
        except Exception as e:
            st.error(f"❌ Erreur de calibration : {e}")
            raise

if "ssvi_params" not in st.session_state:
    st.info("Cliquez pour calibrer la surface SSVI.")
    st.stop()

ssvi_params: SSVIParams = st.session_state.ssvi_params
ssvi_report: dict = st.session_state.ssvi_report

# ─── Paramètres calibrés ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Paramètres SSVI Calibrés")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Terme de structure**")
    st.metric("κ (mean reversion)", f"{ssvi_params.kappa:.4f}")
    st.metric("ν₀ (var. initiale)", f"{ssvi_params.nu0:.4f}")
    st.metric("ν_∞ (var. long terme)", f"{ssvi_params.nu_inf:.4f}")
with col2:
    st.markdown("**Smile**")
    st.metric("ρ (corrélation)", f"{ssvi_params.rho:.4f}")
    st.metric("η (amplitude skew)", f"{ssvi_params.eta:.4f}")
    st.metric("λ (power-law)", f"{ssvi_params.lambda_:.4f}")
with col3:
    st.markdown("**Qualité**")
    step2 = ssvi_report.get("step2", {})
    st.metric("RMSE IV", f"{step2.get('rmse_iv', 0)*100:.3f}%")
    st.metric("R² IV", f"{step2.get('r2_iv', 0):.6f}")
    st.metric("N points", f"{step2.get('n_points', 0):,}")

# ─── Vérification no-arbitrage ────────────────────────────────────────────────
st.subheader("✅ Vérification de Non-Arbitrage")
butterfly_density = check_butterfly_density(ssvi_params)

# Condition analytique butterfly suffisante (Gatheral & Jacquier 2014)
butterfly_analytic_ok = ssvi_params.eta * (1 + abs(ssvi_params.rho)) <= 4
butterfly_val = ssvi_params.eta * (1 + abs(ssvi_params.rho))

# Calendar spread : θ_t croissant ⟺ ν_∞ > 0 et κ > 0
calendar_ok = ssvi_params.nu_inf > 0 and ssvi_params.kappa > 0

col1, col2, col3 = st.columns(3)
col1.metric(
    "Condition butterfly (η(1+|ρ|)≤4)",
    "✅ OK" if butterfly_analytic_ok else "❌ Violée",
    delta=f"η(1+|ρ|) = {butterfly_val:.3f}",
)
col2.metric(
    "Calendar spread (θ_t croissant)",
    "✅ OK" if calendar_ok else "❌ Violée",
)
col3.metric(
    "Densité min (no-butterfly numérique)",
    f"{butterfly_density['min_density']:.4f}",
    delta="OK" if butterfly_density["butterfly_ok"] else "Violation",
)

# ─── Ajout IV SSVI au DataFrame ───────────────────────────────────────────────
df_iv_valid["iv_ssvi"] = ssvi_implied_vol(
    df_iv_valid["log_moneyness"].values,
    df_iv_valid["T"].values,
    ssvi_params,
)
df_iv_valid["iv_error"] = df_iv_valid["iv"] - df_iv_valid["iv_ssvi"]

# ─── Visualisations ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Smile de Volatilité : Marché vs SSVI")

all_mats = sorted(df_iv_valid["T"].unique())
# Même fix que page 05 : utiliser les vraies valeurs float, pas des arrondis
selected_mats = st.multiselect(
    "Maturités à afficher",
    options=all_mats,
    default=all_mats[:min(4, len(all_mats))],
    format_func=lambda t: f"T={t:.4f}a ({t * 365:.0f}j)",
)
if selected_mats:
    # plot_iv_smile filtre internement via np.isclose(T, atol=0.02)
    fig_smile = plot_iv_smile(df_iv_valid, selected_mats, iv_ssvi_col="iv_ssvi")
    st.plotly_chart(fig_smile, use_container_width=True)

# Surface 3D / heatmap
st.subheader("🌋 Surface de Volatilité SSVI")
tab1, tab2 = st.tabs(["Surface 3D", "Heatmap"])

k_grid = np.linspace(-1.5, 1.5, 80)
t_grid = np.linspace(0.05, max(all_mats) * 1.05 if all_mats else 2.0, 30)
K_m, T_m, IV_m = build_ssvi_surface(ssvi_params, k_grid, t_grid)

with tab1:
    fig_surf = plot_iv_surface(k_grid, t_grid, IV_m)
    st.plotly_chart(fig_surf, use_container_width=True)

with tab2:
    fig_heat = plot_iv_heatmap(k_grid, t_grid, IV_m)
    st.plotly_chart(fig_heat, use_container_width=True)

# Erreurs
st.subheader("📊 Erreurs de Calibration (IV marché - IV SSVI)")
fig_err = plot_iv_errors(df_iv_valid)
st.plotly_chart(fig_err, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Erreur médiane", f"{df_iv_valid['iv_error'].median()*100:.3f}%")
col2.metric("RMSE IV", f"{np.sqrt(np.mean(df_iv_valid['iv_error']**2))*100:.3f}%")
col3.metric("Max erreur abs", f"{df_iv_valid['iv_error'].abs().max()*100:.3f}%")

# Mettre à jour session_state avec IV SSVI
st.session_state.df_iv = df_iv_valid

# ─── Exports ──────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "📥 Paramètres SSVI (JSON)",
        data=json.dumps({**ssvi_params.to_dict(), "report": ssvi_report}, indent=2, default=str),
        file_name="ssvi_params.json", mime="application/json",
    )
with col2:
    st.download_button(
        "📥 Comparaison marché vs SSVI (CSV)",
        data=df_iv_valid.to_csv(index=False),
        file_name="iv_ssvi_comparison.csv", mime="text/csv",
    )
