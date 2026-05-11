"""
05_🌊_Volatilité_Implicite.py — Vol implicite, Newton vs Dichotomie, Grecques.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import AppConfig, DATA_PROCESSED_DIR
from src.rates.nelson_siegel import nelson_siegel_rate
from src.pricing.implied_vol import compute_implied_vols, IVMethod
from src.pricing.greeks import compute_greeks_dataframe
from src.utils.io import save_csv
from src.utils.plotting import plot_iv_smile, plot_newton_vs_bisection

st.title("🌊 Volatilités Implicites")
st.markdown(
    """
    Extraction de la volatilité implicite par la **formule de Black (1976)**.
    - **Newton-Raphson** : convergence quadratique (~5 itérations) quand Vega ≫ 0
    - **Dichotomie** : robuste, utilisée en fallback si Newton diverge
    """
)

if "config" not in st.session_state:
    st.session_state.config = AppConfig()
cfg: AppConfig = st.session_state.config

# Vérification des dépendances
if "df_clean" not in st.session_state:
    st.warning("⚠️ Chargez et nettoyez d'abord les données (pages 2-3).")
    st.stop()
if "ns_params" not in st.session_state:
    st.warning("⚠️ Calibrez d'abord Nelson-Siegel (page 4).")
    st.stop()

df_clean: pd.DataFrame = st.session_state.df_clean.copy()
ns_params = st.session_state.ns_params

# Ajout du taux r(T) via Nelson-Siegel
df_clean["rate"] = nelson_siegel_rate(df_clean["T"].values, ns_params)

# ─── Calcul des vol implicites ────────────────────────────────────────────────
st.subheader("1️⃣ Extraction des Volatilités Implicites")

price_col = st.selectbox("Colonne de prix à utiliser", ["mid", "mark_price"], index=0)

if st.button("🌊 Calculer les volatilités implicites", type="primary"):
    with st.spinner("Extraction IV (Newton-Raphson + dichotomie)..."):
        df_iv = compute_implied_vols(df_clean, price_col=price_col, cfg=cfg.implied_vol)
        # Ajout de la variance totale
        df_iv["total_variance"] = df_iv["iv"] ** 2 * df_iv["T"]
        st.session_state.df_iv = df_iv
        st.success(
            f"✅ {df_iv['iv'].notna().sum():,} / {len(df_iv):,} vol implicites extraites"
        )

if "df_iv" not in st.session_state:
    st.info("Cliquez pour extraire les vol implicites.")
    st.stop()

df_iv: pd.DataFrame = st.session_state.df_iv

# ─── Statistiques de convergence ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Diagnostics de Convergence")

n_total = len(df_iv)
n_ok = df_iv["iv"].notna().sum()
n_newton = (df_iv["iv_method"] == IVMethod.NEWTON.value).sum()
n_bisect = (df_iv["iv_method"] == IVMethod.BISECTION.value).sum()
n_failed = (df_iv["iv_method"] == IVMethod.FAILED.value).sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Extraites", f"{n_ok:,}")
col2.metric("Taux réussite", f"{100*n_ok/max(n_total,1):.1f}%")
col3.metric("Newton", f"{n_newton:,}")
col4.metric("Dichotomie", f"{n_bisect:,}")
col5.metric("Échecs", f"{n_failed:,}")

# Graphiques comparatifs
fig_conv = plot_newton_vs_bisection(df_iv)
st.plotly_chart(fig_conv, use_container_width=True)

# ─── Comparaison prix marché vs prix BS ───────────────────────────────────────
st.subheader("📊 Comparaison Marché vs Black-Scholes")
df_valid = df_iv[df_iv["iv"].notna()].copy()

if not df_valid.empty:
    rmse_price = np.sqrt(np.mean((df_valid[price_col] - df_valid["bs_price"]) ** 2))
    st.metric("RMSE prix (marché vs BS)", f"${rmse_price:.4f}")

    # Scatter
    fig_compare = px.scatter(
        df_valid.sample(min(500, len(df_valid))),
        x=price_col, y="bs_price",
        color="option_type",
        title="Prix marché vs Prix Black-Scholes reconstitué",
        labels={price_col: "Prix marché ($)", "bs_price": "Prix BS ($)"},
        template="plotly_white",
        opacity=0.6,
    )
    # Ligne y=x
    max_p = df_valid[price_col].max()
    fig_compare.add_scatter(x=[0, max_p], y=[0, max_p], mode="lines",
                            line=dict(color="black", dash="dash"), name="y=x")
    st.plotly_chart(fig_compare, use_container_width=True)

# ─── Smile de volatilité ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Smile de Volatilité Implicite")

available_maturities = sorted(df_valid["T"].unique())
selected_mats = st.multiselect(
    "Maturités à afficher",
    options=[round(t, 3) for t in available_maturities],
    default=[round(t, 3) for t in available_maturities[:min(4, len(available_maturities))]],
)

if selected_mats:
    df_sub = df_valid[df_valid["T"].isin(selected_mats)]
    fig_smile = plot_iv_smile(df_sub, selected_mats)
    st.plotly_chart(fig_smile, use_container_width=True)

# ─── Grecques ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📐 Calcul des Grecques")

if st.button("📐 Calculer les Grecques"):
    with st.spinner("Calcul des grecques (Δ, Γ, V, θ, ρ)..."):
        df_greeks = compute_greeks_dataframe(df_valid)
        st.session_state.df_greeks = df_greeks
        st.success("✅ Grecques calculées.")

if "df_greeks" in st.session_state:
    df_greeks = st.session_state.df_greeks

    col1, col2, col3 = st.columns(3)
    col1.metric("Δ médian (calls)", f"{df_greeks[df_greeks['option_type']=='C']['delta'].median():.4f}")
    col2.metric("Γ médian", f"{df_greeks['gamma'].median():.6f}")
    col3.metric("V médian", f"{df_greeks['vega'].median():.2f}")

    cols_show = ["instrument_name", "option_type", "strike", "T", "iv",
                 "delta", "gamma", "vega", "theta", "rho"]
    st.dataframe(
        df_greeks[[c for c in cols_show if c in df_greeks.columns]].head(50),
        use_container_width=True,
    )

# ─── Exports ──────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "📥 Vol implicites (CSV)",
        data=df_valid.to_csv(index=False),
        file_name="implied_vols.csv", mime="text/csv",
    )
with col2:
    if "df_greeks" in st.session_state:
        st.download_button(
            "📥 Grecques (CSV)",
            data=st.session_state.df_greeks.to_csv(index=False),
            file_name="greeks.csv", mime="text/csv",
        )
