"""
03_🧹_Nettoyage.py — Nettoyage et diagnostic qualité des données.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import AppConfig, DATA_PROCESSED_DIR
from src.data.cleaning import clean_options
from src.data.validation import compute_spread_pct
from src.utils.io import save_csv
from src.utils.plotting import plot_cleaning_report, plot_spread_distribution

st.title("🧹 Nettoyage et Diagnostics Qualité")

if "config" not in st.session_state:
    st.session_state.config = AppConfig()
cfg: AppConfig = st.session_state.config

if "raw_data" not in st.session_state:
    st.warning("⚠️ Chargez d'abord les données (page 2).")
    st.stop()

# ─── Lancement du nettoyage ───────────────────────────────────────────────────
if st.button("🧹 Lancer le nettoyage", type="primary"):
    with st.spinner("Nettoyage en cours..."):
        df_clean, df_futures, report = clean_options(
            raw_data=st.session_state.raw_data,
            cfg=cfg.cleaning,
        )
        st.session_state.df_clean = df_clean
        st.session_state.df_futures = df_futures
        st.session_state.cleaning_report = report

if "cleaning_report" not in st.session_state:
    st.info("Cliquez sur 'Lancer le nettoyage' pour démarrer.")
    st.stop()

report = st.session_state.cleaning_report
df_clean: pd.DataFrame = st.session_state.df_clean

# ─── Métriques ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Rapport de Nettoyage")

n_init = report.get("initial", 0)
n_ret = report.get("retenues", 0)
n_rej = report.get("total_rejetées", 0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Lignes initiales", f"{n_init:,}")
m2.metric("Retenues", f"{n_ret:,}", delta=f"{100*n_ret/max(n_init,1):.1f}%")
m3.metric("Rejetées", f"{n_rej:,}", delta=f"-{100*n_rej/max(n_init,1):.1f}%", delta_color="inverse")
m4.metric("Maturités (clean)", df_clean["expiry_str"].nunique() if not df_clean.empty else 0)

# ─── Détail des rejets ────────────────────────────────────────────────────────
st.subheader("📋 Détail des Rejets")
rejection_data = {k: v for k, v in report.items()
                  if k not in ("initial", "retenues", "total_rejetées") and v > 0}
if rejection_data:
    fig_report = plot_cleaning_report(rejection_data)
    st.plotly_chart(fig_report, use_container_width=True)

    df_report = pd.DataFrame(
        list(rejection_data.items()), columns=["Motif de rejet", "Nombre de lignes"]
    )
    st.dataframe(df_report, use_container_width=True)
else:
    st.success("Aucun rejet détecté.")

# ─── Distribution spread bid-ask ──────────────────────────────────────────────
if not df_clean.empty and "spread_pct" in df_clean.columns:
    st.markdown("---")
    st.subheader("📊 Distribution des Spreads Bid-Ask (données retenues)")
    fig_spread = plot_spread_distribution(df_clean)
    st.plotly_chart(fig_spread, use_container_width=True)

    q50 = df_clean["spread_pct"].median() * 100
    q95 = df_clean["spread_pct"].quantile(0.95) * 100
    col1, col2 = st.columns(2)
    col1.metric("Spread médian", f"{q50:.2f}%")
    col2.metric("Spread p95", f"{q95:.2f}%")

# ─── Distribution par maturité ────────────────────────────────────────────────
if not df_clean.empty:
    st.markdown("---")
    st.subheader("📊 Distribution par Maturité et Type")

    fig_mat = px.histogram(
        df_clean, x="T", color="option_type", nbins=40,
        barmode="overlay",
        title="Distribution des maturités après nettoyage",
        labels={"T": "Maturité (années)", "option_type": "Type"},
        template="plotly_white",
    )
    st.plotly_chart(fig_mat, use_container_width=True)

    # Distribution des moneyness
    fig_m = px.histogram(
        df_clean, x="moneyness", color="option_type", nbins=50,
        barmode="overlay",
        title="Distribution des moneyness (K/F)",
        template="plotly_white",
    )
    st.plotly_chart(fig_m, use_container_width=True)

# ─── Aperçu ───────────────────────────────────────────────────────────────────
if not df_clean.empty:
    st.markdown("---")
    st.subheader("📋 Données Nettoyées (aperçu)")
    cols = ["instrument_name", "option_type", "strike", "T",
            "bid", "ask", "mid", "spread_pct", "log_moneyness", "moneyness"]
    st.dataframe(
        df_clean[[c for c in cols if c in df_clean.columns]].head(100),
        use_container_width=True,
    )

    # Sauvegarde
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Sauvegarder données nettoyées (CSV)"):
            path = os.path.join(DATA_PROCESSED_DIR,
                                f"options_{cfg.deribit.currency.lower()}_clean.csv")
            save_csv(df_clean, path)
            st.success(f"Sauvegardé : {path}")
    with col2:
        st.download_button(
            "📥 Télécharger données nettoyées (CSV)",
            data=df_clean.to_csv(index=False),
            file_name="options_clean.csv",
            mime="text/csv",
        )
