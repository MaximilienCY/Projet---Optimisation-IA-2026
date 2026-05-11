"""
04_📈_Taux_Nelson_Siegel.py — Extraction de la courbe des taux implicites + Nelson-Siegel.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import AppConfig, DATA_PROCESSED_DIR, DATA_EXPORTS_DIR
from src.rates.put_call_parity import extract_implied_rates
from src.rates.nelson_siegel import (
    calibrate_nelson_siegel, nelson_siegel_rate, NelsonSiegelParams,
)
from src.utils.io import save_csv, save_json
from src.utils.plotting import plot_rates_curve

st.title("📈 Courbe des Taux Implicites & Nelson-Siegel")
st.markdown(
    """
    **Méthode** : Extraction via la parité call-put de Black (1976).
    Pour chaque maturité T avec plusieurs paires call-put disponibles, on estime :
    ```
    r(T) = -ln[(C_mid - P_mid) / (F_T - K)] / T
    ```
    Puis on lisse avec le modèle Nelson-Siegel (1987) par moindres carrés.
    """
)

if "config" not in st.session_state:
    st.session_state.config = AppConfig()
cfg: AppConfig = st.session_state.config

if "df_clean" not in st.session_state:
    st.warning("⚠️ Lancez d'abord le nettoyage (page 3).")
    st.stop()

df_clean: pd.DataFrame = st.session_state.df_clean

# ─── Extraction des taux ──────────────────────────────────────────────────────
st.subheader("1️⃣ Extraction des Taux Implicites")
agg_method = st.radio("Méthode d'agrégation par maturité",
                      ["median", "mean"],
                      help="Médiane : robuste aux outliers (recommandé)")

if st.button("📊 Extraire les taux implicites", type="primary"):
    with st.spinner("Extraction des paires call-put et calcul des taux..."):
        df_rates = extract_implied_rates(df_clean, aggregation=agg_method)
        st.session_state.df_rates = df_rates

if "df_rates" not in st.session_state:
    st.info("Cliquez sur 'Extraire les taux implicites' pour démarrer.")
    st.stop()

df_rates: pd.DataFrame = st.session_state.df_rates

if df_rates.empty:
    st.error("Aucun taux implicite extrait. Vérifiez les données.")
    st.stop()

# ─── Affichage des taux empiriques ────────────────────────────────────────────
st.success(f"✅ {len(df_rates)} taux implicites extraits sur {len(df_rates)} maturités.")

st.dataframe(
    df_rates.assign(
        rate_pct=df_rates["rate"] * 100,
        rate_std_pct=df_rates["rate_std"] * 100,
    )[["expiry_str", "T", "rate_pct", "rate_std_pct", "n_pairs_used"]].rename(
        columns={"rate_pct": "Taux (%)", "rate_std_pct": "Std (%)",
                 "n_pairs_used": "Nb paires", "T": "Maturité (a)"}
    ),
    use_container_width=True,
)

# ─── Calibration Nelson-Siegel ────────────────────────────────────────────────
st.markdown("---")
st.subheader("2️⃣ Calibration Nelson-Siegel")

# Poids optionnels : 1/n_pairs (plus de poids aux maturités avec plus de paires)
use_weights = st.checkbox("Utiliser des poids (1/std²)", value=True)
weights = None
if use_weights:
    std_arr = df_rates["rate_std"].values
    # Évite la division par 0
    safe_std = np.where(std_arr < 1e-8, 1e-8, std_arr)
    weights = 1.0 / safe_std ** 2
    weights = weights / weights.sum()

if st.button("🎯 Calibrer Nelson-Siegel", type="primary"):
    with st.spinner("Calibration en cours (multi-start + évolution différentielle)..."):
        ns_params, ns_metrics = calibrate_nelson_siegel(
            maturities=df_rates["T"].values,
            rates=df_rates["rate"].values,
            weights=weights,
            cfg=cfg.nelson_siegel,
        )
        st.session_state.ns_params = ns_params
        st.session_state.ns_metrics = ns_metrics

if "ns_params" not in st.session_state:
    st.info("Cliquez sur 'Calibrer Nelson-Siegel' pour ajuster le modèle.")
    st.stop()

ns_params: NelsonSiegelParams = st.session_state.ns_params
ns_metrics: dict = st.session_state.ns_metrics

# ─── Résultats ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Résultats de la Calibration")

col1, col2, col3, col4 = st.columns(4)
col1.metric("β₀ (niveau long terme)", f"{ns_params.beta0*100:.4f}%")
col2.metric("β₁ (pente)", f"{ns_params.beta1*100:.4f}%")
col3.metric("β₂ (courbure)", f"{ns_params.beta2*100:.4f}%")
col4.metric("λ (décroissance)", f"{ns_params.lambda_:.4f}")

col5, col6 = st.columns(2)
col5.metric("RMSE", f"{ns_metrics.get('rmse', 0)*10000:.2f} bps")
col6.metric("R²", f"{ns_metrics.get('r2', 0):.6f}")

# ─── Graphique ────────────────────────────────────────────────────────────────
T_plot = np.linspace(0.01, max(df_rates["T"].max() * 1.1, 2.0), 200)
r_fitted = nelson_siegel_rate(T_plot, ns_params)

fig = plot_rates_curve(
    maturities_empirical=df_rates["T"].values,
    rates_empirical=df_rates["rate"].values,
    maturities_fitted=T_plot,
    rates_fitted=r_fitted,
    ns_params=ns_params.to_dict(),
)
st.plotly_chart(fig, use_container_width=True)

# ─── Extrapolation ────────────────────────────────────────────────────────────
st.subheader("🔍 Extrapolation")
T_query = st.number_input("Calculer r(T) pour T =", 0.01, 10.0, 1.0, 0.01, format="%.2f")
r_at_T = float(nelson_siegel_rate(np.array([T_query]), ns_params)[0])
st.info(f"r({T_query:.2f} an) = **{r_at_T*100:.4f}%**")

# ─── Exports ──────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.download_button(
        "📥 Taux empiriques (CSV)",
        data=df_rates.to_csv(index=False),
        file_name="rates_empirical.csv", mime="text/csv",
    )
with col2:
    ns_dict = {**ns_params.to_dict(), **ns_metrics}
    import json
    st.download_button(
        "📥 Paramètres NS (JSON)",
        data=json.dumps(ns_dict, indent=2, default=str),
        file_name="nelson_siegel_params.json", mime="application/json",
    )
with col3:
    T_export = np.arange(0.05, 5.05, 0.05)
    r_export = nelson_siegel_rate(T_export, ns_params)
    df_export = pd.DataFrame({"T": T_export, "rate_ns": r_export})
    st.download_button(
        "📥 Courbe NS (CSV)",
        data=df_export.to_csv(index=False),
        file_name="nelson_siegel_curve.csv", mime="text/csv",
    )
