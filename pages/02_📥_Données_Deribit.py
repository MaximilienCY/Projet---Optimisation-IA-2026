"""
02_📥_Données_Deribit.py — Récupération des données de marché.
"""

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import plotly.express as px

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import AppConfig, DATA_RAW_DIR
from src.api.deribit_client import DeribitClient, DeribitAPIError
from src.utils.io import save_json, save_csv, file_exists, load_json
from src.data.loaders import build_options_dataframe, build_futures_dataframe

st.title("📥 Données Deribit")

# ─── Config ───────────────────────────────────────────────────────────────────
if "config" not in st.session_state:
    st.session_state.config = AppConfig()
cfg: AppConfig = st.session_state.config

# ─── Sidebar info ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Paramètres actifs")
    st.markdown(f"**Sous-jacent** : {cfg.deribit.currency}")
    st.markdown(f"**Timeout** : {cfg.deribit.timeout}s")

# ─── Actions ──────────────────────────────────────────────────────────────────
st.subheader("🔄 Récupération des Données")
col1, col2 = st.columns(2)

with col1:
    fetch_btn = st.button("🌐 Télécharger depuis Deribit", type="primary",
                          help="Appelle l'API publique Deribit en temps réel")
with col2:
    # Charger données sauvegardées si elles existent
    raw_file = os.path.join(DATA_RAW_DIR, f"raw_{cfg.deribit.currency.lower()}.json")
    load_btn = st.button("📂 Charger données sauvegardées",
                         disabled=not file_exists(raw_file))

# ─── Fetch ────────────────────────────────────────────────────────────────────
if fetch_btn:
    with st.spinner(f"Récupération des données {cfg.deribit.currency} sur Deribit..."):
        try:
            client = DeribitClient(cfg.deribit)
            raw_data = client.fetch_all_options_data(currency=cfg.deribit.currency)
            st.session_state.raw_data = raw_data

            # Sauvegarde brute
            save_json(raw_data, raw_file)
            st.success(
                f"✅ {len(raw_data['options'])} options et {len(raw_data['futures'])} futures "
                f"récupérés. Spot = **{raw_data['spot']:,.0f} USD**"
            )
        except DeribitAPIError as e:
            st.error(f"❌ Erreur API Deribit : {e}")
        except Exception as e:
            st.error(f"❌ Erreur inattendue : {e}")

elif load_btn:
    with st.spinner("Chargement des données sauvegardées..."):
        try:
            raw_data = load_json(raw_file)
            st.session_state.raw_data = raw_data
            ts = raw_data.get("timestamp", "inconnue")
            st.success(f"✅ Données chargées (timestamp : {ts})")
        except Exception as e:
            st.error(f"❌ Impossible de charger : {e}")

# ─── Affichage ────────────────────────────────────────────────────────────────
if "raw_data" in st.session_state:
    raw = st.session_state.raw_data
    spot = raw.get("spot", 0)
    options = raw.get("options", [])
    futures = raw.get("futures", [])
    timestamp = raw.get("timestamp", "")

    # Métriques
    st.markdown("---")
    st.subheader("📊 Résumé")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spot BTC/USD", f"${spot:,.0f}")
    m2.metric("Options brutes", f"{len(options):,}")
    m3.metric("Futures actifs", f"{len(futures):,}")
    m4.metric("Timestamp", timestamp[:19] if timestamp else "—")

    # DataFrames structurés
    st.markdown("---")
    st.subheader("📋 DataFrames Structurés")

    tab1, tab2 = st.tabs(["Options", "Futures"])

    with tab1:
        with st.spinner("Construction du DataFrame options..."):
            df_raw = build_options_dataframe(
                raw_options=options,
                spot=spot,
                futures_data=futures,
            )
        if not df_raw.empty:
            st.success(f"{len(df_raw):,} options parsées")

            # Statistiques rapides
            col1, col2, col3 = st.columns(3)
            col1.metric("Maturités distinctes", df_raw["expiry_str"].nunique())
            col2.metric("Strikes min/max",
                        f"{df_raw['strike'].min():,.0f} / {df_raw['strike'].max():,.0f}")
            col3.metric("Calls / Puts",
                        f"{(df_raw['option_type']=='C').sum()} / {(df_raw['option_type']=='P').sum()}")

            # Aperçu
            cols_show = ["instrument_name", "option_type", "strike", "T",
                         "bid", "ask", "mid", "forward_price", "log_moneyness"]
            st.dataframe(
                df_raw[[c for c in cols_show if c in df_raw.columns]].head(50),
                use_container_width=True,
            )

            # Distribution des maturités
            fig = px.histogram(
                df_raw, x="T", nbins=30, color="option_type",
                title="Distribution des maturités (années)",
                labels={"T": "Maturité (années)", "count": "Nb options"},
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Sauvegarde DataFrame brut
            st.session_state.df_raw = df_raw
            csv_path = os.path.join(DATA_RAW_DIR, f"options_{cfg.deribit.currency.lower()}_raw.csv")
            if st.button("💾 Sauvegarder le DataFrame options (CSV)"):
                save_csv(df_raw, csv_path)
                st.success(f"Sauvegardé : {csv_path}")
        else:
            st.warning("Aucune option parsée correctement.")

    with tab2:
        with st.spinner("Construction du DataFrame futures..."):
            df_futures = build_futures_dataframe(raw_futures=futures, spot=spot)

        if not df_futures.empty:
            st.success(f"{len(df_futures)} futures parsés")
            st.dataframe(df_futures[["instrument_name", "T", "bid", "ask", "mid"]],
                         use_container_width=True)
            st.session_state.df_futures = df_futures
        else:
            st.info("Aucun future à terme trouvé (perpetual exclu).")

else:
    st.info("👆 Téléchargez ou chargez les données pour commencer.")
