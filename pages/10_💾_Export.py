"""
10_💾_Export.py — Export de tous les résultats du projet.
"""

import json
from datetime import datetime

import pandas as pd
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import AppConfig

st.title("💾 Export des Résultats")
st.markdown(
    "Téléchargez tous les artefacts du projet. "
    "Les fichiers sont prêts pour votre rapport LaTeX."
)

if "config" not in st.session_state:
    st.session_state.config = AppConfig()


def _csv_btn(key: str, label: str, fname: str) -> None:
    if key in st.session_state:
        data = st.session_state[key]
        if isinstance(data, pd.DataFrame):
            st.download_button(label, data=data.to_csv(index=False),
                               file_name=fname, mime="text/csv", key=f"dl_{key}")
        else:
            st.caption(f"Type non-DataFrame pour {key}")
    else:
        st.caption(f"⚠️ {label} — pas encore calculé")


def _json_btn(obj_or_key, label: str, fname: str, key_suffix: str) -> None:
    if isinstance(obj_or_key, str):
        obj = st.session_state.get(obj_or_key)
    else:
        obj = obj_or_key
    if obj is None:
        st.caption(f"⚠️ {label} — pas encore calculé")
        return
    if hasattr(obj, "to_dict"):
        raw = json.dumps(obj.to_dict(), indent=2, default=str)
    elif isinstance(obj, dict):
        raw = json.dumps(obj, indent=2, default=str)
    else:
        raw = json.dumps(str(obj))
    st.download_button(label, data=raw, file_name=fname,
                       mime="application/json", key=f"dl_{key_suffix}")


# ─── Données brutes et nettoyées ─────────────────────────────────────────────
st.subheader("📦 Données Brutes et Nettoyées")
col1, col2 = st.columns(2)
with col1:
    _csv_btn("df_raw", "📥 Données brutes (CSV)", "deribit_raw.csv")
with col2:
    _csv_btn("df_clean", "📥 Données nettoyées (CSV)", "deribit_clean.csv")

if "cleaning_report" in st.session_state:
    _json_btn(st.session_state.cleaning_report, "📥 Rapport de nettoyage (JSON)",
              "cleaning_report.json", "clean_rep")

st.markdown("---")
st.subheader("📈 Taux Implicites et Nelson-Siegel")
col1, col2 = st.columns(2)
with col1:
    _csv_btn("df_rates", "📥 Taux implicites (CSV)", "implied_rates.csv")
with col2:
    _json_btn("ns_params", "📥 Paramètres Nelson-Siegel (JSON)", "ns_params.json", "ns_params")

st.markdown("---")
st.subheader("🌊 Volatilités Implicites et Grecques")
col1, col2 = st.columns(2)
with col1:
    _csv_btn("df_iv", "📥 Vol implicites + SSVI (CSV)", "implied_vols.csv")
with col2:
    _csv_btn("df_greeks", "📥 Grecques par option (CSV)", "greeks.csv")

st.markdown("---")
st.subheader("🎯 Calibration SSVI")
col1, col2 = st.columns(2)
with col1:
    _json_btn("ssvi_params", "📥 Paramètres SSVI (JSON)", "ssvi_params.json", "ssvi_params")
with col2:
    if "ssvi_report" in st.session_state:
        _json_btn(st.session_state.ssvi_report, "📥 Rapport calibration SSVI (JSON)",
                  "ssvi_calibration_report.json", "ssvi_rep")
    else:
        st.caption("⚠️ Rapport SSVI — pas encore calculé")

st.markdown("---")
st.subheader("💎 Produit Dérivé")
if "product" in st.session_state:
    p = st.session_state.product
    g = st.session_state.get("product_greeks")
    summary = {
        "produit": "bull_call_spread",
        "K1": p.K1, "K2": p.K2, "T": p.T,
        "r": st.session_state.get("product_r"),
        "F": st.session_state.get("product_F"),
        "prix": st.session_state.get("product_price"),
        "greeks": g.to_dict() if g else None,
    }
    _json_btn(summary, "📥 Fiche produit (JSON)", "product_callspread.json", "product")
else:
    st.caption("⚠️ Produit non configuré")

st.markdown("---")
st.subheader("🛡️ Portefeuille de Couverture")
if "portfolio" in st.session_state:
    port = st.session_state.portfolio
    rows = []
    for inst in port.instruments:
        rows.append({
            "instrument": inst.instrument_name,
            "type": inst.option_type,
            "strike": inst.strike if inst.option_type != "F" else None,
            "T": inst.T,
            "quantity": round(inst.quantity, 6),
            "mid_price": inst.mid_price,
            "cost": round(inst.quantity * inst.mid_price, 4),
            "delta": inst.greeks.delta,
            "gamma": inst.greeks.gamma,
            "vega": inst.greeks.vega,
        })
    df_port = pd.DataFrame(rows)
    st.download_button(
        "📥 Portefeuille de couverture (CSV)",
        data=df_port.to_csv(index=False),
        file_name="hedge_portfolio.csv", mime="text/csv",
        key="dl_portfolio",
    )
else:
    st.caption("⚠️ Portefeuille non calculé")

st.markdown("---")
st.subheader("💥 P&L Scénario de Stress")
if "stress_result" in st.session_state:
    r = st.session_state.stress_result
    # Flatten le résultat (exclure le DataFrame hedge_details)
    flat = {k: v for k, v in r.items() if not isinstance(v, pd.DataFrame)}
    df_stress = pd.DataFrame([flat])
    st.download_button(
        "📥 Résultats stress (CSV)",
        data=df_stress.to_csv(index=False),
        file_name="stress_pnl.csv", mime="text/csv",
        key="dl_stress",
    )
    if "hedge_details" in r and isinstance(r["hedge_details"], pd.DataFrame):
        st.download_button(
            "📥 Détail instruments (CSV)",
            data=r["hedge_details"].to_csv(index=False),
            file_name="stress_instruments.csv", mime="text/csv",
            key="dl_stress_det",
        )
else:
    st.caption("⚠️ Résultats de stress non calculés")

# ─── Rapport global ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Rapport Global")
if st.button("📋 Générer le rapport global (JSON)", type="secondary"):
    ts = datetime.utcnow().isoformat()
    report: dict = {"generated_at": ts}

    cfg = st.session_state.get("config", AppConfig())
    report["config"] = {
        "underlying": cfg.deribit.currency,
        "spread_threshold": cfg.cleaning.max_spread_pct,
    }
    if "ns_params" in st.session_state:
        report["nelson_siegel"] = st.session_state.ns_params.to_dict()
    if "ssvi_params" in st.session_state:
        report["ssvi"] = st.session_state.ssvi_params.to_dict()
    if "product" in st.session_state:
        p = st.session_state.product
        g = st.session_state.get("product_greeks")
        report["product"] = {
            "K1": p.K1, "K2": p.K2, "T": p.T,
            "r": st.session_state.get("product_r"),
            "F": st.session_state.get("product_F"),
            "price": st.session_state.get("product_price"),
            "greeks": g.to_dict() if g else None,
        }
    if "stress_result" in st.session_state:
        r = st.session_state.stress_result
        report["stress"] = {k: v for k, v in r.items() if not isinstance(v, pd.DataFrame)}

    st.download_button(
        "📥 Rapport global (JSON)",
        data=json.dumps(report, indent=2, default=str),
        file_name=f"global_report_{ts[:10]}.json", mime="application/json",
        key="dl_global",
    )
    st.success("✅ Rapport généré — cliquez pour télécharger.")
