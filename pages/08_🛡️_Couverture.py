"""
08_🛡️_Couverture.py — Portefeuille Delta-Gamma-Vega neutre par optimisation.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import AppConfig
from src.hedge.portfolio import HedgeInstrument, HedgePortfolio, select_hedge_instruments
from src.hedge.optimizer import build_hedge_portfolio
from src.pricing.greeks import Greeks
from src.volatility.ssvi import SSVIParams

st.title("🛡️ Portefeuille de Couverture Delta-Gamma-Vega")
st.markdown(
    r"""
    **Objectif** : construire un portefeuille $\Pi_{hedge}$ tel que :
    $$\Delta_{hedge} = \Delta_{produit},\quad \Gamma_{hedge} = \Gamma_{produit},\quad \mathcal{V}_{hedge} = \mathcal{V}_{produit}$$

    On résout $G\,q = g_{produit}$ par **SLSQP** (moindres carrés régularisés en fallback).
    """
)

if "config" not in st.session_state:
    st.session_state.config = AppConfig()
cfg: AppConfig = st.session_state.config

for dep, label in [
    ("product", "le produit (page 7)"),
    ("product_greeks", "le produit (page 7)"),
    ("df_iv", "les vol implicites (page 5)"),
    ("ssvi_params", "la calibration SSVI (page 6)"),
]:
    if dep not in st.session_state:
        st.warning(f"⚠️ Calculez d'abord {label}.")
        st.stop()

product = st.session_state.product
product_greeks: Greeks = st.session_state.product_greeks
product_price: float = st.session_state.get("product_price", 0.0)
product_T: float = st.session_state.get("product_T", 0.25)
df_iv: pd.DataFrame = st.session_state.df_iv
ssvi_params: SSVIParams = st.session_state.ssvi_params
df_futures: pd.DataFrame = st.session_state.get("df_futures", pd.DataFrame())

# ─── Rappel des grecques du produit ──────────────────────────────────────────
st.subheader("1️⃣ Grecques du Produit Vendu (à couvrir)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Δ produit", f"{product_greeks.delta:.6f}")
col2.metric("Γ produit", f"{product_greeks.gamma:.8f}")
col3.metric("V produit", f"{product_greeks.vega:.4f}")
col4.metric("Prix", f"${product_price:,.4f}")

# ─── Paramètres de sélection ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("2️⃣ Sélection des Instruments de Couverture")

col1, col2 = st.columns(2)
with col1:
    n_instruments = st.number_input("Nombre d'options à inclure", 3, 15, 6, 1)
    add_futures = st.checkbox("Inclure le future le plus proche", value=True)
with col2:
    max_pos = st.number_input("Position max par instrument (|q|)", 1.0, 200.0, 50.0, 1.0)

if st.button("🛡️ Construire le Portefeuille de Couverture", type="primary"):
    with st.spinner("Sélection des instruments + optimisation DGV..."):
        try:
            # 1. Sélection des instruments candidats (DataFrame)
            df_candidates = select_hedge_instruments(
                df_options=df_iv[df_iv["iv"].notna()],
                product_T=product_T,
                max_instruments=int(n_instruments),
                use_futures=add_futures,
                df_futures=df_futures if not df_futures.empty else None,
            )

            # 2. Conversion en HedgeInstrument
            instruments: list[HedgeInstrument] = []
            for _, row in df_candidates.iterrows():
                inst = HedgeInstrument(
                    instrument_name=str(row.get("instrument_name", f"inst_{_}")),
                    option_type=str(row.get("option_type", "C")),
                    strike=float(row.get("strike", 0.0)),
                    T=float(row.get("T", product_T)),
                    forward_price=float(row.get("forward_price", 50000.0)),
                    rate=float(row.get("rate", 0.0)),
                    iv=float(row.get("iv", 0.0)),
                    mid_price=float(row.get("mid", row.get("mid_price", 0.0))),
                )
                inst.compute_and_set_greeks(ssvi_params=ssvi_params)
                instruments.append(inst)

            if not instruments:
                st.error("❌ Aucun instrument de couverture disponible.")
                st.stop()

            # 3. Optimisation
            cfg_hedge = cfg.hedge
            cfg_hedge.max_position_size = max_pos
            portfolio = build_hedge_portfolio(
                product_greeks=product_greeks,
                instruments=instruments,
                cfg=cfg_hedge,
            )
            report = getattr(portfolio, "_opt_report", {})

            # Stocker les prix initiaux des instruments
            initial_prices = [inst.mid_price for inst in portfolio.instruments]

            st.session_state.portfolio = portfolio
            st.session_state.hedge_report = report
            st.session_state.hedge_initial_prices = initial_prices
            st.success(f"✅ Portefeuille construit avec {len(instruments)} instruments !")
        except Exception as e:
            st.error(f"❌ Erreur : {e}")
            raise

if "portfolio" not in st.session_state:
    st.info("Cliquez pour lancer l'optimisation.")
    st.stop()

portfolio: HedgePortfolio = st.session_state.portfolio
report: dict = st.session_state.get("hedge_report", {})

# ─── Méthode ─────────────────────────────────────────────────────────────────
converged = report.get("converged", False)
method_msg = "SLSQP ✅" if converged else "lstsq (SLSQP n'a pas convergé) ⚠️"
st.info(f"Méthode d'optimisation : **{method_msg}**")

# ─── Vérification de neutralité ──────────────────────────────────────────────
st.markdown("---")
st.subheader("3️⃣ Vérification de la Neutralité DGV")

hedge_d = sum(inst.greeks.delta * inst.quantity for inst in portfolio.instruments)
hedge_g = sum(inst.greeks.gamma * inst.quantity for inst in portfolio.instruments)
hedge_v = sum(inst.greeks.vega * inst.quantity for inst in portfolio.instruments)

net_d = product_greeks.delta + hedge_d
net_g = product_greeks.gamma + hedge_g
net_v = product_greeks.vega + hedge_v

# Afficher residuals si disponibles
res_d = report.get("residual_delta", net_d)
res_g = report.get("residual_gamma", net_g)
res_v = report.get("residual_vega", net_v)

df_check = pd.DataFrame({
    "Grecque": ["Delta Δ", "Gamma Γ", "Vega V"],
    "Produit (cible)": [product_greeks.delta, product_greeks.gamma, product_greeks.vega],
    "Hedge (atteint)": [hedge_d, hedge_g, hedge_v],
    "Résidu net": [net_d, net_g, net_v],
    "Résidu relatif (%)": [
        abs(net_d) / (abs(product_greeks.delta) + 1e-12) * 100,
        abs(net_g) / (abs(product_greeks.gamma) + 1e-12) * 100,
        abs(net_v) / (abs(product_greeks.vega) + 1e-12) * 100,
    ],
})
st.dataframe(df_check.round(8), use_container_width=True, hide_index=True)

col1, col2, col3 = st.columns(3)
col1.metric("Résidu |Δ|", f"{abs(net_d):.2e}")
col2.metric("Résidu |Γ|", f"{abs(net_g):.2e}")
col3.metric("Résidu |V|", f"{abs(net_v):.2e}")

# ─── Composition ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("4️⃣ Composition du Portefeuille de Couverture")

rows = []
for inst in portfolio.instruments:
    rows.append({
        "Instrument": inst.instrument_name,
        "Type": inst.option_type,
        "Strike": round(inst.strike, 2) if inst.option_type != "F" else "—",
        "T (an)": round(inst.T, 4),
        "Quantité": round(inst.quantity, 4),
        "Prix unitaire ($)": round(inst.mid_price, 4),
        "Coût total ($)": round(inst.quantity * inst.mid_price, 2),
        "Δ unitaire": round(inst.greeks.delta, 6),
        "Γ unitaire": round(inst.greeks.gamma, 8),
        "V unitaire": round(inst.greeks.vega, 4),
    })
df_portfolio = pd.DataFrame(rows)
st.dataframe(df_portfolio, use_container_width=True)

total_cost = sum(inst.quantity * inst.mid_price for inst in portfolio.instruments)
col1, col2 = st.columns(2)
col1.metric("Coût total du hedge ($)", f"${total_cost:,.2f}")
col2.metric("Nb instruments", str(len(portfolio.instruments)))

# Bar chart des quantités
fig_qty = go.Figure([
    go.Bar(
        x=df_portfolio["Instrument"],
        y=df_portfolio["Quantité"],
        text=df_portfolio["Quantité"].round(3),
        textposition="auto",
        marker_color=["#4CAF50" if q > 0 else "#F44336"
                      for q in df_portfolio["Quantité"]],
    )
])
fig_qty.update_layout(
    title="Quantités des instruments de couverture",
    xaxis_title="Instrument", yaxis_title="Quantité",
    template="plotly_white",
    xaxis=dict(tickangle=-35),
)
st.plotly_chart(fig_qty, use_container_width=True)

# ─── Export ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.download_button(
    "📥 Portefeuille de couverture (CSV)",
    data=df_portfolio.to_csv(index=False),
    file_name="hedge_portfolio.csv", mime="text/csv",
)
