"""
07_💎_Produit_Dérivé.py — Call Spread, Payoff, Prix et Grecques.
"""

import json

import numpy as np
import pandas as pd
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import AppConfig
from src.products.structures import CallSpread
from src.products.payoff import payoff_grid
from src.utils.plotting import plot_payoff, plot_greeks_bar
from src.volatility.ssvi import SSVIParams
from src.rates.nelson_siegel import nelson_siegel_rate

st.title("💎 Produit Dérivé — Bull Call Spread")
st.markdown(
    """
    Un **bull call spread** = achat call K₁ + vente call K₂ (K₁ < K₂).
    - Coût limité (prime = C(K₁) − C(K₂) > 0)
    - Profit maximal borné à K₂ − K₁ − prime
    - Adapté pour parier sur une hausse modérée du BTC
    - Teste deux points de la surface SSVI à des strikes différents
    """
)

if "config" not in st.session_state:
    st.session_state.config = AppConfig()
cfg: AppConfig = st.session_state.config

for dep, label in [
    ("df_iv", "vol implicites (page 5)"),
    ("ssvi_params", "calibration SSVI (page 6)"),
    ("ns_params", "Nelson-Siegel (page 4)"),
]:
    if dep not in st.session_state:
        st.warning(f"⚠️ Calculez d'abord les {label}.")
        st.stop()

df_iv: pd.DataFrame = st.session_state.df_iv
ssvi_params: SSVIParams = st.session_state.ssvi_params
ns_params = st.session_state.ns_params
spot = st.session_state.get("spot_price", 50_000.0)

# ─── Sélection de la maturité ─────────────────────────────────────────────────
st.subheader("1️⃣ Paramètres du Produit")

available_maturities = sorted(df_iv["T"].unique())
T_options = {f"{t:.3f} an ({t*365:.0f}j)": t for t in available_maturities}
selected_label = st.selectbox(
    "Maturité",
    list(T_options.keys()),
    index=min(1, len(T_options) - 1),
)
T = T_options[selected_label]
r = float(nelson_siegel_rate(np.array([T]), ns_params)[0])
F = spot * np.exp(r * T)

col1, col2, col3 = st.columns(3)
col1.metric("Forward F_T", f"${F:,.2f}")
col2.metric("Taux r(T)", f"{r*100:.3f}%")
col3.metric("Spot", f"${spot:,.2f}")

# Strikes
col1, col2 = st.columns(2)
with col1:
    k1_pct = st.slider("K₁ (% du Forward)", 70, 105, 95) / 100
with col2:
    k2_pct = st.slider("K₂ (% du Forward)", 100, 140, 110) / 100

K1 = k1_pct * F
K2 = k2_pct * F
st.info(f"K₁ = ${K1:,.2f} ({k1_pct*100:.0f}% de F),   K₂ = ${K2:,.2f} ({k2_pct*100:.0f}% de F)")

# ─── Calcul ───────────────────────────────────────────────────────────────────
if st.button("💎 Pricer le Call Spread", type="primary"):
    if K1 >= K2:
        st.error("K₁ doit être strictement inférieur à K₂.")
        st.stop()
    try:
        product = CallSpread(K1=K1, K2=K2, T=T)
        price = product.price(F=F, r=r, ssvi_params=ssvi_params)
        greeks = product.greeks(F=F, r=r, ssvi_params=ssvi_params)

        st.session_state.product = product
        st.session_state.product_price = price
        st.session_state.product_greeks = greeks
        st.session_state.product_T = T
        st.session_state.product_r = r
        st.session_state.product_F = F
        st.success(f"✅ Prix du call spread : ${price:,.4f}")
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        raise

if "product" not in st.session_state:
    st.info("Définissez les paramètres et cliquez pour pricer.")
    st.stop()

product: CallSpread = st.session_state.product
price: float = st.session_state.product_price
greeks = st.session_state.product_greeks
T = st.session_state.product_T
r = st.session_state.product_r
F = st.session_state.product_F

# ─── Résultats ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("2️⃣ Prix et Grecques")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Prix ($)", f"${price:,.4f}")
col2.metric("Delta (Δ)", f"{greeks.delta:.4f}")
col3.metric("Gamma (Γ)", f"{greeks.gamma:.6f}")
col4.metric("Vega (V)", f"{greeks.vega:.4f}")
col5.metric("Theta (θ)", f"{greeks.theta:.4f}")

df_greeks_display = pd.DataFrame({
    "Grecque": ["Delta Δ", "Gamma Γ", "Vega V", "Theta θ", "Rho ρ"],
    "Valeur": [greeks.delta, greeks.gamma, greeks.vega, greeks.theta, greeks.rho],
    "Interprétation": [
        "+1$ sur F → +Δ$ sur la prime",
        "+1$ sur F → variation de Δ de +Γ",
        "+1 pt de vol → +V$ sur la prime",
        "Décroissance de valeur par an (∂/∂T)",
        "Sensibilité de 1% de taux",
    ],
})
st.dataframe(df_greeks_display, use_container_width=True, hide_index=True)

# Bar chart
fig_greeks = plot_greeks_bar(greeks.to_dict(), title="Grecques du Call Spread")
st.plotly_chart(fig_greeks, use_container_width=True)

# ─── Vol implicites par jambe ─────────────────────────────────────────────────
iv1, iv2 = product.ivols(F=F, ssvi_params=ssvi_params)
col1, col2 = st.columns(2)
col1.metric("σ(K₁) via SSVI", f"{iv1*100:.2f}%")
col2.metric("σ(K₂) via SSVI", f"{iv2*100:.2f}%")

# ─── Payoff à maturité ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("3️⃣ Payoff à Maturité")

spots, payoffs = payoff_grid(
    K1=product.K1, K2=product.K2,
    spot=F, product_type="call_spread", n_points=400, range_pct=0.55,
)
fig_payoff = plot_payoff(
    spots=spots,
    payoff=payoffs,
    label="Call Spread BTC",
    current_spot=F,
    price=price,
)
st.plotly_chart(fig_payoff, use_container_width=True)

breakeven = product.K1 + price
col1, col2, col3 = st.columns(3)
col1.info(f"📍 Breakeven ≈ **${breakeven:,.2f}** ({(breakeven/F-1)*100:+.1f}% du forward)")
col2.info(f"📍 Profit max = **${product.K2 - product.K1 - price:,.2f}**")
col3.info(f"📍 Perte max = **${-price:,.4f}** (prime)")

# ─── Export ───────────────────────────────────────────────────────────────────
st.markdown("---")
summary = {
    "produit": "bull_call_spread",
    "K1": product.K1, "K2": product.K2,
    "T": product.T, "r": r, "F": F,
    "prix": price,
    "vol_K1_pct": round(iv1 * 100, 4),
    "vol_K2_pct": round(iv2 * 100, 4),
    "delta": greeks.delta, "gamma": greeks.gamma,
    "vega": greeks.vega, "theta": greeks.theta, "rho": greeks.rho,
    "breakeven": breakeven,
    "profit_max": product.K2 - product.K1 - price,
}
st.download_button(
    "📥 Fiche produit (JSON)",
    data=json.dumps(summary, indent=2),
    file_name="product_callspread.json", mime="application/json",
)
