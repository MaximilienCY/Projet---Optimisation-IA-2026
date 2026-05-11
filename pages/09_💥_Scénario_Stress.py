"""
09_💥_Scénario_Stress.py — P&L sous choc spot +10% / vol -10%.
"""

import pandas as pd
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import AppConfig, StressConfig
from src.hedge.pnl import compute_stress_pnl
from src.utils.plotting import plot_pnl_waterfall

st.title("💥 Scénario de Stress — P&L à 1 Semaine")
st.markdown(
    """
    **Choc imposé par le sujet** :
    - Forward : **+10% relatif** → F_new = F × 1.10
    - Vol implicite : **−10 pts absolus** → σ_new = σ_SSVI − 0.10
    - Temps : avancement de **1 semaine** → T_new = T − 1/52

    *Convention «−10 pts absolus» : si σ_SSVI = 80%, après choc σ = 70%.*
    *Plus conservateur que −10% relatif (72%), car reflète une normalisation post-spike.*
    """
)

if "config" not in st.session_state:
    st.session_state.config = AppConfig()
cfg: AppConfig = st.session_state.config

for dep, label in [
    ("product", "le produit (page 7)"),
    ("product_price", "le prix du produit (page 7)"),
    ("portfolio", "le portefeuille de couverture (page 8)"),
    ("ssvi_params", "la calibration SSVI (page 6)"),
    ("ns_params", "Nelson-Siegel (page 4)"),
    ("hedge_initial_prices", "le portefeuille de couverture (page 8)"),
]:
    if dep not in st.session_state:
        st.warning(f"⚠️ Construisez d'abord {label}.")
        st.stop()

product = st.session_state.product
product_price: float = st.session_state.product_price
portfolio = st.session_state.portfolio
ssvi_params = st.session_state.ssvi_params
hedge_initial_prices: list = st.session_state.hedge_initial_prices
F_initial: float = st.session_state.get("product_F", 50_000.0)
r_initial: float = st.session_state.get("product_r", 0.0)

# ─── Paramètres du scénario ───────────────────────────────────────────────────
st.subheader("1️⃣ Paramètres du Scénario de Stress")

col1, col2, col3 = st.columns(3)
with col1:
    spot_shock_pct = st.number_input(
        "Choc spot (%, relatif)", -50.0, 100.0,
        float(cfg.stress.spot_shock_pct * 100),
    ) / 100
with col2:
    vol_shock_abs_display = st.number_input(
        "Choc vol (pts absolus, ex: -10 = −10pp)",
        -50.0, 50.0,
        float(cfg.stress.vol_shock_abs * 100),
    ) / 100
with col3:
    horizon_weeks = st.number_input(
        "Horizon (semaines)", 1, 52,
        int(cfg.stress.horizon_weeks),
    )

stress_cfg = StressConfig(
    spot_shock_pct=spot_shock_pct,
    vol_shock_abs=vol_shock_abs_display,
    horizon_weeks=float(horizon_weeks),
)

F_new_preview = F_initial * (1.0 + spot_shock_pct)
st.info(
    f"F : ${F_initial:,.0f} → **${F_new_preview:,.0f}** ({spot_shock_pct*100:+.1f}%)  |  "
    f"σ : σ_SSVI + **{vol_shock_abs_display*100:+.1f}pp**  |  "
    f"T : T − **{horizon_weeks/52:.4f}** an"
)

# ─── Calcul P&L ───────────────────────────────────────────────────────────────
if st.button("💥 Calculer le P&L de Stress", type="primary"):
    with st.spinner("Revalorisation sous scénario de stress..."):
        try:
            result = compute_stress_pnl(
                product=product,
                product_price_initial=product_price,
                portfolio=portfolio,
                hedge_instruments_initial_prices=hedge_initial_prices,
                F_initial=F_initial,
                r=r_initial,
                ssvi_params=ssvi_params,
                cfg=stress_cfg,
            )
            st.session_state.stress_result = result
            st.success("✅ P&L calculé !")
        except Exception as e:
            st.error(f"❌ Erreur : {e}")
            raise

if "stress_result" not in st.session_state:
    st.info("Cliquez pour lancer le calcul de stress.")
    st.stop()

result: dict = st.session_state.stress_result

# ─── Résumé ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("2️⃣ Résultats du P&L")

col1, col2, col3 = st.columns(3)
col1.metric(
    "P&L Produit (vendu)",
    f"${result['pnl_product']:+,.2f}",
    delta=f"Prix init. ${result['product_price_initial']:,.4f} → après ${result['product_price_new']:,.4f}",
)
col2.metric("P&L Hedge", f"${result['pnl_hedge']:+,.2f}")
col3.metric(
    "P&L Net Total",
    f"${result['pnl_total']:+,.2f}",
    delta="✅ Couverture partielle" if abs(result["pnl_total"]) < abs(result["pnl_product"]) else "⚠️",
)

# ─── Waterfall ────────────────────────────────────────────────────────────────
pnl_breakdown = {
    "P&L produit\n(position vendue)": result["pnl_product"],
    "P&L couverture\n(hedge)": result["pnl_hedge"],
    "P&L net total": result["pnl_total"],
}
fig_wf = plot_pnl_waterfall(pnl_breakdown)
st.plotly_chart(fig_wf, use_container_width=True)

# ─── Tableau de revalorisation ───────────────────────────────────────────────
st.subheader("3️⃣ Tableau de Revalorisation")

df_reprice = pd.DataFrame({
    "": ["Avant stress", "Après stress", "Variation"],
    "Forward F_T ($)": [
        f"${result['F_initial']:,.2f}",
        f"${result['F_new']:,.2f}",
        f"{(result['F_new']/result['F_initial']-1)*100:+.2f}%",
    ],
    "σ(K₁)": [
        f"{result['iv1_initial']*100:.2f}%",
        f"{result['iv1_stressed']*100:.2f}%",
        f"{(result['iv1_stressed']-result['iv1_initial'])*100:+.2f}pp",
    ],
    "σ(K₂)": [
        f"{result['iv2_initial']*100:.2f}%",
        f"{result['iv2_stressed']*100:.2f}%",
        f"{(result['iv2_stressed']-result['iv2_initial'])*100:+.2f}pp",
    ],
    "Prix produit ($)": [
        f"${result['product_price_initial']:,.4f}",
        f"${result['product_price_new']:,.4f}",
        f"${result['product_price_new']-result['product_price_initial']:+,.4f}",
    ],
    "P&L": ["—", "—", f"${result['pnl_product']:+,.2f} (position vendue)"],
})
st.dataframe(df_reprice, use_container_width=True, hide_index=True)

# ─── Détail hedge ─────────────────────────────────────────────────────────────
st.subheader("4️⃣ Détail par Instrument de Couverture")
hedge_details = result.get("hedge_details")
if hedge_details is not None and not (
    isinstance(hedge_details, pd.DataFrame) and hedge_details.empty
):
    if not isinstance(hedge_details, pd.DataFrame):
        hedge_details = pd.DataFrame(hedge_details)
    st.dataframe(hedge_details.round(6), use_container_width=True, hide_index=True)

# ─── Commentaire pédagogique ──────────────────────────────────────────────────
st.markdown("---")
with st.expander("💬 Interprétation du scénario de stress", expanded=True):
    pnl_net = result["pnl_total"]
    pnl_prod = result["pnl_product"]
    pnl_hedge = result["pnl_hedge"]
    eff = (1 - abs(pnl_net) / (abs(pnl_prod) + 1e-12)) * 100

    st.markdown(
        f"""
        **Résumé** :
        - La position vendue sur le call spread génère un P&L de **${pnl_prod:+,.2f}**.
        - Le portefeuille de couverture compense **${pnl_hedge:+,.2f}**.
        - Le **résidu net** est de **${pnl_net:+,.2f}** (efficacité estimée : {eff:.1f}%).

        **Sources du résidu** :
        - Approximation d'ordre 1–2 : Δ, Γ, V ne capturent pas les non-linéarités au-delà
        - Basis risk : maturités et strikes des instruments ≠ ceux du produit
        - Choc simultané sur spot et vol (cross-gamma non neutralisé)
        - Recalibrage post-choc simplifié (σ_SSVI décalée rigidement de {vol_shock_abs_display*100:+.1f}pp)
        """
    )

# ─── Export ───────────────────────────────────────────────────────────────────
st.markdown("---")
df_export = pd.DataFrame([{
    "scenario_spot_shock_pct": spot_shock_pct * 100,
    "scenario_vol_shock_abs_pp": vol_shock_abs_display * 100,
    "horizon_weeks": horizon_weeks,
    "F_initial": result["F_initial"],
    "F_new": result["F_new"],
    "T_initial": result["T_initial"],
    "T_new": result["T_new"],
    "iv1_initial_pct": result["iv1_initial"] * 100,
    "iv1_stressed_pct": result["iv1_stressed"] * 100,
    "iv2_initial_pct": result["iv2_initial"] * 100,
    "iv2_stressed_pct": result["iv2_stressed"] * 100,
    "product_price_initial": result["product_price_initial"],
    "product_price_new": result["product_price_new"],
    "pnl_product": result["pnl_product"],
    "pnl_hedge": result["pnl_hedge"],
    "pnl_total": result["pnl_total"],
}])
st.download_button(
    "📥 Résultats P&L de stress (CSV)",
    data=df_export.to_csv(index=False),
    file_name="stress_pnl.csv", mime="text/csv",
)
