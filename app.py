"""
app.py — Point d'entrée Streamlit.

Lancement : streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="OptimIA — Options Deribit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Style global ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    h1 { color: #1565C0; }
    h2 { color: #1976D2; }
    h3 { color: #1E88E5; }
    .stMetricValue { font-size: 1.4rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Page d'accueil ────────────────────────────────────────────────────────────
st.title("🔷 OptimIA — Pricing et Couverture d'Options Crypto")
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(
        """
        ## Présentation du Projet

        Ce projet implémente une **chaîne complète de pricing et gestion des risques**
        sur les options Bitcoin (BTC) cotées sur **Deribit**, dans le cadre du cours
        *Optimisation et IA* — CY Tech, ING 2, 2025-2026.

        ### Méthodologie
        1. **Données de marché** : récupération via l'API publique Deribit, filtrage et nettoyage
        2. **Courbe des taux** : extraction par parité call-put + lissage Nelson-Siegel
        3. **Volatilités implicites** : Newton-Raphson & dichotomie (Black 1976)
        4. **Surface SSVI** : calibration en deux étapes (terme de structure + smile)
        5. **Produit dérivé** : Bull Call Spread — prix, payoff et grecques via SSVI
        6. **Couverture** : portefeuille Delta-Gamma-Vega neutre (optimisation SLSQP)
        7. **Scénario de stress** : choc spot +10%, vol -10% abs, horizon 1 semaine

        ### Navigation
        Utilisez la **barre latérale** pour accéder aux différentes sections du projet.
        Chaque section est indépendante mais les données circulent via la session.
        """
    )

with col2:
    st.markdown("### 📌 Accès rapide")
    st.page_link("pages/01_⚙️_Paramètres.py",         label="⚙️ Paramètres globaux")
    st.page_link("pages/02_📥_Données_Deribit.py",     label="📥 Données Deribit")
    st.page_link("pages/03_🧹_Nettoyage.py",           label="🧹 Nettoyage")
    st.page_link("pages/04_📈_Taux_Nelson_Siegel.py",  label="📈 Taux & Nelson-Siegel")
    st.page_link("pages/05_🌊_Volatilité_Implicite.py",label="🌊 Vol Implicite")
    st.page_link("pages/06_🎯_Surface_SSVI.py",        label="🎯 Surface SSVI")
    st.page_link("pages/07_💎_Produit_Dérivé.py",      label="💎 Produit Dérivé")
    st.page_link("pages/08_🛡️_Couverture.py",          label="🛡️ Couverture DGV")
    st.page_link("pages/09_💥_Scénario_Stress.py",     label="💥 Scénario Stress")
    st.page_link("pages/10_💾_Export.py",               label="💾 Export")

st.markdown("---")
st.markdown(
    """
    **Sous-jacent** : BTC — *Choix justifié par sa liquidité supérieure et la profondeur
    du marché d'options sur Deribit (> 3 000 contrats actifs)*

    **Formule de pricing** : Black 1976 (forward-based) | **Modèle de surface** : SSVI
    (Gatheral & Jacquier 2014) | **Optimisation** : SLSQP via SciPy

    ---
    *Projet académique — CY Tech ING 2 — Optimisation et IA 2025-2026*
    """,
    unsafe_allow_html=False,
)
