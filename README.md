# Deribit Options Pricing — Optimisation & IA 2026

> Projet académique — CY Tech ING 2 — S2 2025-2026

Pipeline complet de pricing et gestion des risques sur options Deribit (BTC), depuis la récupération des données jusqu'au calcul de P&L sous scénario de stress.

---

## Objectif

Construire une chaîne quantitative complète incluant :
1. Récupération et nettoyage des données de marché (Deribit public API)
2. Extraction de la courbe des taux implicites + lissage Nelson-Siegel
3. Extraction des volatilités implicites par Newton-Raphson & dichotomie
4. Calibration du modèle SSVI (Gatheral & Jacquier 2014)
5. Pricing d'un bull call spread BTC
6. Couverture Delta-Gamma-Vega neutre par optimisation SLSQP
7. Scénario de stress à 1 semaine : spot +10%, vol −10pp

---

## Structure du Projet

```
.
├── app.py                        # Page d'accueil Streamlit
├── pages/
│   ├── 01_⚙️_Paramètres.py       # Configuration globale
│   ├── 02_📥_Données_Deribit.py   # Fetch API Deribit
│   ├── 03_🧹_Nettoyage.py        # Nettoyage & diagnostics
│   ├── 04_📈_Taux_Nelson_Siegel.py# Taux implicites + NS
│   ├── 05_🌊_Volatilité_Implicite.py # Vol implicite (NR + bisect)
│   ├── 06_🎯_Surface_SSVI.py      # Calibration SSVI
│   ├── 07_💎_Produit_Dérivé.py    # Bull Call Spread
│   ├── 08_🛡️_Couverture.py       # Hedge DGV
│   ├── 09_💥_Scénario_Stress.py   # P&L stress
│   └── 10_💾_Export.py            # Exports CSV/JSON
├── config/
│   └── settings.py               # Tous les paramètres (dataclasses)
├── src/
│   ├── api/deribit_client.py      # Client REST Deribit (public)
│   ├── data/
│   │   ├── loaders.py             # Parsing JSON → DataFrame
│   │   ├── cleaning.py            # Pipeline de nettoyage
│   │   ├── validation.py          # Filtres + rapport qualité
│   │   └── transforms.py          # log-moneyness, variance totale…
│   ├── rates/
│   │   ├── put_call_parity.py     # Extraction taux implicite par maturité
│   │   └── nelson_siegel.py       # Modèle NS + calibration diff. evolution
│   ├── pricing/
│   │   ├── black_scholes.py       # Formule de Black 1976
│   │   ├── implied_vol.py         # Newton-Raphson + dichotomie
│   │   └── greeks.py              # Δ, Γ, V, θ, ρ
│   ├── volatility/
│   │   ├── ssvi.py                # Modèle SSVI paramétrique
│   │   ├── calibration_objectives.py  # Calibration 2-étapes
│   │   └── arbitrage_checks.py    # Conditions no-arbitrage
│   ├── products/
│   │   ├── structures.py          # CallSpread, PutSpread
│   │   └── payoff.py              # Fonctions de payoff
│   ├── hedge/
│   │   ├── portfolio.py           # HedgeInstrument, HedgePortfolio
│   │   ├── optimizer.py           # SLSQP Delta-Gamma-Vega
│   │   └── pnl.py                 # Scénario de stress + P&L
│   └── utils/
│       ├── constants.py
│       ├── math_utils.py          # bisection, Newton-Raphson, RMSE, R²
│       ├── dates.py               # Parsing dates Deribit, time_to_maturity
│       ├── io.py                  # Sauvegarde CSV/Parquet/JSON
│       └── plotting.py            # Graphiques Plotly réutilisables
├── tests/
│   ├── test_black_scholes.py
│   ├── test_implied_vol.py
│   ├── test_nelson_siegel.py
│   └── test_ssvi.py
├── data/
│   ├── raw/                       # Données brutes API
│   ├── processed/                 # Données nettoyées
│   └── exports/                   # Exports finaux
├── reports/
│   └── latex/                     # Squelette rapport LaTeX
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Cloner / extraire le projet
cd "Projet - Optimisation&IA 2026"

# Environnement virtuel (recommandé)
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Dépendances
pip install -r requirements.txt
```

---

## Lancement de l'Application

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`.  
Naviguez de gauche à droite en suivant les étapes de 1 à 10.

---

## Workflow

```
Page 1  ─ Paramètres globaux (sous-jacent, seuils, bornes…)
Page 2  ─ Téléchargement des données Deribit (API publique)
Page 3  ─ Nettoyage + diagnostics qualité
Page 4  ─ Taux implicites + calibration Nelson-Siegel
Page 5  ─ Volatilités implicites (Newton-Raphson + dichotomie)
Page 6  ─ Calibration SSVI + visualisation surface
Page 7  ─ Bull Call Spread : prix, grecques, payoff
Page 8  ─ Couverture Delta-Gamma-Vega (SLSQP)
Page 9  ─ Scénario de stress (spot +10%, vol −10pp) + P&L
Page 10 ─ Export de tous les résultats
```

---

## Modèles Mathématiques

### Nelson-Siegel
$$r(T) = \beta_0 + \beta_1 \frac{1-e^{-\lambda T}}{\lambda T} + \beta_2\left(\frac{1-e^{-\lambda T}}{\lambda T} - e^{-\lambda T}\right)$$

### Black 1976
$$C = e^{-rT}[F\,N(d_1) - K\,N(d_2)],\quad d_1 = \frac{\ln(F/K) + \frac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}$$

### SSVI (Gatheral & Jacquier)
$$w(k,t) = \frac{\theta_t}{2}\left\{1 + \rho\,\phi(\theta_t)\,k + \sqrt{[\phi(\theta_t)\,k+\rho]^2 + (1-\rho^2)}\right\}$$

avec $\theta_t = \nu_\infty t + \frac{\nu_0 - \nu_\infty}{\kappa}(1-e^{-\kappa t})$ et $\phi(\theta) = \frac{\eta}{\theta^\lambda(1+\theta)^{1-\lambda}}$.

### Couverture DGV
Minimisation de $\|q\|^2$ sous contrainte $G\,q = g_{produit}$  
où $G \in \mathbb{R}^{3\times n}$ contient les grecques $(\Delta_i, \Gamma_i, \mathcal{V}_i)$.

---

## Tests

```bash
pytest tests/ -v
```

Couverture minimale :
- `test_black_scholes.py` — parité call-put, valeur intrinsèque, vega
- `test_implied_vol.py` — Newton-Raphson, bisection, aller-retour BS↔IV
- `test_nelson_siegel.py` — forme de la courbe, calibration roundtrip
- `test_ssvi.py` — positivité, condition no-arbitrage, formule ATM

---

## Sorties Générées

| Fichier | Contenu |
|---------|---------|
| `deribit_raw.csv` | Données brutes options/futures |
| `deribit_clean.csv` | Données filtrées |
| `cleaning_report.json` | Rapport de nettoyage |
| `implied_rates.csv` | Taux implicites par maturité |
| `ns_params.json` | Paramètres Nelson-Siegel calibrés |
| `implied_vols.csv` | Vol implicites + comparaison SSVI |
| `greeks.csv` | Grecques par option |
| `ssvi_params.json` | Paramètres SSVI calibrés |
| `product_callspread.json` | Fiche du produit dérivé |
| `hedge_portfolio.csv` | Portefeuille de couverture optimal |
| `stress_pnl.csv` | P&L sous scénario de stress |
| `global_report_{date}.json` | Rapport complet JSON |

---

## Limites Connues

- La calibration SSVI peut être sensible au conditionnement numérique pour des maturités très courtes (< 1 semaine) ou très longues (> 2 ans).
- Le fallback lstsq pour la couverture peut produire des résidus non nuls si les instruments disponibles ne permettent pas une neutralisation exacte (cas sous-déterminé).
- La mise à jour de la surface après choc est simplifiée (décalage rigide de σ_SSVI), sans recalibrage.
- Les données Deribit sont des snapshots : les résultats dépendent du moment de la requête.

---

## Pistes d'Amélioration

- Ajout d'un modèle SVI slice-par-slice pour meilleure flexibilité
- Couverture dynamique multi-période
- Calcul du VaR/CVaR du portefeuille sous distribution log-normale
- Recalibrage post-choc de la surface (itération SSVI après stress)
- Interface pour stratégies plus complexes (straddle, butterfly, condor)

---

## Dépendances Principales

| Librairie | Version min | Usage |
|-----------|-------------|-------|
| numpy | 1.26 | Calcul numérique |
| pandas | 2.2 | Manipulation données |
| scipy | 1.13 | Optimisation, statistiques |
| streamlit | 1.35 | Interface utilisateur |
| plotly | 5.22 | Visualisations |
| requests | 2.32 | API Deribit |

---

*Projet réalisé dans le cadre du cours Optimisation & IA — CY Tech ING 2 — 2025-2026*
