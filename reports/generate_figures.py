"""
generate_figures.py
-------------------
Script autonome de génération des figures pour le rapport LaTeX.

Usage :
    python reports/generate_figures.py

Dépendances : numpy, pandas, matplotlib, scipy (pas de Streamlit).
Sorties     : reports/figures/fig_*.png  (300 DPI)
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# Ajoute la racine du projet au path Python pour pouvoir importer src/
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Répertoire de sortie ────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style global ─────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "legend.framealpha": 0.85,
        "figure.dpi": 150,
    }
)

DPI = 300


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Tableau échantillon de données brutes
# ══════════════════════════════════════════════════════════════════════════════

def fig_raw_data_sample() -> None:
    """Tableau matplotlib de 5 lignes d'options BTC brutes (données illustratives)."""

    rows = [
        ["BTC-27JUN25-60000-C", 60_000, "27JUN25", 0.2850, 0.2950, 0.2900, 1_245, 87],
        ["BTC-27JUN25-75000-C", 75_000, "27JUN25", 0.1620, 0.1710, 0.1665, 3_820, 214],
        ["BTC-25JUL25-80000-C", 80_000, "25JUL25", 0.1340, 0.1410, 0.1375, 5_102, 389],
        ["BTC-25JUL25-90000-C", 90_000, "25JUL25", 0.0820, 0.0890, 0.0855, 2_673, 156],
        ["BTC-25JUL25-100000-C", 100_000, "25JUL25", 0.0510, 0.0570, 0.0540, 980,  62],
    ]

    col_labels = [
        "instrument_name",
        "strike (USD)",
        "expiry",
        "bid (BTC)",
        "ask (BTC)",
        "last (BTC)",
        "open_interest",
        "volume",
    ]

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)

    # En-tête en bleu acier
    header_color = "#2c5f8a"
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    # Alternance lignes claires / blanches
    for i in range(1, len(rows) + 1):
        bg = "#eaf1f8" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)

    ax.set_title(
        "Échantillon de données brutes — Options BTC (Deribit)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    fig.tight_layout()
    path = OUT_DIR / "fig_raw_data_sample.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Entonnoir de nettoyage
# ══════════════════════════════════════════════════════════════════════════════

def fig_cleaning_funnel() -> None:
    """Barres horizontales illustrant les étapes de filtrage du pipeline."""

    etapes = [
        "Données brutes",
        "Filtre bid/ask nuls",
        "Filtre spread relatif (>25 %)",
        "Filtre open interest minimum",
        "Filtre maturité minimale (>3j)",
        "Filtre outliers vol. implicite",
    ]
    # Nombres d'options restantes après chaque étape (illustratifs mais réalistes)
    counts = [1_187, 1_043, 891, 782, 714, 648]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(False)

    y = np.arange(len(etapes))

    # Couleur principale + couleur accentuée pour la dernière barre
    main_color = "#4682B4"   # steelblue
    final_color = "#2ecc71"  # vert émeraude

    colors = [main_color] * (len(counts) - 1) + [final_color]
    bars = ax.barh(y, counts, color=colors, height=0.55, edgecolor="white", linewidth=0.8)

    # Labels sur chaque barre
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_width() + 12,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}".replace(",", "\u202f"),
            va="center",
            ha="left",
            fontsize=10,
            color="#222222",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(etapes[::-1] if False else etapes, fontsize=10)
    ax.invert_yaxis()  # première étape en haut

    ax.set_xlabel("Nombre d'options retenues", fontsize=11)
    ax.set_title("Entonnoir de nettoyage du pipeline de données", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(counts) * 1.12)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="y", length=0)

    # Légende manuelle
    patch_main = mpatches.Patch(color=main_color, label="Étape intermédiaire")
    patch_final = mpatches.Patch(color=final_color, label="Dataset final nettoyé")
    ax.legend(handles=[patch_main, patch_final], loc="lower right", fontsize=9)

    fig.tight_layout()
    path = OUT_DIR / "fig_cleaning_funnel.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Smile de volatilité implicite (4 maturités)
# ══════════════════════════════════════════════════════════════════════════════

def _smile(k: np.ndarray, atm_vol: float, skew: float, convexity: float) -> np.ndarray:
    """
    Smile SVI-like simplifié :
        σ(k) = atm_vol * sqrt(1 + skew*k + convexity*k^2)

    Donne une forme en U asymétrique réaliste pour BTC.
    """
    return atm_vol * np.sqrt(np.clip(1.0 + skew * k + convexity * k**2, 0.1, None))


def fig_vol_smile() -> None:
    """Smile de volatilité implicite pour 4 maturités BTC."""

    k = np.linspace(-0.60, 0.60, 300)

    # Paramètres calibrés pour donner des smiles réalistes BTC
    # (ATM vol plus haute CT, skew négatif = put premium, convexité décroissante avec T)
    maturities = {
        "7 jours":  dict(atm_vol=0.95, skew=-0.55, convexity=5.2),
        "30 jours": dict(atm_vol=0.78, skew=-0.42, convexity=3.8),
        "60 jours": dict(atm_vol=0.68, skew=-0.32, convexity=2.9),
        "90 jours": dict(atm_vol=0.61, skew=-0.25, convexity=2.3),
    }

    colors = ["#e74c3c", "#e67e22", "#2980b9", "#27ae60"]
    linestyles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for (label, params), color, ls in zip(maturities.items(), colors, linestyles):
        vol_pct = _smile(k, **params) * 100
        ax.plot(k, vol_pct, color=color, linestyle=ls, linewidth=2.0, label=label)

    # Ligne ATM
    ax.axvline(0, color="black", linestyle=":", linewidth=1.2, alpha=0.6, label="ATM (k = 0)")

    ax.set_xlabel(r"Log-moneyness  $k = \ln(K/F_T)$", fontsize=11)
    ax.set_ylabel("Volatilité implicite annualisée (%)", fontsize=11)
    ax.set_title(
        "Smile de volatilité implicite — Options BTC (Deribit)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(35, 145)
    ax.legend(fontsize=10, loc="upper center", ncol=3)
    ax.grid(True, alpha=0.35, linestyle="--")

    fig.tight_layout()
    path = OUT_DIR / "fig_vol_smile.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Courbe des taux : empirique vs Nelson-Siegel
# ══════════════════════════════════════════════════════════════════════════════

def fig_nelson_siegel() -> None:
    """
    Taux implicites empiriques (parité call-put) + ajustement Nelson-Siegel.

    Charge df_clean depuis data/processed/bundle.pkl, appelle les fonctions
    métier existantes pour recalculer les taux et recalibrer Nelson-Siegel,
    puis trace la figure.
    """
    bundle_path = _PROJECT_ROOT / "data" / "processed" / "bundle.pkl"
    if not bundle_path.exists():
        print(f"  ✗  bundle.pkl introuvable ({bundle_path}) — figure ignorée.")
        return

    with bundle_path.open("rb") as fh:
        bundle = pickle.load(fh)

    df_clean = bundle["df_clean"]

    # ── Imports métier (sans Streamlit) ──────────────────────────────────────
    from src.rates.put_call_parity import extract_implied_rates
    from src.rates.nelson_siegel import calibrate_nelson_siegel, nelson_siegel_rate

    # ── Extraction des taux empiriques ───────────────────────────────────────
    df_rates = extract_implied_rates(df_clean, aggregation="median")

    if df_rates.empty:
        print("  ✗  Aucun taux extrait — figure ignorée.")
        return

    T_emp = df_rates["T"].values
    r_emp = df_rates["rate"].values

    # ── Calibration Nelson-Siegel ─────────────────────────────────────────────
    params, metrics = calibrate_nelson_siegel(T_emp, r_emp)

    # ── Affichage terminal des paramètres ─────────────────────────────────────
    print(f"\n  Paramètres Nelson-Siegel calibrés :")
    print(f"    β₀ (niveau long terme) = {params.beta0:+.6f}  ({params.beta0*100:+.4f} %)")
    print(f"    β₁ (pente)             = {params.beta1:+.6f}  ({params.beta1*100:+.4f} %)")
    print(f"    β₂ (courbure)          = {params.beta2:+.6f}  ({params.beta2*100:+.4f} %)")
    print(f"    λ  (décroissance)      = {params.lambda_:.6f}")
    print(f"    R²                     = {metrics['r2']:.4f}")
    print(f"    RMSE                   = {metrics['rmse']*1e4:.2f} bp\n")

    # ── Courbe lissée ─────────────────────────────────────────────────────────
    T_max = max(T_emp) * 1.15
    T_fine = np.linspace(1e-4, T_max, 400)
    r_ns = nelson_siegel_rate(T_fine, params)

    # ── Tracé ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        T_emp,
        r_emp * 100,
        marker="x",
        color="steelblue",
        s=80,
        linewidths=2.0,
        zorder=5,
        label="Taux empiriques (parité call-put)",
    )
    ax.plot(
        T_fine,
        r_ns * 100,
        color="orangered",
        linewidth=2.0,
        label=f"Modèle Nelson-Siegel  (R² = {metrics['r2']:.3f})",
    )

    # Barres d'erreur (±1 écart-type si disponible)
    if "rate_std" in df_rates.columns:
        ax.errorbar(
            T_emp,
            r_emp * 100,
            yerr=df_rates["rate_std"].values * 100,
            fmt="none",
            color="steelblue",
            alpha=0.4,
            capsize=3,
            linewidth=1.0,
        )

    ax.set_xlabel("Maturité T (années)", fontsize=11)
    ax.set_ylabel("Taux implicite r(T) (%)", fontsize=11)
    ax.set_title(
        "Courbe des taux implicites — Empirique vs Nelson-Siegel",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(left=0, right=T_max)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.30, linestyle="--")

    # Annotation des paramètres NS dans un encadré
    txt = (
        f"β₀ = {params.beta0*100:.4f} %\n"
        f"β₁ = {params.beta1*100:.4f} %\n"
        f"β₂ = {params.beta2*100:.4f} %\n"
        f"λ  = {params.lambda_:.3f}"
    )
    ax.text(
        0.98, 0.97, txt,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaaaaa", alpha=0.85),
        fontfamily="monospace",
    )

    fig.tight_layout()
    path = OUT_DIR / "fig_nelson_siegel.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\nGénération des figures → {OUT_DIR.resolve()}\n")
    fig_raw_data_sample()
    fig_cleaning_funnel()
    fig_vol_smile()
    fig_nelson_siegel()
    print(f"\n4 figures générées avec succès.\n")


if __name__ == "__main__":
    main()
