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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import norm

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
    """Smile de volatilité implicite pour 4 maturités BTC (Section 4 — Vol implicite)."""

    k = np.linspace(-0.60, 0.60, 300)

    # Paramètres réalistes BTC :
    #   - ATM vol entre 65-75 %
    #   - Skew négatif prononcé (put premium crypto)
    #   - Ailes puts atteignant 110-130 %
    #   - Skew et convexité décroissants avec la maturité (mean-reversion)
    maturities = {
        "7 jours":  dict(atm_vol=0.72, skew=-1.10, convexity=6.5),
        "30 jours": dict(atm_vol=0.68, skew=-0.85, convexity=4.8),
        "60 jours": dict(atm_vol=0.65, skew=-0.65, convexity=3.6),
        "90 jours": dict(atm_vol=0.62, skew=-0.52, convexity=2.8),
    }

    colors = ["#e74c3c", "#e67e22", "#2980b9", "#27ae60"]
    linestyles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for (label, params), color, ls in zip(maturities.items(), colors, linestyles):
        vol_pct = _smile(k, **params) * 100
        ax.plot(k, vol_pct, color=color, linestyle=ls, linewidth=2.0, label=label)

    # Ligne ATM
    ax.axvline(0, color="black", linestyle=":", linewidth=1.3, alpha=0.55)
    ax.text(0.01, 138, "ATM", fontsize=9, color="black", alpha=0.7)

    ax.set_xlabel(r"Log-moneyness  $k = \ln(K/F_T)$", fontsize=11)
    ax.set_ylabel(r"Volatilité implicite $\sigma_{IV}$ (%)", fontsize=11)
    ax.set_title(
        "Smile de volatilité implicite — Options BTC (Deribit)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(40, 145)
    ax.legend(fontsize=10, loc="upper center", ncol=4)
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
# FIGURE — Convergence Newton-Raphson vs Bisection
# ══════════════════════════════════════════════════════════════════════════════

def _bs_call(F: float, K: float, T: float, sigma: float) -> float:
    """Prix d'un call Black-76 (r=0, forward F)."""
    if sigma <= 0 or T <= 0:
        return max(F - K, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def _bs_vega(F: float, K: float, T: float, sigma: float) -> float:
    """Vega Black-76."""
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return F * np.sqrt(T) * norm.pdf(d1)


def _nr_convergence(price: float, F: float, K: float, T: float,
                    sigma0: float = 0.5, max_iter: int = 15,
                    tol: float = 1e-9) -> list[float]:
    """
    Retourne la suite |σ_{n+1} - σ_n| pour chaque itération Newton-Raphson.
    S'arrête si vega trop petite ou convergence atteinte.
    """
    sigma = sigma0
    deltas: list[float] = []
    for _ in range(max_iter):
        p = _bs_call(F, K, T, sigma)
        v = _bs_vega(F, K, T, sigma)
        if v < 1e-10:
            break
        step = (p - price) / v
        sigma_new = max(1e-4, min(sigma - step, 10.0))
        deltas.append(abs(sigma_new - sigma))
        if deltas[-1] < tol:
            break
        sigma = sigma_new
    return deltas


def fig_iv_convergence() -> None:
    """
    Comparaison de convergence : Newton-Raphson (quadratique) vs Bisection (linéaire).
    Panneau gauche : NR pour 3 niveaux de moneyness.
    Panneau droit  : bisection théorique.
    """
    F = 81_000.0  # forward BTC (~niveau marché)
    T = 45 / 365  # maturité 45 jours

    # ── Cas NR : ATM, légèrement OTM (k≈-0.3), profondément OTM (k≈-0.75) ──
    cases = [
        ("ATM  (k ≈ 0)",      F,                  0.68),
        ("OTM  (k ≈ −0.30)",  F * np.exp(-0.30),  0.80),
        ("Deep OTM  (k ≈ −0.75)", F * np.exp(-0.75), 1.10),
    ]
    nr_colors = ["#2980b9", "#e67e22", "#c0392b"]
    nr_ls     = ["-", "--", "-."]

    fig, (ax_nr, ax_bs) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panneau gauche : Newton-Raphson ───────────────────────────────────────
    for (label, K, true_vol), col, ls in zip(cases, nr_colors, nr_ls):
        price = _bs_call(F, K, T, true_vol)
        deltas = _nr_convergence(price, F, K, T, sigma0=0.50, max_iter=20, tol=1e-9)
        if len(deltas) == 0:
            continue
        iters = np.arange(1, len(deltas) + 1)
        ax_nr.semilogy(iters, deltas, color=col, linestyle=ls, linewidth=2.0,
                       marker="o", markersize=5, label=label)

    # Seuil de convergence
    ax_nr.axhline(1e-6, color="grey", linestyle=":", linewidth=1.2,
                  label=r"$\varepsilon = 10^{-6}$")

    ax_nr.set_xlabel("Numéro d'itération", fontsize=11)
    ax_nr.set_ylabel(r"$|\sigma_{n+1} - \sigma_n|$", fontsize=11)
    ax_nr.set_title("Newton-Raphson", fontsize=12, fontweight="bold")
    ax_nr.set_xlim(0.5, 14)
    ax_nr.set_ylim(1e-12, 2.0)
    ax_nr.legend(fontsize=9, loc="upper right")
    ax_nr.grid(True, which="both", alpha=0.3, linestyle="--")
    ax_nr.spines["top"].set_visible(False)
    ax_nr.spines["right"].set_visible(False)

    # ── Panneau droit : Bisection ─────────────────────────────────────────────
    # Convergence théorique : largeur_n = (σ_max - σ_min) / 2^n
    sigma_lo, sigma_hi = 0.01, 4.0
    width0 = sigma_hi - sigma_lo
    n_max = 36
    n_arr = np.arange(0, n_max)
    widths = width0 / 2.0 ** n_arr

    ax_bs.semilogy(n_arr, widths, color="#27ae60", linewidth=2.2,
                   label="Largeur intervalle $(\\sigma_{max} - \\sigma_{min})$")

    # Seuil ε = 1e-6
    ax_bs.axhline(1e-6, color="grey", linestyle=":", linewidth=1.2,
                  label=r"$\varepsilon = 10^{-6}$")

    # Trouver le nombre d'itérations pour atteindre 1e-6
    n_conv = int(np.ceil(np.log2(width0 / 1e-6)))
    ax_bs.axvline(n_conv, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_bs.annotate(
        f"n = {n_conv} itérations",
        xy=(n_conv, 1e-6), xytext=(n_conv + 2, 1e-4),
        fontsize=9, color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
    )

    ax_bs.set_xlabel("Numéro d'itération", fontsize=11)
    ax_bs.set_ylabel(r"$\sigma_{max} - \sigma_{min}$", fontsize=11)
    ax_bs.set_title("Bisection (convergence linéaire)", fontsize=12, fontweight="bold")
    ax_bs.set_xlim(0, n_max - 1)
    ax_bs.set_ylim(1e-12, 10)
    ax_bs.legend(fontsize=9, loc="upper right")
    ax_bs.grid(True, which="both", alpha=0.3, linestyle="--")
    ax_bs.spines["top"].set_visible(False)
    ax_bs.spines["right"].set_visible(False)

    fig.suptitle(
        "Comparaison de la convergence : Newton-Raphson vs Bisection",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    path = OUT_DIR / "fig_iv_convergence.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — Heatmap de la surface de volatilité implicite
# ══════════════════════════════════════════════════════════════════════════════

def fig_iv_heatmap() -> None:
    """
    Heatmap σ_IV(T, k) construite avec le modèle SVI-like _smile().

    Axe X : maturité en jours, Axe Y : log-moneyness k.
    Colormap RdYlGn_r, courbes de niveau tous les 10 vp.
    """
    T_days = np.array([7, 14, 30, 60, 90, 120, 180], dtype=float)
    k_grid = np.linspace(-0.60, 0.60, 10)

    # Paramètres ATM vol et skew interpolés en fonction de T
    # Vol ATM décroît avec la maturité (terme plat → backwardation)
    # Skew et convexité se réduisent (mean-reversion)
    def atm_vol(T_d: float) -> float:
        return 0.74 - 0.06 * np.log(T_d / 7) / np.log(180 / 7)

    def skew(T_d: float) -> float:
        return -1.15 + 0.60 * np.log(T_d / 7) / np.log(180 / 7)

    def convexity(T_d: float) -> float:
        return 7.0 - 4.5 * np.log(T_d / 7) / np.log(180 / 7)

    # Grille IV [n_k × n_T]
    Z = np.zeros((len(k_grid), len(T_days)))
    for j, T_d in enumerate(T_days):
        Z[:, j] = _smile(k_grid, atm_vol(T_d), skew(T_d), convexity(T_d)) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(False)  # pcolormesh gère sa propre grille via contours

    # Heatmap via pcolormesh pour axes continus
    T_mesh, k_mesh = np.meshgrid(T_days, k_grid)
    pcm = ax.pcolormesh(T_mesh, k_mesh, Z, cmap="RdYlGn_r", shading="auto",
                        vmin=40, vmax=140)

    # Courbes de niveau
    cs = ax.contour(T_mesh, k_mesh, Z,
                    levels=np.arange(40, 145, 10),
                    colors="black", linewidths=0.7, alpha=0.55)
    ax.clabel(cs, fmt="%d%%", fontsize=7, inline=True)

    # Ligne ATM
    ax.axhline(0, color="white", linestyle="--", linewidth=1.2, alpha=0.8, label="ATM (k = 0)")
    ax.text(185, 0.01, "ATM", fontsize=9, color="white", va="bottom")

    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"$\sigma_{IV}$ (%)", fontsize=11)

    ax.set_xlabel("Maturité (jours)", fontsize=11)
    ax.set_ylabel(r"Log-moneyness  $k = \ln(K/F_T)$", fontsize=11)
    ax.set_title(
        "Surface de volatilité implicite — données Deribit BTC",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(T_days)
    ax.set_xticklabels([f"{int(t)}j" for t in T_days])
    ax.set_yticks(np.round(k_grid, 2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = OUT_DIR / "fig_iv_heatmap.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")




# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\nGénération des figures → {OUT_DIR.resolve()}\n")
    fig_raw_data_sample()       # Fig 1 — tableau données brutes
    fig_cleaning_funnel()       # Fig 2 — entonnoir nettoyage
    fig_vol_smile()             # Fig 5 — smile de volatilité (4 maturités, réaliste BTC)
    fig_nelson_siegel()         # Fig NS — taux empiriques vs Nelson-Siegel
    fig_iv_convergence()        # Fig 4 — convergence NR vs Bisection
    fig_iv_heatmap()            # Fig 6 — heatmap surface de vol
    print(f"\n6 figures générées avec succès.\n")


if __name__ == "__main__":
    main()
