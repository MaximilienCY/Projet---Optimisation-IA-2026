"""
generate_ssvi_figures.py
------------------------
Script autonome de génération des figures SSVI pour le rapport LaTeX.

Usage (depuis la racine du dépôt) :
    python reports/generate_ssvi_figures.py

Dépendances : numpy, pandas, matplotlib, scipy (pas de Streamlit).
Sorties     : reports/figures/fig_ssvi_surface.png
              reports/figures/fig_ssvi_fit.png      (300 DPI)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── Résolution des chemins ───────────────────────────────────────────────────
# Ajouter la racine du projet au PYTHONPATH pour les imports src.*
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.volatility.ssvi import (
    SSVIParams,
    build_ssvi_surface,
    ssvi_implied_vol,
    ssvi_theta,
)

# ── Constantes ───────────────────────────────────────────────────────────────
OUT_DIR = ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
RNG = np.random.default_rng(seed=42)

# Paramètres SSVI calibrés (valeurs par défaut — à remplacer par les vraies
# valeurs après calibration sur données Deribit BTC)
DEFAULT_PARAMS = SSVIParams(
    kappa=1.5,
    nu0=0.8,
    nu_inf=0.35,
    rho=-0.35,
    eta=1.2,
    lambda_=0.45,
)

# Style matplotlib académique
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass  # fallback sur le style par défaut matplotlib


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Surface 3D de la volatilité implicite SSVI
# ═══════════════════════════════════════════════════════════════════════════════

def fig_ssvi_surface(params: SSVIParams = DEFAULT_PARAMS) -> None:
    """
    Surface 3D σ_SSVI(k, T) en % via matplotlib plot_surface.

    Axe X : log-moneyness k ∈ [-0.8, 0.8]
    Axe Y : maturité T ∈ {0.083, 0.25, 0.5, 1.0, 2.0} ans
    Axe Z : vol implicite (%)
    """
    k_grid = np.linspace(-0.8, 0.8, 80)
    t_grid = np.array([0.083, 0.25, 0.5, 1.0, 2.0])

    K_mesh, T_mesh, IV_mesh = build_ssvi_surface(params, k_grid=k_grid, t_grid=t_grid)
    Z = IV_mesh * 100.0  # décimal → %

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        K_mesh, T_mesh, Z,
        cmap="RdYlGn_r",
        linewidth=0.0,
        antialiased=True,
        alpha=0.90,
    )

    # Plan vertical semi-transparent ATM (k = 0)
    t_plane = np.linspace(t_grid.min(), t_grid.max(), 2)
    z_plane = np.linspace(Z.min(), Z.max(), 2)
    T_pl, Z_pl = np.meshgrid(t_plane, z_plane)
    K_pl = np.zeros_like(T_pl)
    ax.plot_surface(
        K_pl, T_pl, Z_pl,
        color="grey",
        alpha=0.18,
        linewidth=0,
        zorder=0,
    )
    # Ligne verticale ATM en k=0 sur le plan de base
    ax.plot(
        [0, 0], [t_grid.min(), t_grid.max()], [Z.min(), Z.min()],
        color="grey", linewidth=1.2, linestyle="--", alpha=0.6,
    )

    cb = fig.colorbar(surf, ax=ax, pad=0.1, shrink=0.55, aspect=14)
    cb.set_label("Vol implicite (%)", fontsize=11)

    ax.set_xlabel("Log-moneyness  k", fontsize=11, labelpad=10)
    ax.set_ylabel("Maturité T (années)", fontsize=11, labelpad=10)
    ax.set_zlabel("Vol implicite (%)", fontsize=11, labelpad=8)
    ax.set_title("Surface de volatilité implicite SSVI", fontsize=13, fontweight="bold", pad=14)

    ax.view_init(elev=28, azim=-50)

    fig.tight_layout()
    path = OUT_DIR / "fig_ssvi_surface.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Ajustement SSVI vs points de marché simulés
# ═══════════════════════════════════════════════════════════════════════════════

def fig_ssvi_fit(params: SSVIParams = DEFAULT_PARAMS) -> None:
    """
    Pour 4 maturités : points marché simulés (SSVI + bruit) + courbe modèle.

    Les points marché sont construits en ajoutant un bruit gaussien (std=1.5 pp)
    à la surface SSVI pour simuler des IVs observées réalistes.
    """
    maturities = [0.083, 0.25, 0.5, 1.0]           # T en années
    maturity_labels = [
        f"T = {int(round(t * 365))}j" for t in maturities
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    k_market = np.linspace(-0.6, 0.6, 15)           # 15 strikes marché
    k_model = np.linspace(-0.8, 0.8, 200)            # courbe modèle lisse

    fig, ax = plt.subplots(figsize=(10, 6))

    for T, label, color in zip(maturities, maturity_labels, colors):
        # Points "marché" : modèle + bruit gaussien
        iv_model_pts = ssvi_implied_vol(k_market, T, params) * 100.0
        noise = RNG.normal(0.0, 1.5, size=len(k_market))  # std = 1.5 pp
        iv_market = np.clip(iv_model_pts + noise, 1.0, None)

        ax.scatter(
            k_market, iv_market,
            s=45, color=color, alpha=0.70, zorder=3,
            label=f"{label} — marché",
        )

        # Courbe modèle SSVI
        iv_curve = ssvi_implied_vol(k_model, T, params) * 100.0
        ax.plot(
            k_model, iv_curve,
            color=color, linewidth=2.0, zorder=4,
            label=f"{label} — SSVI",
        )

    ax.set_xlabel(r"Log-moneyness  $k = \ln(K/F_T)$", fontsize=11)
    ax.set_ylabel("Volatilité implicite (%)", fontsize=11)
    ax.set_xlim(-0.85, 0.85)

    # Ligne ATM (après que les courbes ont fixé ylim)
    ax.axvline(0, color="black", linestyle=":", linewidth=1.2, alpha=0.6)
    ymin, ymax = ax.get_ylim()
    ax.text(0.005, ymax - (ymax - ymin) * 0.04,
            "ATM", fontsize=9, va="top", ha="left", color="black", alpha=0.7)

    ax.set_title(
        "Calibration SSVI — Ajustement modèle vs marché",
        fontsize=13, fontweight="bold",
    )

    # Légende compacte : regrouper "marché" et "SSVI" par maturité
    handles, labels_leg = ax.get_legend_handles_labels()
    # Construire une légende propre : 1 entrée par maturité avec icône combinée
    from matplotlib.lines import Line2D
    legend_handles = []
    for i, (T, label, color) in enumerate(zip(maturities, maturity_labels, colors)):
        handle = Line2D(
            [0], [0],
            color=color, linewidth=2,
            marker="o", markersize=5, markerfacecolor=color, alpha=0.85,
            label=label,
        )
        legend_handles.append(handle)

    ax.legend(handles=legend_handles, loc="upper right", fontsize=10,
              framealpha=0.92, edgecolor="grey")

    ax.grid(True, alpha=0.35, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = OUT_DIR / "fig_ssvi_fit.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Résumé des paramètres calibrés
# ═══════════════════════════════════════════════════════════════════════════════

def _print_params(params: SSVIParams) -> None:
    print("\n  Paramètres SSVI utilisés :")
    print(f"    κ       (mean-reversion)  = {params.kappa:.4f}")
    print(f"    ν₀      (var ATM court T) = {params.nu0:.4f}  "
          f"  → σ_ATM(T→0) ≈ {(params.nu0**0.5)*100:.1f}%")
    print(f"    ν_∞     (var ATM long T)  = {params.nu_inf:.4f}  "
          f"  → σ_ATM(T→∞) ≈ {(params.nu_inf**0.5)*100:.1f}%")
    print(f"    ρ       (skew)            = {params.rho:.4f}")
    print(f"    η       (niveau skew)     = {params.eta:.4f}")
    print(f"    λ       (régime skew)     = {params.lambda_:.4f}")

    # Vérification butterfly : η·(1+|ρ|) ≤ 4
    butterfly_val = params.eta * (1 + abs(params.rho))
    ok = "✅" if butterfly_val <= 4.0 else "❌"
    print(f"\n  Non-arbitrage butterfly : η·(1+|ρ|) = {butterfly_val:.3f} ≤ 4 → {ok}")


# ═══════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\nGénération des figures SSVI → {OUT_DIR.resolve()}\n")
    _print_params(DEFAULT_PARAMS)
    print()
    fig_ssvi_surface()
    fig_ssvi_fit()
    print(f"\n2 figures SSVI générées avec succès.\n")


if __name__ == "__main__":
    main()
