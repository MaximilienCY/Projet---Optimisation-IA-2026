"""
plotting.py — Fonctions de visualisation Plotly communes au projet.

Conventions :
  - Toutes les fonctions retournent un objet go.Figure.
  - Thème sobre adapté au rapport académique.
  - Les titres, axes et légendes sont toujours renseignés.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Thème de base
_TEMPLATE = "plotly_white"
_FONT_FAMILY = "Computer Modern, serif"


def _base_layout(title: str, **kwargs) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16)),
        template=_TEMPLATE,
        font=dict(family=_FONT_FAMILY, size=12),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#cccccc", borderwidth=1),
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Données & nettoyage
# ─────────────────────────────────────────────────────────────────────────────

def plot_cleaning_report(report: dict) -> go.Figure:
    """Barres empilées : lignes initiales / rejetées / retenues."""
    labels = list(report.keys())
    values = list(report.values())
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#2196F3"))
    fig.update_layout(**_base_layout("Rapport de nettoyage des données"),
                      xaxis_title="Motif", yaxis_title="Nombre de lignes")
    return fig


def plot_spread_distribution(df: pd.DataFrame, spread_col: str = "spread_pct") -> go.Figure:
    """Histogramme des spreads bid-ask en %."""
    fig = go.Figure(go.Histogram(x=df[spread_col] * 100, nbinsx=60,
                                 marker_color="#FF5722", opacity=0.8))
    fig.update_layout(**_base_layout("Distribution des spreads bid-ask (%)"),
                      xaxis_title="Spread (% du mid)", yaxis_title="Fréquence")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Courbe des taux
# ─────────────────────────────────────────────────────────────────────────────

def plot_rates_curve(
    maturities_empirical: np.ndarray,
    rates_empirical: np.ndarray,
    maturities_fitted: np.ndarray,
    rates_fitted: np.ndarray,
    ns_params: Optional[dict] = None,
) -> go.Figure:
    """Compare les taux implicites empiriques et la courbe Nelson-Siegel."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=maturities_empirical, y=rates_empirical * 100,
        mode="markers", name="Taux implicites (marché)",
        marker=dict(color="#E53935", size=8),
    ))
    fig.add_trace(go.Scatter(
        x=maturities_fitted, y=rates_fitted * 100,
        mode="lines", name="Nelson-Siegel (calibré)",
        line=dict(color="#1565C0", width=2),
    ))
    title = "Courbe des taux implicites vs Nelson-Siegel"
    if ns_params:
        subtitle = (
            f"β₀={ns_params.get('beta0', 0):.4f}, "
            f"β₁={ns_params.get('beta1', 0):.4f}, "
            f"β₂={ns_params.get('beta2', 0):.4f}, "
            f"λ={ns_params.get('lambda_', 0):.4f}"
        )
        title += f"<br><sup>{subtitle}</sup>"
    fig.update_layout(**_base_layout(title),
                      xaxis_title="Maturité (années)", yaxis_title="Taux (%)")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Smile de volatilité
# ─────────────────────────────────────────────────────────────────────────────

def plot_iv_smile(
    df_iv: pd.DataFrame,
    maturities: Sequence[float],
    moneyness_col: str = "log_moneyness",
    iv_col: str = "iv",
    iv_ssvi_col: Optional[str] = "iv_ssvi",
) -> go.Figure:
    """
    Smile de volatilité pour plusieurs maturités.
    Compare marché vs SSVI si iv_ssvi_col est fourni.
    """
    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    for i, T in enumerate(maturities):
        sub = df_iv[np.isclose(df_iv["T"], T, atol=0.02)]
        if sub.empty:
            continue
        sub = sub.sort_values(moneyness_col)
        color = colors[i % len(colors)]
        label = f"T={T:.2f}a"
        fig.add_trace(go.Scatter(
            x=sub[moneyness_col], y=sub[iv_col] * 100,
            mode="markers", name=f"Marché {label}",
            marker=dict(color=color, size=7, symbol="circle"),
        ))
        if iv_ssvi_col and iv_ssvi_col in sub.columns:
            fig.add_trace(go.Scatter(
                x=sub[moneyness_col], y=sub[iv_ssvi_col] * 100,
                mode="lines", name=f"SSVI {label}",
                line=dict(color=color, width=2, dash="dash"),
            ))
    fig.update_layout(**_base_layout("Smile de volatilité implicite"),
                      xaxis_title="Log-moneyness k = ln(K/F)", yaxis_title="Vol implicite (%)")
    return fig


def plot_iv_surface(
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    iv_surface: np.ndarray,
    title: str = "Surface de volatilité implicite",
) -> go.Figure:
    """Surface 3D de volatilité implicite."""
    fig = go.Figure(go.Surface(
        x=k_grid, y=T_grid, z=iv_surface * 100,
        colorscale="Viridis",
        colorbar=dict(title="σ (%)"),
    ))
    fig.update_layout(
        **_base_layout(title),
        scene=dict(
            xaxis_title="Log-moneyness k",
            yaxis_title="Maturité T (années)",
            zaxis_title="Vol implicite (%)",
        ),
        height=600,
    )
    return fig


def plot_iv_heatmap(
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    iv_surface: np.ndarray,
    title: str = "Heatmap de volatilité implicite",
) -> go.Figure:
    """Heatmap 2D de la surface de volatilité."""
    fig = go.Figure(go.Heatmap(
        x=k_grid,
        y=T_grid,
        z=iv_surface * 100,
        colorscale="RdYlGn_r",
        colorbar=dict(title="σ (%)"),
    ))
    fig.update_layout(**_base_layout(title),
                      xaxis_title="Log-moneyness k = ln(K/F)",
                      yaxis_title="Maturité T (années)")
    return fig


def plot_iv_errors(
    df_iv: pd.DataFrame,
    error_col: str = "iv_error",
    moneyness_col: str = "log_moneyness",
) -> go.Figure:
    """Scatter des erreurs de calibration SSVI (IV marché - IV SSVI)."""
    fig = go.Figure(go.Scatter(
        x=df_iv[moneyness_col],
        y=df_iv[error_col] * 100,
        mode="markers",
        marker=dict(
            color=df_iv[error_col] * 100,
            colorscale="RdBu",
            cmid=0,
            size=7,
            showscale=True,
            colorbar=dict(title="Erreur (%)"),
        ),
        text=df_iv.get("instrument_name", None),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(**_base_layout("Erreurs de calibration SSVI (σ_marché - σ_SSVI)"),
                      xaxis_title="Log-moneyness k", yaxis_title="Erreur de vol (%)")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Payoff et produit
# ─────────────────────────────────────────────────────────────────────────────

def plot_payoff(
    spots: np.ndarray,
    payoff: np.ndarray,
    label: str = "Payoff",
    current_spot: Optional[float] = None,
    price: Optional[float] = None,
) -> go.Figure:
    """Courbe de payoff à maturité d'un produit dérivé."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spots, y=payoff,
        mode="lines", name=label,
        line=dict(color="#1565C0", width=2),
        fill="tozeroy", fillcolor="rgba(21, 101, 192, 0.1)",
    ))
    if current_spot is not None:
        fig.add_vline(x=current_spot, line_dash="dot", line_color="#E53935",
                      annotation_text=f"Spot={current_spot:,.0f}")
    if price is not None:
        fig.add_hline(y=-price, line_dash="dash", line_color="#FF8F00",
                      annotation_text=f"Prime={price:.4f}")
    fig.update_layout(**_base_layout(f"Payoff à maturité : {label}"),
                      xaxis_title="Prix spot à maturité", yaxis_title="P&L ($)")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Grecques
# ─────────────────────────────────────────────────────────────────────────────

def plot_greeks_bar(greeks_dict: dict, title: str = "Grecques du produit") -> go.Figure:
    """Bar chart des grecques."""
    labels = list(greeks_dict.keys())
    values = [float(v) for v in greeks_dict.values()]
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
    fig.update_layout(**_base_layout(title), xaxis_title="Grecque", yaxis_title="Valeur")
    return fig


def plot_hedge_composition(hedge_df: pd.DataFrame) -> go.Figure:
    """Bar chart des quantités du portefeuille de couverture."""
    fig = go.Figure(go.Bar(
        x=hedge_df["instrument"],
        y=hedge_df["quantity"],
        marker_color=hedge_df["quantity"].apply(lambda q: "#4CAF50" if q >= 0 else "#F44336"),
        text=hedge_df["quantity"].round(4),
        textposition="auto",
    ))
    fig.update_layout(**_base_layout("Composition du portefeuille de couverture"),
                      xaxis_title="Instrument", yaxis_title="Quantité",
                      xaxis=dict(tickangle=-35))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# P&L de stress
# ─────────────────────────────────────────────────────────────────────────────

def plot_pnl_waterfall(pnl_breakdown: dict) -> go.Figure:
    """Waterfall chart du P&L de stress."""
    labels = list(pnl_breakdown.keys())
    values = list(pnl_breakdown.values())
    measure = ["relative"] * (len(labels) - 1) + ["total"]
    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measure,
        connector=dict(line=dict(color="#9E9E9E")),
        increasing=dict(marker=dict(color="#4CAF50")),
        decreasing=dict(marker=dict(color="#F44336")),
        totals=dict(marker=dict(color="#2196F3")),
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(**_base_layout("Décomposition du P&L sous scénario de stress"),
                      xaxis_title="Composante", yaxis_title="P&L ($)")
    return fig


def plot_newton_vs_bisection(df_conv: pd.DataFrame) -> go.Figure:
    """
    Comparaison convergence Newton-Raphson vs Dichotomie.

    Utilise les colonnes 'iv_method' (enum string) et 'iv_iters' (int)
    produites par compute_implied_vols().
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Itérations Newton-Raphson", "Itérations Dichotomie"),
    )

    if "iv_method" not in df_conv.columns or "iv_iters" not in df_conv.columns:
        # Colonnes manquantes — retourne figure vide avec message
        fig.add_annotation(
            text="Données de convergence non disponibles",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        fig.update_layout(**_base_layout("Convergence : Newton-Raphson vs Dichotomie"))
        return fig

    nr_iters = df_conv.loc[df_conv["iv_method"] == "newton", "iv_iters"].dropna()
    bis_iters = df_conv.loc[df_conv["iv_method"] == "bisection", "iv_iters"].dropna()

    if not nr_iters.empty:
        fig.add_trace(
            go.Histogram(
                x=nr_iters, nbinsx=20,
                marker_color="#1565C0", opacity=0.8,
                name=f"Newton ({len(nr_iters)} options)",
            ),
            row=1, col=1,
        )
    if not bis_iters.empty:
        fig.add_trace(
            go.Histogram(
                x=bis_iters, nbinsx=30,
                marker_color="#E65100", opacity=0.8,
                name=f"Dichotomie ({len(bis_iters)} options)",
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title_text="Nombre d'itérations", row=1, col=1)
    fig.update_xaxes(title_text="Nombre d'itérations", row=1, col=2)
    fig.update_yaxes(title_text="Nombre d'options", row=1, col=1)
    fig.update_layout(**_base_layout("Convergence : Newton-Raphson vs Dichotomie"))
    return fig
