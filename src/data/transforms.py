"""
transforms.py — Transformations du DataFrame options (enrichissement).

Ajoute des colonnes dérivées utiles pour les étapes suivantes :
  - variance totale implicite w = σ² * T
  - log-moneyness normalisé
  - identification des paires call-put (pour parité)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_total_variance(df: pd.DataFrame, iv_col: str = "iv") -> pd.DataFrame:
    """Ajoute la colonne 'total_variance' = iv² * T."""
    df = df.copy()
    df["total_variance"] = df[iv_col] ** 2 * df["T"]
    return df


def add_atm_flag(df: pd.DataFrame, atm_band: float = 0.05) -> pd.DataFrame:
    """
    Ajoute un flag booléen 'is_atm' si |log_moneyness| < atm_band.
    Par défaut : 5% autour de ATM.
    """
    df = df.copy()
    df["is_atm"] = df["log_moneyness"].abs() < atm_band
    return df


def get_call_put_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un DataFrame des paires call-put matchées
    sur (expiry_str, strike, T).

    Note : on ne merge PAS sur forward_price car il peut légèrement
    différer entre call et put (underlying_price différent par item API).
    On prend la moyenne des deux comme forward de référence.

    Colonnes de sortie :
      expiry_str, strike, T, forward_price,
      bid_c, ask_c, mid_c, bid_p, ask_p, mid_p
    """
    calls = df[df["option_type"] == "C"][
        ["expiry_str", "strike", "T", "forward_price", "bid", "ask", "mid"]
    ].rename(columns={"bid": "bid_c", "ask": "ask_c", "mid": "mid_c",
                       "forward_price": "fp_c"})

    puts = df[df["option_type"] == "P"][
        ["expiry_str", "strike", "T", "forward_price", "bid", "ask", "mid"]
    ].rename(columns={"bid": "bid_p", "ask": "ask_p", "mid": "mid_p",
                       "forward_price": "fp_p"})

    pairs = pd.merge(calls, puts, on=["expiry_str", "strike", "T"])
    # Forward price = moyenne des deux (robustesse si légère différence)
    pairs["forward_price"] = 0.5 * (pairs["fp_c"] + pairs["fp_p"])
    pairs = pairs.drop(columns=["fp_c", "fp_p"])
    return pairs.reset_index(drop=True)


def pivot_surface(df: pd.DataFrame, iv_col: str = "iv") -> pd.DataFrame:
    """
    Pivote le DataFrame IV en une matrice (T x k) pour visualisation.

    Retourne un DataFrame avec T en index et k (arrondi) en colonnes.
    """
    df = df.copy()
    df["k_round"] = df["log_moneyness"].round(2)
    pivot = df.pivot_table(index="T", columns="k_round", values=iv_col, aggfunc="mean")
    return pivot
