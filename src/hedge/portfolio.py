"""
portfolio.py — Représentation du portefeuille de couverture.

Logique :
  - On vend 1 unité du produit (call spread ou put spread).
  - On construit un portefeuille de couverture avec des options cotées + futures.
  - Les instruments de couverture sont sélectionnés depuis le DataFrame d'options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.pricing.greeks import Greeks, compute_greeks
from src.volatility.ssvi import SSVIParams, ssvi_implied_vol


@dataclass
class HedgeInstrument:
    """Un instrument de couverture avec sa quantité et ses grecques."""
    instrument_name: str
    option_type: str        # "C", "P", ou "F" (future)
    strike: float
    T: float
    forward_price: float
    rate: float
    iv: float               # vol implicite (0 pour future)
    quantity: float = 0.0
    greeks: Greeks = field(default_factory=Greeks)
    mid_price: float = 0.0

    def compute_and_set_greeks(self, ssvi_params: Optional[SSVIParams] = None) -> None:
        """Recalcule les grecques avec iv actuelle ou via SSVI."""
        if self.option_type == "F":
            # Future : Delta = e^{-rT} ≈ 1, Gamma = Vega = 0
            disc = np.exp(-self.rate * self.T) if self.T > 0 else 1.0
            self.greeks = Greeks(delta=disc, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
        else:
            iv = self.iv
            if ssvi_params is not None and iv == 0:
                k = np.log(self.strike / self.forward_price)
                iv = float(ssvi_implied_vol(k, self.T, ssvi_params))
            self.greeks = compute_greeks(
                self.forward_price, self.strike, self.T, self.rate, iv, self.option_type
            )

    @property
    def weighted_greeks(self) -> Greeks:
        """Grecques pondérées par la quantité."""
        return self.greeks * self.quantity


@dataclass
class HedgePortfolio:
    """
    Portefeuille de couverture complet.

    Contient :
      - 1 position vendeuse sur le produit (call spread)
      - n instruments de couverture (options + futures)
    """
    product_greeks: Greeks = field(default_factory=Greeks)
    instruments: list[HedgeInstrument] = field(default_factory=list)

    def net_delta(self) -> float:
        """Delta net du portefeuille (produit + couverture)."""
        hedge_delta = sum(i.weighted_greeks.delta for i in self.instruments)
        return -self.product_greeks.delta + hedge_delta

    def net_gamma(self) -> float:
        hedge_gamma = sum(i.weighted_greeks.gamma for i in self.instruments)
        return -self.product_greeks.gamma + hedge_gamma

    def net_vega(self) -> float:
        hedge_vega = sum(i.weighted_greeks.vega for i in self.instruments)
        return -self.product_greeks.vega + hedge_vega

    def summary_df(self) -> pd.DataFrame:
        """DataFrame de synthèse du portefeuille."""
        rows = [{
            "instrument": i.instrument_name,
            "type": i.option_type,
            "strike": i.strike,
            "T": i.T,
            "quantity": i.quantity,
            "delta": i.greeks.delta,
            "gamma": i.greeks.gamma,
            "vega": i.greeks.vega,
            "w_delta": i.weighted_greeks.delta,
            "w_gamma": i.weighted_greeks.gamma,
            "w_vega": i.weighted_greeks.vega,
            "mid_price": i.mid_price,
        } for i in self.instruments]

        df = pd.DataFrame(rows)
        if not df.empty:
            # Ligne de totaux
            totals = {
                "instrument": "TOTAL HEDGE",
                "type": "",
                "strike": np.nan,
                "T": np.nan,
                "quantity": np.nan,
                "delta": np.nan,
                "gamma": np.nan,
                "vega": np.nan,
                "w_delta": df["w_delta"].sum(),
                "w_gamma": df["w_gamma"].sum(),
                "w_vega": df["w_vega"].sum(),
                "mid_price": np.nan,
            }
            df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
        return df

    def hedge_cost(self) -> float:
        """Coût total du portefeuille de couverture (en USD)."""
        return sum(i.quantity * i.mid_price for i in self.instruments)


def select_hedge_instruments(
    df_options: pd.DataFrame,
    product_T: float,
    max_instruments: int = 10,
    use_futures: bool = True,
    df_futures: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Sélectionne les instruments de couverture les plus liquides.

    Critères :
      - Options de maturité proche du produit (± 2 semaines)
      - Triés par open_interest décroissant
      - Maximum max_instruments instruments

    Returns
    -------
    DataFrame des instruments sélectionnés.
    """
    # Options : maturité proche
    T_tol = 4 / 52.0  # ± 4 semaines
    sub = df_options[
        (np.abs(df_options["T"] - product_T) <= T_tol) &
        (df_options["iv"].notna()) &
        (df_options["iv"] > 0)
    ].copy()

    # Tri par liquidité (open_interest puis volume)
    sub["liquidity_score"] = (
        sub.get("open_interest", 0).fillna(0) * 0.7 +
        sub.get("volume", 0).fillna(0) * 0.3
    )
    sub = sub.sort_values("liquidity_score", ascending=False)
    sub = sub.head(max_instruments - (1 if use_futures else 0))

    if use_futures and df_futures is not None and not df_futures.empty:
        # Ajoute le future le plus proche en maturité
        fut = df_futures.copy()
        fut["T_diff"] = np.abs(fut["T"] - product_T)
        closest_fut = fut.sort_values("T_diff").head(1)
        closest_fut = closest_fut.assign(option_type="F", iv=0.0, strike=0.0)
        sub = pd.concat([sub, closest_fut], ignore_index=True)

    return sub.reset_index(drop=True)
