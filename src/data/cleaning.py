"""
cleaning.py — Pipeline complet de nettoyage des données options.

Agrège loaders.py + validation.py en un pipeline prêt à l'emploi.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from config.settings import DataCleaningConfig, CONFIG
from src.data.loaders import build_options_dataframe, build_futures_dataframe
from src.data.validation import run_validation_report

logger = logging.getLogger(__name__)


def clean_options(
    raw_data: dict[str, Any],
    cfg: DataCleaningConfig | None = None,
    reference_dt: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Pipeline de nettoyage complet.

    Parameters
    ----------
    raw_data     : dict retourné par DeribitClient.fetch_all_options_data()
    cfg          : configuration de nettoyage (défaut = CONFIG.cleaning)
    reference_dt : instant de référence pour le calcul de T

    Returns
    -------
    (df_clean, df_futures, cleaning_report)
    """
    cfg = cfg or CONFIG.cleaning

    spot = float(raw_data.get("spot", 0.0))
    raw_options = raw_data.get("options", [])
    raw_futures = raw_data.get("futures", [])

    # Construction des DataFrames bruts
    df_raw = build_options_dataframe(
        raw_options=raw_options,
        spot=spot,
        futures_data=raw_futures,
        reference_dt=reference_dt,
    )
    df_futures = build_futures_dataframe(
        raw_futures=raw_futures,
        spot=spot,
        reference_dt=reference_dt,
    )

    if df_raw.empty:
        logger.warning("Aucune donnée option disponible après parsing.")
        return df_raw, df_futures, {"initial": 0, "retenues": 0, "total_rejetées": 0}

    # Validation et filtrage
    df_clean, report = run_validation_report(
        df=df_raw,
        max_spread_pct=cfg.max_spread_pct,
        min_T=cfg.min_time_to_maturity,
        max_T=cfg.max_time_to_maturity,
        min_moneyness=cfg.min_moneyness,
        max_moneyness=cfg.max_moneyness,
    )

    return df_clean, df_futures, report
