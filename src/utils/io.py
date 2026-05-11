"""
io.py — Utilitaires d'entrées/sorties (CSV, Parquet, JSON).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Crée le répertoire (et ses parents) s'il n'existe pas."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Sauvegarde un DataFrame en CSV."""
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False, **kwargs)
    logger.info("CSV sauvegardé : %s (%d lignes)", path, len(df))


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Charge un CSV en DataFrame."""
    df = pd.read_csv(path, **kwargs)
    logger.info("CSV chargé : %s (%d lignes)", path, len(df))
    return df


def save_parquet(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Sauvegarde un DataFrame en Parquet (compression snappy)."""
    ensure_dir(Path(path).parent)
    df.to_parquet(path, index=False, compression="snappy", **kwargs)
    logger.info("Parquet sauvegardé : %s (%d lignes)", path, len(df))


def load_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Charge un fichier Parquet en DataFrame."""
    df = pd.read_parquet(path, **kwargs)
    logger.info("Parquet chargé : %s (%d lignes)", path, len(df))
    return df


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Sauvegarde un objet Python en JSON."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=str)
    logger.info("JSON sauvegardé : %s", path)


def load_json(path: str | Path) -> Any:
    """Charge un fichier JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("JSON chargé : %s", path)
    return data


def file_exists(path: str | Path) -> bool:
    """Vérifie si un fichier existe."""
    return Path(path).is_file()


def latest_file(directory: str | Path, pattern: str = "*.csv") -> Path | None:
    """Retourne le fichier le plus récent correspondant au pattern, ou None."""
    files = sorted(Path(directory).glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None
