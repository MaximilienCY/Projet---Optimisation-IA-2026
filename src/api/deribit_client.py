"""
deribit_client.py — Client HTTP robuste pour l'API publique Deribit.

Utilise uniquement les endpoints publics (pas d'authentification requise
pour les données de marché).

Documentation API : https://docs.deribit.com/
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import DeribitConfig

logger = logging.getLogger(__name__)


class DeribitAPIError(Exception):
    """Erreur retournée par l'API Deribit."""


class DeribitClient:
    """
    Client pour l'API publique Deribit v2.

    Gère les retries automatiques, les timeouts et la journalisation.
    Toutes les méthodes retournent des structures Python nues (dict/list).
    """

    def __init__(self, config: DeribitConfig | None = None) -> None:
        self.config = config or DeribitConfig()
        self.session = self._build_session()

    # ─── Session HTTP ────────────────────────────────────────────────────────

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.headers.update({"Content-Type": "application/json"})
        return session

    # ─── Méthode de bas niveau ───────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """
        Effectue un GET sur l'API Deribit et retourne le champ 'result'.

        Raises DeribitAPIError si l'API retourne une erreur.
        """
        url = f"{self.config.base_url}/{endpoint}"
        logger.debug("GET %s %s", url, params)
        response = self.session.get(url, params=params, timeout=self.config.timeout)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise DeribitAPIError(
                f"Deribit API error [{data['error'].get('code')}]: "
                f"{data['error'].get('message')}"
            )
        return data.get("result")

    # ─── Endpoints de marché ─────────────────────────────────────────────────

    def get_instruments(self, currency: str | None = None, kind: str = "option") -> list[dict]:
        """
        Récupère la liste des instruments actifs.

        Parameters
        ----------
        currency : "BTC" ou "ETH" (défaut : config.currency)
        kind     : "option" ou "future"
        """
        ccy = currency or self.config.currency
        result = self._get(
            "public/get_instruments",
            params={"currency": ccy, "kind": kind, "expired": "false"},
        )
        logger.info("Instruments %s/%s : %d récupérés", ccy, kind, len(result or []))
        return result or []

    def get_book_summary_by_currency(
        self, currency: str | None = None, kind: str = "option"
    ) -> list[dict]:
        """
        Récupère les résumés de carnet d'ordres pour tous les instruments
        d'une devise/catégorie.

        Retourne best_bid, best_ask, mark_price, underlying_price, etc.
        """
        ccy = currency or self.config.currency
        result = self._get(
            "public/get_book_summary_by_currency",
            params={"currency": ccy, "kind": kind},
        )
        logger.info(
            "Book summary %s/%s : %d instruments", ccy, kind, len(result or [])
        )
        return result or []

    def get_ticker(self, instrument_name: str) -> dict:
        """Ticker détaillé pour un instrument (best_bid, best_ask, greeks, etc.)."""
        result = self._get("public/ticker", params={"instrument_name": instrument_name})
        return result or {}

    def get_index_price(self, index_name: str | None = None) -> float:
        """
        Retourne le prix spot de l'index (ex: 'btc_usd').
        """
        ccy = self.config.currency.lower()
        idx = index_name or f"{ccy}_usd"
        result = self._get("public/get_index_price", params={"index_name": idx})
        return float(result.get("index_price", 0.0))

    def get_futures_summary(self, currency: str | None = None) -> list[dict]:
        """Retourne les résumés des futures actifs (pour extraire les prix forward)."""
        return self.get_book_summary_by_currency(currency=currency, kind="future")

    # ─── Méthode de collecte complète ────────────────────────────────────────

    def fetch_all_options_data(self, currency: str | None = None) -> dict[str, Any]:
        """
        Récupère en une seule passe toutes les données nécessaires :
          - spot index
          - book summary des options
          - book summary des futures

        Returns
        -------
        dict avec clés : 'spot', 'options', 'futures', 'currency', 'timestamp'
        """
        import datetime

        ccy = currency or self.config.currency
        logger.info("Récupération complète des données %s…", ccy)

        spot = self.get_index_price()
        options = self.get_book_summary_by_currency(currency=ccy, kind="option")
        futures = self.get_futures_summary(currency=ccy)

        return {
            "currency": ccy,
            "spot": spot,
            "options": options,
            "futures": futures,
            "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        }
