"""
constants.py — Constantes mathématiques et financières du projet.
"""

import math

# ─── Mathématiques ─────────────────────────────────────────────────────────
SQRT_2PI: float = math.sqrt(2 * math.pi)
INV_SQRT_2PI: float = 1.0 / SQRT_2PI
SQRT_2: float = math.sqrt(2)
INV_SQRT_2: float = 1.0 / SQRT_2

# ─── Conventions temporelles ───────────────────────────────────────────────
DAYS_PER_YEAR: int = 365
WEEKS_PER_YEAR: float = 52.0
HOURS_PER_DAY: int = 24

# ─── Deribit ───────────────────────────────────────────────────────────────
# Taille minimale du tick de volatilité sur Deribit (en fraction)
DERIBIT_MIN_IV: float = 0.001          # 0.1%
DERIBIT_MAX_IV: float = 5.0            # 500%

# Dividendes supposés nuls pour BTC/ETH
CONTINUOUS_DIVIDEND: float = 0.0

# ─── Numérique ─────────────────────────────────────────────────────────────
EPS: float = 1e-10
LARGE: float = 1e10
