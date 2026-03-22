"""
config.py
=========
Configuration for the Pairs Trading implementation based on Gorter (2006).

Synthetic pairs use the Section 6.1 model:
    xt  = xt-1 + sigma_x * w_t                 [random walk]
    eps_t = beta*eps_{t-1} + beta1*D_eps + eta  [AR spread, beta<1 => coint.]
    yt  = alpha * xt + eps_t                    [no constant!]
"""

# ─── Kappa table (Table 2.5) ──────────────────────────────────────────────────
# (hi, lo, kappa_pct) — number of formation-period trades => κ percentage
KAPPA_TABLE = [
    (float('inf'), 16, 0),   # > 15 trades  → κ = 0%
    (15, 10, 1),              # 10–15 trades → κ = 1%
    (9,   7, 2),              # 7–9  trades  → κ = 2%
    (6,   5, 4),              # 5–6  trades  → κ = 4%
    (4,   3, 6),              # 3–4  trades  → κ = 6%
    (2,   2, 8),              # 2    trades  → κ = 8%
    (1,   1, 10),             # 1    trade   → κ = 10%
    (0,   0, 10),             # 0    trades  → κ = 10%
]

# ─── Gamma search grid (Section 2.4) ─────────────────────────────────────────
# Try thresholds at these percentages of max|spread| in formation period.
# Cap at 50% of max|spread| (GAMMA_MAX_FRACTION).
GAMMA_PERCENTAGES  = list(range(5, 55, 5))   # [5, 10, 15, ..., 50]
GAMMA_MAX_FRACTION = 0.50

# ─── Synthetic pairs ─────────────────────────────────────────────────────────
# Four pairs covering the IMC-style range from Chapter 7:
#   Pair A – strong cointegration (beta << 1, many trades)
#   Pair B – moderate cointegration (beta close to 1)
#   Pair C – no cointegration (independent random walks)
#   Pair D – structural break at mid-sample

PAIRS = [
    {
        "name":   "Pair A (Good – Cointegrated)",
        "source": "synthetic",
        "X":      "Stock X",
        "Y":      "Stock Y",
        "synthetic_params": {
            "T":          520,
            "seed":       42,
            "x0":         30.0,
            "alpha":      1.50,   # true cointegrating ratio
            "sigma_x":    0.30,   # daily std of x increments
            "sigma_eps":  0.05,   # daily std of spread innovation
            "beta":       0.95,   # strong mean reversion => good pair
            "beta1":     -0.10,
            "independent": False,
            "structural_break": False,
        },
    },
    {
        "name":   "Pair B (Mediocre)",
        "source": "synthetic",
        "X":      "Stock X",
        "Y":      "Stock Y",
        "synthetic_params": {
            "T":          520,
            "seed":       7,
            "x0":         25.0,
            "alpha":      2.00,
            "sigma_x":    0.30,
            "sigma_eps":  0.12,   # noisier spread
            "beta":       0.985,  # weaker mean reversion
            "beta1":     -0.05,
            "independent": False,
            "structural_break": False,
        },
    },
    {
        "name":   "Pair C (Bad – Not Cointegrated)",
        "source": "synthetic",
        "X":      "Stock X",
        "Y":      "Stock Y",
        "synthetic_params": {
            "T":          520,
            "seed":       13,
            "x0":         20.0,
            "alpha":      1.20,
            "sigma_x":    0.35,
            "sigma_eps":  0.35,   # not used (independent=True)
            "beta":       1.0,    # unit root in spread => no cointegration
            "beta1":      0.0,
            "independent": True,  # completely independent random walks
            "sigma_y":    0.35,
            "structural_break": False,
        },
    },
    {
        "name":   "Pair D (Structural Break)",
        "source": "synthetic",
        "X":      "Stock X",
        "Y":      "Stock Y",
        "synthetic_params": {
            "T":          520,
            "seed":       99,
            "x0":         40.0,
            "alpha":      0.80,
            "sigma_x":    0.25,
            "sigma_eps":  0.06,
            "beta":       0.97,
            "beta1":     -0.08,
            "independent": False,
            "structural_break": True,
            "break_magnitude":  0.35,  # alpha increases 35% at mid-sample
        },
    },
]

# ─── General parameters ───────────────────────────────────────────────────────
LOOKBACK_YEARS = 2       # default date range for yfinance data
ADF_MAX_LAGS   = 10      # maximum AR lags in AIC search (Chapter 5)
TRADING_DAYS_PER_YEAR = 260  # per Gorter (2006): formation + backtest = 520 obs
