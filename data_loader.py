"""
data_loader.py
==============
Load and prepare price data for pairs trading analysis.

Supports three sources:
  1. 'synthetic' – generate cointegrated or random-walk pairs
  2. 'csv'       – load from CSV files (columns: Date, Price)
  3. 'yfinance'  – download from Yahoo Finance (requires network + yfinance)

Dividend adjustment (section 2.2):
  Bloomberg adjusted-close prices already incorporate dividends.
  For 'yfinance' we use Adj Close which is also dividend-adjusted.
  For 'csv', the user is expected to provide already-adjusted prices, or
  the file may contain an optional 'Dividend' column, in which case we
  apply the book's factor method.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple


# ── Synthetic data generation ──────────────────────────────────────────────────

def _generate_synthetic_pair(params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic pair of stock prices using the model from
    Chapter 4 of Gorter (2006).

    The model:
        xt = xt-1 * exp(drift + sigma_common * wt + sigma_idio * u1t)
        yt = yt-1 * exp(drift + sigma_common * wt + sigma_idio * u2t)
                   + mean_revert_strength * (r_true * xt-1 - yt-1)   [error correction]

    where wt, u1t, u2t ~ i.i.d. N(0,1) are independent innovations.

    When mean_revert_strength > 0 the spread yt - r_true*xt is stationary
    (pair is cointegrated). When mean_revert_strength = 0 both series are
    independent random walks.

    For 'structural_break' pairs the true ratio shifts at T/2.
    """
    T = params.get("T", 520)
    seed = params.get("seed", 0)
    rho_c = params.get("rho_common", 0.8)         # weight of common factor
    mrs = params.get("mean_revert_strength", 0.1)  # error-correction speed
    sig_c = params.get("sigma_common", 0.01)
    sig_i = params.get("sigma_idio", 0.005)
    x0 = params.get("x0", 30.0)
    y0 = params.get("y0", 45.0)
    drift = params.get("drift", 0.0)
    do_break = params.get("structural_break", False)
    break_mag = params.get("break_magnitude", 0.3)

    rng = np.random.default_rng(seed)

    x = np.zeros(T + 1)
    y = np.zeros(T + 1)
    x[0], y[0] = x0, y0
    true_ratio = y0 / x0   # initial long-run ratio

    for t in range(1, T + 1):
        # Mid-sample structural break: ratio changes at t = T//2
        if do_break and t == T // 2 + 1:
            true_ratio *= (1 + break_mag)

        w  = rng.standard_normal()   # common factor shock
        u1 = rng.standard_normal()   # idiosyncratic shock X
        u2 = rng.standard_normal()   # idiosyncratic shock Y

        eps_x = rho_c * sig_c * w + (1 - rho_c) * sig_i * u1
        eps_y = rho_c * sig_c * w + (1 - rho_c) * sig_i * u2

        # Error-correction term: pulls Y back to r * X
        ec = mrs * (true_ratio * x[t - 1] - y[t - 1])

        x[t] = x[t - 1] * np.exp(drift + eps_x)
        y[t] = y[t - 1] * np.exp(drift + eps_y) + ec

        # Keep prices positive
        x[t] = max(x[t], 0.01)
        y[t] = max(y[t], 0.01)

    return x[1:], y[1:]   # return T observations


# ── Dividend adjustment (section 2.2) ─────────────────────────────────────────

def adjust_for_dividends(prices: pd.Series,
                          dividends: pd.Series) -> pd.Series:
    """
    Apply the book's dividend reinvestment adjustment (Table 2.3).

    For each ex-dividend date the factor is updated:
        factor_new = factor_old + dividend / price_on_ex_date

    All prices from that date forward are multiplied by factor_new.

    Parameters
    ----------
    prices    : pd.Series indexed by date, raw Bloomberg prices
    dividends : pd.Series indexed by ex-dividend dates with dividend amounts

    Returns
    -------
    pd.Series of adjusted prices
    """
    adj = prices.copy().astype(float)
    factor = 1.0

    for date, div_amount in dividends.sort_index().items():
        if date in adj.index:
            price_on_date = adj[date]
            if price_on_date > 0:
                factor = factor + div_amount / price_on_date
            # Apply factor to all prices from this date forward
            adj[adj.index >= date] = prices[prices.index >= date] * factor

    return adj


# ── CSV loader ─────────────────────────────────────────────────────────────────

def _load_from_csv(filepath: str) -> pd.Series:
    """
    Load price series from CSV.
    Expected columns: Date (parseable), Price (or Adj Close / Close).
    Optional column: Dividend – if present, dividend adjustment is applied.
    """
    df = pd.read_csv(filepath, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Find price column
    price_col = None
    for col in ["Adj Close", "adj_close", "AdjClose", "Price", "price", "Close", "close"]:
        if col in df.columns:
            price_col = col
            break
    if price_col is None:
        raise ValueError(f"Cannot find price column in {filepath}. "
                         f"Columns: {list(df.columns)}")

    prices = df[price_col].dropna()

    if "Dividend" in df.columns or "dividend" in df.columns:
        div_col = "Dividend" if "Dividend" in df.columns else "dividend"
        divs = df[div_col].dropna()
        divs = divs[divs > 0]
        if not divs.empty:
            prices = adjust_for_dividends(prices, divs)

    return prices


# ── Yahoo Finance loader ───────────────────────────────────────────────────────

_CACHE_DIR = "data_cache"


def _cache_path(ticker: str, start: str, end: str):
    """Return parquet cache path for this ticker/date range."""
    import os
    os.makedirs(_CACHE_DIR, exist_ok=True)
    key = f"{ticker}_{start}_{end}".replace("/", "-").replace(":", "")
    return os.path.join(_CACHE_DIR, f"{key}.parquet")


def _load_from_yfinance(ticker: str, start: str, end: str) -> pd.Series:
    """
    Download adjusted closing prices from Yahoo Finance.
    Results are cached locally as parquet to avoid repeat downloads.
    Requires: pip install yfinance pyarrow
    """
    cp = _cache_path(ticker, start, end)

    # Return from cache if available
    if os.path.exists(cp):
        cached = pd.read_parquet(cp)
        series = cached.iloc[:, 0]
        series.index = pd.to_datetime(series.index)
        return series.dropna()

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is not installed. Install it with: pip install yfinance\n"
            "Or set source='csv' or source='synthetic' in config.py."
        )
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    series = df["Close"].dropna()
    # Cache to parquet
    try:
        pd.DataFrame({ticker: series}).to_parquet(cp)
    except Exception:
        pass  # cache write failure is non-fatal
    return series


# ── Main public interface ──────────────────────────────────────────────────────

def load_pair(pair_cfg: Dict[str, Any],
              start_date: Optional[str] = None,
              end_date: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Load a pair of price series according to config.

    Returns
    -------
    (x, y, name_x, name_y)
        x, y   : 1-D numpy arrays of closing prices (length T ≈ 520)
        name_x : label for stock X
        name_y : label for stock Y
    """
    source = pair_cfg.get("source", "synthetic")
    name_x = pair_cfg.get("X", "X")
    name_y = pair_cfg.get("Y", "Y")

    if source == "synthetic":
        params = pair_cfg.get("synthetic_params", {})
        x, y = _generate_synthetic_pair(params)

    elif source == "csv":
        x_path = pair_cfg.get("X")
        y_path = pair_cfg.get("Y")
        if not x_path or not y_path:
            raise ValueError("For source='csv', 'X' and 'Y' must be file paths.")
        sx = _load_from_csv(x_path)
        sy = _load_from_csv(y_path)
        # Align on common dates
        common = sx.index.intersection(sy.index)
        if len(common) < 50:
            raise ValueError("Too few common dates between the two CSV files.")
        x = sx[common].values
        y = sy[common].values

    elif source == "yfinance":
        if not start_date or not end_date:
            raise ValueError("Provide start_date and end_date for yfinance source.")
        sx = _load_from_yfinance(pair_cfg["X"], start_date, end_date)
        sy = _load_from_yfinance(pair_cfg["Y"], start_date, end_date)
        common = sx.index.intersection(sy.index)
        x = sx[common].values
        y = sy[common].values

    else:
        raise ValueError(f"Unknown source: '{source}'. "
                         f"Choose 'synthetic', 'csv', or 'yfinance'.")

    # Validate
    if len(x) < 100:
        raise ValueError(f"Not enough observations ({len(x)}). Need at least 100.")

    return x, y, name_x, name_y
