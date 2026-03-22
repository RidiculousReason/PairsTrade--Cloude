"""
trading_strategy.py
===================
Implements the pairs trading strategy from Gorter (2006).

Statistical foundation (Chapters 3–7)
--------------------------------------
Hedge ratio:  α̂ = Σ(xt·yt) / Σ(xt²)   [OLS without intercept, Section 6.1]
Formation spread:   et  = yt − α̂·xt
Backtest EMA ratio: r̃_t = (1−κ)·r̃_{t-1} + κ·(yt/xt),  r̃_0 = α̂
Backtest spread:    s̃_t = yt − r̃_t·xt
Strategy II: enter at ±Γ, reverse at opposite Γ (Section 2.3)
P&L per trade: position × [(ΔY) − r̃_entry·(ΔX)]   [r̃ locked at ENTRY]

Risk controls (from audit, not in Gorter):
  max_holding_days  — force-close if position open too long
  stop_mult_gamma   — force-close if |spread| > stop_mult × Γ

Transaction costs (from audit):
  costs_bps         — commission in basis points on each leg notional
  bid_ask_spread_pct — half-spread added to buy / subtracted from sell
  slippage_bps      — market-impact slippage in basis points
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import config


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """One completed (or force-closed) trade."""
    trade_num:     int
    t_entry:       int
    t_exit:        int
    s_entry:       float
    s_exit:        float
    price_y_entry: float
    price_x_entry: float
    price_y_exit:  float
    price_x_exit:  float
    ratio_entry:   float      # r̃ locked at entry — used for P&L
    ratio_exit:    float
    position:      int        # +1 = long Y / short X,  −1 = short Y / long X
    holding_days:  int        # calendar days (index steps here)
    gross_pnl:     float      # before costs
    costs:         float      # total transaction costs
    profit:        float      # net P&L = gross_pnl − costs
    forced_close:  bool       # True if stop-loss or time-stop triggered


@dataclass
class StrategyResult:
    """Complete result for one pair."""
    pair_name: str
    name_x:    str
    name_y:    str

    # Full price series
    x: np.ndarray
    y: np.ndarray

    # Formation parameters
    T_half:    int
    alpha_hat: float      # EG OLS hedge ratio = trading hedge ratio
    gamma:     float      # threshold Γ
    kappa:     float      # smoothing κ (decimal, e.g. 0.02)
    m:         float      # max|spread| in formation period

    # Risk / cost params actually used
    max_holding_days:   int
    stop_mult_gamma:    float
    costs_bps:          float
    bid_ask_spread_pct: float
    slippage_bps:       float

    # Spread series
    spread_formation: np.ndarray   # et = yt − α̂·xt
    spread_backtest:  np.ndarray   # s̃t = yt − r̃t·xt
    r_tilde:          np.ndarray   # EMA ratio in backtest
    r_actual:         np.ndarray   # actual ratio yt/xt in backtest

    # Γ search
    gamma_search: List[Dict]

    # Trades
    trades: List[Trade]

    # Performance
    total_profit:     float
    total_costs:      float
    num_trades:       int
    avg_profit:       float
    sharpe_estimate:  float   # trade-level: mean/std of clean reversals
    sharpe_daily:     float   # annualised daily Sharpe (√252 × μ/σ)
    max_drawdown:     float
    forced_close_count: int


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kappa_from_trades(n_trades: int) -> float:
    """Map number of formation-period trades to κ via Gorter Table 2.5."""
    for hi, lo, kappa_pct in config.KAPPA_TABLE:
        if hi == float('inf'):
            if n_trades > 15:
                return kappa_pct / 100.0
        else:
            if lo <= n_trades <= hi:
                return kappa_pct / 100.0
    return 0.10


def _exec_price(px: float, is_buy: bool,
                bid_ask_spread_pct: float, slippage_bps: float) -> float:
    """
    Adjust raw price for bid/ask spread and slippage.
    Buy:  pay Ask = px × (1 + half_spread) + slippage
    Sell: receive Bid = px × (1 − half_spread) − slippage
    """
    slip = px * slippage_bps / 10_000.0
    half = px * bid_ask_spread_pct / 2.0
    return (px + half + slip) if is_buy else (px - half - slip)


def _transaction_costs(ey: float, ex: float, r: float,
                        y: float, x: float,
                        costs_bps: float) -> float:
    """
    Commission cost = costs_bps / 10000 × sum of all leg notionals (entry + exit).
    Notionals: |Y| and |r·X| at entry and exit.
    """
    return (costs_bps / 10_000.0) * (abs(ey) + abs(r * ex) + abs(y) + abs(r * x))


def _simulate_strategy_ii(spread: np.ndarray, gamma: float) -> Tuple[int, float]:
    """
    Fast Strategy II simulation for the Γ search (formation phase).
    No costs, no stops — pure min-profit criterion from Gorter Section 2.4.

    Returns (n_trades, min_profit) where min_profit = (n_trades−1) × 2Γ.
    """
    position = 0
    n_trades = 0
    for s in spread:
        if position == 0:
            if s >= gamma:
                position = -1
                n_trades += 1
            elif s <= -gamma:
                position = +1
                n_trades += 1
        elif position == +1:
            if s >= gamma:
                position = -1
                n_trades += 1
        else:
            if s <= -gamma:
                position = +1
                n_trades += 1
    if n_trades < 2:
        return n_trades, 0.0
    return n_trades, (n_trades - 1) * 2.0 * gamma


# ─────────────────────────────────────────────────────────────────────────────
# Formation phase
# ─────────────────────────────────────────────────────────────────────────────

def formation_phase(
    x_half: np.ndarray,
    y_half: np.ndarray,
    alpha_hat_override: Optional[float] = None,
) -> Tuple[float, float, float, List[Dict], float]:
    """
    Estimate α̂, Γ, κ from the formation (first-half) data.

    α̂ = Σ(xy)/Σ(x²)  [OLS without intercept, Gorter Section 6.1]
    Γ  chosen by max min-profit across grid {5%,…,50%} of max|spread|
    κ  from Table 2.5 via number of formation-period trades at chosen Γ

    If alpha_hat_override is provided (from EG winning direction in main.py),
    it is used directly instead of re-estimating via OLS. This ensures the
    trading hedge ratio is the same coefficient that passed the cointegration test.

    Returns (alpha_hat, gamma, kappa, search_results, m)
    """
    if alpha_hat_override is not None:
        alpha_hat = alpha_hat_override
    else:
        alpha_hat = float(np.dot(x_half, y_half) / np.dot(x_half, x_half))
    spread    = y_half - alpha_hat * x_half
    m         = float(np.max(np.abs(spread)))

    search_results: List[Dict] = []
    best_profit = -np.inf
    best_gamma  = m * config.GAMMA_MAX_FRACTION

    for pct in config.GAMMA_PERCENTAGES:
        gamma_try = (pct / 100.0) * m
        if gamma_try > config.GAMMA_MAX_FRACTION * m:
            break
        n_tr, profit = _simulate_strategy_ii(spread, gamma_try)
        search_results.append({
            "percentage": pct,
            "gamma":      round(gamma_try, 6),
            "trades":     n_tr,
            "profit":     round(profit, 6),
        })
        if profit > best_profit:
            best_profit = profit
            best_gamma  = gamma_try

    n_form_trades, _ = _simulate_strategy_ii(spread, best_gamma)
    kappa = _kappa_from_trades(n_form_trades)

    return alpha_hat, best_gamma, kappa, search_results, m


# ─────────────────────────────────────────────────────────────────────────────
# Backtest phase
# ─────────────────────────────────────────────────────────────────────────────

def backtest_phase(
    x_back:             np.ndarray,
    y_back:             np.ndarray,
    alpha_hat:          float,
    gamma:              float,
    kappa:              float,
    max_holding_days:   int   = 60,
    stop_mult_gamma:    float = 3.0,
    costs_bps:          float = 0.0,
    bid_ask_spread_pct: float = 0.0,
    slippage_bps:       float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Trade]]:
    """
    Execute Strategy II on the backtest (second-half) data.

    EMA ratio initialised at α̂, then updated each day:
        r̃_t = (1−κ)·r̃_{t-1} + κ·(yt/xt)

    Position sizing: 1 unit Y vs r̃_entry units X (ratio locked at entry).

    Risk controls:
        - Reversal at opposite Γ (Strategy II core)
        - Force-close if |spread| > stop_mult × Γ  (stop-loss)
        - Force-close if holding_days ≥ max_holding_days (time-stop)
        - Force-close at end of period if still open

    P&L formula (r̃_entry locked):
        gross = position × [(ΔY) − r̃_entry·(ΔX)]
        net   = gross − costs

    Returns (r_tilde, r_actual, spread_adj, trades)
    """
    T = len(x_back)
    r_actual   = y_back / x_back
    r_tilde    = np.zeros(T)
    spread_adj = np.zeros(T)

    r_tilde[0]    = alpha_hat
    spread_adj[0] = y_back[0] - r_tilde[0] * x_back[0]
    for t in range(1, T):
        r_tilde[t]    = (1.0 - kappa) * r_tilde[t - 1] + kappa * r_actual[t]
        spread_adj[t] = y_back[t] - r_tilde[t] * x_back[t]

    trades:        List[Trade] = []
    position      = 0
    cooling_off   = False   # True after forced close until spread re-enters band
    t_entry       = 0
    s_entry       = 0.0
    y_entry       = 0.0
    x_entry       = 0.0
    ratio_entry   = 0.0   # r̃ locked at entry
    trade_counter = 0

    def _close_trade(t_exit: int, forced: bool) -> Trade:
        """Build and return a closed Trade record."""
        nonlocal trade_counter
        ye = y_back[t_exit]
        xe = x_back[t_exit]
        re = r_tilde[t_exit]

        # Gross P&L: position × [(ΔY) − r̃_entry·(ΔX)]
        dY    = ye - y_entry
        dX    = xe - x_entry
        gross = position * (dY - ratio_entry * dX)

        # Execution prices (bid/ask + slippage)
        if position == +1:
            # Entered: bought Y, sold ratio_entry·X
            # Exiting: sell Y, buy ratio_entry·X
            entry_y_px = _exec_price(y_entry, True,  bid_ask_spread_pct, slippage_bps)
            entry_x_px = _exec_price(ratio_entry * x_entry, False, bid_ask_spread_pct, slippage_bps)
            exit_y_px  = _exec_price(ye,      False, bid_ask_spread_pct, slippage_bps)
            exit_x_px  = _exec_price(ratio_entry * xe, True,  bid_ask_spread_pct, slippage_bps)
        else:
            # Entered: sold Y, bought ratio_entry·X
            # Exiting: buy Y, sell ratio_entry·X
            entry_y_px = _exec_price(y_entry, False, bid_ask_spread_pct, slippage_bps)
            entry_x_px = _exec_price(ratio_entry * x_entry, True,  bid_ask_spread_pct, slippage_bps)
            exit_y_px  = _exec_price(ye,      True,  bid_ask_spread_pct, slippage_bps)
            exit_x_px  = _exec_price(ratio_entry * xe, False, bid_ask_spread_pct, slippage_bps)

        # Re-compute gross with execution prices
        if position == +1:
            gross = (exit_y_px - entry_y_px) - (exit_x_px - entry_x_px)
        else:
            gross = (entry_y_px - exit_y_px) - (entry_x_px - exit_x_px)

        cost  = _transaction_costs(y_entry, x_entry, ratio_entry,
                                   ye, xe, costs_bps)
        trade_counter += 1
        return Trade(
            trade_num=trade_counter,
            t_entry=t_entry,
            t_exit=t_exit,
            s_entry=s_entry,
            s_exit=spread_adj[t_exit],
            price_y_entry=y_entry,
            price_x_entry=x_entry,
            price_y_exit=ye,
            price_x_exit=xe,
            ratio_entry=ratio_entry,
            ratio_exit=re,
            position=position,
            holding_days=t_exit - t_entry,
            gross_pnl=round(gross + cost, 8),  # before subtract costs
            costs=round(cost, 8),
            profit=round(gross, 8),             # net = gross already with exec prices
            forced_close=forced,
        )

    for t in range(T):
        s = spread_adj[t]

        if position == 0:
            # After a forced close (stop-loss or time-stop), require the
            # spread to return inside [-Γ, +Γ] before allowing re-entry.
            # This prevents immediately re-entering into a broken spread.
            if cooling_off:
                if abs(s) < gamma:
                    cooling_off = False   # spread back inside band — ready
                continue                  # no entry while cooling off

            if s >= gamma:
                position    = -1
                t_entry     = t
                s_entry     = s
                y_entry     = y_back[t]
                x_entry     = x_back[t]
                ratio_entry = r_tilde[t]
            elif s <= -gamma:
                position    = +1
                t_entry     = t
                s_entry     = s
                y_entry     = y_back[t]
                x_entry     = x_back[t]
                ratio_entry = r_tilde[t]
            continue

        # ── Check exit conditions ─────────────────────────────────────────────
        holding       = t - t_entry
        reversal_hit  = (position == +1 and s >= gamma) or \
                        (position == -1 and s <= -gamma)
        stop_hit      = abs(s) > stop_mult_gamma * gamma
        time_hit      = holding >= max_holding_days

        if reversal_hit or stop_hit or time_hit:
            forced = stop_hit or time_hit
            trades.append(_close_trade(t, forced))

            if reversal_hit and not forced:
                # Immediately reverse — clean signal, no cooldown needed
                cooling_off = False
                position    = -position
                t_entry     = t
                s_entry     = s
                y_entry     = y_back[t]
                x_entry     = x_back[t]
                ratio_entry = r_tilde[t]
            else:
                # Stop-loss or time-stop: go flat and wait for spread to normalise
                cooling_off = True
                position    = 0

    # ── Force-close open position at end of period ────────────────────────────
    if position != 0:
        trades.append(_close_trade(T - 1, forced=True))

    return r_tilde, r_actual, spread_adj, trades


# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance(
    trades: List[Trade],
    n_backtest_days: int = 260,
) -> Tuple[float, float, int, float, float, float, float, int]:
    """
    Compute performance metrics.

    total_profit / total_costs: ALL trades (including forced closes).
    avg_profit / sharpe (trade-level): clean reversals only.
    sharpe_daily: annualised using daily P&L series.
    max_drawdown: over all trades.

    Returns
    -------
    (total_profit, total_costs, num_trades, avg_profit,
     sharpe_estimate, sharpe_daily, max_drawdown, forced_close_count)
    """
    if not trades:
        return 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0

    all_profits = np.array([t.profit for t in trades])
    all_costs   = np.array([t.costs  for t in trades])
    total       = float(np.sum(all_profits))
    costs_total = float(np.sum(all_costs))
    n           = len(trades)

    # Trade-level Sharpe: clean reversals only
    clean = [t for t in trades if not t.forced_close]
    if len(clean) > 1:
        cp     = np.array([t.profit for t in clean])
        avg    = float(np.mean(cp))
        std    = float(np.std(cp, ddof=1))
        sharpe = avg / std if std > 0 else 0.0
    elif len(clean) == 1:
        avg, sharpe = float(clean[0].profit), 0.0
    else:
        avg, sharpe = 0.0, 0.0

    # Daily Sharpe: spread P&L across backtest days
    daily = np.zeros(n_backtest_days)
    for t in trades:
        if 0 <= t.t_exit < n_backtest_days:
            daily[t.t_exit] += t.profit
    if daily.std(ddof=1) > 0:
        sharpe_daily = float(np.sqrt(252) * daily.mean() / daily.std(ddof=1))
    else:
        sharpe_daily = 0.0

    cum         = np.cumsum(all_profits)
    running_max = np.maximum.accumulate(cum)
    max_dd      = float(np.max(running_max - cum))

    forced = int(sum(t.forced_close for t in trades))

    return total, costs_total, n, avg, sharpe, sharpe_daily, max_dd, forced


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_strategy(
    x:                  np.ndarray,
    y:                  np.ndarray,
    pair_name:          str   = "Pair",
    name_x:             str   = "X",
    name_y:             str   = "Y",
    max_holding_days:   int   = 60,
    stop_mult_gamma:    float = 3.0,
    costs_bps:          float = 0.0,
    bid_ask_spread_pct: float = 0.0,
    slippage_bps:       float = 0.0,
    eg_alpha_hat:       Optional[float] = None,
) -> StrategyResult:
    """
    Run the full pairs trading strategy.

    Formation half: estimate α̂, Γ, κ.
    Backtest half:  execute Strategy II with risk controls and costs.

    eg_alpha_hat: if provided, use this coefficient directly as the hedge ratio
    instead of re-estimating via OLS. Should come from the winning EG direction
    in main.py to ensure the test and trading use the same α̂.
    """
    T      = len(x)
    T_half = T // 2

    x_form, y_form = x[:T_half], y[:T_half]
    x_back, y_back = x[T_half:], y[T_half:]

    alpha_hat, gamma, kappa, gamma_search, m = formation_phase(
        x_form, y_form, alpha_hat_override=eg_alpha_hat
    )
    spread_form = y_form - alpha_hat * x_form

    r_tilde, r_actual, spread_back, trades = backtest_phase(
        x_back, y_back, alpha_hat, gamma, kappa,
        max_holding_days=max_holding_days,
        stop_mult_gamma=stop_mult_gamma,
        costs_bps=costs_bps,
        bid_ask_spread_pct=bid_ask_spread_pct,
        slippage_bps=slippage_bps,
    )

    (total_profit, total_costs, num_trades, avg_profit,
     sharpe, sharpe_daily, max_dd, forced) = compute_performance(
        trades, n_backtest_days=len(x_back)
    )

    return StrategyResult(
        pair_name=pair_name,
        name_x=name_x,
        name_y=name_y,
        x=x,
        y=y,
        T_half=T_half,
        alpha_hat=alpha_hat,
        gamma=gamma,
        kappa=kappa,
        m=m,
        max_holding_days=max_holding_days,
        stop_mult_gamma=stop_mult_gamma,
        costs_bps=costs_bps,
        bid_ask_spread_pct=bid_ask_spread_pct,
        slippage_bps=slippage_bps,
        spread_formation=spread_form,
        spread_backtest=spread_back,
        r_tilde=r_tilde,
        r_actual=r_actual,
        gamma_search=gamma_search,
        trades=trades,
        total_profit=total_profit,
        total_costs=total_costs,
        num_trades=num_trades,
        avg_profit=avg_profit,
        sharpe_estimate=sharpe,
        sharpe_daily=sharpe_daily,
        max_drawdown=max_dd,
        forced_close_count=forced,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────────────────────

def print_strategy_summary(res: StrategyResult) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  STRATEGY RESULTS: {res.pair_name}")
    print(f"  X = {res.name_x}   Y = {res.name_y}")
    print(sep)

    print(f"\n  FORMATION PARAMETERS  (first {res.T_half} observations)")
    print(f"    EG OLS hedge ratio  α̂  = {res.alpha_hat:.6f}  [Σ(xy)/Σ(x²)]")
    print(f"    Max |spread|        m  = {res.m:.6f}")
    print(f"    Threshold           Γ  = {res.gamma:.6f}  "
          f"({100*res.gamma/res.m:.1f}% of m)")
    print(f"    Smoothing           κ  = {res.kappa*100:.0f}%")

    print(f"\n  RISK / COST PARAMETERS")
    print(f"    Max holding days    = {res.max_holding_days}")
    print(f"    Stop multiplier     = {res.stop_mult_gamma}×Γ  "
          f"(= {res.stop_mult_gamma * res.gamma:.4f})")
    print(f"    Commission          = {res.costs_bps:.1f} bps")
    print(f"    Bid/ask half-spread = {res.bid_ask_spread_pct*100:.3f}%")
    print(f"    Slippage            = {res.slippage_bps:.1f} bps")

    print(f"\n  GAMMA OPTIMISATION  (Strategy II on formation spread)")
    print(f"  {'Pct':>5}  {'Γ':>10}  {'Trades':>8}  {'Min Profit':>12}")
    print("  " + "─" * 44)
    for row in res.gamma_search:
        sel = " ←" if abs(row["gamma"] - res.gamma) < 1e-10 else ""
        print(f"  {row['percentage']:>5}  {row['gamma']:>10.6f}  "
              f"{row['trades']:>8}  {row['profit']:>12.6f}{sel}")

    print(f"\n  BACKTEST TRADES  (second {len(res.spread_backtest)} observations)")
    if not res.trades:
        print("  (no trades)")
    else:
        print(f"  {'#':>4}  {'Entry':>7}  {'Exit':>7}  {'Hold':>5}  {'Pos':>7}  "
              f"{'s_entry':>9}  {'s_exit':>9}  {'Gross':>8}  {'Costs':>7}  "
              f"{'Net':>9}  Note")
        print("  " + "─" * 90)
        for tr in res.trades:
            pos_str = "+Y/−X" if tr.position == 1 else "−Y/+X"
            note    = "FORCED" if tr.forced_close else ""
            print(f"  {tr.trade_num:>4}  {tr.t_entry:>7}  {tr.t_exit:>7}  "
                  f"{tr.holding_days:>5}  {pos_str:>7}  "
                  f"{tr.s_entry:>9.4f}  {tr.s_exit:>9.4f}  "
                  f"{tr.gross_pnl:>8.4f}  {tr.costs:>7.4f}  "
                  f"{tr.profit:>9.4f}  {note}")

    print(f"\n  PERFORMANCE SUMMARY")
    print(f"    Total net profit     : {res.total_profit:>12.4f}")
    print(f"    Total costs          : {res.total_costs:>12.4f}")
    print(f"    Total trades         : {res.num_trades:>12d}")
    print(f"    Forced closes        : {res.forced_close_count:>12d}")
    print(f"    Avg net / trade      : {res.avg_profit:>12.4f}  (clean reversals)")
    print(f"    Sharpe (trade-level) : {res.sharpe_estimate:>12.4f}")
    print(f"    Sharpe (daily ann.)  : {res.sharpe_daily:>12.4f}")
    print(f"    Max drawdown         : {res.max_drawdown:>12.4f}")
    print()
