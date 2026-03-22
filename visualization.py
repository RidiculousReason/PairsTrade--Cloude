"""
visualization.py
================
All charts for the pairs trading analysis, reproducing figures from
Gorter (2006) and extending them for practical use.

Figures produced per pair:
  Fig 1: Price series (both stocks over full period)
  Fig 2: Spread in formation period with Γ threshold
  Fig 3: Ratio rt, moving-average r̃t and r̄ in backtest period
  Fig 4: Spread in backtest period with trade signals
  Fig 5: Γ optimisation table (profit vs Γ)
  Fig 6: Cumulative P&L curve
  Fig 7: EG residuals (spread proxy) with ADF diagnostic
  Fig 8: Summary dashboard (combined figure)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from typing import List, Optional

from trading_strategy import StrategyResult, Trade
from cointegration import CointegrationSummary


# ── Style defaults ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       120,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
})

COLOUR_X    = "#2196F3"   # blue
COLOUR_Y    = "#F44336"   # red
COLOUR_SPRD = "#333333"   # dark
COLOUR_RTIL = "#9C27B0"   # purple
COLOUR_RBAR = "#795548"   # brown
COLOUR_RACT = "#FF9800"   # orange
COLOUR_LONG = "#4CAF50"   # green – buy Y
COLOUR_SHRT = "#F44336"   # red   – sell Y
COLOUR_FLAT = "#9E9E9E"   # grey  – flat
COLOUR_GAMMA = "#E91E63"  # pink dashed lines


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Prices
# ─────────────────────────────────────────────────────────────────────────────

def plot_prices(res: StrategyResult, ax: Optional[plt.Axes] = None,
                standalone: bool = True) -> Optional[plt.Figure]:
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3.5))

    T = len(res.x)
    t = np.arange(T)
    ax2 = ax.twinx()

    ax.plot(t, res.x, color=COLOUR_X, lw=1.2, label=res.name_x)
    ax2.plot(t, res.y, color=COLOUR_Y, lw=1.2, label=res.name_y, alpha=0.85)

    ax.axvline(res.T_half, color="k", lw=0.8, ls=":", alpha=0.7, label="Formation | Backtest")
    ax.set_xlabel("t (trading days)")
    ax.set_ylabel(f"{res.name_x} price", color=COLOUR_X)
    ax2.set_ylabel(f"{res.name_y} price", color=COLOUR_Y)
    ax.set_title(f"{res.pair_name} – Price Series")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax.grid(True)

    if standalone and fig:
        fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Formation spread
# ─────────────────────────────────────────────────────────────────────────────

def plot_formation_spread(res: StrategyResult, ax: Optional[plt.Axes] = None,
                          standalone: bool = True) -> Optional[plt.Figure]:
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))

    T_half = res.T_half
    t = np.arange(T_half)
    s = res.spread_formation

    ax.plot(t, s, color=COLOUR_SPRD, lw=1, label=r"$s_t = y_t - \hat{\alpha}x_t$")
    ax.axhline(0, color="k", lw=0.8, ls="-")
    ax.axhline(res.gamma,  color=COLOUR_GAMMA, lw=1.2, ls="--",
               label=f"Γ = {res.gamma:.3f}")
    ax.axhline(-res.gamma, color=COLOUR_GAMMA, lw=1.2, ls="--")

    ax.fill_between(t, s, res.gamma,  where=(s > res.gamma),
                    alpha=0.15, color=COLOUR_SHRT)
    ax.fill_between(t, s, -res.gamma, where=(s < -res.gamma),
                    alpha=0.15, color=COLOUR_LONG)

    ax.set_xlabel("t (formation period)")
    ax.set_ylabel("Spread")
    ax.set_title(f"{res.pair_name} – Formation Spread"
                 f"  [α̂={res.alpha_hat:.4f}, Γ={res.gamma:.4f}, κ={res.kappa*100:.0f}%]")
    ax.legend()
    ax.grid(True)

    if standalone and fig:
        fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Ratio evolution in backtest
# ─────────────────────────────────────────────────────────────────────────────

def plot_ratio(res: StrategyResult, ax: Optional[plt.Axes] = None,
               standalone: bool = True) -> Optional[plt.Figure]:
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))

    T_back = len(res.r_tilde)
    t_back = np.arange(T_back)

    ax.plot(t_back, res.r_actual, color=COLOUR_RACT, lw=0.7, alpha=0.6,
            label=r"$r_t = y_t/x_t$ (actual)")
    kappa_pct = res.kappa * 100
    ax.plot(t_back, res.r_tilde,  color=COLOUR_RTIL, lw=1.5,
            label=r"$\tilde{r}_t$" + f" (EMA ratio, kappa={kappa_pct:.0f}%)")
    ax.axhline(res.alpha_hat, color=COLOUR_RBAR, lw=1.2, ls=":",
               label=r"$\bar{r}$" + f" = {res.alpha_hat:.4f}")

    ax.set_xlabel("t (backtest period)")
    ax.set_ylabel("Ratio")
    ax.set_title(f"{res.pair_name} – Ratio Evolution in Backtest")
    ax.legend()
    ax.grid(True)

    if standalone and fig:
        fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Backtest spread with trade signals
# ─────────────────────────────────────────────────────────────────────────────

def plot_backtest_spread(res: StrategyResult, ax: Optional[plt.Axes] = None,
                         standalone: bool = True) -> Optional[plt.Figure]:
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3.5))

    T_back = len(res.spread_backtest)
    t_back = np.arange(T_back)
    s = res.spread_backtest

    ax.plot(t_back, s, color=COLOUR_SPRD, lw=0.9, label=r"$\tilde{s}_t$")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(res.gamma,  color=COLOUR_GAMMA, lw=1.2, ls="--",
               label=f"±Γ = {res.gamma:.3f}")
    ax.axhline(-res.gamma, color=COLOUR_GAMMA, lw=1.2, ls="--")

    # Fill between thresholds
    ax.fill_between(t_back, s, res.gamma, where=(s > res.gamma),
                    alpha=0.12, color=COLOUR_SHRT)
    ax.fill_between(t_back, s, -res.gamma, where=(s < -res.gamma),
                    alpha=0.12, color=COLOUR_LONG)

    # Mark trade entries & exits
    for tr in res.trades:
        colour = COLOUR_LONG if tr.position == 1 else COLOUR_SHRT
        ax.scatter(tr.t_entry, tr.s_entry, marker="^" if tr.position == 1 else "v",
                   color=colour, s=60, zorder=5)
        if tr.t_exit is not None and tr.s_exit is not None:
            ax.scatter(tr.t_exit, tr.s_exit, marker="s",
                       color=colour, s=40, zorder=5, alpha=0.7)
            ax.plot([tr.t_entry, tr.t_exit], [tr.s_entry, tr.s_exit],
                    color=colour, lw=0.6, alpha=0.4)

    ax.set_xlabel("t (backtest period)")
    ax.set_ylabel(r"Spread $\tilde{s}_t$")
    ax.set_title(f"{res.pair_name} – Backtest Spread & Trades  "
                 f"[{res.num_trades} trades, profit={res.total_profit:.2f}]")
    ax.legend()
    ax.grid(True)

    if standalone and fig:
        fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Gamma optimisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_gamma_optimisation(res: StrategyResult, ax: Optional[plt.Axes] = None,
                            standalone: bool = True) -> Optional[plt.Figure]:
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.5))

    if not res.gamma_search:
        ax.text(0.5, 0.5, "No gamma search data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    gammas  = [row["gamma"]  for row in res.gamma_search]
    profits = [row["profit"] for row in res.gamma_search]
    trades  = [row["trades"] for row in res.gamma_search]

    ax.bar(gammas, profits, width=np.diff(gammas + [gammas[-1]*1.1])*0.7,
           align="edge", alpha=0.6, color=COLOUR_X, label="Min Profit")
    ax.axvline(res.gamma, color=COLOUR_GAMMA, lw=1.5, ls="--",
               label=f"Chosen Γ = {res.gamma:.3f}")

    ax2 = ax.twinx()
    ax2.plot(gammas, trades, "o-", color=COLOUR_Y, lw=1.2, ms=4,
             label="# Trades")
    ax2.set_ylabel("# Trades", color=COLOUR_Y)

    ax.set_xlabel("Γ  (threshold)")
    ax.set_ylabel("Minimum profit (gross)")
    ax.set_title(f"{res.pair_name} – Γ Optimisation (Formation Period)")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax.grid(True, axis="y")

    if standalone and fig:
        fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Cumulative P&L
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative_pnl(res: StrategyResult, ax: Optional[plt.Axes] = None,
                        standalone: bool = True) -> Optional[plt.Figure]:
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))

    closed = [t for t in res.trades if t.profit is not None]
    if not closed:
        ax.text(0.5, 0.5, "No closed trades", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    trade_nums = [t.trade_num for t in closed]
    cum_pnl    = np.cumsum([t.profit for t in closed])

    ax.plot(trade_nums, cum_pnl, "o-", color=COLOUR_SPRD, lw=1.5, ms=4)
    ax.axhline(0, color="k", lw=0.6)
    ax.fill_between(trade_nums, 0, cum_pnl,
                    where=(np.array(cum_pnl) >= 0), alpha=0.2, color=COLOUR_LONG)
    ax.fill_between(trade_nums, 0, cum_pnl,
                    where=(np.array(cum_pnl) < 0),  alpha=0.2, color=COLOUR_SHRT)

    ax.set_xlabel("Trade number")
    ax.set_ylabel("Cumulative P&L")
    ax.set_title(f"{res.pair_name} – Cumulative P&L  (Total: {res.total_profit:.4f})")
    ax.grid(True)

    if standalone and fig:
        fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: EG residuals
# ─────────────────────────────────────────────────────────────────────────────

def plot_eg_residuals(cs: CointegrationSummary, ax: Optional[plt.Axes] = None,
                      standalone: bool = True) -> Optional[plt.Figure]:
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))

    eg   = cs.engle_granger
    resid = eg.residuals
    t    = np.arange(len(resid))

    ax.plot(t, resid, color=COLOUR_SPRD, lw=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.fill_between(t, resid, 0, alpha=0.15, color=COLOUR_X)

    verdict = "Cointegrated ✓" if cs.eg_cointegrated_5pct else "Not cointegrated ✗"
    stat    = eg.adf_result.test_stat
    ax.set_title(f"{cs.pair_name} – EG Residuals  "
                 f"[ADF={stat:.3f}, {verdict}]")
    ax.set_xlabel("t")
    ax.set_ylabel("Residual")
    ax.grid(True)

    if standalone and fig:
        fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Full dashboard per pair
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(res: StrategyResult,
                   cs: CointegrationSummary,
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Multi-panel dashboard combining all key plots for one pair.
    Layout:
      Row 0: Prices        | Gamma optimisation
      Row 1: Formation spread | Ratio evolution
      Row 2: Backtest spread  | Cumulative P&L
      Row 3: EG residuals   | (text summary)
    """
    fig = plt.figure(figsize=(16, 20))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_prices    = fig.add_subplot(gs[0, 0])
    ax_gamma     = fig.add_subplot(gs[0, 1])
    ax_form      = fig.add_subplot(gs[1, 0])
    ax_ratio     = fig.add_subplot(gs[1, 1])
    ax_backtest  = fig.add_subplot(gs[2, 0])
    ax_pnl       = fig.add_subplot(gs[2, 1])
    ax_eg        = fig.add_subplot(gs[3, 0])
    ax_text      = fig.add_subplot(gs[3, 1])

    plot_prices(res, ax=ax_prices, standalone=False)
    plot_gamma_optimisation(res, ax=ax_gamma, standalone=False)
    plot_formation_spread(res, ax=ax_form, standalone=False)
    plot_ratio(res, ax=ax_ratio, standalone=False)
    plot_backtest_spread(res, ax=ax_backtest, standalone=False)
    plot_cumulative_pnl(res, ax=ax_pnl, standalone=False)
    plot_eg_residuals(cs, ax=ax_eg, standalone=False)

    # ── Text summary panel ─────────────────────────────────────────────────────
    ax_text.axis("off")
    eg  = cs.engle_granger
    joh = cs.johansen

    def yn(b): return "Yes ✓" if b else "No  ✗"

    lines = [
        f"PAIR SUMMARY: {res.pair_name}",
        "",
        "── STRATEGY ──────────────────────────",
        f"  α̂ (hedge ratio)   : {res.alpha_hat:.4f}",
        f"  Γ (threshold)     : {res.gamma:.4f}",
        f"  κ (smoothing)     : {res.kappa*100:.0f}%",
        f"  Total trades      : {res.num_trades}",
        f"  Forced closes     : {res.forced_close_count}",
        f"  Total net profit  : {res.total_profit:.4f}",
        f"  Total costs       : {res.total_costs:.4f}",
        f"  Avg profit/trade  : {res.avg_profit:.4f}  (clean)",
        f"  Sharpe (daily)    : {res.sharpe_daily:.4f}",
        f"  Max drawdown      : {res.max_drawdown:.4f}",
        f"  Stop: {res.stop_mult_gamma}×Γ  MaxHold: {res.max_holding_days}d",
        "",
        "── UNIT ROOT TESTS (I(1)) ────────────",
        f"  {res.name_x} level non-stat : {yn(not cs.adf_x.is_stationary)}",
        f"  {res.name_y} level non-stat : {yn(not cs.adf_y.is_stationary)}",
        f"  Δ{res.name_x} stationary    : {yn(cs.adf_dx.is_stationary)}",
        f"  Δ{res.name_y} stationary    : {yn(cs.adf_dy.is_stationary)}",
        f"  Both I(1)         : {yn(cs.both_i1)}",
        "",
        "── ENGLE-GRANGER (Y~X / X~Y) ─────────",
        f"  α̂ Y~X             : {eg.alpha:.4f}",
        f"  ADF(resid) Y~X    : {eg.adf_result.test_stat:.4f}",
        f"  Coint Y~X 5%      : {yn(eg.cointegrated_5pct)}",
        f"  Coint X~Y 5%      : {yn(cs.engle_granger_yx.cointegrated_5pct)}",
        f"  Combined 5%       : {yn(cs.eg_cointegrated_5pct)}",
        "",
        "── JOHANSEN (three-test) ─────────────",
        f"  λ₁={joh.eigenvalues[0]:.4f}  λ₂={joh.eigenvalues[1]:.4f}",
        f"  Test1 Trace g=2   : {yn(joh.test1_reject)}",
        f"  Test2 MaxEig g=2  : {yn(joh.test2_reject)}",
        f"  Test3 rank≤1 OK   : {yn(not joh.test3_reject)}",
        f"  Cointegrated 5%   : {yn(joh.cointegrated_5pct)}",
    ]

    ax_text.text(0.03, 0.97, "\n".join(lines),
                 transform=ax_text.transAxes,
                 fontsize=8, verticalalignment="top",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))

    fig.suptitle(f"Pairs Trading Analysis: {res.pair_name}", fontsize=13, y=0.995)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"    Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Cross-pair comparison chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_pair_comparison(results: list,
                               coint_summaries: list,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Side-by-side comparison of all pairs:
      - Profitability ranking
      - Cointegration test results
    """
    n = len(results)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, 1.5 * n)))
    names   = [r.pair_name.split("(")[0].strip() for r in results]
    profits = [r.total_profit for r in results]
    n_trades= [r.num_trades   for r in results]

    # ── Panel 1: Profit ranking ────────────────────────────────────────────────
    ax = axes[0]
    colours = [COLOUR_LONG if p >= 0 else COLOUR_SHRT for p in profits]
    bars = ax.barh(names, profits, color=colours, alpha=0.75)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Total Profit")
    ax.set_title("Profitability Ranking")
    ax.grid(True, axis="x")
    for bar, val in zip(bars, profits):
        ax.text(val + 0.002 * max(abs(p) for p in profits),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=7)

    # ── Panel 2: Number of trades ──────────────────────────────────────────────
    ax = axes[1]
    ax.barh(names, n_trades, color=COLOUR_X, alpha=0.7)
    ax.set_xlabel("Number of Trades")
    ax.set_title("Trading Frequency")
    ax.grid(True, axis="x")

    # ── Panel 3: Cointegration heatmap ────────────────────────────────────────
    ax = axes[2]
    tests = ["EG Y~X 5%", "EG X~Y 5%", "EG comb 5%", "Joh 5%"]
    data  = []
    for cs in coint_summaries:
        eg  = cs.engle_granger
        joh = cs.johansen
        data.append([
            int(eg.cointegrated_5pct),
            int(cs.engle_granger_yx.cointegrated_5pct),
            int(cs.eg_cointegrated_5pct),
            int(joh.cointegrated_5pct),
        ])
    data_arr = np.array(data)

    im = ax.imshow(data_arr, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels(tests, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names)
    ax.set_title("Cointegration Tests")

    for i in range(n):
        for j in range(len(tests)):
            txt = "✓" if data_arr[i, j] else "✗"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="white" if data_arr[i, j] else "black")

    fig.suptitle("Cross-Pair Comparison", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"    Saved: {save_path}")

    return fig
