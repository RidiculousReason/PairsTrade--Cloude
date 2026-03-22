"""
main.py
=======
Entry point for the Pairs Trading analysis project (Gorter 2006).

Usage
-----
    python main.py                        # reads pairs.txt, downloads via yfinance
    python main.py my_pairs.txt           # custom pairs file
    python main.py --demo                 # synthetic demo, no internet needed

    # With risk controls and transaction costs:
    python main.py --costs_bps 1 --bid_ask_spread_pct 0.0005 --max_holding_days 60

    # Custom date range:
    python main.py --start 2019-01-01 --end 2023-12-31

pairs.txt format (one pair per line):
    XOM     CVX     Exxon vs Chevron   # space or comma separated; name optional
    XOM,CVX                            # comma also works
    # this is a comment
"""

import os
import sys
import argparse
import csv as csv_mod
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np

import config
from data_loader import load_pair
from trading_strategy import run_strategy, print_strategy_summary, StrategyResult
from cointegration import (
    full_cointegration_analysis, print_cointegration_summary, CointegrationSummary
)
from visualization import plot_dashboard, plot_cross_pair_comparison


# ─────────────────────────────────────────────────────────────────────────────
# Ranking table  (Chapter 7 style)
# ─────────────────────────────────────────────────────────────────────────────

def rank_pairs(results: List[StrategyResult],
               coint_summaries: List[CointegrationSummary]) -> None:
    sep = "=" * 95
    print(f"\n{sep}")
    print("  COMBINED RANKING TABLE  (Chapter 7 style)")
    print(sep)
    print(f"\n  {'Pair':<28} {'Γ':>7} {'κ%':>4} {'Trades':>7} {'Forced':>7} "
          f"{'Profit':>9} {'Costs':>7} {'Sh.Daily':>9} {'EG5%':>6} {'Joh5%':>6} {'Verdict':>8}")
    print("  " + "─" * 93)

    order = sorted(range(len(results)),
                   key=lambda i: results[i].total_profit, reverse=True)

    for rank, i in enumerate(order, 1):
        r  = results[i]
        cs = coint_summaries[i]
        eg_ok  = "Yes" if cs.eg_cointegrated_5pct      else "No"
        joh_ok = "Yes" if cs.johansen.cointegrated_5pct else "No"
        verdict = "★ GOOD" if (r.total_profit > 0 and r.num_trades >= 4) else "✗ BAD"

        print(f"  {rank}. {r.pair_name:<26} {r.gamma:>7.3f} "
              f"{r.kappa*100:>4.0f} {r.num_trades:>7} {r.forced_close_count:>7} "
              f"{r.total_profit:>9.4f} {r.total_costs:>7.4f} "
              f"{r.sharpe_daily:>9.3f} "
              f"{eg_ok:>6} {joh_ok:>6} {verdict:>8}")
    print()


def print_overall_conclusion(results: List[StrategyResult],
                             coint_summaries: List[CointegrationSummary]) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  CONCLUSION: COINTEGRATION vs. PROFITABILITY")
    print(sep)

    profitable   = [r for r in results if r.total_profit > 0 and r.num_trades >= 4]
    unprofitable = [r for r in results if r not in profitable]

    print(f"\n  Profitable pairs ({len(profitable)}):")
    for r in profitable:
        i  = [res.pair_name for res in results].index(r.pair_name)
        cs = coint_summaries[i]
        eg = "cointegrated" if cs.eg_cointegrated_5pct else "NOT cointegrated"
        print(f"    {r.pair_name}  →  {eg}  (EG 5%)")

    print(f"\n  Unprofitable pairs ({len(unprofitable)}):")
    for r in unprofitable:
        i  = [res.pair_name for res in results].index(r.pair_name)
        cs = coint_summaries[i]
        eg = "cointegrated" if cs.eg_cointegrated_5pct else "NOT cointegrated"
        print(f"    {r.pair_name}  →  {eg}  (EG 5%)")

    agree = sum(
        1 for r, cs in zip(results, coint_summaries)
        if (r.total_profit > 0 and r.num_trades >= 4) == cs.eg_cointegrated_5pct
    )
    total = len(results)
    print(f"\n  Agreement (profitability ↔ EG cointegration): "
          f"{agree}/{total} ({100*agree/total:.0f}%)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
    pairs_cfg:          list,
    start_date:         str   = None,
    end_date:           str   = None,
    output_dir:         str   = "output",
    verbose:            bool  = True,
    max_holding_days:   int   = 60,
    stop_mult_gamma:    float = 3.0,
    costs_bps:          float = 0.0,
    bid_ask_spread_pct: float = 0.0,
    slippage_bps:       float = 0.0,
) -> Tuple[List, List]:
    """
    Full pipeline: load data → strategy → cointegration → report.

    Parameters
    ----------
    pairs_cfg          : list of pair config dicts (from read_pairs_file or config.PAIRS)
    start_date/end_date: date range for yfinance downloads
    output_dir         : where to save figures, CSV, summary
    verbose            : if True, print per-pair detailed output
    max_holding_days   : force-close a position after this many days
    stop_mult_gamma    : force-close if |spread| > stop_mult × Γ
    costs_bps          : commission in basis points per leg notional
    bid_ask_spread_pct : bid/ask half-spread fraction
    slippage_bps       : market-impact slippage in basis points
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    results_all: List[StrategyResult]       = []
    coint_all:   List[CointegrationSummary] = []
    csv_rows:    list                       = []

    for pair_cfg in pairs_cfg:
        pair_name = pair_cfg.get("name", "Unknown Pair")
        print(f"\n{'─'*65}")
        print(f"  Processing: {pair_name}")
        print(f"{'─'*65}")

        # ── Load data ──────────────────────────────────────────────────────────
        try:
            x, y, name_x, name_y = load_pair(pair_cfg, start_date, end_date)
        except Exception as e:
            print(f"  ERROR loading pair '{pair_name}': {e}")
            continue

        print(f"  Loaded {len(x)} observations  "
              f"({config.TRADING_DAYS_PER_YEAR} obs ≈ 1 year)")

        # ── Trading strategy ────────────────────────────────────────────────────
        print("  Running trading strategy ...")
        strat = run_strategy(
            x, y,
            pair_name=pair_name,
            name_x=name_x,
            name_y=name_y,
            max_holding_days=max_holding_days,
            stop_mult_gamma=stop_mult_gamma,
            costs_bps=costs_bps,
            bid_ask_spread_pct=bid_ask_spread_pct,
            slippage_bps=slippage_bps,
        )
        if verbose:
            print_strategy_summary(strat)

        # ── Cointegration analysis ─────────────────────────────────────────────
        print("  Running cointegration tests ...")
        coint = full_cointegration_analysis(
            x, y, pair_name=pair_name, name_x=name_x, name_y=name_y
        )
        if verbose:
            print_cointegration_summary(coint)

        results_all.append(strat)
        coint_all.append(coint)

        # ── Per-pair dashboard ─────────────────────────────────────────────────
        safe_name = pair_name.replace(" ", "_").replace("/", "-")
        dash_path = os.path.join(output_dir, f"{safe_name}_dashboard.png")
        try:
            fig = plot_dashboard(strat, coint, save_path=dash_path)
            plt.close(fig)
        except Exception as e:
            print(f"  Warning: could not save dashboard: {e}")

        # ── Accumulate CSV row ─────────────────────────────────────────────────
        csv_rows.append({
            "pair":              pair_name,
            "ticker_x":          name_x,
            "ticker_y":          name_y,
            "n_obs":             len(x),
            "alpha_hat":         round(strat.alpha_hat, 6),
            "gamma":             round(strat.gamma, 6),
            "kappa_pct":         round(strat.kappa * 100, 1),
            "n_trades":          strat.num_trades,
            "forced_closes":     strat.forced_close_count,
            "total_profit":      round(strat.total_profit, 4),
            "total_costs":       round(strat.total_costs, 4),
            "avg_profit_clean":  round(strat.avg_profit, 4),
            "sharpe_trade":      round(strat.sharpe_estimate, 4),
            "sharpe_daily":      round(strat.sharpe_daily, 4),
            "max_drawdown":      round(strat.max_drawdown, 4),
            "i1_pass":           int(coint.both_i1),
            "eg_coint_5pct":     int(coint.eg_cointegrated_5pct),
            "eg_stat_xy":        round(coint.engle_granger.adf_result.test_stat, 4),
            "eg_stat_yx":        round(coint.engle_granger_yx.adf_result.test_stat, 4),
            "joh_coint_5pct":    int(coint.johansen.cointegrated_5pct),
        })

    if not results_all:
        print("\n  No pairs analysed successfully.")
        return results_all, coint_all

    # ── Ranking & conclusion ────────────────────────────────────────────────────
    rank_pairs(results_all, coint_all)
    print_overall_conclusion(results_all, coint_all)

    # ── Cross-pair comparison chart ────────────────────────────────────────────
    comp_path = os.path.join(output_dir, "cross_pair_comparison.png")
    try:
        fig = plot_cross_pair_comparison(results_all, coint_all, save_path=comp_path)
        if fig:
            plt.close(fig)
    except Exception as e:
        print(f"  Warning: could not save comparison chart: {e}")

    # ── CSV results ────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Results CSV saved to: {csv_path}")

    # ── Text summary ───────────────────────────────────────────────────────────
    summary_path = os.path.join(output_dir, "results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PAIRS TRADING RESULTS SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Costs: {costs_bps:.1f}bps comm, "
                f"{bid_ask_spread_pct*100:.3f}% bid/ask, "
                f"{slippage_bps:.1f}bps slippage\n")
        f.write(f"Risk:  max_hold={max_holding_days}d, "
                f"stop_mult={stop_mult_gamma}×Γ\n")
        f.write("=" * 80 + "\n\n")
        for row in csv_rows:
            f.write(
                f"{row['pair']:<35} "
                f"Γ={row['gamma']:.3f} κ={row['kappa_pct']:.0f}% "
                f"trades={row['n_trades']} forced={row['forced_closes']} "
                f"profit={row['total_profit']:.4f} costs={row['total_costs']:.4f} "
                f"Sh.daily={row['sharpe_daily']:.3f} "
                f"EG={'Y' if row['eg_coint_5pct'] else 'N'} "
                f"Joh={'Y' if row['joh_coint_5pct'] else 'N'}\n"
            )
    print(f"  Summary saved to:    {summary_path}")
    print(f"  Dashboards saved to: {output_dir}/")

    return results_all, coint_all


# ─────────────────────────────────────────────────────────────────────────────
# pairs.txt parser
# ─────────────────────────────────────────────────────────────────────────────

def read_pairs_file(path: str) -> list:
    """
    Parse pairs.txt into a list of pair config dicts.

    Supported formats (one pair per line):
        XOM  CVX                  space-separated
        XOM, CVX                  comma-separated
        XOM  CVX  Exxon vs Chev   optional label after the two tickers
        # comment                  ignored
    """
    pairs = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Strip inline comments
            if " #" in line:
                line = line[:line.index(" #")].rstrip()
            # Normalise comma → space so split() handles both
            line  = line.replace(",", " ")
            parts = line.split()
            if len(parts) < 2:
                print(f"  [pairs.txt] Line {lineno}: skipping '{raw.rstrip()}' "
                      f"(need at least 2 tickers)")
                continue
            x, y = parts[0].upper(), parts[1].upper()
            name = " ".join(parts[2:]) if len(parts) > 2 else f"{x} / {y}"
            pairs.append({"name": name, "X": x, "Y": y, "source": "yfinance"})
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

PAIRS_FILE = "pairs.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pairs Trading Analysis – Gorter (2006)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "pairs.txt format (one pair per line, space or comma separated):\n"
            "  XOM  CVX            # minimal\n"
            "  XOM, CVX            # comma also works\n"
            "  XOM  CVX  My label  # optional name\n"
            "  # comment\n\n"
            "Examples:\n"
            "  python main.py\n"
            "  python main.py my_pairs.txt --start 2019-01-01 --end 2023-12-31\n"
            "  python main.py --costs_bps 1 --bid_ask_spread_pct 0.0005 "
            "--max_holding_days 60\n"
            "  python main.py --demo\n"
        ),
    )
    # Positional
    parser.add_argument(
        "pairs_file", nargs="?", default=PAIRS_FILE,
        help=f"Path to pairs file (default: {PAIRS_FILE}).",
    )
    # Date range
    parser.add_argument("--start", default=None,
                        help="Start date YYYY-MM-DD (default: 4 years before --end).")
    parser.add_argument("--end",   default=None,
                        help="End date YYYY-MM-DD (default: today).")
    # Output
    parser.add_argument("--output", default="output",
                        help="Output directory (default: ./output).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-pair verbose output.")
    parser.add_argument("--demo",  action="store_true",
                        help="Run built-in synthetic demo (no internet needed).")
    # Transaction costs
    parser.add_argument("--costs_bps", type=float, default=0.0,
                        help="Commission in basis points per leg notional (default: 0).")
    parser.add_argument("--bid_ask_spread_pct", type=float, default=0.0,
                        help="Bid/ask half-spread as fraction (default: 0). "
                             "E.g. 0.0005 = 0.05%%.")
    parser.add_argument("--slippage_bps", type=float, default=0.0,
                        help="Market-impact slippage in basis points (default: 0).")
    # Risk controls
    parser.add_argument("--max_holding_days", type=int, default=60,
                        help="Force-close position after this many days (default: 60).")
    parser.add_argument("--stop_mult", type=float, default=3.0,
                        help="Stop-loss multiplier: close if |spread| > mult×Γ "
                             "(default: 3.0).")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # ── Pairs source ──────────────────────────────────────────────────────────
    if args.demo:
        pairs_cfg    = config.PAIRS
        use_yfinance = False
    else:
        if not os.path.exists(args.pairs_file):
            print(f"\n  ERROR: pairs file not found: '{args.pairs_file}'")
            print(f"  Create it with one pair per line, e.g.:\n")
            print(f"    XOM  CVX  Exxon vs Chevron")
            print(f"    BP   SHEL")
            print(f"\n  Or run the synthetic demo:  python main.py --demo\n")
            sys.exit(1)
        pairs_cfg = read_pairs_file(args.pairs_file)
        if not pairs_cfg:
            print(f"\n  ERROR: no valid pairs found in '{args.pairs_file}'.\n")
            sys.exit(1)
        use_yfinance = True

    # ── Date range (4-year default → ~1040 trading days) ─────────────────────
    end_date   = args.end or datetime.today().strftime("%Y-%m-%d")
    start_date = args.start or (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=int(4 * 365.25))
    ).strftime("%Y-%m-%d")

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  PAIRS TRADING ANALYSIS  (Gorter 2006)")
    if use_yfinance:
        print(f"  Pairs file   : {args.pairs_file}")
        print(f"  Data source  : Yahoo Finance (yfinance)")
        print(f"  Date range   : {start_date} → {end_date}")
    else:
        print(f"  Data source  : Synthetic demo")
    print(f"  Costs        : {args.costs_bps:.1f} bps comm | "
          f"{args.bid_ask_spread_pct*100:.3f}% bid/ask | "
          f"{args.slippage_bps:.1f} bps slippage")
    print(f"  Risk         : max_hold={args.max_holding_days}d | "
          f"stop={args.stop_mult}×Γ")
    print(f"  Pairs ({len(pairs_cfg)}):")
    for p in pairs_cfg:
        label = p['name'] if p['name'] != f"{p['X']} / {p['Y']}" else ""
        print(f"    {p['X']:10} / {p['Y']:10}  {label}")
    print(f"{'='*65}\n")

    results, coint = run_analysis(
        pairs_cfg=pairs_cfg,
        start_date=start_date,
        end_date=end_date,
        output_dir=args.output,
        verbose=not args.quiet,
        max_holding_days=args.max_holding_days,
        stop_mult_gamma=args.stop_mult,
        costs_bps=args.costs_bps,
        bid_ask_spread_pct=args.bid_ask_spread_pct,
        slippage_bps=args.slippage_bps,
    )

    print(f"\nDone. Analysed {len(results)} pairs.")
    print(f"Output: {args.output}/\n")
