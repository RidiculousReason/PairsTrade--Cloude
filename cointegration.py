"""
cointegration.py
================
Statistical tests for pairs trading, following Gorter (2006) Chapters 3-6.

Implemented from scratch (no statsmodels):
  1. OLS regression
  2. Augmented Dickey-Fuller (ADF) test – Cases 1, 2, 3
  3. Engle-Granger two-step cointegration test (both directions)
  4. Johansen cointegration test (trace and max-eigenvalue)
  5. Information criteria: AIC + BIC + HIC for AR / VAR lag selection

Critical value sources:
  - ADF/EG: Case 2, T=260 (formation half only — tests do NOT see backtest data).
             MacKinnon (1994) finite-sample values for T=250.
  - Johansen: Tables 4.1 / 4.2 of Gorter (2006), Case 1, T=400
             (asymptotic; conservative for T=260 but standard in the literature).

NOTE: Gorter (2006) runs tests on full T=520. We run on T=260 (formation half)
to avoid look-ahead bias. This requires slightly adjusted critical values.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import config


# ---------------------------------------------------------------------------
# Critical value tables  (all from Gorter 2006)
# ---------------------------------------------------------------------------

# ADF critical values, Case 2 (with constant, no trend)
# T=260 (formation half): MacKinnon (1994) finite-sample for T=250.
# Slightly more negative than T=520 values — correct conservative adjustment.
ADF_CRITICAL_VALUES = {
    "case1": {"1%": -2.5899, "5%": -1.9439, "10%": -1.6177},  # no const, no trend
    "case2": {"1%": -3.46,   "5%": -2.88,   "10%": -2.57},    # with const, T=260
    "case3": {"1%": -3.99,   "5%": -3.43,   "10%": -3.13},    # const + trend
}

# Engle-Granger critical values (residual ADF, Case 2, T=260)
# Gorter Chapter 7: same distribution as DF Case 2.
# We use T=260 finite-sample values to match our formation-half sample size.
EG_CRITICAL_VALUES = {
    "1%":  -3.46,
    "5%":  -2.88,
    "10%": -2.57,
}

# Johansen critical values – Case 1 (no constant, no trend), T=400
# Source: Tables 4.1 (trace) and 4.2 (max-eigenvalue) of Gorter (2006)
#
# Key = g = n - h  (number of non-cointegrating relations under H0)
# For n=2 stocks: testing rank=0  -> g=2; testing rank<=1 -> g=1

JOHANSEN_TRACE_CV = {
    1: {"1%":  6.51, "5%":  3.84, "10%":  2.86},
    2: {"1%": 16.31, "5%": 12.53, "10%": 10.47},
    3: {"1%": 29.75, "5%": 24.31, "10%": 21.63},
}

JOHANSEN_MAXEIG_CV = {
    1: {"1%":  6.51, "5%":  3.84, "10%":  2.86},
    2: {"1%": 15.69, "5%": 11.44, "10%":  9.52},
    3: {"1%": 22.99, "5%": 17.89, "10%": 15.59},
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ADFResult:
    """Result of an Augmented Dickey-Fuller test."""
    series_name:    str
    case:           str
    test_stat:      float
    p_lags:         int
    critical_values: Dict[str, float]
    reject_1pct:    bool
    reject_5pct:    bool
    reject_10pct:   bool
    is_stationary:  bool   # reject at 5%


@dataclass
class EngleGrangerResult:
    """Result of the Engle-Granger two-step test (one direction)."""
    direction:          str        # e.g. "Y ~ X"
    alpha:              float      # OLS slope (no intercept, Section 4.3)
    alpha0:             float      # always 0.0 (no intercept)
    residuals:          np.ndarray
    adf_result:         ADFResult
    cointegrated_1pct:  bool
    cointegrated_5pct:  bool
    cointegrated_10pct: bool


@dataclass
class JohansenResult:
    """Result of the Johansen cointegration test (three-test verdict)."""
    trace_stats:    np.ndarray
    max_eig_stats:  np.ndarray
    eigenvalues:    np.ndarray
    eigenvectors:   np.ndarray
    # Three-test verdict (Section 4.4)
    test1_reject:   bool   # trace  g=2 rejects (H0: rank=0 vs H1: rank>=1)
    test2_reject:   bool   # maxeig g=2 rejects (H0: rank=0 vs H1: rank=1)
    test3_reject:   bool   # either g=1 rejects (H0: rank<=1 -- should NOT)
    cointegrated_5pct: bool  # test1 AND test2 AND NOT test3


@dataclass
class CointegrationSummary:
    """Complete cointegration analysis for one pair."""
    pair_name:             str
    name_x:                str
    name_y:                str
    adf_x:                 ADFResult   # ADF on level of X (should NOT reject)
    adf_y:                 ADFResult   # ADF on level of Y (should NOT reject)
    adf_dx:                ADFResult   # ADF on first diff of X (SHOULD reject)
    adf_dy:                ADFResult   # ADF on first diff of Y (SHOULD reject)
    both_i1:               bool        # levels non-stationary AND diffs stationary
    engle_granger:         EngleGrangerResult    # primary: Y ~ X
    engle_granger_yx:      EngleGrangerResult    # reverse: X ~ Y
    eg_cointegrated_5pct:  bool                  # True if EITHER direction rejects
    johansen:              JohansenResult


# ---------------------------------------------------------------------------
# OLS helper
# ---------------------------------------------------------------------------

def _ols(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """OLS: regress y on columns of X. Returns (beta, residuals)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


# ---------------------------------------------------------------------------
# AR lag selection by AIC + BIC + HIC (Chapter 3)
# Final lag = round(mean(argmin_AIC, argmin_BIC, argmin_HIC))
# ---------------------------------------------------------------------------

def _ar_criteria(z: np.ndarray, p: int) -> tuple:
    """Return (AIC, BIC, HIC) for AR(p) fit on z."""
    T = len(z)
    if p == 0:
        sigma2 = np.var(z)
        n_obs  = T
    else:
        lags   = np.column_stack([z[p - i - 1: T - i - 1] for i in range(p)])
        _, e   = _ols(z[p:], lags)
        sigma2 = np.mean(e ** 2)
        n_obs  = T - p
    if sigma2 <= 0:
        return np.inf, np.inf, np.inf
    log_lik = -n_obs / 2 * (np.log(2 * np.pi) + np.log(sigma2) + 1)
    k   = p + 1   # number of parameters (AR coefficients + variance)
    aic = -2 * log_lik + 2 * k
    bic = -2 * log_lik + k * np.log(n_obs)
    hic = -2 * log_lik + 2 * k * np.log(np.log(n_obs)) if n_obs > 2 else np.inf
    return aic, bic, hic


def _select_ar_lags(z: np.ndarray, max_lags: int) -> int:
    """
    Select AR lag order as round(mean(k_AIC, k_BIC, k_HIC)).
    Each criterion picks its own argmin; the three are averaged and rounded.
    This is more conservative than AIC alone and matches the GitHub reference.
    """
    criteria = [_ar_criteria(z, p) for p in range(max_lags + 1)]
    aic_vals = [c[0] for c in criteria]
    bic_vals = [c[1] for c in criteria]
    hic_vals = [c[2] for c in criteria]
    k_aic = int(np.argmin(aic_vals))
    k_bic = int(np.argmin(bic_vals))
    k_hic = int(np.argmin(hic_vals))
    return max(0, int(round(np.mean([k_aic, k_bic, k_hic]))))


# ---------------------------------------------------------------------------
# Augmented Dickey-Fuller test (Chapter 5)
# ---------------------------------------------------------------------------

def adf_test(z: np.ndarray,
             series_name: str = "z",
             case: str = "case2",
             max_lags: Optional[int] = None,
             critical_values: Optional[Dict] = None) -> ADFResult:
    """
    ADF unit root test.

    H0: rho=1 (I(1))   H1: rho<1 (stationary)
    Test stat = (rho_hat - 1) / se(rho_hat)

    Critical values default to Gorter (2006), T=520, Case 2.
    """
    if max_lags is None:
        max_lags = config.ADF_MAX_LAGS

    T  = len(z)
    dz = np.diff(z)
    n_aug_lags = _select_ar_lags(dz, max_lags)

    n_start = n_aug_lags + 1
    y_reg   = dz[n_start:]
    z_lag1  = z[n_start: T - 1]
    lag_cols = [dz[n_start - i: T - 1 - i] for i in range(1, n_aug_lags + 1)]

    if case == "case1":
        X = np.column_stack([z_lag1] + lag_cols) if lag_cols else z_lag1.reshape(-1, 1)
    elif case == "case2":
        ones = np.ones(len(y_reg))
        X = np.column_stack([ones, z_lag1] + lag_cols) if lag_cols \
            else np.column_stack([ones, z_lag1])
    elif case == "case3":
        ones  = np.ones(len(y_reg))
        trend = np.arange(1, len(y_reg) + 1, dtype=float)
        X = np.column_stack([ones, trend, z_lag1] + lag_cols) if lag_cols \
            else np.column_stack([ones, trend, z_lag1])
    else:
        raise ValueError(f"Unknown case: {case}")

    n_obs = len(y_reg)
    beta, resid = _ols(y_reg, X)

    k       = X.shape[1]
    sigma2  = np.sum(resid ** 2) / max(n_obs - k, 1)
    XtX_inv = np.linalg.inv(X.T @ X)

    rho_idx   = {"case1": 0, "case2": 1, "case3": 2}[case]
    se_rho    = np.sqrt(sigma2 * XtX_inv[rho_idx, rho_idx])
    test_stat = beta[rho_idx] / se_rho   # = (rho_hat - 1) / se

    if critical_values is None:
        cv = ADF_CRITICAL_VALUES[case]
    else:
        cv = critical_values

    return ADFResult(
        series_name=series_name,
        case=case,
        test_stat=float(test_stat),
        p_lags=n_aug_lags,
        critical_values=dict(cv),
        reject_1pct=  test_stat < cv["1%"],
        reject_5pct=  test_stat < cv["5%"],
        reject_10pct= test_stat < cv["10%"],
        is_stationary=test_stat < cv["5%"],
    )


# ---------------------------------------------------------------------------
# Engle-Granger two-step cointegration test (Chapter 4.3)
# ---------------------------------------------------------------------------

def engle_granger_test(x: np.ndarray,
                       y: np.ndarray,
                       name_x: str = "X",
                       name_y: str = "Y",
                       max_lags: Optional[int] = None) -> EngleGrangerResult:
    """
    Engle-Granger two-step test (Y ~ X direction).

    Step 1: OLS without intercept  yt = alpha * xt + et
              alpha_hat = sum(x*y) / sum(x^2)   (Section 4.3 cash neutrality)

    Step 2: ADF Case 2 on residuals with EG critical values
            (Chapter 7: same distribution as DF Case 2 for T=520)
    """
    if max_lags is None:
        max_lags = config.ADF_MAX_LAGS

    # OLS without intercept
    alpha = float(np.dot(x, y) / np.dot(x, x))
    resid = y - alpha * x

    # ADF on residuals
    adf_res = adf_test(
        resid,
        series_name=f"EG residuals ({name_y}~{name_x})",
        case="case2",
        max_lags=max_lags,
        critical_values=EG_CRITICAL_VALUES,
    )

    return EngleGrangerResult(
        direction=f"{name_y} ~ {name_x}",
        alpha=alpha,
        alpha0=0.0,
        residuals=resid,
        adf_result=adf_res,
        cointegrated_1pct=adf_res.reject_1pct,
        cointegrated_5pct=adf_res.reject_5pct,
        cointegrated_10pct=adf_res.reject_10pct,
    )


# ---------------------------------------------------------------------------
# Johansen cointegration test (Chapter 4.4)
# ---------------------------------------------------------------------------

def _johansen_VAR_lags(Y: np.ndarray, max_lags: int = 5) -> int:
    """
    AIC-based VAR lag selection.
    Returns at least 2 (minimum: first step fits VAR(p-1) on differences).
    """
    T, k = Y.shape
    best_aic = np.inf
    best_p   = 2

    for p in range(1, max_lags + 1):
        n     = T - p
        Z     = Y[p:]
        Xlags = np.hstack([Y[p - i - 1: T - i - 1] for i in range(p)])
        B, *_ = np.linalg.lstsq(Xlags, Z, rcond=None)
        resid = Z - Xlags @ B
        Sigma = resid.T @ resid / n
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            continue
        aic = n * logdet + 2 * p * k ** 2
        if aic < best_aic:
            best_aic = aic
            best_p   = p

    return max(2, best_p)   # enforce minimum p=2


def johansen_test(x: np.ndarray,
                  y: np.ndarray,
                  name_x: str = "X",
                  name_y: str = "Y",
                  max_lags: int = 5) -> JohansenResult:
    """
    Johansen cointegration test, Case 1 (no constant, no trend).

    Three-test verdict (Section 4.4):
      Test 1: trace  stat, g=2 CVs  ->  H0: rank=0  vs H1: rank>=1
      Test 2: maxeig stat, g=2 CVs  ->  H0: rank=0  vs H1: rank=1
      Test 3: both   stats, g=1 CVs ->  H0: rank<=1 vs H1: rank=2  (should NOT reject)

    Cointegrated (rank=1) iff:
      Test1 AND Test2 reject, AND Test3 does NOT reject.

    Critical values: Tables 4.1 / 4.2 of Gorter (2006), T=400.
    """
    T = len(x)
    Y = np.column_stack([x, y])
    k = 2

    p = _johansen_VAR_lags(Y, max_lags)

    DY   = np.diff(Y, axis=0)
    Ylag = Y[:-1]

    if p > 1:
        Z  = np.hstack([DY[p - 1 - j: T - 1 - j] for j in range(1, p)])
        R0 = DY[p - 1:]   - Z @ np.linalg.lstsq(Z, DY[p - 1:],   rcond=None)[0]
        R1 = Ylag[p - 1:] - Z @ np.linalg.lstsq(Z, Ylag[p - 1:], rcond=None)[0]
    else:
        R0 = DY
        R1 = Ylag

    n_eff = R0.shape[0]
    S00 = R0.T @ R0 / n_eff
    S11 = R1.T @ R1 / n_eff
    S01 = R0.T @ R1 / n_eff
    S10 = S01.T

    try:
        M = np.linalg.inv(S11) @ S10 @ np.linalg.inv(S00) @ S01
        eigenvalues, eigenvectors = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        eigenvalues  = np.zeros(k)
        eigenvectors = np.eye(k)

    order        = np.argsort(-np.real(eigenvalues))
    eigenvalues  = np.real(eigenvalues[order])
    eigenvectors = np.real(eigenvectors[:, order])
    eigenvalues  = np.clip(eigenvalues, 1e-10, 1 - 1e-10)

    trace_stats   = np.array([
        -n_eff * np.sum(np.log(1 - eigenvalues[r0:])) for r0 in range(k)
    ])
    max_eig_stats = np.array([
        -n_eff * np.log(1 - eigenvalues[r0]) for r0 in range(k)
    ])

    # Three-test verdict
    test1_reject = trace_stats[0]   > JOHANSEN_TRACE_CV[2]["5%"]
    test2_reject = max_eig_stats[0] > JOHANSEN_MAXEIG_CV[2]["5%"]
    test3_reject = (trace_stats[1]   > JOHANSEN_TRACE_CV[1]["5%"] or
                    max_eig_stats[1] > JOHANSEN_MAXEIG_CV[1]["5%"])

    cointegrated = test1_reject and test2_reject and (not test3_reject)

    return JohansenResult(
        trace_stats=trace_stats,
        max_eig_stats=max_eig_stats,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        test1_reject=test1_reject,
        test2_reject=test2_reject,
        test3_reject=test3_reject,
        cointegrated_5pct=cointegrated,
    )


# ---------------------------------------------------------------------------
# Full cointegration analysis (Chapter 7 procedure)
# ---------------------------------------------------------------------------

def full_cointegration_analysis(x: np.ndarray,
                                y: np.ndarray,
                                pair_name: str = "Pair",
                                name_x: str = "X",
                                name_y: str = "Y") -> CointegrationSummary:
    """
    Complete cointegration analysis:
      1. ADF test on each series (levels + first differences) — check I(1)
      2. Engle-Granger in BOTH directions (test is not symmetric)
      3. Johansen (three-test verdict, diagnostic)

    Called with formation-half data only (x_form, y_form) from main.py
    to avoid look-ahead bias. Critical values calibrated for T=260.
    """
    # Step 1a: ADF on levels (should NOT reject — series should be I(1))
    adf_x = adf_test(x, series_name=name_x, case="case2")
    adf_y = adf_test(y, series_name=name_y, case="case2")

    # Step 1b: ADF on first differences (SHOULD reject — diffs should be I(0))
    dx = np.diff(x)
    dy = np.diff(y)
    adf_dx = adf_test(dx, series_name=f"Δ{name_x}", case="case2")
    adf_dy = adf_test(dy, series_name=f"Δ{name_y}", case="case2")

    # Both I(1): levels non-stationary AND first diffs stationary
    both_i1 = (
        (not adf_x.is_stationary) and (not adf_y.is_stationary) and
        adf_dx.is_stationary and adf_dy.is_stationary
    )

    # Step 2: Engle-Granger in both directions
    eg_xy = engle_granger_test(x, y, name_x=name_x, name_y=name_y)
    eg_yx = engle_granger_test(y, x, name_x=name_y, name_y=name_x)
    eg_coint = eg_xy.cointegrated_5pct or eg_yx.cointegrated_5pct

    # Step 3: Johansen (three-test verdict)
    joh = johansen_test(x, y, name_x=name_x, name_y=name_y)

    return CointegrationSummary(
        pair_name=pair_name,
        name_x=name_x,
        name_y=name_y,
        adf_x=adf_x,
        adf_y=adf_y,
        adf_dx=adf_dx,
        adf_dy=adf_dy,
        both_i1=both_i1,
        engle_granger=eg_xy,
        engle_granger_yx=eg_yx,
        eg_cointegrated_5pct=eg_coint,
        johansen=joh,
    )


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _stars(reject_10: bool, reject_5: bool, reject_1: bool) -> str:
    if reject_1:  return "***"
    if reject_5:  return "**"
    if reject_10: return "*"
    return ""


def print_adf_result(r: ADFResult) -> None:
    print(f"    Series       : {r.series_name}")
    print(f"    Case         : {r.case}")
    print(f"    Lags chosen  : {r.p_lags}  (AIC)")
    print(f"    Test stat    : {r.test_stat:.4f}")
    print(f"    Crit. values : 1% = {r.critical_values['1%']:.4f}, "
          f"5% = {r.critical_values['5%']:.4f}, "
          f"10% = {r.critical_values['10%']:.4f}")
    stars   = _stars(r.reject_10pct, r.reject_5pct, r.reject_1pct)
    verdict = "Stationary I(0)" if r.is_stationary else "Unit root I(1)"
    print(f"    Verdict      : {verdict}  {stars}")
    print(f"    (*** 1%  ** 5%  * 10%)")


def _print_eg(eg: EngleGrangerResult) -> None:
    print(f"    Direction    : {eg.direction}")
    print(f"    alpha_hat    : {eg.alpha:.6f}  (no intercept)")
    print(f"    ADF on residuals:")
    print_adf_result(eg.adf_result)
    stars   = _stars(eg.cointegrated_10pct, eg.cointegrated_5pct, eg.cointegrated_1pct)
    verdict = "COINTEGRATED" if eg.cointegrated_5pct else "NOT cointegrated"
    print(f"    Verdict      : {verdict}  {stars}")


def print_cointegration_summary(cs: CointegrationSummary) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  COINTEGRATION ANALYSIS: {cs.pair_name}")
    print(f"  X = {cs.name_x}   Y = {cs.name_y}")
    print(sep)

    print(f"\n  UNIT ROOT TESTS  (I(1) = levels non-stationary AND diffs stationary)")
    print(f"\n  [Level ADF — H0: unit root, should NOT reject]")
    print(f"\n  {cs.name_x}")
    print_adf_result(cs.adf_x)
    print(f"\n  {cs.name_y}")
    print_adf_result(cs.adf_y)
    print(f"\n  [First-difference ADF — H0: unit root, SHOULD reject]")
    print_adf_result(cs.adf_dx)
    print_adf_result(cs.adf_dy)
    status = "YES" if cs.both_i1 else "NO"
    print(f"\n  Both series I(1): {status}  (levels non-stationary AND diffs stationary)")

    print(f"\n  ENGLE-GRANGER TEST  (no intercept; both directions)")
    print(f"\n  [Direction 1]")
    _print_eg(cs.engle_granger)
    print(f"\n  [Direction 2]")
    _print_eg(cs.engle_granger_yx)
    eg_v = "COINTEGRATED" if cs.eg_cointegrated_5pct else "NOT cointegrated"
    print(f"\n  Combined EG (5%): {eg_v}")

    joh = cs.johansen
    print(f"\n  JOHANSEN TEST  (Case 1: no const)")
    print(f"    Eigenvalues : lam1={joh.eigenvalues[0]:.6f}  lam2={joh.eigenvalues[1]:.6f}")
    tr_cv2  = JOHANSEN_TRACE_CV[2]["5%"]
    me_cv2  = JOHANSEN_MAXEIG_CV[2]["5%"]
    tr_cv1  = JOHANSEN_TRACE_CV[1]["5%"]
    me_cv1  = JOHANSEN_MAXEIG_CV[1]["5%"]
    r1 = "YES" if joh.test1_reject else "No"
    r2 = "YES" if joh.test2_reject else "No"
    r3 = "YES" if joh.test3_reject else "No"
    print(f"\n    Test  Statistic  CV(5%)  Reject?  Description")
    print(f"    " + "-" * 55)
    print(f"    1  Tr={joh.trace_stats[0]:>8.4f}  {tr_cv2:>6.4f}   {r1:<5}  H0:rank=0 (Trace g=2)")
    print(f"    2  ME={joh.max_eig_stats[0]:>8.4f}  {me_cv2:>6.4f}   {r2:<5}  H0:rank=0 (MaxEig g=2)")
    print(f"    3  Tr={joh.trace_stats[1]:>8.4f}  {tr_cv1:>6.4f}   {r3:<5}  H0:rank<=1 (should NOT reject)")
    print(f"       ME={joh.max_eig_stats[1]:>8.4f}  {me_cv1:>6.4f}")
    jv = "COINTEGRATED" if joh.cointegrated_5pct else "NOT cointegrated"
    cond = (f"T1={'Y' if joh.test1_reject else 'N'} "
            f"T2={'Y' if joh.test2_reject else 'N'} "
            f"T3={'N(good)' if not joh.test3_reject else 'Y(bad)'}")
    print(f"\n    Johansen (5%): {jv}  [{cond}]")
    print()
