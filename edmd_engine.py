"""
edmd_engine.py — Extended Dynamic Mode Decomposition (EDMD) Engine
========================================================================

Theory
------
**The Koopman operator.** For a discrete-time dynamical system
x_{t+1} = F(x_t) — possibly highly nonlinear — the Koopman operator K acts
on scalar observable functions g of the state via (Kg)(x) = g(F(x)). Even
though F may be nonlinear, K is LINEAR (it just happens to act on an
infinite-dimensional space of functions rather than on the state itself).
If we can find a good finite-dimensional approximation of K, we get to
analyze genuinely nonlinear dynamics with all the machinery of linear
spectral theory — eigenvalues, eigenvectors, modes.

**DMD vs. EDMD.** Standard Dynamic Mode Decomposition approximates K using
only the raw state as the observable, i.e. it implicitly assumes the
dynamics are linear. Extended DMD (Williams, Kevrekidis & Rowley, 2015)
instead lifts the state into a richer DICTIONARY of nonlinear observables:

    psi(x) = [1, x_1..x_d, x_1^2..x_d^2, x_i*x_j (i<j), RBF_1(x)..RBF_M(x)]

and fits a matrix A (via regularized least squares over snapshot pairs)
such that:

    psi(x_{t+1}) ~= A @ psi(x_t)

DMD is recovered exactly as the special case where the dictionary is only
the linear terms — EDMD is a strict generalization, and provably a better
finite-dimensional approximation of K whenever the true dynamics are
nonlinear (which return dynamics generally are).

**Delay embedding.** A scalar return series isn't a complete "state" on
its own — by Takens' embedding theorem, stacking DELAY_DIM consecutive
lagged returns into a vector reconstructs a genuine state space in which
the return series behaves like a proper (deterministic-plus-noise)
dynamical system:

    x_t = [r_t, r_{t-1}, ..., r_{t-d+1}]

**Fitting A.** Given snapshot pairs (psi(x_t), psi(x_{t+1})) across the
training window:

    G     = Psi_X^T Psi_X / N + lambda*I        (regularized Gram matrix)
    Ahat  = Psi_X^T Psi_Y / N
    M     = G^{-1} Ahat  ,   A = M^T             (so psi(x_{t+1}) ~= A psi(x_t))

**Spectral decomposition.** Eigendecompose A = V diag(lambda) V^{-1}.
Each eigenvalue directly gives a growth/decay rate of the reconstructed
dynamics: |lambda| > 1 is an expanding/trending mode, |lambda| < 1 is a
contracting/mean-reverting mode, arg(lambda) gives an oscillation
frequency. This is literal spectral analysis of the dynamics, not a
statistical fit interpreted after the fact.

**Forecasting.** Project today's dictionary vector onto the eigenbasis
(c = V^+ psi(x_today)), then evolve forward purely via the diagonal
spectral dynamics:

    psi(x_{t+n}) ~= V @ (lambda^n * c)

and read the forecasted return off the corresponding dictionary
coordinate (index of the raw x_1 = r_t term).

**Score construction**

    score = 0.50*forecast_signal + 0.30*growth_rate*sign(forecast_signal) + 0.20*fit_quality

| Component        | Meaning                                                                |
|--------------------|--------------------------------------------------------------------------|
| forecast_signal    | Mean of the H-step-ahead return path via spectral propagation          |
| growth_rate        | Weighted-average log|eigenvalue|, weighted by relevance to the return observable right now |
| fit_quality        | R^2 of the EDMD regression itself — a natural byproduct, not an auxiliary add-on |

No gradient descent, no training epochs — the entire model is a single
regularized least-squares solve plus one eigendecomposition of a small
matrix. Known EDMD pitfall: a rich dictionary fit on finite noisy samples
can produce spurious eigenvalues far from anything physically plausible;
FORECAST_CLIP_STD and GROWTH_RATE_CLIP guard against exactly that.

References
----------
- Williams, M., Kevrekidis, I. & Rowley, C. (2015). A Data-Driven
  Approximation of the Koopman Operator: Extending Dynamic Mode
  Decomposition. Journal of Nonlinear Science.
- Koopman, B.O. (1931). Hamiltonian Systems and Transformation in Hilbert
  Space. PNAS.
- Takens, F. (1981). Detecting Strange Attractors in Turbulence. Lecture
  Notes in Mathematics.
"""

import numpy as np
import pandas as pd
from typing import List

import config


# ── Dictionary construction ───────────────────────────────────────────────────

def build_delay_embedding(log_ret: np.ndarray, d: int) -> np.ndarray:
    """Returns X (n,d): row k = [r_t, r_{t-1}, ..., r_{t-d+1}] with t = k+d-1."""
    T = len(log_ret)
    n = T - d + 1
    X = np.zeros((n, d))
    for k in range(n):
        X[k] = log_ret[k:k + d][::-1]
    return X


def rbf_bandwidth(X: np.ndarray) -> float:
    """Median-heuristic RBF bandwidth from pairwise distances in X."""
    n = len(X)
    diffs = X[:, None, :] - X[None, :, :]
    dist2 = np.sum(diffs ** 2, axis=2)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(np.clip(dist2[iu], 0, None))
    med = np.median(dists) if len(dists) > 0 else 1.0
    return max(med, 1e-8) * config.RBF_SIGMA_SCALE


def build_dictionary(X: np.ndarray, centers: np.ndarray, sigma: float):
    """
    psi(x) = [1, x_1..x_d, x_1^2..x_d^2, x_i*x_j (i<j), RBF_1(x)..RBF_M(x)]
    Returns (Psi (n,K), return_idx) where return_idx is the dictionary
    column corresponding to x_1 = r_t (today's return, the most recent lag).
    """
    n, d = X.shape
    const = np.ones((n, 1))
    linear = X
    quad_sq = X ** 2

    cross_cols = []
    for i in range(d):
        for j in range(i + 1, d):
            cross_cols.append((X[:, i] * X[:, j])[:, None])
    cross = np.hstack(cross_cols) if cross_cols else np.zeros((n, 0))

    diffs = X[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diffs ** 2, axis=2)
    rbf = np.exp(-dist2 / (2 * sigma ** 2))

    Psi = np.hstack([const, linear, quad_sq, cross, rbf])
    return_idx = 1   # column 0 is the constant; column 1 is x_1 = r_t
    return Psi, return_idx


# ── EDMD fit ───────────────────────────────────────────────────────────────────

def fit_edmd(Psi_X: np.ndarray, Psi_Y: np.ndarray):
    """Regularized least-squares Koopman approximation: psi(x_{t+1}) ~= A @ psi(x_t)."""
    N, K = Psi_X.shape
    G = (Psi_X.T @ Psi_X) / N + config.REGULARIZATION * np.eye(K)
    Ahat = (Psi_X.T @ Psi_Y) / N
    M = np.linalg.solve(G, Ahat)
    A = M.T

    Psi_Y_hat = Psi_X @ M
    resid = Psi_Y - Psi_Y_hat
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((Psi_Y - Psi_Y.mean(axis=0, keepdims=True)) ** 2)
    fit_quality = float(1.0 - np.clip(ss_res / (ss_tot + 1e-10), 0.0, 1.0))

    return A, fit_quality


# ── Per-ticker forecast + diagnostics ───────────────────────────────────────────

def forecast_and_diagnose(prices: pd.DataFrame, ticker: str, window: int, rng: np.random.Generator):
    """
    Fit EDMD for one ticker using only data present in `prices` (the caller
    controls the as-of cutoff simply by how much history `prices` contains)
    and return {forecast_signal, growth_rate, fit_quality} for the forecast
    made from the LAST row of `prices`. Returns None on failure.

    This is the single source of truth for per-ticker EDMD scoring.
    """
    d, H = config.DELAY_DIM, config.PRED_HORIZON

    ps = prices[ticker].dropna()
    if len(ps) < window + d + config.N_RBF_CENTERS + 10:
        return None

    log_ret_full = np.log(ps / ps.shift(1)).dropna().values
    log_ret = log_ret_full[-(window + d):]   # training slice: `window` snapshots worth + embedding lookback
    if len(log_ret) < d + config.N_RBF_CENTERS + 10:
        return None

    ret_mu, ret_sd = log_ret.mean(), log_ret.std() + 1e-8

    X = build_delay_embedding(log_ret, d)         # (n, d)
    if len(X) < config.N_RBF_CENTERS + 5:
        return None

    Psi_X_full = X[:-1]   # states t = 0..n-2
    Psi_Y_full = X[1:]    # states t = 1..n-1  (one-step-ahead targets, still raw delay-states here)

    n_train = len(Psi_X_full)
    n_centers = min(config.N_RBF_CENTERS, n_train)
    center_idx = rng.choice(n_train, size=n_centers, replace=False)
    centers = Psi_X_full[center_idx]
    sigma = rbf_bandwidth(Psi_X_full)

    Psi_X, return_idx = build_dictionary(Psi_X_full, centers, sigma)
    Psi_Y, _          = build_dictionary(Psi_Y_full, centers, sigma)

    try:
        A, fit_quality = fit_edmd(Psi_X, Psi_Y)
        eigvals, V = np.linalg.eig(A)
    except np.linalg.LinAlgError as e:
        print(f"    Failed {ticker}: {e}")
        return None

    x_today = X[-1]
    psi_today, _ = build_dictionary(x_today[None, :], centers, sigma)
    psi_today = psi_today[0]

    c, *_ = np.linalg.lstsq(V, psi_today, rcond=None)

    forecast_returns = []
    for step in range(1, H + 1):
        psi_fwd = V @ (c * (eigvals ** step))
        r_fwd = float(np.real(psi_fwd[return_idx]))
        r_fwd = float(np.clip(r_fwd, -config.FORECAST_CLIP_STD * ret_sd, config.FORECAST_CLIP_STD * ret_sd))
        forecast_returns.append(r_fwd)
    forecast_signal = float(np.mean(forecast_returns))

    contributions = c * V[return_idx, :]
    weights = np.abs(contributions)
    weights = weights / (np.sum(weights) + 1e-10)
    growth_rate = float(np.sum(weights * np.log(np.abs(eigvals) + 1e-10)))
    growth_rate = float(np.clip(growth_rate, -config.GROWTH_RATE_CLIP, config.GROWTH_RATE_CLIP))

    return {
        "forecast_signal": forecast_signal,
        "growth_rate": growth_rate,
        "fit_quality": fit_quality,
        "sign": np.sign(forecast_signal) if forecast_signal != 0 else 1.0,
    }


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_edmd_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.DataFrame:
    """
    Fit EDMD per ETF (pure univariate dynamical-systems reconstruction via
    delay embedding — no macro conditioning) and extract a spectral
    forecast signal. Returns a DataFrame of score + diagnostics
    (cross-sectional z-scored on the composite).
    """
    cols = ["score", "forecast_signal", "growth_rate", "fit_quality"]
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.DataFrame(columns=cols)

    rng = np.random.default_rng(42)
    raw_scores = {}

    for ticker in avail:
        print(f"    Fitting EDMD for {ticker}")
        diag = forecast_and_diagnose(prices, ticker, window, rng)
        if diag is None:
            continue

        forecast_signal = diag["forecast_signal"]
        growth_rate     = diag["growth_rate"]
        fit_quality     = diag["fit_quality"]
        sign            = diag["sign"]

        print(f"    {ticker}: forecast={forecast_signal:.5f}  "
              f"growth_rate={growth_rate:.4f}  fit={fit_quality:.3f}")

        composite = (
            config.WEIGHT_FORECAST * forecast_signal
            + config.WEIGHT_GROWTH   * growth_rate * sign
            + config.WEIGHT_FIT       * fit_quality
        )
        raw_scores[ticker] = {
            "composite": composite,
            "forecast_signal": forecast_signal,
            "growth_rate": growth_rate,
            "fit_quality": fit_quality,
        }

    if not raw_scores:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(raw_scores).T
    mu_s, std_s = df["composite"].mean(), df["composite"].std()
    if std_s < 1e-10:
        df["score"] = 0.0
    else:
        df["score"] = (df["composite"] - mu_s) / std_s
    return df[cols]
