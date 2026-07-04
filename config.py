import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-edmd-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── EDMD hyperparameters ──────────────────────────────────────────────────────
# Williams, Kevrekidis & Rowley (2015) "A Data-Driven Approximation of the
# Koopman Operator: Extending Dynamic Mode Decomposition". The Koopman
# operator K acts LINEARLY on observables of a (possibly nonlinear) dynamical
# system: (Kg)(x) = g(F(x)). Standard DMD approximates K using only the raw
# state as the observable — equivalent to assuming the dynamics are linear.
#
# EDMD instead lifts the state into a rich DICTIONARY of nonlinear
# observables psi(x) = [1, x, x^2, x_i*x_j, RBF(x)...] and fits a finite
# matrix A such that psi(x_{t+1}) ~= A @ psi(x_t) via regularized least
# squares. This is a strictly richer, provably better finite-dimensional
# approximation of K for genuinely nonlinear systems — standard DMD is
# recovered exactly as the special case where the dictionary is just the
# linear terms.
#
# The eigenvalues of A give the growth/decay rates of the underlying return
# dynamics directly: |lambda| > 1 is an expanding/trending mode, |lambda| < 1
# is a contracting/mean-reverting mode, and arg(lambda) gives an oscillation
# frequency. This is genuine spectral analysis of the dynamics, not a
# statistical fit to be interpreted after the fact.
#
# State construction uses DELAY EMBEDDING (Takens' theorem): the state at
# time t is the last DELAY_DIM lagged returns, not just today's return alone
# — this is what allows a scalar return series to be treated as a proper
# (reconstructed) dynamical system in the first place.

DELAY_DIM      = 5     # d: delay-embedding dimension (state = last d returns)
N_RBF_CENTERS  = 12     # RBF centers sampled from the training snapshots
RBF_SIGMA_SCALE = 1.0    # multiplier on the median-heuristic RBF bandwidth
REGULARIZATION  = 1e-6   # Tikhonov ridge added to the Gram matrix for stability

PRED_HORIZON = 21        # H: forecast horizon, propagated via Koopman spectral evolution

# Regularized least-squares on a rich dictionary with finite noisy samples
# can produce spurious eigenvalues far from anything physically plausible
# for daily returns. These are safety clips against that known EDMD
# pitfall — not a tuning knob for the signal itself.
FORECAST_CLIP_STD = 5.0   # clip each forecasted step to +/- this many training-return std devs
GROWTH_RATE_CLIP  = 2.0   # clip the growth-rate diagnostic to +/- this value

# ── Score construction ────────────────────────────────────────────────────────
# forecast_signal : mean of the H-step-ahead return path, propagated purely
#                   via linear evolution in Koopman/spectral space
#                   (psi(x_{t+n}) = V @ (eigvals^n * c))
# growth_rate     : weighted-average log|eigenvalue| across modes, weighted
#                   by each mode's contribution to reconstructing the return
#                   observable right now — positive means the locally
#                   relevant dynamics are expanding (trending/momentum),
#                   negative means contracting (mean-reverting)
# fit_quality     : R^2 of the EDMD regression itself (how well A actually
#                   reconstructs psi(x_{t+1}) from psi(x_t) on the training
#                   snapshots) — a natural, undistorted diagnostic since it
#                   IS the regression's own objective, not an auxiliary add-on

WEIGHT_FORECAST = 0.50
WEIGHT_GROWTH    = 0.30
WEIGHT_FIT       = 0.20

TOP_N = 3
