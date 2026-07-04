# 🌀 P2-ETF-EDMD

**Extended Dynamic Mode Decomposition Engine — Williams, Kevrekidis & Rowley (2015)**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine fits an **EDMD** (Extended Dynamic Mode Decomposition) model
per ETF — a finite-dimensional approximation of the Koopman operator
governing return dynamics, built from a rich dictionary of nonlinear
observables rather than the raw state alone. Unlike every ML engine in
this suite, **there is no training loop at all**: the entire model is a
single regularized least-squares solve plus one eigendecomposition of a
small matrix. Eigenvalues give the growth/decay rates of the underlying
return dynamics directly — this is genuine spectral analysis, not a
statistical fit interpreted after the fact.

---

## Theory

### The Koopman Operator

For a discrete-time dynamical system `x_{t+1} = F(x_t)` — possibly highly
nonlinear — the Koopman operator K acts linearly on observable functions
of the state: `(Kg)(x) = g(F(x))`. Even though F may be nonlinear, K is
LINEAR (it acts on an infinite-dimensional function space rather than on
the state itself). A good finite-dimensional approximation of K gives
access to the full machinery of linear spectral theory for genuinely
nonlinear dynamics.

### DMD vs. EDMD

Standard Dynamic Mode Decomposition approximates K using only the raw
state as the observable — implicitly assuming linear dynamics. EDMD
instead lifts the state into a richer dictionary:

```
psi(x) = [1, x_1..x_d, x_1^2..x_d^2, x_i*x_j (i<j), RBF_1(x)..RBF_M(x)]
```

and fits a matrix A via regularized least squares such that
`psi(x_{t+1}) ~= A @ psi(x_t)`. **DMD is recovered exactly** as the
special case where the dictionary is only the linear terms — EDMD is a
strict generalization, provably a better finite-dimensional approximation
of K whenever the true dynamics are nonlinear.

### Delay Embedding

A scalar return series isn't a complete "state" on its own. By **Takens'
embedding theorem**, stacking `DELAY_DIM` consecutive lagged returns into
a vector reconstructs a genuine state space:

```
x_t = [r_t, r_{t-1}, ..., r_{t-d+1}]
```

in which the return series behaves like a proper dynamical system.

### Fitting A

Given snapshot pairs `(psi(x_t), psi(x_{t+1}))` across the training window:

```
G     = Psi_X^T Psi_X / N + lambda*I        (regularized Gram matrix)
Ahat  = Psi_X^T Psi_Y / N
M     = G^-1 Ahat  ,   A = M^T               (so psi(x_{t+1}) ~= A psi(x_t))
```

### Spectral Decomposition

Eigendecompose `A = V diag(lambda) V^-1`. Each eigenvalue directly gives a
growth/decay rate: `|lambda| > 1` is an expanding/trending mode,
`|lambda| < 1` is a contracting/mean-reverting mode, `arg(lambda)` gives
an oscillation frequency.

Note: the constant dictionary term `[1]` is always a trivial Koopman
eigenfunction with eigenvalue exactly 1 (`K(1)(x) = 1(F(x)) = 1`) — this
is expected and correctly down-weighted by the growth_rate calculation,
which weights each mode by its relevance to reconstructing the *return*
observable specifically, not just any observable.

### Forecasting

Project today's dictionary vector onto the eigenbasis
(`c = V^+ psi(x_today)`), then evolve forward purely via diagonal spectral
dynamics:

```
psi(x_{t+n}) ~= V @ (lambda^n * c)
```

and read the forecasted return off the dictionary coordinate corresponding
to `x_1 = r_t`.

### Score Construction

```
score = 0.50*forecast_signal + 0.30*growth_rate*sign(forecast_signal) + 0.20*fit_quality
```

| Component | Meaning |
|-----------|---------|
| forecast_signal | Mean of the H-step-ahead return path via spectral propagation |
| growth_rate | Weighted-average log\|eigenvalue\|, weighted by relevance to the return observable right now |
| fit_quality | R² of the EDMD regression itself — a natural byproduct, not an auxiliary add-on |

### Known Pitfall: Spurious Eigenvalues

A rich dictionary fit on finite noisy samples can produce spurious
eigenvalues far from anything physically plausible for daily returns.
`FORECAST_CLIP_STD` and `GROWTH_RATE_CLIP` are safety nets against exactly
this documented EDMD limitation — not tuning knobs for the signal itself.

### Validation

Recovered `growth_rate` was validated against synthetic AR(1) processes
with known ground-truth decay rates (`log|phi|`): correct sign in every
case, correct magnitude-independence from the sign of phi (φ=0.7 and
φ=-0.7 give nearly identical growth rates, as they should), and correct
monotonic ordering as φ approaches a unit root.

---

## Distinction from Other Engines in the Suite

| Engine | Training | Core mechanism |
|--------|----------|-----------------|
| Decision Transformer | Gradient descent (Adam) | Causal sequence modelling (offline RL) |
| OT-FM | Gradient descent (Adam) | Flow matching with optimal coupling |
| N-HiTS | Gradient descent (Adam) | Multi-rate hierarchical decomposition |
| **EDMD (this engine)** | **None — closed-form least squares** | **Koopman spectral decomposition** |

EDMD is the only engine in this group that requires no training loop at
all, and the only one whose diagnostics (eigenvalues) have a direct
physical interpretation as growth/decay rates rather than being learned
representations.

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure

```
P2-ETF-EDMD/
├── config.py          # Universes, EDMD hyperparameters, score weights
├── data_manager.py    # HuggingFace loader
├── edmd_engine.py      # Core: delay embedding, dictionary, Koopman regression, spectral forecast
├── trainer.py          # Orchestrator
├── push_results.py     # HfApi.upload_file wrapper
├── streamlit_app.py     # Two-tab Streamlit dashboard
├── us_calendar.py      # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml   # Single job
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-EDMD
cd P2-ETF-EDMD
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-edmd-results`

---

## References

- Williams, M., Kevrekidis, I. & Rowley, C. (2015). A Data-Driven
  Approximation of the Koopman Operator: Extending Dynamic Mode
  Decomposition. Journal of Nonlinear Science.
- Koopman, B.O. (1931). Hamiltonian Systems and Transformation in Hilbert
  Space. PNAS.
- Takens, F. (1981). Detecting Strange Attractors in Turbulence. Lecture
  Notes in Mathematics.
