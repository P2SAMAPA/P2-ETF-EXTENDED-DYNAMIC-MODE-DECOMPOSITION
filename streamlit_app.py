import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="EDMD Engine", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#1e2749; margin-bottom:0.3rem; }
.sub-header  { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
.uni-title   { font-size:1.4rem; font-weight:600; margin-top:1rem; margin-bottom:0.8rem;
               padding-left:0.5rem; border-left:5px solid #4a6fa5; }
.etf-card    { background:linear-gradient(135deg,#1e2749 0%,#4a6fa5 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.win-card    { background:linear-gradient(135deg,#1e2749 0%,#2f3e6b 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.etf-ticker  { font-size:1.3rem; font-weight:bold; }
.etf-score   { font-size:0.88rem; margin-top:0.25rem; opacity:0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌀 EDMD Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Williams, Kevrekidis & Rowley (2015) Extended Dynamic Mode Decomposition · '
    'Rich dictionary (monomials + RBFs + delay embedding), regularized least-squares Koopman fit · '
    'No training — pure linear algebra, eigenvalues give growth/decay rates directly · '
    'Multi-window cross-sectional z-score</div>',
    unsafe_allow_html=True)

st.sidebar.markdown("## EDMD Engine")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(
    f"**Dictionary:** delay dim={config.DELAY_DIM} | RBF centers={config.N_RBF_CENTERS} | "
    f"regularization={config.REGULARIZATION}")
st.sidebar.markdown(
    f"**Forecast:** horizon={config.PRED_HORIZON}d | "
    f"clips: forecast=±{config.FORECAST_CLIP_STD}σ, growth_rate=±{config.GROWTH_RATE_CLIP}")
st.sidebar.markdown(
    f"**Weights:** Forecast {config.WEIGHT_FORECAST:.0%} | "
    f"Growth {config.WEIGHT_GROWTH:.0%} | "
    f"Fit {config.WEIGHT_FIT:.0%}")

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        return [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                          detail=True, recursive=True)
                if f["type"] == "file"]
    except Exception as e:
        return [f"Error: {e}"]


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f],
                     reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


files     = list_repo_files()
tab1_path = find_latest(files, "edmd_engine_2")
tab2_path = find_latest(files, "edmd_engine_windows_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — Koopman Spectral Forecast Signal")

    with st.expander("EDMD Methodology", expanded=True):
        st.markdown("""
**The Koopman operator** acts linearly on observables of a (possibly
nonlinear) dynamical system, even when the underlying dynamics itself
isn't linear. Standard DMD approximates it using only the raw state —
implicitly assuming linear dynamics. **EDMD** lifts the state into a much
richer dictionary of nonlinear observables:

```
psi(x) = [1, x_1..x_d, x_1^2..x_d^2, x_i*x_j (i<j), RBF_1(x)..RBF_M(x)]
```

and fits a matrix A via regularized least squares such that
`psi(x_{t+1}) ~= A @ psi(x_t)`. DMD is recovered exactly as the special
case where the dictionary is only the linear terms — EDMD is a strict,
provably better generalization whenever the true dynamics are nonlinear.

**Delay embedding** (Takens' theorem): the state is the last `DELAY_DIM`
lagged returns, not just today's value alone — this is what lets a scalar
return series be treated as a genuine reconstructed dynamical system.

**Spectral decomposition:** eigendecompose `A = V diag(lambda) V^-1`.
Each eigenvalue directly gives a growth/decay rate: `|lambda| > 1` is an
expanding/trending mode, `|lambda| < 1` is contracting/mean-reverting,
`arg(lambda)` gives an oscillation frequency. This is literal spectral
analysis of the dynamics, not a statistical fit interpreted after the fact.

**No training.** The entire model is one regularized least-squares solve
plus one eigendecomposition of a small matrix — no gradient descent, no
epochs.

**Signal:**

```
score = 0.50*forecast_signal + 0.30*growth_rate*sign(forecast_signal) + 0.20*fit_quality
```

- `forecast_signal` — mean of the H-step-ahead return path, propagated
  purely via linear evolution in Koopman/spectral space
- `growth_rate` — weighted-average log|eigenvalue|, weighted by each
  mode's relevance to reconstructing the return observable right now
- `fit_quality` — R² of the EDMD regression itself: a natural byproduct
  of the fit, not an auxiliary add-on

**Known pitfall:** a rich dictionary fit on finite noisy samples can
produce spurious eigenvalues far from anything physically plausible for
daily returns — `forecast_signal` and `growth_rate` are clipped as a
safety net against exactly that, not as a tuning knob.
        """)

    for universe_name, uni_data in universes1.items():
        top_etfs = uni_data.get("top_etfs", [])
        if not top_etfs:
            continue
        st.markdown(
            f'<div class="uni-title">{universe_name.replace("_"," ").title()}</div>',
            unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="etf-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">EDMD score = {etf['edmd_score']:.4f}</div>
  <div class="etf-score">best window = {etf.get('best_window','N/A')}d</div>
  <div class="etf-score">growth rate = {etf.get('growth_rate', float('nan')):.3f}</div>
  <div class="etf-score">fit quality = {etf.get('fit_quality', float('nan')):.2f}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {universe_name}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    rows.append({
                        "ETF": t,
                        "EDMD Score": info.get("score"),
                        "Best Window (d)": info.get("best_window", "N/A"),
                        "Forecast Signal": info.get("forecast_signal"),
                        "Growth Rate": info.get("growth_rate"),
                        "Fit Quality": info.get("fit_quality"),
                    })
                df = pd.DataFrame(rows).sort_values("EDMD Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(
        f"Run date: {data1.get('run_date','?')} · "
        "Williams, Kevrekidis & Rowley (2015) EDMD · "
        "Scores are cross-sectional z-scores.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore EDMD Rankings by Window")

    if not universes2:
        st.warning("Window-level detail not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data available.")
        st.stop()

    default_idx  = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: f"{w}d  (~{round(w/21)} months)",
    )
    win_key = str(selected_win)

    with st.expander("Window guidance", expanded=False):
        st.markdown("""
- **63d** — few snapshots relative to dictionary size; Gram matrix regression is less stable; reactive, noisier
- **126d** — 6-month window; recommended minimum for a well-conditioned fit
- **252d** — 1-year window; most stable spectral decomposition; recommended primary signal
- **504d** — 2-year window; structural regime dynamics; slow-moving signal, may blend multiple regimes
        """)

    st.markdown(f"### EDMD Rankings at **{selected_win}d** window")

    for universe_name in ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]:
        label = {
            "FI_COMMODITIES": "🏦 FI & Commodities",
            "EQUITY_SECTORS": "📈 Equity Sectors",
            "COMBINED":       "🌐 Combined",
        }.get(universe_name, universe_name)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_etfs", [])):
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">EDMD score = {etf['edmd_score']:.4f}</div>
  <div class="etf-score">window = {selected_win}d</div>
  <div class="etf-score">growth rate = {etf.get('growth_rate', float('nan')):.3f}</div>
  <div class="etf-score">fit quality = {etf.get('fit_quality', float('nan')):.2f}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df = pd.DataFrame(
                    rows,
                    columns=["ETF", "EDMD Score", "Forecast Signal", "Growth Rate", "Fit Quality"],
                )
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
