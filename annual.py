import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import RerunException, RerunData
import plotly.graph_objects as go
from textwrap import dedent
import inspect

PRETTY_LBM = {'LBM 100E':'100% Equity','LBM 90E':'90% Equity','LBM 80E':'80% Equity','LBM 70E':'70% Equity',
              'LBM 60E':'60% Equity','LBM 50E':'50% Equity','LBM 40E':'40% Equity','LBM 30E':'30% Equity',
              'LBM 20E':'20% Equity','LBM 10E':'10% Equity','LBM 100F':'100% Fixed'}
PRETTY_SPX = {f"spx{p}e": f"{p}% Equity" for p in [100,90,80,70,60,50,40,30,20,10,0]}
GENERIC_ORDER = ["100% Equity","90% Equity","80% Equity","70% Equity","60% Equity","50% Equity",
                 "40% Equity","30% Equity","20% Equity","10% Equity","100% Fixed"]
CPI_FILE_PATH = "cpi_mo_returns_factors.xlsx"

##############################
# App: Required Annual Invest
# Goal: For each allocation column in a worksheet, compute the required
#       constant annual contribution to reach a target (e.g., $1,000,000)
#       with a chosen confidence over N years, using historical factors.
##############################

###

st.set_page_config(layout="wide")
st.title("Required Annual Investment by Allocation (Worksheet-Driven)")
st.caption("Computes the annual contribution required to reach BOTH an Ideal Goal (at its confidence) AND an Acceptable Goal (at 100%), using historical factor windows.")

# ------------------------------
# Inputs (you can change defaults)
# ------------------------------
file_path = "global_factors.xlsx"
sheet_name = "global_factors"
spx_file_path = "spx_factors.xlsx"
spx_sheet_name = "spx_factors"

sb = st.sidebar
current_alloc_choice_global = None
current_alloc_choice_spx = None
annual_alloc_choice_global = None
annual_alloc_choice_spx = None
data_choice = sb.selectbox(
    "Data source",
    ["Global Equity", "S&P 500", "Both Global & SP500"],
    index=0,
    help="Choose the factor set: LBM workbook (Excel) or S&P 500 workbook (spx_factors.xlsx).",
)
returns_basis = sb.radio(
    "Returns basis",
    ["Real", "Nominal"],
    index=1,
    help=(
        "Real returns are already inflation-adjusted. "
        "Nominal returns multiply each historical window by CPI inflation factors "
        f"({CPI_FILE_PATH}) to reflect the dollar value of past periods."
    ),
)
nominal_mode = (returns_basis == "Nominal")
num_years = sb.number_input("Years", min_value=1, max_value=60, value=6)
current_portfolio_value = sb.number_input(
    "Current Portfolio Value ($)", min_value=0, step=50_000, value=1_100_000,
    help="How much is already invested today. It compounds through the historical windows alongside new contributions.",
    format="%i"
)
annual_invest_sim = sb.number_input(
    "Annual investment to simulate ($)",
    min_value=0,
    value=240_000,
    step=25_000,
    format="%i",
    help="Used for the percentile chart (no floor is added)."
)
fee_pct = sb.slider(
    "Annual fee (%)", min_value=0.0, max_value=1.0, value=0.20, step=0.1,
    help="Applied once per 12-month factor: net = gross × (1 − fee). 0.20 % = 20 basis points."
)


row_increment = 12  # Data is monthly, so step 12 rows per year

st.divider()
# Load factors (LBM Excel or SPX Excel)
if data_choice.startswith("Both"):
    src_kind = "BOTH"
elif data_choice.startswith("Global"):
    src_kind = "LBM"
else:
    src_kind = "SPX"

df_lbm, df_spx = None, None
df_spx_base = None
earliest_begin_month = None
spx_needed = src_kind in ("SPX", "BOTH") or nominal_mode
if src_kind in ("LBM", "BOTH"):
    try:
        df_lbm = pd.read_excel(file_path, sheet_name=sheet_name)
        df_lbm = df_lbm.dropna(how="all").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading LBM factors: {e}")
if spx_needed:
    try:
        df_spx_base = pd.read_excel(spx_file_path, sheet_name=spx_sheet_name)
        df_spx_base = df_spx_base.dropna(how="all").reset_index(drop=True)
        try:
            earliest_begin_month = pd.to_datetime(df_spx_base["begin month"]).min()
            if pd.notna(earliest_begin_month):
                earliest_begin_month = earliest_begin_month.strftime("%Y-%m")
            else:
                earliest_begin_month = None
        except Exception:
            earliest_begin_month = None
        if src_kind in ("SPX", "BOTH"):
            df_spx = df_spx_base.copy()
    except Exception as e:
        if src_kind in ("SPX", "BOTH"):
            st.error(f"Error loading SPX factors: {e}")
        elif nominal_mode:
            st.error(f"Nominal returns require SPX data for CPI alignment: {e}")
def _load_cpi_monthly(path):
    """Return a cleaned Date/inflation-factor frame or None if it cannot be built."""
    try:
        df = pd.read_excel(path)
    except Exception:
        return None
    df = df.rename(columns=lambda c: str(c).strip())
    if "Date" not in df.columns or "cpi" not in df.columns:
        return None
    df = df[["Date", "cpi"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["cpi"] = pd.to_numeric(df["cpi"], errors="coerce")
    df = df.dropna(subset=["Date", "cpi"])
    if df.empty:
        return None
    df = df.sort_values("Date").reset_index(drop=True)
    df["inflation_factor"] = 1.0 + df["cpi"]
    return df[["Date", "inflation_factor"]]


def _build_nominal_inflation_series(spx_dates_df, cpi_df):
    """Return inflation-factor series aligned with the SPX window rows or None."""
    if spx_dates_df is None or cpi_df is None:
        return None
    if "begin month" not in spx_dates_df.columns or "end month" not in spx_dates_df.columns:
        return None
    begins = pd.to_datetime(spx_dates_df["begin month"], errors="coerce")
    ends = pd.to_datetime(spx_dates_df["end month"], errors="coerce")
    monthly = cpi_df[["Date", "inflation_factor"]].copy()
    result = []
    for start, end in zip(begins, ends):
        if pd.isna(start) or pd.isna(end):
            result.append(np.nan)
            continue
        window = monthly[(monthly["Date"] >= start) & (monthly["Date"] <= end)]
        if window.shape[0] != 12:
            result.append(np.nan)
            continue
        result.append(float(window["inflation_factor"].prod()))
    return pd.Series(result, index=spx_dates_df.index)


def _apply_nominal_adjustment(df, cols, inflation_series, label):
    """Multiply the requested columns by CPI inflation factors if any are available."""
    if df is None or not cols or inflation_series is None:
        return False
    aligned = inflation_series.reindex(df.index)
    if aligned.isna().all():
        return False
    df.loc[:, cols] = df[cols].multiply(aligned, axis=0)
    return True
allocation_cols_lbm, allocation_cols_spx = [], []
allocation_meta_lbm, allocation_meta_spx = [], []
if df_lbm is not None:
    df_lbm.columns = df_lbm.columns.astype(str).str.strip().str.replace("  ", " ")
    allocation_cols_lbm = [c for c in df_lbm.columns if c.upper().startswith("LBM ")]
    for c in allocation_cols_lbm:
        df_lbm[c] = pd.to_numeric(df_lbm[c], errors='coerce')
        allocation_meta_lbm.append({"raw": c, "clean": c.strip()})
if df_spx is not None:
    df_spx.columns = df_spx.columns.astype(str).str.strip().str.replace("  ", " ")
    allocation_cols_spx = [c for c in df_spx.columns if c.upper().startswith("SPX")]
    for c in allocation_cols_spx:
        df_spx[c] = pd.to_numeric(df_spx[c], errors='coerce')
        allocation_meta_spx.append({"raw": c, "clean": c.strip()})
if src_kind in ("LBM","BOTH") and not allocation_cols_lbm:
    st.warning("No allocation columns found in LBM (expected headers starting with 'LBM ').")
if src_kind in ("SPX","BOTH") and not allocation_cols_spx:
    st.warning("No allocation columns found in SPX (expected headers like 'spx60e', 'spx40e', etc.).")

inflation_series = None
if nominal_mode:
    if df_spx_base is None:
        st.error("Nominal returns require SPX factor windows (spx_factors.xlsx) for CPI alignment.")
    else:
        cpi_df = _load_cpi_monthly(CPI_FILE_PATH)
        if cpi_df is None:
            st.error(f"Could not load CPI data from {CPI_FILE_PATH}; nominal returns stay disabled.")
        else:
            inflation_series = _build_nominal_inflation_series(df_spx_base, cpi_df)
            if inflation_series is None or inflation_series.dropna().empty:
                st.warning("Nominal returns are enabled but CPI coverage does not span the available windows.")
    applied_nominal = False
    if inflation_series is not None:
        if df_lbm is not None:
            applied_nominal |= _apply_nominal_adjustment(df_lbm, allocation_cols_lbm, inflation_series, "Global")
        if df_spx is not None:
            applied_nominal |= _apply_nominal_adjustment(df_spx, allocation_cols_spx, inflation_series, "SP500")
        if not applied_nominal and (df_lbm is not None or df_spx is not None):
            st.warning("Nominal returns requested but no windows matched the CPI data; results remain real.")

def _meta_by_clean(metas, clean):
    if not metas or clean is None:
        return None
    for meta in metas:
        if meta["clean"] == clean:
            return meta
    return None

def _clean_list(metas):
    return [m["clean"] for m in metas] if metas else []

def _sync_state(key, options):
    pending_key = f"pending_{key}"
    if pending_key in st.session_state:
        st.session_state[key] = st.session_state.pop(pending_key)
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = options[0] if options else None

options_curr_global = _clean_list(allocation_meta_lbm)
options_curr_spx = _clean_list(allocation_meta_spx)
options_ann_global = _clean_list(allocation_meta_lbm)
options_ann_spx = _clean_list(allocation_meta_spx)

if options_curr_global:
    _sync_state("current_alloc_global", options_curr_global)
if options_curr_spx:
    _sync_state("current_alloc_spx", options_curr_spx)
if options_ann_global:
    _sync_state("annual_alloc_global", options_ann_global)
if options_ann_spx:
    _sync_state("annual_alloc_spx", options_ann_spx)

def _rerun():
    raise RerunException(RerunData(None))

_PLOTLY_SUPPORTS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters

def _render_plotly(fig):
    kwargs = {}
    if _PLOTLY_SUPPORTS_WIDTH:
        kwargs["width"] = "stretch"
    else:
        kwargs["use_container_width"] = True
    st.plotly_chart(fig, **kwargs)

def _fmt_currency(val):
    if val is None:
        return "N/A"
    try:
        if not np.isfinite(val):
            return "N/A"
    except TypeError:
        return "N/A"
    return f"${val:,.0f}"

CPI_FILE_PATH = "cpi_mo_returns_factors.xlsx"

def _load_cpi_monthly(path):
    """Return a cleaned Date/inflation-factor frame or None if it cannot be built."""
    try:
        df = pd.read_excel(path)
    except Exception:
        return None
    df = df.rename(columns=lambda c: str(c).strip())
    if "Date" not in df.columns or "cpi" not in df.columns:
        return None
    df = df[["Date", "cpi"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["cpi"] = pd.to_numeric(df["cpi"], errors="coerce")
    df = df.dropna(subset=["Date", "cpi"])
    if df.empty:
        return None
    df = df.sort_values("Date").reset_index(drop=True)
    df["inflation_factor"] = 1.0 + df["cpi"]
    return df[["Date", "inflation_factor"]]


def _build_nominal_inflation_series(spx_dates_df, cpi_df):
    """Return inflation-factor series aligned with the SPX window rows or None."""
    if spx_dates_df is None or cpi_df is None:
        return None
    if "begin month" not in spx_dates_df.columns or "end month" not in spx_dates_df.columns:
        return None
    begins = pd.to_datetime(spx_dates_df["begin month"], errors="coerce")
    ends = pd.to_datetime(spx_dates_df["end month"], errors="coerce")
    monthly = cpi_df[["Date", "inflation_factor"]].copy()
    result = []
    for start, end in zip(begins, ends):
        if pd.isna(start) or pd.isna(end):
            result.append(np.nan)
            continue
        window = monthly[(monthly["Date"] >= start) & (monthly["Date"] <= end)]
        if window.shape[0] != 12:
            result.append(np.nan)
            continue
        result.append(float(window["inflation_factor"].prod()))
    return pd.Series(result, index=spx_dates_df.index)


def _apply_nominal_adjustment(df, cols, inflation_series, label):
    """Multiply the requested columns by CPI inflation factors if any are available."""
    if df is None or not cols or inflation_series is None:
        return False
    aligned = inflation_series.reindex(df.index)
    if aligned.isna().all():
        return False
    df.loc[:, cols] = df[cols].multiply(aligned, axis=0)
    return True

if ((src_kind in ("LBM","BOTH") and allocation_meta_lbm) or
    (src_kind in ("SPX","BOTH") and allocation_meta_spx)):
    sb.markdown("### Current Portfolio Allocation")
    sb.caption("Choose the mix that represents how your existing money is invested today.")
    if src_kind in ("LBM","BOTH") and allocation_meta_lbm:
        current_alloc_choice_global_clean = sb.selectbox(
            "Global data",
            options=options_curr_global,
            format_func=lambda clean: PRETTY_LBM.get(clean, clean),
            key="current_alloc_global"
        )
        current_alloc_choice_global = _meta_by_clean(allocation_meta_lbm, current_alloc_choice_global_clean)
    if src_kind in ("SPX","BOTH") and allocation_meta_spx:
        current_alloc_choice_spx_clean = sb.selectbox(
            "S&P 500 data",
            options=options_curr_spx,
            format_func=lambda clean: PRETTY_SPX.get(clean, clean),
            key="current_alloc_spx"
        )
        current_alloc_choice_spx = _meta_by_clean(allocation_meta_spx, current_alloc_choice_spx_clean)

if ((src_kind in ("LBM","BOTH") and allocation_meta_lbm) or
    (src_kind in ("SPX","BOTH") and allocation_meta_spx)):
    sb.markdown("### Annual Contribution Allocation")
    sb.caption("Pick the mix for new contributions going in each year.")
    if src_kind in ("LBM","BOTH") and allocation_meta_lbm:
        annual_alloc_choice_global_clean = sb.selectbox(
            "Global data (annual)",
            options=options_ann_global,
            format_func=lambda clean: PRETTY_LBM.get(clean, clean),
            key="annual_alloc_global"
        )
        annual_alloc_choice_global = _meta_by_clean(allocation_meta_lbm, annual_alloc_choice_global_clean)
    if src_kind in ("SPX","BOTH") and allocation_meta_spx:
        annual_alloc_choice_spx_clean = sb.selectbox(
            "S&P 500 data (annual)",
            options=options_ann_spx,
            format_func=lambda clean: PRETTY_SPX.get(clean, clean),
            key="annual_alloc_spx"
        )
        annual_alloc_choice_spx = _meta_by_clean(allocation_meta_spx, annual_alloc_choice_spx_clean)

fee = float(fee_pct)/100.0
current_portfolio_value = float(current_portfolio_value)
if fee > 0:
    if df_lbm is not None and allocation_cols_lbm:
        df_lbm[allocation_cols_lbm] = df_lbm[allocation_cols_lbm] * (1.0 - fee)
    if df_spx is not None and allocation_cols_spx:
        df_spx[allocation_cols_spx] = df_spx[allocation_cols_spx] * (1.0 - fee)

# ------------------------------
# Core math
# ------------------------------

def _quantile_linear(arr: np.ndarray, q: float) -> float:
    """Helper that safely computes a lower-tail quantile using linear interpolation."""
    if arr.size == 0:
        return float('nan')
    try:
        return float(np.quantile(arr, q, method="linear"))
    except TypeError:
        return float(np.quantile(arr, q, interpolation="linear"))


def simulate_annuity_and_lumpsum(factors: pd.Series, years: int, step: int) -> tuple[list, list]:
    """Return ending values per $1 for both annuity contributions and a lump sum."""
    annuity_vals, lump_vals = [], []
    n = len(factors)
    max_start = n - (step * (years - 1))
    if max_start <= 0:
        return annuity_vals, lump_vals
    for start in range(max_start):
        inv = 0.0
        lump = 1.0
        valid = True
        for y in range(years):
            idx = start + y * step
            f = factors.iloc[idx]
            if pd.isna(f) or f <= 0:
                valid = False
                break
            inv = (inv + 1.0) * float(f)
            lump *= float(f)
        if valid:
            annuity_vals.append(inv)
            lump_vals.append(lump)
    return annuity_vals, lump_vals


def simulate_ending_values_annuity(factors: pd.Series, years: int, step: int) -> list:
    """For each possible start row, simulate the ending value of an annuity
    that contributes $1 at the beginning of each year for 'years' years,
    compounding by the factor at each step window. Skips windows containing NaNs.

    Returns a list of ending values (one per valid start window).
    """
    vals, _ = simulate_annuity_and_lumpsum(factors, years, step)
    return vals


def required_annual_for_goal(ending_values: list, goal_amount: float, conf: float) -> float:
    """Given ending values per $1 contributed annually, return the annual contribution
    needed to hit goal_amount at the specified confidence. Uses lower-tail quantile
    with linear interpolation to avoid collapsing to a single worst-window."""
    goal_amount = float(goal_amount)
    if goal_amount <= 0:
        return 0.0
    if ending_values is None:
        return float('nan')
    arr = np.array(ending_values, dtype=float)
    if arr.size == 0:
        return float('nan')
    arr = np.sort(arr)
    q = max(0.0, min(1.0, 1.0 - float(conf)))
    ev = _quantile_linear(arr, q)
    if ev <= 0 or not np.isfinite(ev):
        return float('inf')
    return float(goal_amount) / float(ev)

# ------------------------------
# Precompute annuity/lump caches
# ------------------------------

calc_cache = {"Global": {}, "SP500": {}}
lump_cache = {"Global": {}, "SP500": {}}

def _build_caches(df_src, metas, source_label):
    if df_src is None or not metas:
        return
    for meta in metas:
        col_raw = meta["raw"]
        col_clean = meta["clean"]
        evs, lumps = simulate_annuity_and_lumpsum(df_src[col_raw], int(num_years), int(row_increment))
        calc_cache[source_label][col_clean] = {"evs": np.array(evs, dtype=float)}
        lump_cache[source_label][col_clean] = np.array(lumps, dtype=float)

_build_caches(df_lbm, allocation_meta_lbm, "Global")
_build_caches(df_spx, allocation_meta_spx, "SP500")

# ------------------------------
# Scenario percentile chart (primary output)
# ------------------------------
st.header("Outcome Percentiles")
st.caption("Ending value percentiles across historical windows. No floor added; uses the annual amount below.")

annual_amount = float(annual_invest_sim or 0.0)
if annual_amount <= 0:
    st.warning("Enter a positive annual investment to plot outcomes.")

def _options_for_source(source_label):
    return allocation_meta_lbm if source_label == "Global" else allocation_meta_spx

def _scenario_outcomes(source_label, current_clean, annual_clean):
    metas = _options_for_source(source_label)
    if not metas:
        return None, None
    current_meta = _meta_by_clean(metas, current_clean)
    annual_meta = _meta_by_clean(metas, annual_clean)
    if not current_meta or not annual_meta:
        return None, None
    evs_arr = calc_cache[source_label].get(annual_meta["clean"], {}).get("evs")
    lumps_arr = lump_cache[source_label].get(current_meta["clean"])
    if evs_arr is None or lumps_arr is None:
        return None, None
    n = min(len(evs_arr), len(lumps_arr))
    if n <= 0:
        return None, None
    evs_slice = np.array(evs_arr[:n], dtype=float)
    lumps_slice = np.array(lumps_arr[:n], dtype=float)
    mask = np.isfinite(evs_slice) & np.isfinite(lumps_slice)
    if not mask.any():
        return None, None
    evs_slice = evs_slice[mask]
    lumps_slice = lumps_slice[mask]
    idxs = np.arange(n)[mask]
    outcomes = lumps_slice * float(current_portfolio_value) + evs_slice * annual_amount
    finite_mask = np.isfinite(outcomes)
    if not finite_mask.any():
        return None, None
    outcomes = outcomes[finite_mask]
    idxs = idxs[finite_mask]
    return outcomes, idxs

scenarios = []
source_options = []
if allocation_meta_lbm:
    source_options.append("Global")
if allocation_meta_spx:
    source_options.append("SP500")
if not source_options:
    st.error("No allocation data available to run scenarios.")
    st.stop()

selected_source = sb.selectbox(
    "Source for all scenarios",
    options=source_options,
    index=0,
)
metas_for_source = _options_for_source(selected_source)
clean_options = _clean_list(metas_for_source)
default_percents = {1: 100, 2: 80, 3: 60}

def _default_clean_for(source_label: str, pct: int) -> str:
    if source_label == "Global":
        return f"LBM {pct}E"
    return f"spx{pct}e"

chart_view = sb.radio(
    "Chart view",
    ["Percentile slope"],
    index=0,
)

for idx in range(1, 4):
    metas = metas_for_source
    default_clean = _default_clean_for(selected_source, default_percents.get(idx, 100))
    try:
        default_index = clean_options.index(default_clean)
    except ValueError:
        default_index = 0 if clean_options else None
    current_clean = sb.selectbox(
        f"Scenario {idx} current allocation",
        options=clean_options,
        format_func=lambda c: PRETTY_LBM.get(c, PRETTY_SPX.get(c, c)),
        index=default_index if default_index is not None else 0,
        key=f"sc{idx}_current"
    ) if clean_options else None
    annual_clean = sb.selectbox(
        f"Scenario {idx} annual allocation",
        options=clean_options,
        format_func=lambda c: PRETTY_LBM.get(c, PRETTY_SPX.get(c, c)),
        index=default_index if default_index is not None else 0,
        key=f"sc{idx}_annual"
    ) if clean_options else None
    scenarios.append({
        "label": f"Scenario {idx}",
        "source": selected_source,
        "current_clean": current_clean,
        "annual_clean": annual_clean,
    })

band_traces = []
cdf_traces = []
box_traces = []
perc_marker_traces = []
slope_traces = []
low_traces = []
outcomes_map = {}
percentiles = np.linspace(0, 1, 101)
table_percentiles = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
table_rows = []
colors = ["#1f77b4", "#d62728", "#2ca02c"]
def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(31,119,180,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

for sc_idx, sc in enumerate(scenarios):
    name = f"{sc['label']} ({sc['source']})"
    outcomes, idxs = _scenario_outcomes(sc["source"], sc["current_clean"], sc["annual_clean"])
    if outcomes is None:
        continue
    outcomes_map[name] = outcomes
    values = np.quantile(outcomes, percentiles)
    tab_vals = np.quantile(outcomes, table_percentiles)
    color = colors[sc_idx % len(colors)] if colors else None
    lower_val = tab_vals[2]  # P10
    upper_val = tab_vals[6]  # P90
    median_val = tab_vals[4]
    # Band
    band_traces.append(go.Scatter(
        x=[0, 100],
        y=[lower_val, lower_val],
        line=dict(color=color, width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    band_traces.append(go.Scatter(
        x=[0, 100],
        y=[upper_val, upper_val],
        line=dict(color=color, width=0),
        fill="tonexty",
        fillcolor=_rgba(color if color else "#1f77b4", 0.18),
        showlegend=False,
        hoverinfo="skip",
    ))
    # Median line
    band_traces.append(go.Scatter(
        x=[0, 100],
        y=[median_val, median_val],
        mode="lines",
        line=dict(color=color, width=2),
        name=name,
    ))
    # CDF trace
    sorted_outcomes = np.sort(outcomes)
    if sorted_outcomes.size:
        probs = np.linspace(0, 100, sorted_outcomes.size)
        cdf_traces.append(go.Scatter(
            x=sorted_outcomes,
            y=probs,
            mode="lines",
            name=name,
            line=dict(color=color),
        ))
    # Box/violin (box for simplicity)
    box_traces.append(go.Box(
        y=outcomes,
        name=name,
        marker_color=color,
        boxmean=True,
        hovertemplate=f"{name}<br>%{{y:,.0f}}<extra></extra>",
    ))
    # Percentile slope points
    slope_labels = ["P0","P5","P10","P25","P50","P75","P90","P95","P100"]
    slope_values = [
        tab_vals[0], tab_vals[1], tab_vals[2], tab_vals[3], tab_vals[4],
        tab_vals[5], tab_vals[6], tab_vals[7], tab_vals[8]
    ]
    slope_traces.append(go.Scatter(
        x=slope_labels,
        y=slope_values,
        mode="lines+markers",
        name=name,
        line=dict(color=color),
    ))
    low_traces.append(go.Scatter(
        x=["P0","P5","P10"],
        y=[tab_vals[0], tab_vals[1], tab_vals[2]],
        mode="lines+markers",
        name=name,
        line=dict(color=color),
    ))
    # Box with jittered points
    box_traces.append(go.Box(
        y=outcomes,
        name=name,
        marker_color=color,
        boxmean=True,
        boxpoints="all",
        jitter=0.4,
        pointpos=0,
        marker=dict(opacity=0.2, size=4, color=color),
        hovertemplate=f"{name}<br>%{{y:,.0f}}<extra></extra>",
    ))
    p5, p25, p50, p75, p95 = np.percentile(outcomes, [5, 25, 50, 75, 95])
    perc_marker_traces.append(go.Scatter(
        x=[name]*5,
        y=[p5, p25, p50, p75, p95],
        mode="markers",
        showlegend=False,
        marker=dict(
            color=color,
            symbol=["triangle-down","line-ns-open","diamond","line-ns-open","triangle-up"],
            size=[10,8,10,8,10],
            line=dict(color=color, width=1.5),
        ),
        hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
    ))
    p0_begin = None
    if sc["source"] == "SP500" and idxs is not None and idxs.size > 0 and df_spx_base is not None:
        try:
            min_idx = int(idxs[np.argmin(outcomes)])
            if 0 <= min_idx < len(df_spx_base):
                p0_begin = min_idx + 1  # row number (1-based) from the SPX factor table
        except Exception:
            p0_begin = None
    table_rows.append({
        "Scenario": sc["label"],
        "Source": sc["source"],
        "P0": tab_vals[0],
        "P0 Begin": p0_begin,
        "P5": tab_vals[1],
        "P10": tab_vals[2],
        "P25": tab_vals[3],
        "P50": tab_vals[4],
        "P75": tab_vals[5],
        "P90": tab_vals[6],
        "P95": tab_vals[7],
        "P100": tab_vals[8],
    })

if slope_traces and annual_amount > 0:
    fig = go.Figure(slope_traces)
    fig.update_layout(
        xaxis_title="All Percentiles",
        yaxis_title="Ending value ($)",
        yaxis_tickformat=",.0f",
        template="plotly_white",
    )
    _render_plotly(fig)
    # Low-percentile inset
    if low_traces:
        fig_low = go.Figure(low_traces)
        fig_low.update_layout(
            xaxis_title="Zooming intoLower percentiles",
            yaxis_title="Ending value ($)",
            yaxis_tickformat=",.0f",
            template="plotly_white",
        )
        _render_plotly(fig_low)
else:
    st.info("Select allocations and enter a positive annual investment to see the slope view.")

if table_rows and annual_amount > 0:
    tbl = pd.DataFrame(table_rows)
    for col in ["P0","P5","P10","P25","P50","P75","P90","P95","P100"]:
        tbl[col] = tbl[col].apply(_fmt_currency)
    tbl = tbl.rename(columns={"P0 Begin": "P0 Begin Row"})
    st.subheader("Scenario Percentiles")
    st.table(tbl)

st.stop()
