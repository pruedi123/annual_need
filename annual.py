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

col1, col2, col3 = st.columns(3)
current_alloc_choice_global = None
current_alloc_choice_spx = None
annual_alloc_choice_global = None
annual_alloc_choice_spx = None
with col1:
    data_choice = st.selectbox(
        "Data source",
        ["Global Equity", "S&P 500", "Both Global & SP500"],
        index=0,
        help="Choose the factor set: LBM workbook (Excel) or S&P 500 workbook (spx_factors.xlsx).",
    )
    ideal_goal = st.number_input("Ideal Goal ($)", min_value=1, step=50000, value=2_500_000,
                                 help="Today’s dollars: same buying power as money today.",
                                 format="%i")
    conf_pct_ideal = st.slider("Ideal Confidence (%)", min_value=50, step= 10, max_value=100, value=90,
                               help="e.g., 90% means ≥90% of historical windows finish at/above the Ideal Goal.")
    ideal_conf_level = conf_pct_ideal / 100.0
with col2:
    num_years = st.number_input("Years", min_value=1, max_value=60, value=4)
    acceptable_goal = st.number_input("Essential Goal ($)", min_value=1, step=50000, value=2_000_000,
                                      help="A minimum acceptable outcome (floor) which means that you do not want less than this amount.",
                                      format="%i")
    current_portfolio_value = st.number_input(
        "Current Portfolio Value ($)", min_value=0, step=50_000, value=2000000,
        help="How much is already invested today. It compounds through the historical windows alongside new contributions.",
        format="%i"
    )
    current_conf_pct = st.slider(
        "Current Portfolio Confidence (%)",
        min_value=50, max_value=100, value=99, step=1,
        help=(
            "Think of this as how conservative you want to be with the money you already have. "
            "At 100%, you only credit yourself with the very worst historical outcome. "
            "At 99%, you say “I’m comfortable assuming my current savings will at least match what happened in 99 out of 100 similar periods,” "
            "or in other words, only 1% of historical historical periods tested would have produced less than this amount. "
            "which gives more credit to today’s balance and shrinks the required annual contribution."
        ),
    )
    acceptable_conf_level = 1.0  # fixed 100%
with col3:
    fee_pct = st.slider("Annual fee (%)", min_value=0.0, max_value=1.0, value=0.20, step=0.1,
                        help="Applied once per 12-month factor: net = gross × (1 − fee).")

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
try:
    if src_kind in ("LBM", "BOTH"):
        df_lbm = pd.read_excel(file_path, sheet_name=sheet_name)
except Exception as e:
    st.error(f"Error loading LBM factors: {e}")
try:
    if src_kind in ("SPX", "BOTH"):
        df_spx = pd.read_excel(spx_file_path, sheet_name=spx_sheet_name)
except Exception as e:
    st.error(f"Error loading SPX factors: {e}")

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

if ((src_kind in ("LBM","BOTH") and allocation_meta_lbm) or
    (src_kind in ("SPX","BOTH") and allocation_meta_spx)):
    st.markdown("#### Current Portfolio Allocation")
    help_text = "Select which allocation the existing portfolio follows. It stays constant over the full horizon."
    st.caption(help_text)
    cols_alloc = st.columns(2)
    if src_kind in ("LBM","BOTH") and allocation_meta_lbm:
        with cols_alloc[0]:
            current_alloc_choice_global_clean = st.selectbox(
                "Global data",
                options=options_curr_global,
                format_func=lambda clean: PRETTY_LBM.get(clean, clean),
                key="current_alloc_global"
            )
            current_alloc_choice_global = _meta_by_clean(allocation_meta_lbm, current_alloc_choice_global_clean)
    if src_kind in ("SPX","BOTH") and allocation_meta_spx:
        target_col = cols_alloc[0] if src_kind == "SPX" else cols_alloc[1]
        with target_col:
            current_alloc_choice_spx_clean = st.selectbox(
                "S&P 500 data",
                options=options_curr_spx,
                format_func=lambda clean: PRETTY_SPX.get(clean, clean),
                key="current_alloc_spx"
            )
            current_alloc_choice_spx = _meta_by_clean(allocation_meta_spx, current_alloc_choice_spx_clean)

if ((src_kind in ("LBM","BOTH") and allocation_meta_lbm) or
    (src_kind in ("SPX","BOTH") and allocation_meta_spx)):
    st.markdown("#### Annual Contribution Allocation")
    st.caption("Choose the allocation applied to ongoing annual contributions for each data source.")
    cols_alloc_ann = st.columns(2)
    if src_kind in ("LBM","BOTH") and allocation_meta_lbm:
        with cols_alloc_ann[0]:
            annual_alloc_choice_global_clean = st.selectbox(
                "Global data (annual)",
                options=options_ann_global,
                format_func=lambda clean: PRETTY_LBM.get(clean, clean),
                key="annual_alloc_global"
            )
            annual_alloc_choice_global = _meta_by_clean(allocation_meta_lbm, annual_alloc_choice_global_clean)
    if src_kind in ("SPX","BOTH") and allocation_meta_spx:
        target_col = cols_alloc_ann[0] if src_kind == "SPX" else cols_alloc_ann[1]
        with target_col:
            annual_alloc_choice_spx_clean = st.selectbox(
                "S&P 500 data (annual)",
                options=options_ann_spx,
                format_func=lambda clean: PRETTY_SPX.get(clean, clean),
                key="annual_alloc_spx"
            )
            annual_alloc_choice_spx = _meta_by_clean(allocation_meta_spx, annual_alloc_choice_spx_clean)

fee = float(fee_pct)/100.0
current_portfolio_value = float(current_portfolio_value)
current_conf_tail = max(0.0, min(1.0, 1.0 - float(current_conf_pct) / 100.0))
ideal_tail = max(0.0, min(1.0, 1.0 - float(ideal_conf_level)))
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
# Current portfolio floor calc
# ------------------------------

lump_floor = {"Global": 0.0, "SP500": 0.0}
lump_label = {"Global": None, "SP500": None}
if current_alloc_choice_global:
    lump_label["Global"] = PRETTY_LBM.get(current_alloc_choice_global["clean"], current_alloc_choice_global["clean"])
if current_alloc_choice_spx:
    lump_label["SP500"] = PRETTY_SPX.get(current_alloc_choice_spx["clean"], current_alloc_choice_spx["clean"])

def _floor_for_choice(source_label, choice_meta):
    if current_portfolio_value <= 0 or not choice_meta:
        return 0.0
    lumps_arr = lump_cache[source_label].get(choice_meta["clean"])
    if lumps_arr is None or lumps_arr.size == 0:
        return 0.0
    floor_factor = _quantile_linear(lumps_arr, current_conf_tail)
    return float(floor_factor * current_portfolio_value)

if current_alloc_choice_global:
    lump_floor["Global"] = _floor_for_choice("Global", current_alloc_choice_global)
if current_alloc_choice_spx:
    lump_floor["SP500"] = _floor_for_choice("SP500", current_alloc_choice_spx)

def _solve_source(source_label, metas):
    if not metas:
        return None
    best = None
    for current_meta in metas:
        floor_val = _floor_for_choice(source_label, current_meta)
        shortfall_ideal = max(float(ideal_goal) - floor_val, 0.0)
        shortfall_accept = max(float(acceptable_goal) - floor_val, 0.0)
        for annual_meta in metas:
            evs_arr = calc_cache[source_label].get(annual_meta["clean"], {}).get("evs")
            if evs_arr is None or evs_arr.size == 0:
                continue
            req_i = required_annual_for_goal(evs_arr, shortfall_ideal, float(ideal_conf_level))
            req_a = required_annual_for_goal(evs_arr, shortfall_accept, 1.0)
            req = max(req_i, req_a)
            if not np.isfinite(req):
                continue
            total_arr = evs_arr * float(req)
            ending_val = float('nan')
            if total_arr.size:
                ending_val = _quantile_linear(np.array(total_arr, dtype=float), ideal_tail) + floor_val
            display_current = PRETTY_LBM.get(current_meta["clean"], current_meta["clean"]) if source_label == "Global" else PRETTY_SPX.get(current_meta["clean"], current_meta["clean"])
            display_annual = PRETTY_LBM.get(annual_meta["clean"], annual_meta["clean"]) if source_label == "Global" else PRETTY_SPX.get(annual_meta["clean"], annual_meta["clean"])
            entry = {
                "Source": source_label,
                "Current Allocation": display_current,
                "Annual Allocation": display_annual,
                "Floor": floor_val,
                "Required Annual": float(req),
                "Ending Value": ending_val,
                "current_clean": current_meta["clean"],
                "annual_clean": annual_meta["clean"],
            }
            if best is None or entry["Required Annual"] < best["Required Annual"]:
                best = entry
    return best

if "solver_rows_display" not in st.session_state:
    st.session_state["solver_rows_display"] = None

solver_cols = st.columns([2, 2, 3])
with solver_cols[0]:
    if st.button("Apply Manual Selection", width="stretch"):
        st.session_state["solver_rows_display"] = None
        _rerun()
with solver_cols[1]:
    run_solver = st.button("Run Solver (Min Required Annual)", width="stretch")
with solver_cols[2]:
    show_outputs = st.radio(
        "View detailed results?",
        options=["Show", "Hide"],
        index=0,
        horizontal=True,
        key="toggle_outputs"
    ) == "Show"

if run_solver:
    solver_rows = []
    updated = False
    if src_kind in ("LBM","BOTH"):
        best_global = _solve_source("Global", allocation_meta_lbm)
        if best_global:
            solver_rows.append(best_global)
            if best_global.get("current_clean"):
                st.session_state["pending_current_alloc_global"] = best_global["current_clean"]
            if best_global.get("annual_clean"):
                st.session_state["pending_annual_alloc_global"] = best_global["annual_clean"]
            updated = True
    if src_kind in ("SPX","BOTH"):
        best_spx = _solve_source("SP500", allocation_meta_spx)
        if best_spx:
            solver_rows.append(best_spx)
            if best_spx.get("current_clean"):
                st.session_state["pending_current_alloc_spx"] = best_spx["current_clean"]
            if best_spx.get("annual_clean"):
                st.session_state["pending_annual_alloc_spx"] = best_spx["annual_clean"]
            updated = True
    if solver_rows:
        st.session_state["solver_rows_display"] = solver_rows
    if updated:
        _rerun()
    else:
        st.warning("Solver could not identify a feasible allocation based on the current inputs.")

# ------------------------------
# Run (auto)
# ------------------------------
have_any = (
    (src_kind in ("LBM","BOTH") and df_lbm is not None and allocation_cols_lbm) or
    (src_kind in ("SPX","BOTH") and df_spx is not None and allocation_cols_spx)
)
if have_any:
    rows = []
    selected_rows = {"Global": None, "SP500": None}
    selected_labels = {"Global": None, "SP500": None}
    # LBM
    if src_kind in ("LBM","BOTH") and allocation_meta_lbm:
        lump_floor_global = float(lump_floor.get("Global", 0.0))
        shortfall_ideal_global = max(float(ideal_goal) - lump_floor_global, 0.0)
        shortfall_accept_global = max(float(acceptable_goal) - lump_floor_global, 0.0)
        for meta in allocation_meta_lbm:
            col_clean = meta["clean"]
            evs_arr = calc_cache["Global"].get(col_clean, {}).get("evs")
            if evs_arr is None:
                continue
            calc_cache["Global"][col_clean] = {"evs": evs_arr}
            if evs_arr.size == 0:
                req = np.nan
                ending_val = np.nan
            else:
                req_i = required_annual_for_goal(evs_arr, shortfall_ideal_global, float(ideal_conf_level))
                req_a = required_annual_for_goal(evs_arr, shortfall_accept_global, float(1.0))
                req = max(req_i, req_a)
                ending_val = np.nan
                if np.isfinite(req):
                    total_arr = evs_arr * float(req)
                    if total_arr.size:
                        ending_val = _quantile_linear(np.array(total_arr, dtype=float), ideal_tail) + lump_floor_global
            rows.append({
                "Allocation": col_clean,
                "Required Annual": np.nan if pd.isna(req) else float(req),
                "Ending Value": np.nan if pd.isna(ending_val) else float(ending_val),
                "Current Portfolio Floor": lump_floor_global,
            })
            if annual_alloc_choice_global and col_clean == annual_alloc_choice_global["clean"]:
                selected_rows["Global"] = rows[-1].copy()
                selected_labels["Global"] = PRETTY_LBM.get(col_clean, col_clean)
    # SPX
    if src_kind in ("SPX","BOTH") and allocation_meta_spx:
        lump_floor_spx = float(lump_floor.get("SP500", 0.0))
        shortfall_ideal_spx = max(float(ideal_goal) - lump_floor_spx, 0.0)
        shortfall_accept_spx = max(float(acceptable_goal) - lump_floor_spx, 0.0)
        for meta in allocation_meta_spx:
            col_clean = meta["clean"]
            evs_arr = calc_cache["SP500"].get(col_clean, {}).get("evs")
            if evs_arr is None:
                continue
            calc_cache["SP500"][col_clean] = {"evs": evs_arr}
            if evs_arr.size == 0:
                req = np.nan
                ending_val = np.nan
            else:
                req_i = required_annual_for_goal(evs_arr, shortfall_ideal_spx, float(ideal_conf_level))
                req_a = required_annual_for_goal(evs_arr, shortfall_accept_spx, float(1.0))
                req = max(req_i, req_a)
                ending_val = np.nan
                if np.isfinite(req):
                    total_arr = evs_arr * float(req)
                    if total_arr.size:
                        ending_val = _quantile_linear(np.array(total_arr, dtype=float), ideal_tail) + lump_floor_spx
            rows.append({
                "Allocation": col_clean,
                "Required Annual": np.nan if pd.isna(req) else float(req),
                "Ending Value": np.nan if pd.isna(ending_val) else float(ending_val),
                "Current Portfolio Floor": lump_floor_spx,
            })
            if annual_alloc_choice_spx and col_clean == annual_alloc_choice_spx["clean"]:
                selected_rows["SP500"] = rows[-1].copy()
                selected_labels["SP500"] = PRETTY_SPX.get(col_clean, col_clean)

    # Build pretty maps and wide table

    results = pd.DataFrame(rows)
    # Determine source for each row and generic label
    def _generic_label(a: str) -> str:
        label = PRETTY_LBM.get(a, PRETTY_SPX.get(a, a))
        # Normalize 0% Equity (spx0e) to the shared row label used in the table
        if isinstance(label, str) and label.strip().startswith("0% Equity"):
            return "100% Fixed"
        return label
    tmp = results.copy()
    tmp["Source"] = np.where(tmp["Allocation"].str.upper().str.startswith("LBM "), "Global",
                             np.where(tmp["Allocation"].str.lower().str.startswith("spx"), "SP500", None))
    tmp["Generic"] = tmp["Allocation"].map(_generic_label)

    def _build_wide(value_col: str) -> pd.DataFrame:
        w = tmp.pivot_table(index="Generic", columns="Source", values=value_col, aggfunc="first")
        w = w.reindex([g for g in GENERIC_ORDER if g in w.index])
        w = w.rename_axis(None, axis=1).reset_index().rename(columns={"Generic":"Allocation"})
        return w

    wide_req = _build_wide("Required Annual")

    # Format for display
    display_results = wide_req.copy()
    for col in ["Global","SP500"]:
        if col in display_results.columns:
            display_results[col] = display_results[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    # Dynamic columns
    cols = ["Allocation"]
    if "Global" in display_results.columns: cols.append("Global")
    if "SP500" in display_results.columns: cols.append("SP500")
    display_results = display_results[cols]

    # Current portfolio floor summary
    if show_outputs and current_portfolio_value > 0:
        floor_rows = []
        floor_col_name = f"Floor ({current_conf_pct:.0f}%)"
        if src_kind in ("LBM","BOTH") and lump_label.get("Global"):
            floor_rows.append({
                "Source": "Global",
                "Allocation": lump_label["Global"],
                floor_col_name: f"${lump_floor.get('Global', 0.0):,.0f}",
            })
        if src_kind in ("SPX","BOTH") and lump_label.get("SP500"):
            floor_rows.append({
                "Source": "SP500",
                "Allocation": lump_label["SP500"],
                floor_col_name: f"${lump_floor.get('SP500', 0.0):,.0f}",
            })
        if floor_rows:
            st.subheader(f"Current Portfolio Floor ({current_conf_pct:.0f}% Confidence)")
            st.caption("Historical outcome for today's balance at the selected confidence, assuming it remains in the chosen allocation.")
            st.table(pd.DataFrame(floor_rows))

    summary_rows = []
    summary_details = []
    for source in ["Global","SP500"]:
        sel = selected_rows.get(source)
        if not sel:
            continue
        req = sel.get("Required Annual")
        end_val = sel.get("Ending Value")
        floor_val = lump_floor.get(source, 0.0)
        current_label = lump_label.get(source) or "—"
        annual_label = selected_labels.get(source) or (sel.get("Allocation") if sel else "—")
        summary_rows.append({
            "Source": source,
            "Annual Allocation": annual_label,
            "Required Annual": "" if req is None or not np.isfinite(req) else f"${req:,.0f}",
            "Current Allocation": current_label,
            f"Current Floor ({current_conf_pct:.0f}%)": f"${floor_val:,.0f}",
            f"Ending Value @ {conf_pct_ideal:.0f}%": "" if end_val is None or not np.isfinite(end_val) else f"${end_val:,.0f}",
        })
        summary_details.append({
            "source": source,
            "current_label": current_label,
            "annual_label": annual_label,
            "floor": floor_val,
            "required": req if req is not None and np.isfinite(req) else None,
            "ending": end_val if end_val is not None and np.isfinite(end_val) else None,
        })
    if show_outputs:
        if summary_rows:
            st.subheader("Selected Allocation Results")
            st.caption("Compares the chosen annual contribution allocation with the current-portfolio floor.")
            st.table(pd.DataFrame(summary_rows))
        else:
            st.info("Select at least one annual contribution allocation above to view results.")

    solver_rows_display = st.session_state.get("solver_rows_display")
    if show_outputs and solver_rows_display:
        solver_df = pd.DataFrame(solver_rows_display)
        for col in ["Floor","Required Annual","Ending Value"]:
            if col in solver_df.columns:
                solver_df[col] = solver_df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
        st.subheader("Solver Recommendation")
        st.caption("These allocations were applied to the selectors above to minimize the required annual contribution.")
        st.table(solver_df.drop(columns=["current_clean","annual_clean"], errors="ignore"))
        if st.button("Clear solver recommendation", key="clear_solver"):
            st.session_state["solver_rows_display"] = None
            _rerun()

    # ------------------------------------------------------------
    # Optimization helper (optional)
    # ------------------------------------------------------------
    st.markdown("#### Optimization (Min Required Annual)")

    def _solve_source(source_label, metas):
        if not metas:
            return None
        best = None
        for current_meta in metas:
            floor_val = _floor_for_choice(source_label, current_meta)
            shortfall_ideal = max(float(ideal_goal) - floor_val, 0.0)
            shortfall_accept = max(float(acceptable_goal) - floor_val, 0.0)
            for annual_meta in metas:
                evs_arr = calc_cache[source_label].get(annual_meta["clean"], {}).get("evs")
                if evs_arr is None or evs_arr.size == 0:
                    continue
                req_i = required_annual_for_goal(evs_arr, shortfall_ideal, float(ideal_conf_level))
                req_a = required_annual_for_goal(evs_arr, shortfall_accept, 1.0)
                req = max(req_i, req_a)
                if not np.isfinite(req):
                    continue
                total_arr = evs_arr * float(req)
                ending_val = float('nan')
                if total_arr.size:
                    ending_val = _quantile_linear(np.array(total_arr, dtype=float), ideal_tail) + floor_val
                entry = {
                    "Source": source_label,
                    "Current Allocation": PRETTY_LBM.get(current_meta["clean"], current_meta["clean"]) if source_label == "Global" else PRETTY_SPX.get(current_meta["clean"], current_meta["clean"]),
                    "Annual Allocation": PRETTY_LBM.get(annual_meta["clean"], annual_meta["clean"]) if source_label == "Global" else PRETTY_SPX.get(annual_meta["clean"], annual_meta["clean"]),
                    "Floor": floor_val,
                    "Required Annual": float(req),
                    "Ending Value": ending_val,
                }
                if best is None or entry["Required Annual"] < best["Required Annual"]:
                    best = entry
        return best


    # ------------------------------------------------------------
    # Failure Distribution (when investing the Required Annual) for selected allocation(s)
    # ------------------------------------------------------------
    if show_outputs:
        st.markdown("#### Failure Distribution (Selected Allocation)")
        st.caption(f"Adds the {current_conf_pct:.0f}% confidence current-portfolio floor to each outcome before measuring failures.")
    failure_rows = []

    def _failure_stats_ann(raw_col: str, stats: dict, required_amt: float, label_source: str, floor_val: float):
        evs_arr = stats.get("evs") if stats else None
        if evs_arr is None or evs_arr.size == 0 or not np.isfinite(required_amt):
            return
        arr = np.array(evs_arr, dtype=float) * float(required_amt)
        if floor_val:
            arr = arr + float(floor_val)
        total = int(arr.size)
        fails = arr < float(ideal_goal)
        num_fail = int(fails.sum())
        if num_fail == 0:
            failure_rows.append({
                "Source": label_source,
                "Allocation": raw_col,
                "Windows": total,
                "Failures": 0,
                "Failure Rate": "0.0%",
                "Worst": "",
                "P25": "",
                "Median": "",
                "P75": ""
            })
            return
        failed = arr[fails]
        p25 = np.percentile(failed, 25)
        p50 = np.percentile(failed, 50)
        p75 = np.percentile(failed, 75)
        worst = failed.min()
        failure_rows.append({
            "Source": label_source,
            "Allocation": raw_col,
            "Windows": total,
            "Failures": num_fail,
            "Failure Rate": f"{(num_fail/total):.1%}",
            "Worst": f"${worst:,.0f}",
            "P25": f"${p25:,.0f}",
            "Median": f"${p50:,.0f}",
            "P75": f"${p75:,.0f}",
            "Failure Rate Raw": num_fail / total,
            "Failures Raw": num_fail,
            "Worst Raw": worst,
            "Median Raw": p50,
        })

    # Analyze selected annual allocations
    def _run_failure_for_source(source: str, choice_meta):
        if not choice_meta:
            return
        raw_col = choice_meta["clean"]
        stats = calc_cache[source].get(raw_col)
        sel = selected_rows.get(source)
        if not stats or not sel:
            return
        req_amt = sel.get("Required Annual")
        if req_amt is None or not np.isfinite(req_amt):
            return
        _failure_stats_ann(raw_col, stats, float(req_amt), source, lump_floor.get(source, 0.0))

    _run_failure_for_source("Global", annual_alloc_choice_global)
    _run_failure_for_source("SP500", annual_alloc_choice_spx)

    if show_outputs:
        if failure_rows:
            fail_df = pd.DataFrame(failure_rows)
            # Friendlier allocation label in output
            def _friendly_alloc_fail(raw_name, source):
                if source == "Global":
                    return PRETTY_LBM.get(raw_name, raw_name)
                else:
                    return PRETTY_SPX.get(raw_name, raw_name)
            fail_df["Allocation"] = fail_df.apply(lambda r: _friendly_alloc_fail(r["Allocation"], r["Source"]), axis=1)
            st.data_editor(
                fail_df,
                hide_index=True,
                disabled=True,
                width="stretch",
                column_config={
                    "Source": st.column_config.TextColumn("Source", help="Data source used."),
                    "Allocation": st.column_config.TextColumn("Allocation", help="Selected annual allocation at current settings."),
                    "Windows": st.column_config.NumberColumn("Windows", help="Number of valid rolling windows."),
                    "Failures": st.column_config.NumberColumn("Failures", help="Count of windows that ended below Ideal Goal."),
                    "Failure Rate": st.column_config.TextColumn("Failure Rate", help="Failures / Windows."),
                    "Worst": st.column_config.TextColumn("Worst", help="Worst ending value among failures."),
                    "P25": st.column_config.TextColumn("P25", help="25th percentile of failure endings."),
                    "Median": st.column_config.TextColumn("Median", help="Median failure ending value."),
                    "P75": st.column_config.TextColumn("P75", help="75th percentile (less-bad failure)."),
                }
            )
        else:
            st.info("No failures at the selected settings for the chosen allocation(s).")

    # ------------------------------------------------------------
    # Success Distribution (when investing the Required Annual) for selected allocation(s)
    # ------------------------------------------------------------
    success_rows = []
    if show_outputs:
        st.markdown("#### Success Distribution (Selected Allocation)")
        st.caption(f"Same combined balance: required annual contributions plus the {current_conf_pct:.0f}% confidence current-portfolio floor.")

    def _success_stats_ann(raw_col: str, stats: dict, required_amt: float, label_source: str, floor_val: float):
        evs_arr = stats.get("evs") if stats else None
        if evs_arr is None or evs_arr.size == 0 or not np.isfinite(required_amt):
            return
        arr = np.array(evs_arr, dtype=float) * float(required_amt)
        if floor_val:
            arr = arr + float(floor_val)
        total = int(arr.size)
        succ_mask = arr >= float(ideal_goal)
        num_succ = int(succ_mask.sum())
        if num_succ == 0:
            success_rows.append({
                "Source": label_source,
                "Allocation": raw_col,
                "Windows": total,
                "Successes": 0,
                "Success Rate": "0.0%",
                "Min": "",
                "P25": "",
                "Median": "",
                "P75": "",
                "Best": ""
            })
            return
        succ = arr[succ_mask]
        p25 = np.percentile(succ, 25)
        p50 = np.percentile(succ, 50)
        p75 = np.percentile(succ, 75)
        best = succ.max()
        min_succ = succ.min()
        success_rows.append({
            "Source": label_source,
            "Allocation": raw_col,
            "Windows": total,
            "Successes": num_succ,
            "Success Rate": f"{(num_succ/total):.1%}",
            "Min": f"${min_succ:,.0f}",
            "P25": f"${p25:,.0f}",
            "Median": f"${p50:,.0f}",
            "P75": f"${p75:,.0f}",
            "Best": f"${best:,.0f}",
            "Success Rate Raw": num_succ / total,
            "P25 Raw": p25,
        })

    def _run_success_for_source(source: str, choice_meta):
        if not choice_meta:
            return
        raw_col = choice_meta["clean"]
        stats = calc_cache[source].get(raw_col)
        sel = selected_rows.get(source)
        if not stats or not sel:
            return
        req_amt = sel.get("Required Annual")
        if req_amt is None or not np.isfinite(req_amt):
            return
        _success_stats_ann(raw_col, stats, float(req_amt), source, lump_floor.get(source, 0.0))

    _run_success_for_source("Global", annual_alloc_choice_global)
    _run_success_for_source("SP500", annual_alloc_choice_spx)

    if show_outputs:
        if success_rows:
            succ_df = pd.DataFrame(success_rows)
            def _friendly_alloc_succ(raw_name, source):
                if source == "Global":
                    return PRETTY_LBM.get(raw_name, raw_name)
                else:
                    return PRETTY_SPX.get(raw_name, raw_name)
            succ_df["Allocation"] = succ_df.apply(lambda r: _friendly_alloc_succ(r["Allocation"], r["Source"]), axis=1)
            st.data_editor(
                succ_df,
                hide_index=True,
                disabled=True,
                width="stretch",
                column_config={
                    "Source": st.column_config.TextColumn("Source", help="Data source used."),
                    "Allocation": st.column_config.TextColumn("Allocation", help="Selected annual allocation at current settings."),
                    "Windows": st.column_config.NumberColumn("Windows", help="Number of valid rolling windows."),
                    "Successes": st.column_config.NumberColumn("Successes", help="Count of windows that ended at/above Ideal Goal."),
                    "Success Rate": st.column_config.TextColumn("Success Rate", help="Successes / Windows."),
                    "Min": st.column_config.TextColumn("Min", help="Worst ending value among successes (still ≥ Ideal Goal)."),
                    "P25": st.column_config.TextColumn("P25", help="25th percentile of successful endings."),
                    "Median": st.column_config.TextColumn("Median", help="Median successful ending value."),
                    "P75": st.column_config.TextColumn("P75", help="75th percentile of successful endings."),
                    "Best": st.column_config.TextColumn("Best", help="Best ending value among successes."),
                }
            )
        else:
            st.info("No successes found at the selected settings for the chosen allocation(s).")

    if summary_details:
        fail_lookup = {r["Source"]: r for r in failure_rows}
        succ_lookup = {r["Source"]: r for r in success_rows}
        st.subheader("Result Explanation (plain text)")
        for det in summary_details:
            src = det["source"]
            req_text = _fmt_currency(det["required"])
            succ = succ_lookup.get(src, {})
            fail = fail_lookup.get(src, {})
            success_rate = succ.get("Success Rate", f"{conf_pct_ideal:.0f}% target")
            p25_text = succ.get("P25", "N/A")
            failure_rate = fail.get("Failure Rate", "0.0%")
            num_fail = fail.get("Failures Raw", 0)
            worst_txt = fail.get("Worst", "N/A")
            median_txt = fail.get("Median", "N/A")
            explanation = (
                f"{src}: To achieve your goal under current terms, you will need to invest {req_text} each year for {int(num_years)} years. "
                f"The current allocation would be {det['current_label']} and the required annual contributions stay in {det['annual_label']}. "
                f"Historically, that mix met the Ideal goal of ${ideal_goal:,.0f} with a success rate of {success_rate}; the median outcome was {p25_text}. "
                f"About {failure_rate} of windows (roughly {num_fail} simulations) fell short—the worst ending value was {worst_txt} and the typical shortfall (median failure) was {median_txt}."
            )
            st.text(explanation)

    # Charts (separate), highlight selected allocation (fallback to minimum if none)
    if show_outputs:
        chart_df = wide_req.copy()
        if not chart_df.empty:
            n = len(chart_df)
            # Global chart
            if "Global" in chart_df.columns and chart_df["Global"].notna().any():
                g_vals = chart_df["Global"]
                g_colors = ["#9ecae1"] * n
                target_label = selected_labels.get("Global")
                if target_label and target_label in chart_df["Allocation"].values:
                    g_target_idx = chart_df.index[chart_df["Allocation"] == target_label][0]
                    g_colors[g_target_idx] = "#2ca02c"
                elif g_vals.notna().any():
                    g_min_pos = g_vals[g_vals.notna()].idxmin()
                    try: g_i = chart_df.index.get_loc(g_min_pos)
                    except Exception: g_i = int(g_min_pos) if isinstance(g_min_pos,(int,np.integer)) else None
                    if g_i is not None and 0 <= g_i < n: g_colors[g_i] = "#2ca02c"
                fig_g = go.Figure()
                fig_g.add_bar(name="Global", x=chart_df["Allocation"], y=g_vals, marker_color=g_colors)
                fig_g.update_layout(title="Required Annual — Global", xaxis_title="Allocation", yaxis_title="Required Annual ($)",
                                    yaxis=dict(tickformat=",.0f", tickprefix="$"), showlegend=False)
                _render_plotly(fig_g)
            # SP500 chart
            if "SP500" in chart_df.columns and chart_df["SP500"].notna().any():
                s_vals = chart_df["SP500"]
                s_colors = ["#3182bd"] * n
                target_label = selected_labels.get("SP500")
                if target_label and target_label in chart_df["Allocation"].values:
                    s_target_idx = chart_df.index[chart_df["Allocation"] == target_label][0]
                    s_colors[s_target_idx] = "#D95F02"
                elif s_vals.notna().any():
                    s_min_pos = s_vals[s_vals.notna()].idxmin()
                    try: s_i = chart_df.index.get_loc(s_min_pos)
                    except Exception: s_i = int(s_min_pos) if isinstance(s_min_pos,(int,np.integer)) else None
                    if s_i is not None and 0 <= s_i < n: s_colors[s_i] = "#D95F02"
                fig_s = go.Figure()
                fig_s.add_bar(name="SP500", x=chart_df["Allocation"], y=s_vals, marker_color=s_colors)
                fig_s.update_layout(title="Required Annual — SP500", xaxis_title="Allocation", yaxis_title="Required Annual ($)",
                                    yaxis=dict(tickformat=",.0f", tickprefix="$"), showlegend=False)
                _render_plotly(fig_s)

    # Download (CSV)
    csv = wide_req.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="required_annual_by_allocation.csv", mime="text/csv")


# ------------------------------
# Disclosures Download Section
# ------------------------------
st.divider()
st.subheader("Disclosures")

pdf_candidates = [
    ("Global (LBM)", "DataSource LBM Portfolios.pdf"),
    ("S&P 500 (SPX)", "DataSource SPX_e portfolios.pdf"),
]

for label, pdf_file in pdf_candidates:
    try:
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            f"Download {label} Disclosures (PDF)",
            data=pdf_bytes,
            file_name=pdf_file,
            mime="application/pdf",
        )
    except FileNotFoundError:
        st.info(f"Add `{pdf_file}` to the app folder to enable {label} disclosures.")

st.markdown('[Click here to go to Main Site](https://www.paulruedi.com)')
