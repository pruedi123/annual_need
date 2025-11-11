import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from textwrap import dedent

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
with col1:
    data_choice = st.selectbox(
        "Data source",
        ["Global Equity", "S&P 500", "Both Global & SP500"],
        index=0,
        help="Choose the factor set: LBM workbook (Excel) or S&P 500 workbook (spx_factors.xlsx).",
    )
    ideal_goal = st.number_input("Ideal Goal ($)", min_value=1, step=50000, value=1_000_000,
                                 help="Today’s dollars: same buying power as money today.",
                                 format="%i")
    conf_pct_ideal = st.slider("Ideal Confidence (%)", min_value=50, step= 10, max_value=100, value=100,
                               help="e.g., 90% means ≥90% of historical windows finish at/above the Ideal Goal.")
    ideal_conf_level = conf_pct_ideal / 100.0
with col2:
    num_years = st.number_input("Years", min_value=1, max_value=60, value=30)
    acceptable_goal = st.number_input("Essential Goal ($)", min_value=1, step=50000, value=800_000,
                                      help="A minimum acceptable outcome (floor) sized at 100% confidence.",
                                      format="%i")
    current_portfolio_value = st.number_input(
        "Current Portfolio Value ($)", min_value=0, step=50_000, value=0,
        help="How much is already invested today. It compounds through the historical windows alongside new contributions.",
        format="%i"
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
if df_lbm is not None:
    df_lbm.columns = df_lbm.columns.astype(str).str.strip().str.replace("  ", " ")
    allocation_cols_lbm = [c for c in df_lbm.columns if c.upper().startswith("LBM ")]
    for c in allocation_cols_lbm: df_lbm[c] = pd.to_numeric(df_lbm[c], errors='coerce')
if df_spx is not None:
    df_spx.columns = df_spx.columns.astype(str).str.strip().str.replace("  ", " ")
    allocation_cols_spx = [c for c in df_spx.columns if c.upper().startswith("SPX")]
    for c in allocation_cols_spx: df_spx[c] = pd.to_numeric(df_spx[c], errors='coerce')
if src_kind in ("LBM","BOTH") and not allocation_cols_lbm:
    st.warning("No allocation columns found in LBM (expected headers starting with 'LBM ').")
if src_kind in ("SPX","BOTH") and not allocation_cols_spx:
    st.warning("No allocation columns found in SPX (expected headers like 'spx60e', 'spx40e', etc.).")

fee = float(fee_pct)/100.0
current_portfolio_value = float(current_portfolio_value)
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
# Run (auto)
# ------------------------------
have_any = (
    (src_kind in ("LBM","BOTH") and df_lbm is not None and allocation_cols_lbm) or
    (src_kind in ("SPX","BOTH") and df_spx is not None and allocation_cols_spx)
)
if have_any:
    rows = []
    calc_cache = {"Global": {}, "SP500": {}}
    # LBM
    if src_kind in ("LBM","BOTH") and df_lbm is not None:
        for col in allocation_cols_lbm:
            col_clean = col.strip()
            evs, lumps = simulate_annuity_and_lumpsum(df_lbm[col], int(num_years), int(row_increment))
            evs_arr = np.array(evs, dtype=float)
            lumps_arr = np.array(lumps, dtype=float)
            calc_cache["Global"][col_clean] = {"evs": evs_arr, "lumps": lumps_arr}
            lumps_actual = (
                np.array(lumps_arr, dtype=float) * float(current_portfolio_value)
                if current_portfolio_value > 0 and lumps_arr.size
                else np.zeros_like(lumps_arr, dtype=float)
            )
            lumps_conf_100 = float(lumps_actual.min()) if lumps_actual.size else 0.0
            shortfall_ideal = max(float(ideal_goal) - lumps_conf_100, 0.0)
            shortfall_accept = max(float(acceptable_goal) - lumps_conf_100, 0.0)
            if evs_arr.size == 0:
                req = np.nan
                ending_val = np.nan
            else:
                req_i = required_annual_for_goal(evs_arr, shortfall_ideal, float(ideal_conf_level))
                req_a = required_annual_for_goal(evs_arr, shortfall_accept, float(1.0))
                req = max(req_i, req_a)
                ending_val = np.nan
                if np.isfinite(req):
                    total_arr = evs_arr * float(req)
                    if lumps_actual.size == total_arr.size:
                        total_arr = total_arr + lumps_actual
                    if total_arr.size:
                        ending_val = _quantile_linear(np.array(total_arr, dtype=float), ideal_tail)
            rows.append({
                "Allocation": col_clean,
                "Required Annual": np.nan if pd.isna(req) else float(req),
                "Ending Value": np.nan if pd.isna(ending_val) else float(ending_val),
                "Current Portfolio 100%": float(lumps_conf_100),
            })
    # SPX
    if src_kind in ("SPX","BOTH") and df_spx is not None:
        for col in allocation_cols_spx:
            col_clean = col.strip()
            evs, lumps = simulate_annuity_and_lumpsum(df_spx[col], int(num_years), int(row_increment))
            evs_arr = np.array(evs, dtype=float)
            lumps_arr = np.array(lumps, dtype=float)
            calc_cache["SP500"][col_clean] = {"evs": evs_arr, "lumps": lumps_arr}
            lumps_actual = (
                np.array(lumps_arr, dtype=float) * float(current_portfolio_value)
                if current_portfolio_value > 0 and lumps_arr.size
                else np.zeros_like(lumps_arr, dtype=float)
            )
            lumps_conf_100 = float(lumps_actual.min()) if lumps_actual.size else 0.0
            shortfall_ideal = max(float(ideal_goal) - lumps_conf_100, 0.0)
            shortfall_accept = max(float(acceptable_goal) - lumps_conf_100, 0.0)
            if evs_arr.size == 0:
                req = np.nan
                ending_val = np.nan
            else:
                req_i = required_annual_for_goal(evs_arr, shortfall_ideal, float(ideal_conf_level))
                req_a = required_annual_for_goal(evs_arr, shortfall_accept, float(1.0))
                req = max(req_i, req_a)
                ending_val = np.nan
                if np.isfinite(req):
                    total_arr = evs_arr * float(req)
                    if lumps_actual.size == total_arr.size:
                        total_arr = total_arr + lumps_actual
                    if total_arr.size:
                        ending_val = _quantile_linear(np.array(total_arr, dtype=float), ideal_tail)
            rows.append({
                "Allocation": col_clean,
                "Required Annual": np.nan if pd.isna(req) else float(req),
                "Ending Value": np.nan if pd.isna(ending_val) else float(ending_val),
                "Current Portfolio 100%": float(lumps_conf_100),
            })

    # Build pretty maps and wide table
    order_lbm = ['LBM 100E','LBM 90E','LBM 80E','LBM 70E','LBM 60E','LBM 50E','LBM 40E','LBM 30E','LBM 20E','LBM 10E','LBM 100F']
    pretty_lbm = {'LBM 100E':'100% Equity','LBM 90E':'90% Equity','LBM 80E':'80% Equity','LBM 70E':'70% Equity',
                  'LBM 60E':'60% Equity','LBM 50E':'50% Equity','LBM 40E':'40% Equity','LBM 30E':'30% Equity',
                  'LBM 20E':'20% Equity','LBM 10E':'10% Equity','LBM 100F':'100% Fixed'}
    order_spx = [f"spx{p}e" for p in [100,90,80,70,60,50,40,30,20,10,0]]
    pretty_spx = {f"spx{p}e": f"{p}% Equity" for p in [100,90,80,70,60,50,40,30,20,10,0]}

    results = pd.DataFrame(rows)
    # Determine source for each row and generic label
    def _generic_label(a: str) -> str:
        label = pretty_lbm.get(a, pretty_spx.get(a, a))
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
        generic_order = ["100% Equity","90% Equity","80% Equity","70% Equity","60% Equity","50% Equity","40% Equity","30% Equity","20% Equity","10% Equity","100% Fixed"]
        w = w.reindex([g for g in generic_order if g in w.index])
        w = w.rename_axis(None, axis=1).reset_index().rename(columns={"Generic":"Allocation"})
        return w

    wide_req = _build_wide("Required Annual")
    wide_end = _build_wide("Ending Value")
    wide_lump = _build_wide("Current Portfolio 100%")

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

    st.subheader("Results")
    st.caption("Required Annual satisfies BOTH: Ideal Goal at Ideal Confidence AND Acceptable Goal at 100%.")
    st.write(display_results)

    # Current portfolio only (100% confidence)
    has_lump = not wide_lump.drop(columns=["Allocation"], errors="ignore").empty and \
        wide_lump.drop(columns=["Allocation"], errors="ignore").notna().any().any()
    if has_lump:
        lump_display = wide_lump.copy()
        for col in ["Global","SP500"]:
            if col in lump_display.columns:
                lump_display[col] = lump_display[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
        lump_cols = ["Allocation"]
        if "Global" in lump_display.columns: lump_cols.append("Global")
        if "SP500" in lump_display.columns: lump_cols.append("SP500")
        st.subheader("Current Portfolio Ending Value (100% Confidence)")
        st.caption("Worst-case historical window outcome when investing the current portfolio alone (no new contributions).")
        st.write(lump_display[lump_cols])

    # Combined ending values table
    has_endings = not wide_end.drop(columns=["Allocation"], errors="ignore").empty and \
        wide_end.drop(columns=["Allocation"], errors="ignore").notna().any().any()
    if has_endings:
        ending_display = wide_end.copy()
        for col in ["Global","SP500"]:
            if col in ending_display.columns:
                ending_display[col] = ending_display[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
        end_cols = ["Allocation"]
        if "Global" in ending_display.columns: end_cols.append("Global")
        if "SP500" in ending_display.columns: end_cols.append("SP500")
        st.subheader("Ending Value (Current Portfolio + Required Annual)")
        st.caption(f"{conf_pct_ideal}% confidence ending balance when combining today's portfolio (${current_portfolio_value:,.0f}) with the required annual contribution.")
        st.write(ending_display[end_cols])

    # ------------------------------------------------------------
    # Failure Distribution (when investing the Required Annual) for cheapest allocation(s)
    # ------------------------------------------------------------
    st.markdown("#### Failure Distribution (when investing the Required Annual)")
    st.caption("Includes today's portfolio value compounding through each historical window.")
    failure_rows = []

    # Build maps from generic label -> raw column for each source
    inv_lbm = {v: k for k, v in pretty_lbm.items()}    # e.g., "60% Equity" -> "LBM 60E"
    inv_spx = {v: k for k, v in pretty_spx.items()}    # e.g., "60% Equity" -> "spx60e"
    # Normalize: generic "100% Fixed" maps to LBM 100F and spx0e where applicable
    if "100% Fixed" not in inv_lbm and "LBM 100F" in pretty_lbm:
        inv_lbm["100% Fixed"] = "LBM 100F"
    if "100% Fixed" not in inv_spx and "spx0e" in pretty_spx:
        inv_spx["100% Fixed"] = "spx0e"

    def _failure_stats_ann(raw_col: str, stats: dict, required_amt: float, label_source: str):
        evs_arr = stats.get("evs") if stats else None
        lumps_arr = stats.get("lumps") if stats else None
        if evs_arr is None or evs_arr.size == 0 or not np.isfinite(required_amt):
            return
        arr = np.array(evs_arr, dtype=float) * float(required_amt)
        if current_portfolio_value > 0 and lumps_arr is not None and lumps_arr.size == arr.size:
            arr = arr + np.array(lumps_arr, dtype=float) * float(current_portfolio_value)
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
        })

    # Identify cheapest (min required annual) allocation per source and compute failures
    # Global
    if "Global" in wide_req.columns and wide_req["Global"].notna().any():
        gidx = wide_req["Global"].idxmin()
        generic_g = wide_req.loc[gidx, "Allocation"]
        raw_g = inv_lbm.get(generic_g)
        req_amt_g = wide_req.loc[gidx, "Global"]
        stats_g = calc_cache["Global"].get(raw_g) if raw_g else None
        if raw_g and stats_g:
            _failure_stats_ann(raw_g, stats_g, req_amt_g, "Global")
    # SP500
    if "SP500" in wide_req.columns and wide_req["SP500"].notna().any():
        sidx = wide_req["SP500"].idxmin()
        generic_s = wide_req.loc[sidx, "Allocation"]
        raw_s = inv_spx.get(generic_s)
        req_amt_s = wide_req.loc[sidx, "SP500"]
        stats_s = calc_cache["SP500"].get(raw_s) if raw_s else None
        if raw_s and stats_s:
            _failure_stats_ann(raw_s, stats_s, req_amt_s, "SP500")

    if failure_rows:
        fail_df = pd.DataFrame(failure_rows)
        # Friendlier allocation label in output
        def _friendly_alloc_fail(raw_name, source):
            if source == "Global":
                return pretty_lbm.get(raw_name, raw_name)
            else:
                return pretty_spx.get(raw_name, raw_name)
        fail_df["Allocation"] = fail_df.apply(lambda r: _friendly_alloc_fail(r["Allocation"], r["Source"]), axis=1)
        st.data_editor(
            fail_df,
            hide_index=True,
            disabled=True,
            use_container_width=True,
            column_config={
                "Source": st.column_config.TextColumn("Source", help="Data source used."),
                "Allocation": st.column_config.TextColumn("Allocation", help="Cheapest allocation at current settings."),
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
        st.info("No failures at the selected confidence for the cheapest allocation(s).")

    # ------------------------------------------------------------
    # Success Distribution (when investing the Required Annual) for cheapest allocation(s)
    # ------------------------------------------------------------
    st.markdown("#### Success Distribution (when investing the Required Annual)")
    st.caption("Same combined balance: current portfolio plus required annual contributions.")
    success_rows = []

    def _success_stats_ann(raw_col: str, stats: dict, required_amt: float, label_source: str):
        evs_arr = stats.get("evs") if stats else None
        lumps_arr = stats.get("lumps") if stats else None
        if evs_arr is None or evs_arr.size == 0 or not np.isfinite(required_amt):
            return
        arr = np.array(evs_arr, dtype=float) * float(required_amt)
        if current_portfolio_value > 0 and lumps_arr is not None and lumps_arr.size == arr.size:
            arr = arr + np.array(lumps_arr, dtype=float) * float(current_portfolio_value)
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
        })

    # Compute success stats for the same cheapest allocations
    if "Global" in wide_req.columns and wide_req["Global"].notna().any():
        gidx = wide_req["Global"].idxmin()
        generic_g = wide_req.loc[gidx, "Allocation"]
        raw_g = inv_lbm.get(generic_g)
        req_amt_g = wide_req.loc[gidx, "Global"]
        stats_g = calc_cache["Global"].get(raw_g) if raw_g else None
        if raw_g and stats_g:
            _success_stats_ann(raw_g, stats_g, req_amt_g, "Global")
    if "SP500" in wide_req.columns and wide_req["SP500"].notna().any():
        sidx = wide_req["SP500"].idxmin()
        generic_s = wide_req.loc[sidx, "Allocation"]
        raw_s = inv_spx.get(generic_s)
        req_amt_s = wide_req.loc[sidx, "SP500"]
        stats_s = calc_cache["SP500"].get(raw_s) if raw_s else None
        if raw_s and stats_s:
            _success_stats_ann(raw_s, stats_s, req_amt_s, "SP500")

    if success_rows:
        succ_df = pd.DataFrame(success_rows)
        def _friendly_alloc_succ(raw_name, source):
            if source == "Global":
                return pretty_lbm.get(raw_name, raw_name)
            else:
                return pretty_spx.get(raw_name, raw_name)
        succ_df["Allocation"] = succ_df.apply(lambda r: _friendly_alloc_succ(r["Allocation"], r["Source"]), axis=1)
        st.data_editor(
            succ_df,
            hide_index=True,
            disabled=True,
            use_container_width=True,
            column_config={
                "Source": st.column_config.TextColumn("Source", help="Data source used."),
                "Allocation": st.column_config.TextColumn("Allocation", help="Cheapest allocation at current settings."),
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
        st.info("No successes found (this would occur only at very high fees or extreme settings).")

    # Charts (separate), highlight lowest bar
    chart_df = wide_req.copy()
    if not chart_df.empty:
        n = len(chart_df)
        # Global chart
        if "Global" in chart_df.columns and chart_df["Global"].notna().any():
            g_vals = chart_df["Global"]
            g_colors = ["#9ecae1"] * n
            if g_vals.notna().any():
                g_min_pos = g_vals[g_vals.notna()].idxmin()
                try: g_i = chart_df.index.get_loc(g_min_pos)
                except Exception: g_i = int(g_min_pos) if isinstance(g_min_pos,(int,np.integer)) else None
                if g_i is not None and 0 <= g_i < n: g_colors[g_i] = "#2ca02c"
            fig_g = go.Figure()
            fig_g.add_bar(name="Global", x=chart_df["Allocation"], y=g_vals, marker_color=g_colors)
            fig_g.update_layout(title="Required Annual — Global", xaxis_title="Allocation", yaxis_title="Required Annual ($)",
                                yaxis=dict(tickformat=",.0f", tickprefix="$"), showlegend=False)
            st.plotly_chart(fig_g, use_container_width=True)
        # SP500 chart
        if "SP500" in chart_df.columns and chart_df["SP500"].notna().any():
            s_vals = chart_df["SP500"]
            s_colors = ["#3182bd"] * n
            if s_vals.notna().any():
                s_min_pos = s_vals[s_vals.notna()].idxmin()
                try: s_i = chart_df.index.get_loc(s_min_pos)
                except Exception: s_i = int(s_min_pos) if isinstance(s_min_pos,(int,np.integer)) else None
                if s_i is not None and 0 <= s_i < n: s_colors[s_i] = "#D95F02"
            fig_s = go.Figure()
            fig_s.add_bar(name="SP500", x=chart_df["Allocation"], y=s_vals, marker_color=s_colors)
            fig_s.update_layout(title="Required Annual — SP500", xaxis_title="Allocation", yaxis_title="Required Annual ($)",
                                yaxis=dict(tickformat=",.0f", tickprefix="$"), showlegend=False)
            st.plotly_chart(fig_s, use_container_width=True)

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
