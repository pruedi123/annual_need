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

st.set_page_config(layout="wide")
st.title("Required Annual Investment by Allocation (Worksheet-Driven)")
st.caption("Computes the annual contribution required to reach BOTH an Ideal Goal (at its confidence) AND an Acceptable Goal (at 100%), using historical factor windows.")

# ------------------------------
# Inputs (you can change defaults)
# ------------------------------
file_path = "all_portfolio_annual_factor_20_bps.xlsx"
sheet_name = "allocation_factors"

col1, col2, col3 = st.columns(3)
with col1:
    data_choice = st.selectbox(
        "Data source",
        ["Global (LBM)", "S&P 500 (SPX)", "Both Global + SPX)"],
        index=0,
        help="Choose the factor set: LBM workbook (Excel) or S&P 500 CSV (spx_factors.csv).",
    )
    ideal_goal = st.number_input("Ideal Goal ($)", min_value=1, step=50000, value=1_000_000,
                                 help="Today’s dollars: same buying power as money today.",
                                 format="%i")
    conf_pct_ideal = st.slider("Ideal Confidence (%)", min_value=50, max_value=100, value=90,
                               help="e.g., 90% means ≥90% of historical windows finish at/above the Ideal Goal.")
    ideal_conf_level = conf_pct_ideal / 100.0
with col2:
    num_years = st.number_input("Years", min_value=1, max_value=60, value=30)
    acceptable_goal = st.number_input("Acceptable Goal ($)", min_value=1, step=50000, value=800_000,
                                      help="A minimum acceptable outcome (floor) sized at 100% confidence.",
                                      format="%i")
    acceptable_conf_level = 1.0  # fixed 100%
with col3:
    fee_pct = st.slider("Annual fee (%)", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
                        help="Applied once per 12-month factor: net = gross × (1 − fee).")

row_increment = 12  # Data is monthly, so step 12 rows per year

st.divider()
# Load factors (LBM Excel or SPX CSV)
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
        df_spx = pd.read_csv("spx_factors.csv", sep=None, engine="python")
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
if fee > 0:
    if df_lbm is not None and allocation_cols_lbm:
        df_lbm[allocation_cols_lbm] = df_lbm[allocation_cols_lbm] * (1.0 - fee)
    if df_spx is not None and allocation_cols_spx:
        df_spx[allocation_cols_spx] = df_spx[allocation_cols_spx] * (1.0 - fee)

# ------------------------------
# Core math
# ------------------------------

def simulate_ending_values_annuity(factors: pd.Series, years: int, step: int) -> list:
    """For each possible start row, simulate the ending value of an annuity
    that contributes $1 at the beginning of each year for 'years' years,
    compounding by the factor at each step window. Skips windows containing NaNs.

    Returns a list of ending values (one per valid start window).
    """
    vals = []
    n = len(factors)
    max_start = n - (step * (years - 1))
    if max_start <= 0:
        return vals
    for start in range(max_start):
        inv = 0.0
        valid = True
        for y in range(years):
            idx = start + y * step
            f = factors.iloc[idx]
            if pd.isna(f) or f <= 0:
                valid = False
                break
            inv = (inv + 1.0) * float(f)
        if valid:
            vals.append(inv)
    return vals


def required_annual_for_goal(ending_values: list, goal_amount: float, conf: float) -> float:
    """Given ending values per $1 contributed annually, return the annual contribution
    needed to hit goal_amount at the specified confidence. Uses lower-tail quantile
    with linear interpolation to avoid collapsing to a single worst-window."""
    if not ending_values:
        return float('nan')
    arr = np.sort(np.array(ending_values, dtype=float))
    q = max(0.0, min(1.0, 1.0 - float(conf)))
    try:
        ev = np.quantile(arr, q, method="linear")
    except TypeError:
        ev = np.quantile(arr, q, interpolation="linear")
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
    # LBM
    if src_kind in ("LBM","BOTH") and df_lbm is not None:
        for col in allocation_cols_lbm:
            evs = simulate_ending_values_annuity(df_lbm[col], int(num_years), int(row_increment))
            if not evs:
                req = np.nan
            else:
                req_i = required_annual_for_goal(evs, float(ideal_goal), float(ideal_conf_level))
                req_a = required_annual_for_goal(evs, float(acceptable_goal), float(1.0))
                req = max(req_i, req_a)
            rows.append({"Allocation": col.strip(), "Required Annual": np.nan if pd.isna(req) else float(req)})
    # SPX
    if src_kind in ("SPX","BOTH") and df_spx is not None:
        for col in allocation_cols_spx:
            evs = simulate_ending_values_annuity(df_spx[col], int(num_years), int(row_increment))
            if not evs:
                req = np.nan
            else:
                req_i = required_annual_for_goal(evs, float(ideal_goal), float(ideal_conf_level))
                req_a = required_annual_for_goal(evs, float(acceptable_goal), float(1.0))
                req = max(req_i, req_a)
            rows.append({"Allocation": col.strip(), "Required Annual": np.nan if pd.isna(req) else float(req)})

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
    wide = tmp.pivot_table(index="Generic", columns="Source", values="Required Annual", aggfunc="first")
    # Order rows by common sequence
    generic_order = ["100% Equity","90% Equity","80% Equity","70% Equity","60% Equity","50% Equity","40% Equity","30% Equity","20% Equity","10% Equity","100% Fixed"]
    wide = wide.reindex([g for g in generic_order if g in wide.index])
    wide = wide.rename_axis(None, axis=1).reset_index().rename(columns={"Generic":"Allocation"})
    # Format for display
    display_results = wide.copy()
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

    # Charts (separate), highlight lowest bar
    chart_df = wide.copy()
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
    csv = wide.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="required_annual_by_allocation.csv", mime="text/csv")

st.markdown('[Click here to go to Main Site](https://www.paulruedi.com)')