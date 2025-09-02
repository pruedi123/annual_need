import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

##############################
# App: Required Annual Invest
# Goal: For each allocation column in a worksheet, compute the required
#       constant annual contribution to reach a target (e.g., $1,000,000)
#       with a chosen confidence over N years, using historical factors.
##############################

st.set_page_config(layout="wide")
st.title("Required Annual Investment by Allocation (Worksheet‑Driven)")
st.caption("Computes the annual contribution required to reach your goal with the selected confidence using historical factor windows.")

# ------------------------------
# Inputs (you can change defaults)
# ------------------------------
file_path = st.text_input("Excel file path", value="all_portfolio_annual_factor_20_bps.xlsx")
sheet_name = st.text_input("Worksheet name", value="allocation_factors")

col1, col2, col3 = st.columns(3)
with col1:
    goal = st.number_input("Goal ($)", min_value=1, step=1000, value=1_000_000)
with col2:
    num_years = st.number_input("Years", min_value=1, max_value=60, value=30)
with col3:
    conf_pct = st.slider("Confidence (%)", min_value=50, max_value=100, value=90)
    confidence_level = conf_pct / 100.0

row_increment = 12  # Data is monthly, so step 12 rows per year

st.divider()

# ------------------------------
# Load worksheet
# ------------------------------
df = None
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    st.success("Worksheet loaded.")
except Exception as e:
    st.error(f"Error loading file/sheet: {e}")

# ------------------------------
# Prepare columns (detect allocations, coerce numeric)
# ------------------------------
allocation_cols = []
if df is not None:
    # Normalize headers and coerce numeric for LBM columns
    df.columns = df.columns.astype(str).str.strip().str.replace("  ", " ")
    allocation_cols = [c for c in df.columns if c.upper().startswith("LBM ")]
    for c in allocation_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    if not allocation_cols:
        st.warning("No allocation columns found (expected headers starting with 'LBM ').")

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
    """Given the distribution of ending values per $1 contributed annually,
    compute the annual contribution needed to hit 'goal_amount' at the
    chosen confidence (lower tail). Uses a conservative quantile.
    """
    if not ending_values:
        return float('nan')
    arr = np.array(sorted(ending_values))
    # Lower tail: we want the value such that conf% of outcomes are >= this.
    q = (1.0 - conf)
    # Use 'lower' interpolation to be conservative.
    idx = int(np.floor(q * len(arr)))
    idx = max(0, min(idx, len(arr) - 1))
    ev = arr[idx]
    if ev <= 0:
        return float('inf')
    return goal_amount / ev

# ------------------------------
# Run
# ------------------------------
if st.button("Compute required annual investment"):
    if df is None or not allocation_cols:
        st.error("Load a worksheet with allocation columns first.")
    else:
        rows = []
        for col in allocation_cols:
            evs = simulate_ending_values_annuity(df[col], int(num_years), int(row_increment))
            if not evs:
                req = np.nan
                note = "No valid windows (NaNs or insufficient length)"
            else:
                req = required_annual_for_goal(evs, float(goal), float(confidence_level))
                note = ""
            rows.append({
                "Allocation": col.strip(),
                "Required Annual Investment": np.nan if pd.isna(req) else int(round(req)),
                "Valid Windows": len(evs),
                "Note": note,
            })
        results = pd.DataFrame(rows)

        # Order & pretty labels
        order = [
            'LBM 100E','LBM 90E','LBM 80E','LBM 70E','LBM 60E',
            'LBM 50E','LBM 40E','LBM 30E','LBM 20E','LBM 10E','LBM 100F'
        ]
        pretty = {
            'LBM 100E': '100% Equity','LBM 90E': '90% Equity','LBM 80E': '80% Equity','LBM 70E': '70% Equity',
            'LBM 60E': '60% Equity','LBM 50E': '50% Equity','LBM 40E': '40% Equity','LBM 30E': '30% Equity',
            'LBM 20E': '20% Equity','LBM 10E': '10% Equity','LBM 100F': '100% Fixed'
        }
        results["_key"] = pd.Categorical(results["Allocation"], categories=order, ordered=True)
        results = results.sort_values("_key").drop(columns=["_key"]).copy()
        results["Allocation"] = results["Allocation"].map(pretty).fillna(results["Allocation"])  # fallback

        # Format a copy for display with currency and no decimals
        display_results = results.copy()
        if "Required Annual Investment" in display_results.columns:
            display_results["Required Annual Investment"] = display_results["Required Annual Investment"].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) else ""
            )

        st.subheader("Results")
        st.write(display_results)

        # Bar chart
        plot_df = results.dropna(subset=["Required Annual Investment"]).copy()
        if not plot_df.empty:
            min_val = plot_df["Required Annual Investment"].min()
            colors = ["green" if v == min_val else "blue" for v in plot_df["Required Annual Investment"]]
            fig = go.Figure(data=[go.Bar(
                x=plot_df['Allocation'],
                y=plot_df['Required Annual Investment'],
                marker_color=colors,
                text=[f"${v:,.0f}" for v in plot_df['Required Annual Investment']],
                textposition='outside'
            )])
            fig.update_layout(
                title="Required Annual Investment by Allocation",
                xaxis_title="Allocation",
                yaxis_title="Required Annual Investment ($)",
                uniformtext_minsize=8,
                uniformtext_mode='hide',
                yaxis=dict(tickformat=",.0f", tickprefix="$")
            )
            st.plotly_chart(fig, use_container_width=True)

        # Download
        csv = results.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="required_annual_by_allocation.csv", mime="text/csv")

st.markdown('[Click here to go to Main Site](https://www.paulruedi.com)')