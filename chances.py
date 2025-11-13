import numpy as np
import pandas as pd
import streamlit as st

PRETTY_LBM = {
    "LBM 100E": "100% Equity",
    "LBM 90E": "90% Equity",
    "LBM 80E": "80% Equity",
    "LBM 70E": "70% Equity",
    "LBM 60E": "60% Equity",
    "LBM 50E": "50% Equity",
    "LBM 40E": "40% Equity",
    "LBM 30E": "30% Equity",
    "LBM 20E": "20% Equity",
    "LBM 10E": "10% Equity",
    "LBM 100F": "100% Fixed",
}
PRETTY_SPX = {f"spx{p}e": f"{p}% Equity" for p in [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]}
PERIOD_STEP = 1  # monthly factors, advance one row per month

st.set_page_config(layout="wide")
st.title("Chance of Falling Below a Target")
st.caption("Estimate, for every allocation, the fraction of historical windows that finish below a selected target.")


def _load_factors(file_path: str, sheet: str, prefix: str) -> tuple[pd.DataFrame | None, list[dict]]:
    """Return (dataframe, allocation_meta). Each meta has raw + clean names."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet)
    except Exception as exc:
        st.error(f"Unable to load {file_path} -> {sheet}: {exc}")
        return None, []
    df.columns = df.columns.astype(str).str.strip().str.replace("  ", " ")
    metas = []
    for col in df.columns:
        if col.upper().startswith(prefix):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            metas.append({"raw": col, "clean": col.strip()})
    return df, metas


def _quantile_linear(arr: np.ndarray, q: float) -> float:
    """Helper to get a linear-interpolated quantile; works on NumPy ≥1.23 and older."""
    if arr.size == 0:
        return float("nan")
    try:
        return float(np.quantile(arr, q, method="linear"))
    except TypeError:
        return float(np.quantile(arr, q, interpolation="linear"))


def simulate_lump_values(factors: pd.Series, months: int, step: int) -> np.ndarray:
    """Rolling-window lump-sum growth factors per $1 invested over 'months' months."""
    values = []
    n = len(factors)
    months = int(months)
    if months <= 0:
        return np.array(values, dtype=float)
    max_start = n - (step * (months - 1))
    if max_start <= 0:
        return np.array(values, dtype=float)
    for start in range(max_start):
        total = 1.0
        valid = True
        for period in range(months):
            idx = start + period * step
            if idx >= n:
                valid = False
                break
            f_val = factors.iloc[idx]
            if pd.isna(f_val) or f_val <= 0:
                valid = False
                break
            total *= float(f_val)
        if valid:
            values.append(total)
    return np.array(values, dtype=float)


def build_forward_12m(df_src: pd.DataFrame | None, metas: list[dict], window: int = 12) -> pd.DataFrame | None:
    """From monthly factors, build forward-looking 12-month products (starting each month)."""
    if df_src is None or not metas:
        return None
    rolled_cols = {}
    for meta in metas:
        series = df_src[meta["raw"]]
        arr = series.to_numpy(dtype=float)
        n = arr.size
        if n < window:
            continue
        out = np.full(n - window + 1, np.nan, dtype=float)
        for start in range(n - window + 1):
            window_vals = arr[start : start + window]
            if np.any(~np.isfinite(window_vals)) or np.any(window_vals <= 0):
                continue
            out[start] = float(np.prod(window_vals))
        rolled_cols[meta["clean"]] = out
    if not rolled_cols:
        return None
    return pd.DataFrame(rolled_cols)


def _pretty_name(source: str, clean: str) -> str:
    mapping = PRETTY_LBM if source == "Global" else PRETTY_SPX
    return mapping.get(clean, clean)


def _meta_by_clean(metas: list[dict], clean: str) -> dict | None:
    if not metas or clean is None:
        return None
    for meta in metas:
        if meta["clean"] == clean:
            return meta
    return None


def _fmt_currency(val: float) -> str:
    if val is None:
        return "—"
    try:
        if not np.isfinite(val):
            return "—"
    except TypeError:
        return "—"
    return f"${val:,.0f}"


sb = st.sidebar
data_choice = sb.selectbox(
    "Data source",
    ["Global Equity", "S&P 500", "Both Global & SP500"],
    index=0,
    help="Choose which historical factor set(s) to evaluate.",
)
months_out = sb.number_input("Months from today", min_value=1, max_value=720, value=36)
current_value = sb.number_input(
    "Current portfolio value ($)", min_value=0, step=50_000, value=1_000_000, format="%i"
)
target_value = sb.number_input(
    "Target / floor value ($)",
    min_value=0,
    step=50_000,
    value=800_000,
    format="%i",
    help="Chance is computed on the current balance growing into the future and finishing below this amount.",
)
fee_pct = sb.slider(
    "Annual fee (%)",
    min_value=0.0,
    max_value=1.0,
    step=0.1,
    value=0.20,
    help="Net growth factor = gross factor × (1 − fee). Set to 0 to ignore fees.",
)

src_kind = (
    "BOTH"
    if data_choice.startswith("Both")
    else "LBM"
    if data_choice.startswith("Global")
    else "SPX"
)
df_lbm, metas_lbm = None, []
df_spx, metas_spx = None, []
df_lbm_orig, metas_lbm_orig = None, []
df_spx_orig, metas_spx_orig = None, []
if src_kind in ("LBM", "BOTH"):
    df_lbm, metas_lbm = _load_factors("global_mo_factors.xlsx", "factors_mo", "LBM")
    df_lbm_orig, metas_lbm_orig = _load_factors("global_factors.xlsx", "global_factors", "LBM")
if src_kind in ("SPX", "BOTH"):
    df_spx, metas_spx = _load_factors("spx_mo_factors.xlsx", "factors_mo", "SPX")
    df_spx_orig, metas_spx_orig = _load_factors("spx_factors.xlsx", "spx_factors", "SPX")

annual_fee = float(fee_pct) / 100.0
monthly_fee = annual_fee / 12.0
if annual_fee > 0:
    if df_lbm is not None and metas_lbm:
        df_lbm[[m["raw"] for m in metas_lbm]] = df_lbm[[m["raw"] for m in metas_lbm]] * (1.0 - monthly_fee)
    if df_spx is not None and metas_spx:
        df_spx[[m["raw"] for m in metas_spx]] = df_spx[[m["raw"] for m in metas_spx]] * (1.0 - monthly_fee)
    if df_lbm_orig is not None and metas_lbm_orig:
        df_lbm_orig[[m["raw"] for m in metas_lbm_orig]] = df_lbm_orig[[m["raw"] for m in metas_lbm_orig]] * (1.0 - annual_fee)
    if df_spx_orig is not None and metas_spx_orig:
        df_spx_orig[[m["raw"] for m in metas_spx_orig]] = df_spx_orig[[m["raw"] for m in metas_spx_orig]] * (1.0 - annual_fee)


def _selection_widget(label: str, metas: list[dict], mapping: dict[str, str]) -> list[dict]:
    if not metas:
        return []
    options = [m["clean"] for m in metas]
    selected = sb.multiselect(
        label,
        options=options,
        default=options,
        format_func=lambda clean: mapping.get(clean, clean),
    )
    if not selected:
        return []
    filtered = [m for m in metas if m["clean"] in selected]
    return filtered


if df_lbm is None and df_spx is None:
    st.stop()

selected_lbm = []
selected_spx = []
if df_lbm is not None:
    selected_lbm = _selection_widget("Global allocations", metas_lbm, PRETTY_LBM)
if df_spx is not None:
    selected_spx = _selection_widget("S&P 500 allocations", metas_spx, PRETTY_SPX)

rolled_12m = {
    "Global": build_forward_12m(df_lbm, metas_lbm),
    "SP500": build_forward_12m(df_spx, metas_spx),
}

if not current_value or current_value <= 0:
    st.warning("Enter a positive current portfolio value to see probabilities.")
if not selected_lbm and not selected_spx:
    st.warning("Pick at least one allocation in the sidebar.")


def _probability_rows(
    source_label: str, df_src: pd.DataFrame, metas: list[dict], months: int, curr: float, target: float
) -> list[dict]:
    rows = []
    if df_src is None or not metas:
        return rows
    threshold_factor = float(target) / float(curr) if curr > 0 else float("nan")
    for meta in metas:
        col = meta["raw"]
        factors = df_src[col]
        lump_arr = simulate_lump_values(factors, int(months), PERIOD_STEP)
        lump_arr = lump_arr[np.isfinite(lump_arr)]
        if lump_arr.size == 0:
            continue
        prob = float(np.mean(lump_arr < threshold_factor)) if np.isfinite(threshold_factor) else float("nan")
        worst = float(np.min(lump_arr)) * curr if curr > 0 else float("nan")
        median = _quantile_linear(lump_arr, 0.5) * curr if curr > 0 else float("nan")
        p90 = _quantile_linear(lump_arr, 0.9) * curr if curr > 0 else float("nan")
        rows.append(
            {
                "Source": source_label,
                "Allocation": _pretty_name(source_label, meta["clean"]),
                "Chance Below Target (%)": prob * 100.0,
                "Median Ending ($)": median,
                "90th % Ending ($)": p90,
                "Worst Ending ($)": worst,
                "# of Tests": int(lump_arr.size),
            }
        )
    return rows


def _comparison_dataframe(
    source_label: str,
    alloc_clean: str,
    rolled_df_map: dict,
    orig_df: pd.DataFrame | None,
    orig_metas: list[dict],
) -> pd.DataFrame:
    """Return side-by-side original vs monthly-derived 12m factors."""
    rolled_df = rolled_df_map.get(source_label)
    if rolled_df is None or alloc_clean not in rolled_df.columns:
        return pd.DataFrame()
    orig_meta = _meta_by_clean(orig_metas, alloc_clean)
    if orig_meta is None or orig_df is None:
        return pd.DataFrame()
    orig_series = orig_df[orig_meta["raw"]]
    orig_vals = pd.to_numeric(orig_series, errors="coerce").to_numpy(dtype=float)
    monthly_vals = rolled_df[alloc_clean].to_numpy(dtype=float)
    limit = min(len(orig_vals), len(monthly_vals))
    if limit <= 0:
        return pd.DataFrame()
    orig_slice = orig_vals[:limit]
    monthly_slice = monthly_vals[:limit]
    mask = np.isfinite(orig_slice) & np.isfinite(monthly_slice)
    if not np.any(mask):
        return pd.DataFrame()
    idx = np.arange(limit)[mask]
    comp = pd.DataFrame(
        {
            "Window #": idx,
            "Original 12m factor": orig_slice[mask],
            "Monthly → 12m factor": monthly_slice[mask],
        }
    )
    comp["Difference"] = comp["Monthly → 12m factor"] - comp["Original 12m factor"]
    return comp


result_rows: list[dict] = []
result_rows.extend(_probability_rows("Global", df_lbm, selected_lbm, months_out, current_value, target_value))
result_rows.extend(_probability_rows("SP500", df_spx, selected_spx, months_out, current_value, target_value))

if not result_rows:
    st.info("No valid simulations for the current selections.")
else:
    results_df = pd.DataFrame(result_rows)
    results_df.sort_values(by=["Source", "Chance Below Target (%)"], ascending=[True, False], inplace=True)
    display_df = results_df.copy()
    currency_cols = ["Median Ending ($)", "90th % Ending ($)", "Worst Ending ($)"]
    for col in currency_cols:
        if col in display_df:
            display_df[col] = display_df[col].apply(_fmt_currency)
    st.subheader("Historical chance of finishing below the target")
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Source": st.column_config.Column(width="small"),
            "Allocation": st.column_config.Column(width="medium"),
            "Chance Below Target (%)": st.column_config.NumberColumn(format="%.1f%%", width="small"),
            "Median Ending ($)": st.column_config.Column(width="medium"),
            "90th % Ending ($)": st.column_config.Column(width="medium"),
            "Worst Ending ($)": st.column_config.Column(width="medium"),
            "# of Tests": st.column_config.Column(width="small"),
        },
    )

st.divider()
st.subheader("12-Month Factor Comparison")
comparison_sources = []
if df_lbm_orig is not None and rolled_12m.get("Global") is not None:
    comparison_sources.append("Global")
if df_spx_orig is not None and rolled_12m.get("SP500") is not None:
    comparison_sources.append("SP500")

if not comparison_sources:
    st.info("Monthly-to-12m comparison unavailable (missing data).")
else:
    comp_source = st.selectbox("Source", options=comparison_sources)
    if comp_source == "Global":
        meta_monthly = metas_lbm
        meta_orig = metas_lbm_orig
        df_orig = df_lbm_orig
    else:
        meta_monthly = metas_spx
        meta_orig = metas_spx_orig
        df_orig = df_spx_orig
    monthly_allocs = {m["clean"] for m in (meta_monthly or [])}
    orig_allocs = {m["clean"] for m in (meta_orig or [])}
    common_allocs = sorted(monthly_allocs & orig_allocs)
    if not common_allocs:
        st.info("No overlapping allocations to compare for this source.")
    else:
        comp_alloc = st.selectbox(
            "Allocation",
            options=common_allocs,
            format_func=lambda clean: _pretty_name(comp_source, clean),
        )
        preview_rows = st.slider("Rows to preview", min_value=5, max_value=200, value=25, step=5)
        comp_df = _comparison_dataframe(comp_source, comp_alloc, rolled_12m, df_orig, meta_orig)
        if comp_df.empty:
            st.warning("Could not compute overlapping 12-month factors for this allocation.")
        else:
            diffs = comp_df["Difference"].to_numpy(dtype=float)
            if diffs.size:
                mean_abs = float(np.mean(np.abs(diffs)))
                max_abs = float(np.max(np.abs(diffs)))
                st.caption(f"Mean absolute difference: {mean_abs:.6f} | Max absolute difference: {max_abs:.6f}")
            st.dataframe(
                comp_df.head(preview_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Window #": st.column_config.NumberColumn(width="small"),
                    "Original 12m factor": st.column_config.NumberColumn(format="%.6f"),
                    "Monthly → 12m factor": st.column_config.NumberColumn(format="%.6f"),
                    "Difference": st.column_config.NumberColumn(format="%.6f"),
                },
            )
