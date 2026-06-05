"""
GCN-influence diagnostics for the per-step GCN + LSTM forecaster.

Two entry points, both pure (they take arrays / paths, never touch the
training pipeline internals):

    plot_gcn_influence(...)   — one 3-panel HTML per (product, seed, w, s):
        1. Forecasts: actual vs GCN (z on) vs GCN (z OFF, same weights) vs
           the LSTM baseline.  Per-step neighbours show on hover.
        2. Graph influence: signed area of (z_on − z_off) over the horizon,
           with the per-step neighbour count overlaid — does the graph move
           the prediction where neighbours are present?
        3. z-health: Var(z), ||z|| and ‖∇gcn‖/‖∇lstm‖ vs epoch, read back
           from the per-product diagnostics.csv.

    plot_influence_summary(...) — cross-product aggregate:
        * scatter RMSE(z on) vs RMSE(z off) with the y=x line — points above
          the line are products where zeroing z HURT (graph helped).
        * bar of mean |influence| per product.

The "z off" forecast is an *in-model* ablation: the same trained weights with
``model.ablate_z = True``.  It isolates how much that one model relies on the
graph, unlike a separately trained ablation model.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pcolors
from plotly.subplots import make_subplots


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────
def _align(a: Sequence, n: int) -> np.ndarray:
    """Coerce to float array of length n (truncate or NaN-pad)."""
    a = np.asarray(a, dtype=float).ravel()
    if len(a) < n:
        return np.concatenate([a, np.full(n - len(a), np.nan)])
    return a[:n]


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    m = ~(np.isnan(pred) | np.isnan(true))
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[m] - true[m]) ** 2)))


def _last_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    The diagnostics CSV is appended to per epoch and accumulates across
    re-runs (epoch resets to its minimum each fresh run).  Keep only the
    final contiguous run so the curve reflects the model we just trained.
    """
    if "epoch" not in df.columns or df.empty:
        return df
    ep = pd.to_numeric(df["epoch"], errors="coerce").to_numpy()
    if np.all(np.isnan(ep)):
        return df
    starts = np.where(ep == np.nanmin(ep))[0]
    return df.iloc[starts[-1]:] if len(starts) else df


def _read_z_health(diag_csv_path: Optional[str], seed: Optional[int]) -> Optional[pd.DataFrame]:
    """Read the full-run diagnostics CSV, return the final run for this seed."""
    if not diag_csv_path or not os.path.exists(diag_csv_path):
        return None
    try:
        df = pd.read_csv(diag_csv_path)
    except Exception:
        return None
    if df.empty:
        return None
    if "ablate_z" in df.columns:  # full file should already be z-on only
        df = df[df["ablate_z"].astype(str).isin(("False", "0", "0.0"))]
    if seed is not None and "seed" in df.columns:
        sel = df[df["seed"].astype(str) == str(seed)]
        if not sel.empty:
            df = sel
    return _last_run(df) if not df.empty else None


# ──────────────────────────────────────────────────────────────────────────
# per-product influence plot
# ──────────────────────────────────────────────────────────────────────────
def plot_gcn_influence(
    test,
    test_index,
    full_forecast,
    full_no_z_forecast,
    baseline_forecast=None,
    all_step_neighbours: Optional[dict] = None,
    neighbour_series: Optional[dict] = None,
    diag_csv_path: Optional[str] = None,
    seed: Optional[int] = None,
    title: str = "GCN Influence",
    save_path: Optional[str] = None,
) -> dict:
    """
    Build the 3-panel influence figure and (optionally) save it.

    Returns a stats dict — {rmse_full, rmse_noz, mean_abs_influence,
    max_abs_influence, final_var_z, mean_var_z} — for the aggregate summary.
    """
    n = len(test_index)
    test_arr = _align(test, n)
    full = _align(full_forecast, n)
    noz = _align(full_no_z_forecast, n)
    influence = full - noz  # signed: how much the graph moved the prediction

    # per-step neighbour counts (drive the overlay + hover)
    if all_step_neighbours is not None:
        nbr_counts = np.array(
            [len(all_step_neighbours.get(i, [])) for i in range(n)], dtype=float
        )
        nbr_strs = [
            ", ".join(str(x) for x in all_step_neighbours.get(i, [])) or "—"
            for i in range(n)
        ]
        custom = np.array([[i, s] for i, s in zip(range(n), nbr_strs)], dtype=object)
        hover_step = (
            "Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}"
            "<br>Step: %{customdata[0]}<br>Neighbours: %{customdata[1]}<extra></extra>"
        )
    else:
        nbr_counts = np.zeros(n)
        custom = np.arange(n)
        hover_step = "Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}<br>Step: %{customdata}"

    rmse_full = _rmse(full, test_arr)
    rmse_noz = _rmse(noz, test_arr)
    mean_abs = float(np.nanmean(np.abs(influence)))
    max_abs = float(np.nanmax(np.abs(influence))) if np.any(~np.isnan(influence)) else float("nan")

    zdf = _read_z_health(diag_csv_path, seed)
    final_var_z = mean_var_z = float("nan")
    if zdf is not None and "z_var_mean" in zdf.columns:
        zv = pd.to_numeric(zdf["z_var_mean"], errors="coerce").to_numpy()
        if np.any(~np.isnan(zv)):
            final_var_z = float(zv[~np.isnan(zv)][-1])
            mean_var_z = float(np.nanmean(zv))

    # ── figure ────────────────────────────────────────────────────────────
    delta_rmse = rmse_noz - rmse_full  # >0 → graph helped
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.50, 0.27, 0.23],
        vertical_spacing=0.085,
        subplot_titles=(
            f"Forecasts — RMSE: z-on {rmse_full:.3f} vs z-off {rmse_noz:.3f} "
            f"(Δ={delta_rmse:+.3f}, graph {'helped' if delta_rmse > 0 else 'hurt'})",
            f"Graph influence (z-on − z-off) — mean |Δ|={mean_abs:.3f}, max |Δ|={max_abs:.3f}",
            "z-health over training (Var(z), ||z||, grad ratio)",
        ),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}],
               [{"secondary_y": True}]],
    )

    # row 1 — forecasts
    fig.add_trace(go.Scatter(x=test_index, y=test_arr, name="Actual",
                             mode="lines", line=dict(color="green", width=2),
                             customdata=custom, hovertemplate=hover_step), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_index, y=full, name="GCN (z on)",
                             mode="lines", line=dict(color="red", width=2),
                             customdata=custom, hovertemplate=hover_step), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_index, y=noz, name="GCN (z off, in-model)",
                             mode="lines", line=dict(color="royalblue", width=2, dash="dash"),
                             customdata=custom, hovertemplate=hover_step), row=1, col=1)
    if baseline_forecast is not None:
        fig.add_trace(go.Scatter(x=test_index, y=_align(baseline_forecast, n),
                                 name="LSTM baseline", mode="lines",
                                 line=dict(color="gray", width=1, dash="dot"),
                                 hovertemplate=hover_step, customdata=custom), row=1, col=1)

    # row 2 — influence area + neighbour count
    fig.add_trace(go.Scatter(x=test_index, y=influence, name="Influence (z-on − z-off)",
                             mode="lines", line=dict(color="purple", width=1.5),
                             fill="tozeroy", fillcolor="rgba(128,0,128,0.25)",
                             hovertemplate="Date: %{x|%Y-%m-%d}<br>Δ=%{y:.3f}<extra></extra>"),
                  row=2, col=1, secondary_y=False)
    if all_step_neighbours is not None:
        fig.add_trace(go.Bar(x=test_index, y=nbr_counts, name="# neighbours",
                             marker=dict(color="rgba(30,144,255,0.35)"),
                             hovertemplate="Date: %{x|%Y-%m-%d}<br>#nbrs=%{y}<extra></extra>"),
                      row=2, col=1, secondary_y=True)
    fig.add_hline(y=0.0, line_dash="dot", line_color="black", line_width=1, row=2, col=1)

    # row 3 — z-health
    if zdf is not None and "epoch" in zdf.columns:
        ep = pd.to_numeric(zdf["epoch"], errors="coerce")
        if "z_var_mean" in zdf.columns:
            fig.add_trace(go.Scatter(x=ep, y=pd.to_numeric(zdf["z_var_mean"], errors="coerce"),
                                     name="Var(z)", mode="lines", line=dict(color="crimson")),
                          row=3, col=1, secondary_y=False)
        if "z_norm_mean" in zdf.columns:
            fig.add_trace(go.Scatter(x=ep, y=pd.to_numeric(zdf["z_norm_mean"], errors="coerce"),
                                     name="||z||", mode="lines", line=dict(color="darkorange")),
                          row=3, col=1, secondary_y=False)
        if "grad_ratio_gcn_lstm" in zdf.columns:
            fig.add_trace(go.Scatter(x=ep, y=pd.to_numeric(zdf["grad_ratio_gcn_lstm"], errors="coerce"),
                                     name="‖∇gcn‖/‖∇lstm‖", mode="lines",
                                     line=dict(color="teal", dash="dot")),
                          row=3, col=1, secondary_y=True)
        fig.update_xaxes(title_text="Epoch", row=3, col=1)
        fig.update_yaxes(title_text="Var(z) / ||z||", row=3, col=1, secondary_y=False)
        fig.update_yaxes(title_text="grad ratio", row=3, col=1, secondary_y=True)
    else:
        fig.add_annotation(text="no diagnostics.csv found for this run",
                           xref="x domain", yref="y domain", x=0.5, y=0.5,
                           showarrow=False, row=3, col=1)

    fig.update_yaxes(title_text="value", row=1, col=1)
    fig.update_yaxes(title_text="Δ forecast", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="# neighbours", row=2, col=1, secondary_y=True)
    fig.update_layout(height=1100, width=1500, template="plotly_white",
                      title=dict(text=title, font=dict(size=16)),
                      hovermode="x unified",
                      legend=dict(orientation="v", yanchor="top", y=1.0,
                                  xanchor="left", x=1.02))

    if save_path:
        fig.write_html(save_path) if save_path.endswith(".html") else fig.write_image(save_path)

    return dict(
        rmse_full=rmse_full, rmse_noz=rmse_noz,
        mean_abs_influence=mean_abs, max_abs_influence=max_abs,
        final_var_z=final_var_z, mean_var_z=mean_var_z,
    )


# ──────────────────────────────────────────────────────────────────────────
# aggregate summary across products
# ──────────────────────────────────────────────────────────────────────────
def plot_influence_summary(records: list, save_dir: str) -> None:
    """
    records : list of dicts with at least
        product_id, store_id, seed, rmse_full, rmse_noz, mean_abs_influence
    Writes two HTMLs into save_dir.
    """
    if not records:
        return
    df = pd.DataFrame(records)
    os.makedirs(save_dir, exist_ok=True)
    df["label"] = df.apply(
        lambda r: f"{r.get('product_id')}_{r.get('store_id')} (s{r.get('seed')})", axis=1
    )

    # ── scatter: RMSE(z on) vs RMSE(z off) ────────────────────────────────
    rf = pd.to_numeric(df["rmse_full"], errors="coerce")
    rz = pd.to_numeric(df["rmse_noz"], errors="coerce")
    m = ~(rf.isna() | rz.isna())
    rf, rz, lbl = rf[m], rz[m], df["label"][m]
    fig = go.Figure()
    if len(rf):
        lo = float(min(rf.min(), rz.min()))
        hi = float(max(rf.max(), rz.max()))
        pad = (hi - lo) * 0.05 or 1.0
        fig.add_trace(go.Scatter(x=[lo - pad, hi + pad], y=[lo - pad, hi + pad],
                                 mode="lines", name="y = x",
                                 line=dict(color="black", dash="dash")))
        helped = rz >= rf  # zeroing z hurt → graph helped
        fig.add_trace(go.Scatter(
            x=rf, y=rz, mode="markers+text", text=lbl, textposition="top center",
            name="product", marker=dict(
                size=11,
                color=np.where(helped, "green", "crimson"),
                line=dict(width=1, color="black")),
            hovertemplate="%{text}<br>RMSE z-on=%{x:.3f}<br>RMSE z-off=%{y:.3f}<extra></extra>"))
        n_helped = int(helped.sum())
        sub = (f"{n_helped}/{len(rf)} products above y=x (graph helped). "
               "Points above the line: zeroing z increased RMSE.")
    else:
        sub = "no valid RMSE pairs"
    fig.update_layout(
        title=f"GCN influence: RMSE(z on) vs RMSE(z off)<br>"
              f"<span style='font-size:12px;color:gray'>{sub}</span>",
        xaxis_title="RMSE — GCN (z on)", yaxis_title="RMSE — GCN (z off)",
        template="plotly_white", height=750, width=900)
    fig.write_html(os.path.join(save_dir, "_summary_rmse_zon_vs_zoff.html"))

    # ── bar: mean |influence| per product ─────────────────────────────────
    mai = pd.to_numeric(df["mean_abs_influence"], errors="coerce")
    fig2 = go.Figure(go.Bar(
        x=df["label"], y=mai, marker=dict(color="purple"),
        hovertemplate="%{x}<br>mean |Δ|=%{y:.3f}<extra></extra>"))
    fig2.update_layout(
        title="Mean |graph influence| per product (|z-on − z-off|)",
        xaxis_title="product", yaxis_title="mean |Δ forecast|",
        template="plotly_white", height=600, width=1100)
    fig2.write_html(os.path.join(save_dir, "_summary_mean_influence.html"))
