"""
GCN influence plots — quantify how much the graph embedding contributes
versus the pure-LSTM ablation (z=0 baseline).

Public API
----------
plot_gcn_influence(...)      — per-product interactive Plotly HTML + stats dict
plot_influence_summary(...)  — cross-product summary HTMLs
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── helpers ────────────────────────────────────────────────────────────────
def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


# ── per-product plot ────────────────────────────────────────────────────────
def plot_gcn_influence(
    test: np.ndarray,
    test_index,
    full_forecast: np.ndarray,
    full_no_z_forecast: np.ndarray,
    diag_csv_path: Optional[str] = None,
    seed: int = 0,
    title: str = "GCN Influence",
    save_path: Optional[str] = None,
) -> dict:
    """
    Build an interactive Plotly figure comparing the full GCN+LSTM forecast
    against the ablation baseline (z=0) and return influence statistics.

    Subplots
    --------
    1. Forecast overlay  — ground truth, GCN+LSTM, ablation; RMSE annotated.
    2. GCN delta         — (full − ablation) per step, coloured by sign.
                           Green = GCN pushed the prediction up,
                           Red   = GCN pushed it down.
    3. Step-wise error   — |GCN error| and |ablation error| per step.
    4. Training diagnostics (‖z‖, Var(z)) — only when diag_csv_path exists.

    Returns
    -------
    dict with:
        rmse_gcn, rmse_ablation, rmse_delta (abl − gcn, positive = GCN helps),
        mae_gcn,  mae_ablation,  mae_delta,
        pct_steps_helped, mean_influence, max_influence, mean_delta
    """
    test_arr = np.asarray(test,               dtype=float)
    full_arr = np.asarray(full_forecast,      dtype=float)
    noz_arr  = np.asarray(full_no_z_forecast, dtype=float)

    min_len = min(len(test_arr), len(full_arr), len(noz_arr))
    test_arr = test_arr[:min_len]
    full_arr = full_arr[:min_len]
    noz_arr  = noz_arr[:min_len]

    try:
        x_axis = pd.to_datetime(test_index[:min_len])
    except Exception:
        x_axis = np.arange(min_len)

    # ── stats ──────────────────────────────────────────────────────────────
    rmse_gcn  = _safe_rmse(test_arr, full_arr)
    rmse_abl  = _safe_rmse(test_arr, noz_arr)
    mae_gcn   = _safe_mae(test_arr,  full_arr)
    mae_abl   = _safe_mae(test_arr,  noz_arr)
    rmse_delta = rmse_abl  - rmse_gcn   # positive = GCN helps
    mae_delta  = mae_abl   - mae_gcn

    delta    = full_arr - noz_arr
    err_gcn  = np.abs(full_arr - test_arr)
    err_abl  = np.abs(noz_arr  - test_arr)

    valid_mask = (~np.isnan(err_gcn)) & (~np.isnan(err_abl))
    pct_helped = (
        float((err_gcn[valid_mask] < err_abl[valid_mask]).mean() * 100)
        if valid_mask.any() else float("nan")
    )
    mean_influence = float(np.nanmean(np.abs(delta)))
    max_influence  = float(np.nanmax(np.abs(delta)))
    mean_delta_val = float(np.nanmean(delta))

    stats = {
        "rmse_gcn":        rmse_gcn,
        "rmse_ablation":   rmse_abl,
        "rmse_delta":      rmse_delta,
        "mae_gcn":         mae_gcn,
        "mae_ablation":    mae_abl,
        "mae_delta":       mae_delta,
        "pct_steps_helped": pct_helped,
        "mean_influence":  mean_influence,
        "max_influence":   max_influence,
        "mean_delta":      mean_delta_val,
    }

    # ── training diagnostics (optional) ────────────────────────────────────
    diag_df = None
    if diag_csv_path is not None and os.path.exists(diag_csv_path):
        try:
            _df = pd.read_csv(diag_csv_path)
            # keep only the non-ablation run for this seed
            _abl_col = _df.get("ablate_z", _df.get("ablation", None))
            if _abl_col is not None:
                _mask = ~_abl_col.astype(str).str.lower().isin(["true", "1"])
                _df = _df[_mask]
            if "seed" in _df.columns:
                _df = _df[_df["seed"] == seed]
            if len(_df) > 0:
                diag_df = _df.reset_index(drop=True)
        except Exception:
            diag_df = None

    has_diag = diag_df is not None and len(diag_df) > 0
    n_rows = 4 if has_diag else 3
    row_heights = ([0.38, 0.20, 0.20, 0.22] if has_diag
                   else [0.45, 0.25, 0.30])

    subplot_titles = [
        "Forecast vs Ground Truth",
        "GCN Influence (full − ablation) per step",
        "Step-wise Absolute Error: GCN+LSTM vs Ablation",
    ]
    if has_diag:
        subplot_titles.append("Training Diagnostics (‖z‖ and Var(z) by epoch)")

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=False,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        vertical_spacing=0.06,
    )

    # ── row 1: forecast overlay ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_axis, y=test_arr, name="Ground Truth",
        line=dict(color="black", width=2), mode="lines",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_axis, y=full_arr,
        name=f"GCN+LSTM  RMSE={rmse_gcn:.4f}",
        line=dict(color="royalblue", width=1.5), mode="lines",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_axis, y=noz_arr,
        name=f"Ablation z=0  RMSE={rmse_abl:.4f}",
        line=dict(color="tomato", width=1.5, dash="dash"), mode="lines",
    ), row=1, col=1)

    # ── row 2: delta bar ────────────────────────────────────────────────────
    bar_colors = ["seagreen" if d >= 0 else "crimson" for d in delta]
    fig.add_trace(go.Bar(
        x=x_axis, y=delta,
        name="GCN delta (full−abl)",
        marker_color=bar_colors,
        showlegend=True,
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # ── row 3: step-wise error ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_axis, y=err_gcn,
        name="|err| GCN+LSTM",
        line=dict(color="royalblue", width=1),
        fill="tozeroy", fillcolor="rgba(65,105,225,0.15)",
        mode="lines",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=x_axis, y=err_abl,
        name="|err| Ablation",
        line=dict(color="tomato", width=1),
        fill="tozeroy", fillcolor="rgba(255,99,71,0.15)",
        mode="lines",
    ), row=3, col=1)

    # ── row 4: training diagnostics ─────────────────────────────────────────
    if has_diag:
        epochs = diag_df["epoch"] if "epoch" in diag_df.columns else diag_df.index
        if "z_norm_mean" in diag_df.columns:
            fig.add_trace(go.Scatter(
                x=epochs, y=diag_df["z_norm_mean"],
                name="‖z‖ mean",
                line=dict(color="purple", width=1.5),
                mode="lines",
            ), row=4, col=1)
        if "z_var_mean" in diag_df.columns:
            fig.add_trace(go.Scatter(
                x=epochs, y=diag_df["z_var_mean"],
                name="Var(z)",
                line=dict(color="darkorange", width=1.5, dash="dot"),
                mode="lines",
                yaxis="y4",
            ), row=4, col=1)
        fig.update_xaxes(title_text="Epoch", row=4, col=1)

    # ── summary annotation (top of figure) ─────────────────────────────────
    sign_r = "+" if rmse_delta >= 0 else ""
    sign_m = "+" if mae_delta  >= 0 else ""
    annotation_text = (
        f"ΔRMSE (abl−gcn) = {sign_r}{rmse_delta:.4f}  |  "
        f"ΔMAE = {sign_m}{mae_delta:.4f}  |  "
        f"Steps helped = {pct_helped:.1f}%  |  "
        f"Mean |influence| = {mean_influence:.4f}"
    )
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.01, y=1.01,
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="lightgray",
        borderwidth=1,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=260 * n_rows,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.04,
            xanchor="right",  x=1,
            font=dict(size=10),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    for i in range(1, n_rows + 1):
        fig.update_xaxes(
            showgrid=True, gridcolor="rgba(200,200,200,0.4)", row=i, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(200,200,200,0.4)", row=i, col=1,
        )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.write_html(save_path)
    else:
        fig.show()

    return stats


# ── cross-product summary ──────────────────────────────────────────────────
def plot_influence_summary(influence_records: list, output_dir: str) -> None:
    """
    Aggregate GCN influence statistics across all products and write summary
    HTML plots to ``output_dir``.

    Each entry in ``influence_records`` must contain at minimum:
        product_id, store_id, seed, metric, label,
        rmse_gcn, rmse_ablation, rmse_delta,
        mae_gcn,  mae_ablation,
        pct_steps_helped, mean_influence
    """
    if not influence_records:
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(influence_records)

    # ── 1. RMSE delta bar (sorted, one bar per product×metric avg) ─────────
    if "product_id" in df.columns and "metric" in df.columns:
        agg = (
            df.groupby(["product_id", "metric"])["rmse_delta"]
            .mean()
            .reset_index()
            .sort_values("rmse_delta", ascending=False)
        )
        bar_colors = ["seagreen" if v >= 0 else "crimson"
                      for v in agg["rmse_delta"]]
        fig_bar = go.Figure(go.Bar(
            x=[f"{r.product_id}/{r.metric}" for _, r in agg.iterrows()],
            y=agg["rmse_delta"],
            marker_color=bar_colors,
            text=[f"{v:+.4f}" for v in agg["rmse_delta"]],
            textposition="outside",
        ))
        fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
        n_pos = int((agg["rmse_delta"] >= 0).sum())
        fig_bar.update_layout(
            title=(
                f"ΔRMSE (Ablation − GCN+LSTM) per product/metric — "
                f"positive = GCN helps  ({n_pos}/{len(agg)} configs)"
            ),
            xaxis_tickangle=-45,
            yaxis_title="ΔRMSE",
            height=max(450, 30 * len(agg) + 150),
            plot_bgcolor="white",
        )
        fig_bar.write_html(os.path.join(output_dir, "summary_rmse_delta.html"))

    # ── 2. Distribution of pct_steps_helped ───────────────────────────────
    if "pct_steps_helped" in df.columns:
        psh = df["pct_steps_helped"].dropna()
        fig_hist = go.Figure(go.Histogram(
            x=psh, nbinsx=25,
            marker_color="steelblue", opacity=0.8,
            name="pct_steps_helped",
        ))
        fig_hist.add_vline(
            x=50, line_dash="dash", line_color="red",
            annotation_text="50% (break-even)",
            annotation_position="top right",
        )
        fig_hist.add_vline(
            x=float(psh.mean()), line_dash="dot", line_color="navy",
            annotation_text=f"mean={psh.mean():.1f}%",
            annotation_position="top left",
        )
        fig_hist.update_layout(
            title="% Forecast Steps Where GCN Reduced Absolute Error",
            xaxis_title="% steps helped", yaxis_title="Count",
            height=400, plot_bgcolor="white",
        )
        fig_hist.write_html(os.path.join(output_dir, "summary_pct_helped.html"))

    # ── 3. Scatter: mean_influence vs rmse_delta ───────────────────────────
    if {"mean_influence", "rmse_delta"}.issubset(df.columns):
        dv = df.dropna(subset=["mean_influence", "rmse_delta"])
        hover_text = [
            f"product={r.get('product_id', '?')}<br>"
            f"metric={r.get('metric', '?')}<br>"
            f"seed={r.get('seed', '?')}<br>"
            f"ΔRMSE={r['rmse_delta']:+.4f}<br>"
            f"mean_influence={r['mean_influence']:.4f}"
            for r in dv.to_dict("records")
        ]
        fig_scatter = go.Figure(go.Scatter(
            x=dv["mean_influence"],
            y=dv["rmse_delta"],
            mode="markers",
            hovertext=hover_text,
            hoverinfo="text",
            marker=dict(
                color=dv["rmse_delta"],
                colorscale="RdYlGn",
                size=9,
                colorbar=dict(title="ΔRMSE"),
                showscale=True,
                line=dict(width=0.5, color="gray"),
            ),
        ))
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_scatter.update_layout(
            title=(
                "Mean GCN Influence Magnitude vs RMSE Delta — "
                "positive ΔRMSE = GCN reduces error"
            ),
            xaxis_title="Mean |full forecast − ablation|",
            yaxis_title="ΔRMSE (ablation − GCN+LSTM)",
            height=480, plot_bgcolor="white",
        )
        fig_scatter.write_html(
            os.path.join(output_dir, "summary_influence_vs_rmse.html")
        )

    # ── 4. Summary table ───────────────────────────────────────────────────
    cols_ordered = [
        "product_id", "store_id", "seed", "metric",
        "rmse_gcn", "rmse_ablation", "rmse_delta",
        "mae_gcn", "mae_ablation", "mae_delta",
        "pct_steps_helped", "mean_influence", "max_influence",
    ]
    available = [c for c in cols_ordered if c in df.columns]
    df_tbl = df[available].copy()
    fmt_cols = [c for c in available
                if c not in ("product_id", "store_id", "seed", "metric")]
    for col in fmt_cols:
        df_tbl[col] = df_tbl[col].apply(
            lambda v: f"{v:+.4f}" if pd.notna(v) else ""
        )
    # alternating row colours
    n_rows_tbl = len(df_tbl)
    fill_colors = [["white" if i % 2 == 0 else "aliceblue" for i in range(n_rows_tbl)]]
    fig_tbl = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in df_tbl.columns],
            fill_color="steelblue",
            font=dict(color="white", size=11),
            align="left",
            line_color="white",
        ),
        cells=dict(
            values=[df_tbl[c].tolist() for c in df_tbl.columns],
            fill_color=fill_colors * len(df_tbl.columns),
            align="left",
            font=dict(size=10),
        ),
    ))
    fig_tbl.update_layout(
        title="GCN Influence — Full Results Table",
        height=max(400, 28 * n_rows_tbl + 120),
    )
    fig_tbl.write_html(os.path.join(output_dir, "summary_table.html"))

    # ── console summary ────────────────────────────────────────────────────
    n_pos_pct = int((df["pct_steps_helped"] > 50).sum()) if "pct_steps_helped" in df.columns else 0
    print(
        f"[influence_summary] {len(df)} configs | "
        f"mean delta_RMSE={df['rmse_delta'].mean():+.4f} | "
        f"mean steps_helped={df['pct_steps_helped'].mean():.1f}% | "
        f"{n_pos_pct}/{len(df)} configs where GCN helps >50% of steps"
    )
