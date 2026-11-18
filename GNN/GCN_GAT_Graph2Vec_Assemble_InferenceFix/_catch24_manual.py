"""
Pure NumPy/SciPy implementation of catch24.

Feature order (matches pycatch22.catch22_all with catch24=True):
  0  DN_HistogramMode_5
  1  DN_HistogramMode_10
  2  DN_OutlierInclude_p_001_mdrmd
  3  DN_OutlierInclude_n_001_mdrmd
  4  CO_f1ecac
  5  CO_FirstMin_ac
  6  SP_Summaries_welch_rect_area_5_1
  7  SP_Summaries_welch_rect_centroid
  8  FC_LocalSimple_mean3_stderr
  9  FC_LocalSimple_mean1_tauresrat
 10  MD_hrv_classic_pnn40
 11  SB_BinaryStats_mean_longstretch1
 12  SB_BinaryStats_diff_longstretch0
 13  SB_MotifThree_quantile_hh
 14  CO_HistogramAMI_even_2_5
 15  CO_trev_1_num
 16  IN_AutoMutualInfoStats_40_gaussian_fmmi
 17  SB_TransitionMatrix_3ac_sumdiagcov
 18  PD_PeriodicityWang_th0_01
 19  CO_Embed2_Dist_tau_d_expfit_meandiff
 20  SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1
 21  SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1
 22  DN_Mean                 (catch24 extra)
 23  DN_Spread_Std           (catch24 extra)
"""
from __future__ import annotations
import numpy as np
from scipy import stats as sp_stats


# ── helpers ────────────────────────────────────────────────────────────────

def _z_score(ts: np.ndarray) -> np.ndarray:
    mu, sigma = np.mean(ts), np.std(ts, ddof=1)
    return (ts - mu) / sigma if sigma > 0 else np.zeros_like(ts)


def _autocorr_vec(ts: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased autocorrelation at lags 0..max_lag via FFT (O(n log n))."""
    n = len(ts)
    var = float(np.var(ts))
    if var == 0 or max_lag < 0:
        return np.zeros(max_lag + 1)
    x = ts - np.mean(ts)
    m = 1
    while m < 2 * n:
        m <<= 1
    X = np.fft.rfft(x, n=m)
    acf_full = np.fft.irfft(X * np.conj(X), n=m)[:max_lag + 1].real
    return acf_full / (n * var)


def _longest_run(seq: np.ndarray, target: bool) -> int:
    max_run = current = 0
    for b in seq:
        current = (current + 1) if bool(b) == target else 0
        if current > max_run:
            max_run = current
    return max_run


# ── 0–1. DN_HistogramMode ─────────────────────────────────────────────────

def DN_HistogramMode_5(ts: np.ndarray) -> float:
    return _DN_HistogramMode(ts, 5)


def DN_HistogramMode_10(ts: np.ndarray) -> float:
    return _DN_HistogramMode(ts, 10)


def _DN_HistogramMode(ts: np.ndarray, n_bins: int) -> float:
    z = _z_score(ts)
    counts, edges = np.histogram(z, bins=n_bins)
    i = int(np.argmax(counts))
    return float((edges[i] + edges[i + 1]) / 2.0)


# ── 2–3. DN_OutlierInclude ────────────────────────────────────────────────

def DN_OutlierInclude_p_001_mdrmd(ts: np.ndarray) -> float:
    return _DN_OutlierInclude(ts, sign=+1)


def DN_OutlierInclude_n_001_mdrmd(ts: np.ndarray) -> float:
    return _DN_OutlierInclude(ts, sign=-1)


def _DN_OutlierInclude(ts: np.ndarray, sign: int) -> float:
    n = len(ts)
    sigma = np.std(ts, ddof=1)
    if sigma == 0:
        return 0.0
    z = (ts - np.mean(ts)) / sigma
    step = 0.01
    max_thresh = float(np.max(z) if sign == 1 else -np.min(z))
    if max_thresh <= 0:
        return 0.0
    n_steps = max(1, int(np.ceil(max_thresh / step)))
    mdrmd_vals = []
    for k in range(n_steps):
        thr = (k + 1) * step
        idx = np.where(z >= thr)[0] if sign == 1 else np.where(z <= -thr)[0]
        if len(idx) == 0:
            continue
        r_norm = (idx + 1) / n          # 1-indexed, normalized to [0,1]
        mdrmd_vals.append(float(np.median(np.abs(r_norm - 0.5))))
    return float(np.mean(mdrmd_vals)) if mdrmd_vals else 0.0


# ── 4. CO_f1ecac ──────────────────────────────────────────────────────────

def CO_f1ecac(ts: np.ndarray) -> float:
    n = len(ts)
    if n < 2:
        return 0.0
    thr = 1.0 / np.e
    acf = _autocorr_vec(ts, n - 1)
    for lag in range(1, n):
        if acf[lag] < thr:
            return float(lag)
    return float(n)


# ── 5. CO_FirstMin_ac ─────────────────────────────────────────────────────

def CO_FirstMin_ac(ts: np.ndarray) -> float:
    n = len(ts)
    if n < 3:
        return float(n)
    acf = _autocorr_vec(ts, n - 1)
    for lag in range(1, n - 1):
        if acf[lag] < acf[lag - 1] and acf[lag] < acf[lag + 1]:
            return float(lag)
    return float(n)


# ── 6–7. SP_Summaries_welch_rect ──────────────────────────────────────────

def _welch_rect(ts: np.ndarray):
    """Periodogram (single rectangular window over full signal)."""
    n = len(ts)
    fft_vals = np.fft.rfft(ts)
    psd = (np.abs(fft_vals) ** 2) / n
    freqs = np.fft.rfftfreq(n)
    return freqs, psd


def SP_Summaries_welch_rect_area_5_1(ts: np.ndarray) -> float:
    freqs, psd = _welch_rect(ts)
    total = float(np.trapz(psd, freqs))
    if total == 0:
        return 0.0
    cutoff = freqs[-1] / 5.0
    mask = freqs <= cutoff
    return float(np.trapz(psd[mask], freqs[mask]) / total)


def SP_Summaries_welch_rect_centroid(ts: np.ndarray) -> float:
    freqs, psd = _welch_rect(ts)
    total = float(np.sum(psd))
    return float(np.sum(freqs * psd) / total) if total > 0 else 0.0


# ── 8. FC_LocalSimple_mean3_stderr ────────────────────────────────────────

def FC_LocalSimple_mean3_stderr(ts: np.ndarray) -> float:
    window = 3
    n = len(ts)
    if n <= window:
        return 0.0
    preds = np.array([ts[t - window:t].mean() for t in range(window, n)])
    resid = ts[window:] - preds
    return float(np.std(resid, ddof=1))


# ── 9. FC_LocalSimple_mean1_tauresrat ─────────────────────────────────────

def FC_LocalSimple_mean1_tauresrat(ts: np.ndarray) -> float:
    resid = ts[1:] - ts[:-1]
    if len(resid) == 0:
        return 0.0
    tau = CO_f1ecac(resid)
    return float(tau / len(resid))


# ── 10. MD_hrv_classic_pnn40 ──────────────────────────────────────────────

def MD_hrv_classic_pnn40(ts: np.ndarray) -> float:
    sigma = np.std(ts, ddof=1)
    if sigma == 0:
        return 0.0
    return float(np.mean(np.abs(np.diff(ts)) > 0.4 * sigma))


# ── 11. SB_BinaryStats_mean_longstretch1 ──────────────────────────────────

def SB_BinaryStats_mean_longstretch1(ts: np.ndarray) -> float:
    return float(_longest_run(ts > np.mean(ts), True))


# ── 12. SB_BinaryStats_diff_longstretch0 ──────────────────────────────────

def SB_BinaryStats_diff_longstretch0(ts: np.ndarray) -> float:
    # Longest run of non-increasing steps in the diff-binarized series
    return float(_longest_run(np.diff(ts) > 0, False))


# ── 13. SB_MotifThree_quantile_hh ─────────────────────────────────────────

def SB_MotifThree_quantile_hh(ts: np.ndarray) -> float:
    qs = np.quantile(ts, [1 / 3, 2 / 3])
    binned = np.digitize(ts, qs).clip(0, 2)
    n = len(binned)
    counts: dict = {}
    for i in range(n - 2):
        m = (int(binned[i]), int(binned[i + 1]), int(binned[i + 2]))
        counts[m] = counts.get(m, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return float(-sum((c / total) * np.log(c / total) for c in counts.values()))


# ── 14. CO_HistogramAMI_even_2_5 ──────────────────────────────────────────

def CO_HistogramAMI_even_2_5(ts: np.ndarray) -> float:
    lag, n_bins = 2, 5
    n = len(ts)
    if n <= lag:
        return 0.0
    mn, mx = float(ts.min()), float(ts.max())
    if mn == mx:
        return 0.0
    edges = np.linspace(mn, mx, n_bins + 1)
    edges[-1] += 1e-10
    b1 = np.searchsorted(edges[1:], ts[:n - lag], side='left').clip(0, n_bins - 1)
    b2 = np.searchsorted(edges[1:], ts[lag:],      side='left').clip(0, n_bins - 1)
    joint = np.zeros((n_bins, n_bins))
    np.add.at(joint, (b1, b2), 1)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.where(
            (joint > 0) & (px > 0) & (py > 0),
            np.log(np.where(joint > 0, joint / (px * py), 1.0)),
            0.0,
        )
    return float((joint * log_ratio).sum())


# ── 15. CO_trev_1_num ─────────────────────────────────────────────────────

def CO_trev_1_num(ts: np.ndarray) -> float:
    z = _z_score(ts)
    return float(np.mean((z[1:] - z[:-1]) ** 3))


# ── 16. IN_AutoMutualInfoStats_40_gaussian_fmmi ───────────────────────────

def IN_AutoMutualInfoStats_40_gaussian_fmmi(ts: np.ndarray) -> float:
    max_lag = min(40, len(ts) - 2)
    if max_lag < 2:
        return float(max_lag)
    ami = []
    for lag in range(1, max_lag + 1):
        a, b = ts[:-lag], ts[lag:]
        if np.std(a) == 0 or np.std(b) == 0:
            rho = 0.0
        else:
            rho = float(np.corrcoef(a, b)[0, 1])
        ami.append(-0.5 * np.log(max(1.0 - rho * rho, 1e-10)))
    for i in range(1, len(ami) - 1):
        if ami[i] < ami[i - 1] and ami[i] < ami[i + 1]:
            return float(i + 1)       # 1-indexed lag of first minimum
    return float(len(ami))


# ── 17. SB_TransitionMatrix_3ac_sumdiagcov ────────────────────────────────

def SB_TransitionMatrix_3ac_sumdiagcov(ts: np.ndarray) -> float:
    tau = max(1, int(CO_FirstMin_ac(ts)))
    n_bins = 3
    qs = np.quantile(ts, np.array([1, 2]) / n_bins)
    binned = np.digitize(ts, qs).clip(0, n_bins - 1)
    n = len(binned)
    T = np.zeros((n_bins, n_bins))
    np.add.at(T, (binned[:n - tau], binned[tau:]), 1)
    rs = T.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    T /= rs
    # np.cov(T): rows are treated as variables → trace = sum of row variances
    return float(np.trace(np.cov(T)))


# ── 18. PD_PeriodicityWang_th0_01 ─────────────────────────────────────────

def PD_PeriodicityWang_th0_01(ts: np.ndarray) -> float:
    n = len(ts)
    max_lag = n // 2
    if max_lag < 2:
        return 0.0
    thr = 0.01
    acf = _autocorr_vec(ts, max_lag)
    # Find first local minimum, then first lag after it that exceeds thr
    for i in range(1, max_lag - 1):
        if acf[i] < acf[i - 1] and acf[i] < acf[i + 1]:
            for j in range(i + 1, max_lag + 1):
                if acf[j] > thr:
                    return float(j)
            break
    return 0.0


# ── 19. CO_Embed2_Dist_tau_d_expfit_meandiff ──────────────────────────────

def CO_Embed2_Dist_tau_d_expfit_meandiff(ts: np.ndarray) -> float:
    tau = max(1, int(CO_FirstMin_ac(ts)))
    n = len(ts)
    if n <= tau + 1:
        return 0.0
    x1, x2 = ts[:n - tau], ts[tau:]
    dists = np.sqrt(np.diff(x1) ** 2 + np.diff(x2) ** 2)
    if len(dists) == 0:
        return 0.0
    mu_d = float(np.mean(dists))
    if mu_d == 0:
        return 0.0
    lam = 1.0 / mu_d
    sorted_d = np.sort(dists)
    nd = len(sorted_d)
    ecdf = np.arange(1, nd + 1) / nd
    tcdf = 1.0 - np.exp(-lam * sorted_d)
    return float(np.mean(np.abs(ecdf - tcdf)))


# ── 20–21. SC_FluctAnal ───────────────────────────────────────────────────

def SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(ts: np.ndarray) -> float:
    return _SC_FluctAnal(ts, method='rsrange', poly_order=1)


def SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(ts: np.ndarray) -> float:
    return _SC_FluctAnal(ts, method='dfa', poly_order=2)


def _SC_FluctAnal(ts: np.ndarray, method: str, poly_order: int) -> float:
    n = len(ts)
    min_scale, max_scale = 5, n // 4
    if max_scale < min_scale:
        return 0.0
    scales = np.unique(
        np.round(np.exp(np.linspace(np.log(min_scale), np.log(max_scale), 50))).astype(int)
    )
    scales = scales[(scales >= min_scale) & (scales <= max_scale)]

    x = np.cumsum(ts - ts.mean()) if method == 'dfa' else _z_score(ts)

    log_s, log_F = [], []
    for s in scales:
        n_win = n // s
        if n_win < 2:
            continue
        F_vals = []
        for w in range(n_win):
            seg = x[w * s:(w + 1) * s]
            if method == 'dfa':
                t = np.arange(len(seg), dtype=float)
                p = np.polyfit(t, seg, poly_order)
                resid = seg - np.polyval(p, t)
                F_vals.append(float(np.mean(resid ** 2)))
            else:
                std = float(np.std(seg, ddof=1))
                if std > 0:
                    F_vals.append((seg.max() - seg.min()) / std)
        if not F_vals:
            continue
        if method == 'dfa':
            log_F.append(0.5 * np.log(max(np.mean(F_vals), 1e-10)))
        else:
            log_F.append(np.log(max(np.mean(F_vals), 1e-10)))
        log_s.append(np.log(float(s)))

    if len(log_s) < 3:
        return 0.0
    log_s = np.array(log_s)
    log_F = np.array(log_F)
    slope, intercept, *_ = sp_stats.linregress(log_s, log_F)
    residuals = log_F - (slope * log_s + intercept)
    # prop_r1: proportion of scales whose residual is within 1 unit of the fit
    return float(np.mean(np.abs(residuals) < 1.0))


# ── 22–23. DN_Mean / DN_Spread_Std (catch24 extras) ──────────────────────

def DN_Mean(ts: np.ndarray) -> float:
    return float(np.mean(ts))


def DN_Spread_Std(ts: np.ndarray) -> float:
    return float(np.std(ts, ddof=1))


# ── public API (drop-in for pycatch22) ────────────────────────────────────

CATCH24_NAMES = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "SP_Summaries_welch_rect_area_5_1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
    "FC_LocalSimple_mean1_tauresrat",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "DN_Mean",
    "DN_Spread_Std",
]

_CATCH24_FUNCS = [
    DN_HistogramMode_5,
    DN_HistogramMode_10,
    DN_OutlierInclude_p_001_mdrmd,
    DN_OutlierInclude_n_001_mdrmd,
    CO_f1ecac,
    CO_FirstMin_ac,
    SP_Summaries_welch_rect_area_5_1,
    SP_Summaries_welch_rect_centroid,
    FC_LocalSimple_mean3_stderr,
    FC_LocalSimple_mean1_tauresrat,
    MD_hrv_classic_pnn40,
    SB_BinaryStats_mean_longstretch1,
    SB_BinaryStats_diff_longstretch0,
    SB_MotifThree_quantile_hh,
    CO_HistogramAMI_even_2_5,
    CO_trev_1_num,
    IN_AutoMutualInfoStats_40_gaussian_fmmi,
    SB_TransitionMatrix_3ac_sumdiagcov,
    PD_PeriodicityWang_th0_01,
    CO_Embed2_Dist_tau_d_expfit_meandiff,
    SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
    SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
    DN_Mean,
    DN_Spread_Std,
]


def catch22_all(ts: list | np.ndarray, catch24: bool = False) -> dict:
    """Drop-in replacement for pycatch22.catch22_all."""
    arr = np.asarray(ts, dtype=np.float64)
    width = 24 if catch24 else 22
    values = []
    for f in _CATCH24_FUNCS[:width]:
        try:
            v = float(f(arr))
        except Exception:
            v = 0.0
        values.append(v if np.isfinite(v) else 0.0)
    return {"names": CATCH24_NAMES[:width], "values": values}
