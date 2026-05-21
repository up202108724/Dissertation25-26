
def infer_metric_type(metric):
    distance_metrics = ['euclidean','manhattan', 'hamming', 'amplitude_offset', 'slope_consistency', 'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss']
    if metric in distance_metrics:
        return 'distance'
    else:
        return 'similarity'