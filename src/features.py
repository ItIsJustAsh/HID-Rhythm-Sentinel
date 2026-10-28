import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "mean_iki_ms","median_iki_ms","std_iki_ms","cv_iki",
    "p10_iki_ms","p90_iki_ms","burst_ratio_lt80",
    "ultrafast_ratio_lt35","max_burst_len","timing_entropy",
    "session_length"
]

def timestamps_to_intervals(timestamps):
    ts = np.sort(np.asarray(list(timestamps), dtype=float))
    return np.diff(ts) if len(ts) >= 2 else np.array([])

def timing_entropy(x, bins=10):
    if len(x) == 0 or np.allclose(x, x[0]): return 0.0
    h, _ = np.histogram(x, bins=bins)
    p = h[h > 0].astype(float); p /= p.sum()
    return float(-(p*np.log2(p)).sum())

def extract_features_from_intervals(intervals):
    x = np.asarray(list(intervals), dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if len(x) == 0: return {n: 0.0 for n in FEATURE_NAMES}
    mean = float(x.mean()); std = float(x.std())
    fast = x < 80; ultra = x < 35
    longest = cur = 0
    for f in fast:
        cur = cur + 1 if f else 0
        longest = max(longest, cur)
    return {
        "mean_iki_ms": mean, "median_iki_ms": float(np.median(x)),
        "std_iki_ms": std, "cv_iki": float(std/mean) if mean else 0.0,
        "p10_iki_ms": float(np.percentile(x,10)),
        "p90_iki_ms": float(np.percentile(x,90)),
        "burst_ratio_lt80": float(fast.mean()),
        "ultrafast_ratio_lt35": float(ultra.mean()),
        "max_burst_len": float(longest),
        "timing_entropy": timing_entropy(x),
        "session_length": float(len(x)+1)
    }

def extract_features_from_timestamps(timestamps):
    return extract_features_from_intervals(timestamps_to_intervals(timestamps))

def feature_frame(features):
    return pd.DataFrame([[features[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES)
