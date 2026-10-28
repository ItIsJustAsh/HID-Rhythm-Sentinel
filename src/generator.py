import numpy as np
import pandas as pd

def generate_session(label, n_events=45, seed=42):
    rng = np.random.default_rng(seed)
    if label == "human":
        intervals = np.clip(rng.lognormal(np.log(150), .42, n_events-1), 55, 650)
    elif label == "injection":
        intervals = np.clip(rng.lognormal(np.log(18), .18, n_events-1), 6, 45)
    else:
        raise ValueError("label must be human or injection")
    return pd.DataFrame({
        "timestamp_ms": np.r_[0.0, np.cumsum(intervals)],
        "label": label,
        "event_index": np.arange(n_events)
    })

def make_dataset(human=600, injection=600, events=45, seed=42):
    parts=[]
    for i in range(human):
        d=generate_session("human",events,seed+i); d["session_id"]=i; parts.append(d)
    for i in range(injection):
        d=generate_session("injection",events,seed+human+i); d["session_id"]=human+i; parts.append(d)
    return pd.concat(parts, ignore_index=True)
