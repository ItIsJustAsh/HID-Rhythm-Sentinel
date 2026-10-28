import joblib
from features import extract_features_from_timestamps, feature_frame

def load_bundle(path="models/hid_rhythm_sentinel.joblib"):
    return joblib.load(path)

def predict(timestamps,bundle):
    f=extract_features_from_timestamps(timestamps)
    p=float(bundle["model"].predict_proba(feature_frame(f))[0,1])
    verdict=("Likely Human Typing" if p<.40 else
             "Suspicious Timing Pattern" if p<.70 else
             "Likely HID Injection")
    return f,p,verdict
