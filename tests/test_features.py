from src.features import FEATURE_NAMES,extract_features_from_timestamps
from src.generator import generate_session

def test_features():
    d=generate_session("human",45,1)
    f=extract_features_from_timestamps(d.timestamp_ms)
    assert len(f)==11 and list(f)==FEATURE_NAMES

def test_demo_separation():
    h=extract_features_from_timestamps(generate_session("human",100,2).timestamp_ms)
    i=extract_features_from_timestamps(generate_session("injection",100,3).timestamp_ms)
    assert i["mean_iki_ms"] < h["mean_iki_ms"]
