import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/"src"))
from generator import generate_session
from predict import load_bundle,predict

st.set_page_config(page_title="HID-Rhythm Sentinel",page_icon="⌨️",layout="wide")
st.title("⌨️ HID-Rhythm Sentinel")
st.caption("Behavioral detection of possible USB keyboard-injection attacks using inter-key timing.")
st.info("This is an educational prototype. It analyzes timing metadata, not typed characters. The model is trained on controlled synthetic data.")

@st.cache_resource
def model(): return load_bundle(str(ROOT/"models/hid_rhythm_sentinel.joblib"))

def show(ts):
    f,p,v=predict(ts,model())
    a,b,c=st.columns(3)
    a.metric("Verdict",v); b.metric("Injection probability",f"{p:.1%}"); c.metric("Events",len(ts))
    st.progress(p,text=f"Injection probability: {p:.1%}")
    st.line_chart(pd.DataFrame({"inter_key_interval_ms":np.diff(np.sort(ts))}))
    st.dataframe(pd.DataFrame({"Feature":list(f),"Value":[round(x,4) for x in f.values()]}),hide_index=True,use_container_width=True)

t1,t2,t3=st.tabs(["Demo Simulator","Upload CSV","How It Works"])
with t1:
    kind=st.radio("Session type",["Human-like typing","HID injection-like typing"],horizontal=True)
    seed=st.number_input("Random seed",0,999999,42)
    if st.button("Generate & Analyze",type="primary"):
        label="human" if kind.startswith("Human") else "injection"
        d=generate_session(label,45,int(seed))
        show(d.timestamp_ms.to_numpy())
        st.download_button("Download generated CSV",d[["timestamp_ms"]].to_csv(index=False),"demo_session.csv","text/csv")
with t2:
    st.write("Upload a CSV with a `timestamp_ms` column.")
    st.download_button("Download CSV template","timestamp_ms\n0\n145\n301\n455\n612\n","timestamp_template.csv","text/csv")
    up=st.file_uploader("Timestamp CSV",type="csv")
    if up:
        d=pd.read_csv(up)
        if "timestamp_ms" not in d:
            st.error("Missing timestamp_ms column.")
        elif len(d)<2:
            st.error("At least two timestamps are required.")
        else:
            show(d.timestamp_ms.to_numpy())
with t3:
    st.subheader("Pipeline")
    st.code("HID / keyboard timestamps\n        ↓\nInter-key intervals\n        ↓\n11 timing features\n        ↓\nRandom Forest\n        ↓\nInjection probability\n        ↓\nHuman / Suspicious / HID Injection")
    st.markdown("**Thresholds:** < 0.40 Human • 0.40–0.69 Suspicious • ≥ 0.70 HID Injection")
    st.markdown("**Features:** mean, median, standard deviation, coefficient of variation, P10, P90, burst ratio <80 ms, ultrafast ratio <35 ms, maximum burst length, timing entropy, session length.")
    st.warning("The project's 100% result is only a controlled synthetic demonstration, not real-world accuracy.")
st.divider()
st.caption("HID-Rhythm Sentinel • timing metadata only • use local collection only with permission")
