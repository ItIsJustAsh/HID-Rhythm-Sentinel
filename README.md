# HID-Rhythm Sentinel

Behavioral detection of possible USB keyboard-injection attacks using inter-key timing.

## What this project actually does

```text
Keyboard/HID timestamps
        ↓
Inter-key intervals
        ↓
11 timing features
        ↓
Random Forest
        ↓
Injection probability
        ↓
Human / Suspicious / HID Injection
```

The included demonstration model uses **1,200 controlled synthetic sessions** (600 human-like + 600 injection-like), an **80/20 stratified split**, **11 features**, and a **250-tree Random Forest with max depth 10**.

The 100% evaluation result is a controlled synthetic demonstration and **not** a claim of 100% real-world detection.

## Run it

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

## Real keyboard demonstration

A hosted Streamlit app cannot directly read the physical keyboard of a visitor's computer. For an actual keyboard test, run:

```bash
python local_collector.py
```

Press keys normally and press **ESC** to stop. It saves `recorded_session.csv` containing timestamps only. Upload that CSV into the Streamlit app.

The collector intentionally does not save key names or typed characters.

## 11 features

Mean IKI, median IKI, standard deviation, coefficient of variation, P10, P90, burst ratio <80 ms, ultrafast ratio <35 ms, maximum burst length, timing entropy, and session length.

## Limitations

Synthetic training data, fixed thresholds, keyboard/device variability, legitimate automation, adaptive attackers, and offline completed-session analysis.

## Future work

Real HID traces, online sliding windows, device-specific baselines, adaptive thresholds, explainable alerts, adversarial slow/human-mimicking tests, and endpoint integration.

## Responsible use

Only record keyboard timing on systems and sessions where you have permission. Timing is still behavioral metadata and should be handled responsibly.
