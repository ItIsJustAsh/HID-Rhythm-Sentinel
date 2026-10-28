import time
from pathlib import Path
import pandas as pd
from pynput import keyboard

rows=[]; start=time.perf_counter()
def on_press(_):
    rows.append({"timestamp_ms":(time.perf_counter()-start)*1000})
def on_release(key):
    return key != keyboard.Key.esc

print("Recording keyboard timing only. Press ESC to stop.")
print("Use only on a machine/session where you have permission.")
with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
    listener.join()
out=Path("recorded_session.csv")
pd.DataFrame(rows).to_csv(out,index=False)
print(f"Saved {len(rows)} timestamps to {out.resolve()}")
