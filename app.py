
Physics Constraints
Density: 0.40–0.97
Tensile ≥ 1.90 MPa
EFRF < 0.40
MCC ≤ 8.0%
🧬 Hubryd AI Multi‑objective Optimization
Minimal Working Version · Free Tier

🔄 Training model from scratch...

🖥️ Using device: cpu

Early stopping at epoch 51

✅
Model ready!

API (%)


85.00

95.00

Binder (%)


0.50

5.00

PVPP (%)


0.50

6.00

Mg-St (%)


0.01

1.20

MCC (%)


0.00

8.00

✅
Total: 100.00%

streamlit.errors.StreamlitAPIException: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).

Traceback:
File "/mount/src/ai.hybrid.formula/app.py", line 601, in <module>
    pressure = st.slider("Pressure (MPa)", 80.0, PRESSURE_MAX, st.session_state.pressure, 1.0)
File "/home/adminuser/venv/lib/python3.14/site-packages/streamlit/runtime/metrics_util.py", line 698, in wrapped_func
    result = non_optional_func(*args, **kwargs)
File "/home/adminuser/venv/lib/python3.14/site-packages/streamlit/elements/widgets/slider.py", line 721, in slider
    return self._slider(
           ~~~~~~~~~~~~^
        label=label,
        ^^^^^^^^^^^^
    ...<14 lines>...
        ctx=ctx,
        ^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.14/site-packages/streamlit/elements/widgets/slider.py", line 942, in _slider
    raise StreamlitAPIException(msg)
