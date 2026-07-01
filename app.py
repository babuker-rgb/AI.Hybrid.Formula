"""
Improved PINN Streamlit app.

Main fixes:
- Targets are NOT scaled. The PINN trains on physical units directly.
- Predictions are NOT inverse-transformed.
- LayerNorm replaces BatchNorm for safe single-sample prediction.
- model.eval() is used before inference.
"""

import datetime
import time
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

TENSILE_MIN = 1.90
EFRF_MAX = 0.40
MCC_MAX = 8.0
DENSITY_MAX = 0.99
PRESSURE_MAX = 300.0
BINDER_MIN = 0.5
BINDER_MAX = 5.0

FEATURE_NAMES = [
    "API_%", "MCC_%", "PVPP_%", "MgSt_%",
    "Binder_%", "Pressure_MPa", "Speed_rpm", "Granule_Size_um"
]

DEFAULTS = {
    "api": 90.5, "binder": 2.7, "pvpp": 3.0, "mgst": 0.20,
    "mcc": 3.6, "pressure": 230.0, "speed": 12.0, "granule": 125.0
}


def normalize_formulation(api, mcc, pvpp, mgst, binder):
    api = float(np.clip(api, 85, 95))
    mcc = float(np.clip(mcc, 0, MCC_MAX))
    pvpp = float(np.clip(pvpp, 0.5, 6))
    mgst = float(np.clip(mgst, 0.01, 1.2))
    binder = float(np.clip(binder, BINDER_MIN, BINDER_MAX))

    total = api + mcc + pvpp + mgst + binder
    if total < 100:
        mcc = min(MCC_MAX, mcc + (100 - total))
    elif total > 100:
        scale = 100 / total
        api, mcc, pvpp, mgst, binder = api * scale, mcc * scale, pvpp * scale, mgst * scale, binder * scale

    return api, mcc, pvpp, mgst, binder


def add_interaction_features(x):
    x = np.asarray(x, dtype=np.float32)
    api = x[:, 0:1]
    mcc = x[:, 1:2]
    binder = x[:, 4:5]
    pressure = x[:, 5:6]
    speed = x[:, 6:7]

    return np.concatenate([
        x,
        pressure * binder,
        pressure * api,
        np.clip(pressure / (speed + 0.1), 0, 1000),
        np.clip(api / (mcc + 0.1), 0, 1000),
        np.clip(binder / (speed + 0.1), 0, 100),
    ], axis=1)


def generate_data(n=700, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 8), dtype=np.float32)
    y = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        api = rng.uniform(85, 95)
        binder = rng.uniform(BINDER_MIN, BINDER_MAX)
        pvpp = rng.uniform(0.5, 6)
        mgst = rng.uniform(0.01, 1.2)
        mcc = rng.uniform(0, MCC_MAX)
        pressure = rng.uniform(80, PRESSURE_MAX)
        speed = rng.uniform(1, 50)
        granule = rng.uniform(30, 250)

        api, mcc, pvpp, mgst, binder = normalize_formulation(api, mcc, pvpp, mgst, binder)
        x[i] = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]

        k = max(0.006, 0.032 * (1 - 0.35 * (api - 85) / 10))
        a = 1.0 + 0.12 * (binder - 2.0) - 0.18 * (mgst - 0.2)
        density = np.clip(1 - np.exp(-(k * pressure + a)), 0.40, DENSITY_MAX)

        tensile = (
            2.35 - 0.10 * (api - 90) + 0.32 * (binder - 2.5)
            + 0.006 * (pressure - 180) - 1.05 * (mgst - 0.2)
            - 0.010 * (speed - 10) + 0.035 * (mcc - 3.5)
        )
        tensile = np.clip(tensile + rng.normal(0, 0.08), 0.55, 6.0)

        er = (
            0.55 + 0.035 * (api - 90) + 0.012 * (speed - 10)
            - 0.0025 * (pressure - 180) - 0.045 * (binder - 2.5)
        )
        er = np.clip(er + rng.normal(0, 0.025), 0.12, 1.35)

        y[i] = [density, tensile, er]

    df = pd.DataFrame(x, columns=FEATURE_NAMES)
    df["Density"] = y[:, 0]
    df["Tensile_Strength_MPa"] = y[:, 1]
    df["Elastic_Recovery"] = y[:, 2]
    return df


class PINN(nn.Module):
    def __init__(self, input_dim=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.Tanh(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.Tanh(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.Tanh(),
            nn.Linear(64, 5),
        )
        self.loss_history = {"train": [], "val": []}

    def forward(self, x):
        raw = self.net(x)
        density = DENSITY_MAX * torch.sigmoid(raw[:, 0:1])
        tensile = torch.nn.functional.softplus(raw[:, 1:2]) + 0.05
        er = torch.nn.functional.softplus(raw[:, 2:3]) + 0.02
        k = 0.005 + 0.095 * torch.sigmoid(raw[:, 3:4])
        a = 0.50 + 1.50 * torch.sigmoid(raw[:, 4:5])
        return torch.cat([density, tensile, er, k, a], dim=1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            return self.forward(x)[:, :3].numpy()

    def loss(self, x_scaled, x_raw, y):
        pred = self.forward(x_scaled)
        data_loss = nn.MSELoss()(pred[:, :3], y)

        density = pred[:, 0]
        tensile = pred[:, 1]
        er = pred[:, 2]
        k = pred[:, 3]
        a = pred[:, 4]
        pressure = x_raw[:, 5]

        heckel_left = torch.log(1 / (1 - torch.clamp(density, 0.01, DENSITY_MAX)))
        heckel_right = k * pressure + a
        heckel_loss = torch.mean((heckel_left - heckel_right) ** 2)

        efrf = er / (tensile + 1e-8)
        efrf_loss = torch.mean(torch.relu(efrf - EFRF_MAX) ** 2)

        return data_loss + 0.20 * heckel_loss + 0.30 * efrf_loss


@st.cache_resource(show_spinner=False)
def load_model():
    torch.manual_seed(42)
    df = generate_data()

    x_raw = df[FEATURE_NAMES].values.astype(np.float32)
    y = df[["Density", "Tensile_Strength_MPa", "Elastic_Recovery"]].values.astype(np.float32)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(add_interaction_features(x_raw)).astype(np.float32)

    x_train, x_val, raw_train, raw_val, y_train, y_val = train_test_split(
        x_scaled, x_raw, y, test_size=0.2, random_state=42
    )

    model = PINN(input_dim=x_scaled.shape[1])
    opt = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)

    xt = torch.tensor(x_train, dtype=torch.float32)
    rt = torch.tensor(raw_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    xv = torch.tensor(x_val, dtype=torch.float32)
    rv = torch.tensor(raw_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)

    for epoch in range(600):
        model.train()
        opt.zero_grad()
        train_loss = model.loss(xt, rt, yt)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        model.eval()
        val_loss = model.loss(xv, rv, yv)
        model.loss_history["train"].append(float(train_loss.detach()))
        model.loss_history["val"].append(float(val_loss.detach()))

    model.eval()
    return model, scaler, df


def predict(model, scaler, inputs):
    x = add_interaction_features(np.array([inputs], dtype=np.float32))
    xs = scaler.transform(x).astype(np.float32)
    density, tensile, er = model.predict(xs)[0]
    efrf = er / max(tensile, 1e-8)
    return float(density), float(tensile), float(er), float(efrf)


def compare_models(df, scaler, model):
    x = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["Tensile_Strength_MPa"].values.astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

    x_train = scaler.transform(add_interaction_features(x_train))
    x_test = scaler.transform(add_interaction_features(x_test))

    rows = []
    pinn_pred = model.predict(x_test)[:, 1]
    rows.append(["PINN", r2_score(y_test, pinn_pred), np.sqrt(mean_squared_error(y_test, pinn_pred)), mean_absolute_error(y_test, pinn_pred), "Enforced"])

    for name, mdl in {
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=800, random_state=7),
        "Random Forest": RandomForestRegressor(n_estimators=120, random_state=7),
    }.items():
        mdl.fit(x_train, y_train)
        pred = mdl.predict(x_test)
        rows.append([name, r2_score(y_test, pred), np.sqrt(mean_squared_error(y_test, pred)), mean_absolute_error(y_test, pred), "Not enforced"])

    return pd.DataFrame(rows, columns=["Model", "R2", "RMSE", "MAE", "Physics"])


def main():
    st.set_page_config(page_title="Improved PINN Framework", layout="wide")
    st.title("Improved Physics-Informed Neural Network Framework")

    model, scaler, df = load_model()

    with st.sidebar:
        st.header("Constraints")
        st.write(f"Tensile >= {TENSILE_MIN:.2f} MPa")
        st.write(f"EFRF < {EFRF_MAX:.2f}")
        st.write(f"MCC <= {MCC_MAX:.1f}%")

    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)

    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("Inputs")
        api = st.slider("API (%)", 85.0, 95.0, float(st.session_state.api), 0.1)
        binder = st.slider("Binder (%)", BINDER_MIN, BINDER_MAX, float(st.session_state.binder), 0.1)
        pvpp = st.slider("PVPP (%)", 0.5, 6.0, float(st.session_state.pvpp), 0.1)
        mgst = st.slider("Mg-St (%)", 0.01, 1.2, float(st.session_state.mgst), 0.01)
        mcc = st.slider("MCC (%)", 0.0, MCC_MAX, float(st.session_state.mcc), 0.1)
        pressure = st.slider("Pressure (MPa)", 80.0, PRESSURE_MAX, float(st.session_state.pressure), 1.0)
        speed = st.slider("Speed (rpm)", 1.0, 50.0, float(st.session_state.speed), 0.5)
        granule = st.slider("Granule size (um)", 30.0, 250.0, float(st.session_state.granule), 1.0)

    inputs = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
    density, tensile, er, efrf = predict(model, scaler, inputs)
    status = "PASS" if tensile >= TENSILE_MIN and efrf < EFRF_MAX else "FAIL"

    with right:
        st.subheader("Prediction")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Density", f"{density:.3f}")
        c2.metric("Tensile", f"{tensile:.3f} MPa")
        c3.metric("Elastic recovery", f"{er:.3f}")
        c4.metric("EFRF", f"{efrf:.4f}")

        if status == "PASS":
            st.success("Formulation satisfies the constraints.")
        else:
            st.error("Formulation fails at least one constraint.")

        tabs = st.tabs(["Model comparison", "Training curves", "Physics outputs"])

        with tabs[0]:
            comp = compare_models(df, scaler, model)
            st.dataframe(comp, use_container_width=True, hide_index=True)

        with tabs[1]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=model.loss_history["train"], name="Train loss"))
            fig.add_trace(go.Scatter(y=model.loss_history["val"], name="Validation loss"))
            fig.update_layout(yaxis_type="log", xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            x = scaler.transform(add_interaction_features(np.array([inputs], dtype=np.float32)))
            with torch.no_grad():
                full = model.forward(torch.tensor(x, dtype=torch.float32)).numpy()[0]
            st.write(pd.DataFrame([{
                "k": full[3],
                "A": full[4],
                "Heckel_left": np.log(1 / (1 - density)),
                "Heckel_right": full[3] * pressure + full[4],
            }]))


if __name__ == "__main__":
    main()
