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

TENSILE_MIN = 1.90
EFRF_MAX = 0.40
MCC_MAX = 8.0
DENSITY_MAX = 0.99
PRESSURE_MAX = 300.0
BINDER_MIN = 0.5
BINDER_MAX = 5.0

FEATURE_NAMES = [
    "API_%",
    "MCC_%",
    "PVPP_%",
    "MgSt_%",
    "Binder_%",
    "Pressure_MPa",
    "Speed_rpm",
    "Granule_Size_um",
]

DEFAULTS = {
    "api": 90.5,
    "mcc": 3.6,
    "pvpp": 3.0,
    "mgst": 0.20,
    "binder": 2.7,
    "pressure": 230.0,
    "speed": 12.0,
    "granule": 125.0,
}


def normalize_formulation(api, mcc, pvpp, mgst, binder):
    api = float(np.clip(api, 85.0, 95.0))
    mcc = float(np.clip(mcc, 0.0, MCC_MAX))
    pvpp = float(np.clip(pvpp, 0.5, 6.0))
    mgst = float(np.clip(mgst, 0.01, 1.2))
    binder = float(np.clip(binder, BINDER_MIN, BINDER_MAX))

    total = api + mcc + pvpp + mgst + binder

    if total > 100.0:
        scale = 100.0 / total
        api *= scale
        mcc *= scale
        pvpp *= scale
        mgst *= scale
        binder *= scale
    elif total < 100.0:
        remainder = 100.0 - total
        add_to_mcc = min(MCC_MAX - mcc, remainder)
        mcc += add_to_mcc
        remainder -= add_to_mcc
        api = max(85.0, api - remainder)

    return api, mcc, pvpp, mgst, binder


def add_interaction_features(x_raw):
    x_raw = np.asarray(x_raw, dtype=np.float32)

    api = x_raw[:, 0:1]
    mcc = x_raw[:, 1:2]
    binder = x_raw[:, 4:5]
    pressure = x_raw[:, 5:6]
    speed = x_raw[:, 6:7]

    pressure_binder = pressure * binder
    pressure_api = pressure * api
    pressure_speed = np.clip(pressure / (speed + 0.1), 0.0, 1000.0)
    api_mcc = np.clip(api / (mcc + 0.1), 0.0, 1000.0)
    binder_speed = np.clip(binder / (speed + 0.1), 0.0, 100.0)

    return np.concatenate(
        [
            x_raw,
            pressure_binder,
            pressure_api,
            pressure_speed,
            api_mcc,
            binder_speed,
        ],
        axis=1,
    )


def generate_pinn_data(n_samples=2500, random_state=42):
    rng = np.random.default_rng(random_state)

    x = np.zeros((n_samples, 8), dtype=np.float32)
    y = np.zeros((n_samples, 3), dtype=np.float32)

    for i in range(n_samples):
        if i < n_samples // 2:
            api = rng.uniform(85.0, 95.0)
            mcc = rng.uniform(0.0, MCC_MAX)
            pvpp = rng.uniform(0.5, 6.0)
            mgst = rng.uniform(0.01, 1.2)
            binder = rng.uniform(BINDER_MIN, BINDER_MAX)
            pressure = rng.uniform(80.0, PRESSURE_MAX)
            speed = rng.uniform(1.0, 50.0)
            granule = rng.uniform(30.0, 250.0)
        else:
            api = np.clip(rng.normal(90.5, 1.7), 85.0, 95.0)
            mcc = np.clip(rng.normal(3.6, 1.1), 0.0, MCC_MAX)
            pvpp = np.clip(rng.normal(3.0, 0.55), 0.5, 6.0)
            mgst = np.clip(rng.normal(0.20, 0.06), 0.01, 1.2)
            binder = np.clip(rng.normal(2.7, 0.45), BINDER_MIN, BINDER_MAX)
            pressure = np.clip(rng.normal(230.0, 22.0), 80.0, PRESSURE_MAX)
            speed = np.clip(rng.normal(12.0, 4.0), 1.0, 50.0)
            granule = np.clip(rng.normal(125.0, 25.0), 30.0, 250.0)

        api, mcc, pvpp, mgst, binder = normalize_formulation(
            api,
            mcc,
            pvpp,
            mgst,
            binder,
        )

        x[i] = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]

        k_true = 0.020 + 0.010 * (1.0 - (api - 85.0) / 10.0)
        k_true += 0.003 * (binder - 2.5)
        k_true -= 0.002 * (speed - 12.0) / 12.0
        k_true = max(k_true, 0.006)

        a_true = 0.65 + 0.08 * binder - 0.12 * mgst

        density = 1.0 - np.exp(-(k_true * pressure / 2.7 + a_true))
        density = np.clip(
            density + rng.normal(0.0, 0.006),
            0.40,
            DENSITY_MAX,
        )

        tensile = (
            2.45
            - 0.095 * (api - 90.0)
            + 0.040 * (mcc - 3.5)
            + 0.35 * (binder - 2.5)
            + 0.0065 * (pressure - 180.0)
            - 0.95 * (mgst - 0.2)
            - 0.009 * (speed - 10.0)
            - 0.0008 * (granule - 125.0)
        )
        tensile = np.clip(
            tensile + rng.normal(0.0, 0.025),
            0.55,
            6.0,
        )

        elastic_recovery = (
            0.50
            + 0.030 * (api - 90.0)
            + 0.011 * (speed - 10.0)
            - 0.0022 * (pressure - 180.0)
            - 0.040 * (binder - 2.5)
            + 0.020 * (mgst - 0.2)
            + 0.0004 * (granule - 125.0)
        )
        elastic_recovery = np.clip(
            elastic_recovery + rng.normal(0.0, 0.008),
            0.12,
            1.35,
        )

        y[i] = [density, tensile, elastic_recovery]

    df = pd.DataFrame(x, columns=FEATURE_NAMES)
    df["Density"] = y[:, 0]
    df["Tensile_Strength_MPa"] = y[:, 1]
    df["Elastic_Recovery"] = y[:, 2]

    return df


class MultiTaskPINN(nn.Module):
    def __init__(self, input_dim=13):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(0.03),

            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(0.03),

            nn.Linear(192, 96),
            nn.LayerNorm(96),
            nn.SiLU(),

            nn.Linear(96, 48),
            nn.SiLU(),

            nn.Linear(48, 5),
        )

        self.loss_history = {
            "train": [],
            "val": [],
            "data": [],
            "physics": [],
        }

    def forward(self, x):
        raw = self.network(x)

        density = DENSITY_MAX * torch.sigmoid(raw[:, 0:1])
        tensile = torch.nn.functional.softplus(raw[:, 1:2]) + 0.05
        elastic_recovery = torch.nn.functional.softplus(raw[:, 2:3]) + 0.02

        k = 0.003 + 0.060 * torch.sigmoid(raw[:, 3:4])
        a = 0.20 + 1.30 * torch.sigmoid(raw[:, 4:5])

        return torch.cat([density, tensile, elastic_recovery, k, a], dim=1)

    def predict(self, x_scaled):
        self.eval()

        with torch.no_grad():
            if not isinstance(x_scaled, torch.Tensor):
                x_scaled = torch.tensor(x_scaled, dtype=torch.float32)

            prediction = self.forward(x_scaled)
            return prediction[:, :3].cpu().numpy()

    def compute_loss(
        self,
        x_scaled,
        x_raw,
        y_true,
        epoch=0,
        compute_grad=True,
        record_loss=False,
    ):
        prediction = self.forward(x_scaled)

        density = prediction[:, 0]
        tensile = prediction[:, 1]
        elastic_recovery = prediction[:, 2]
        k = prediction[:, 3]
        a = prediction[:, 4]

        pressure = x_raw[:, 5]

        data_loss = nn.MSELoss()(prediction[:, :3], y_true)

        density_clamped = torch.clamp(density, 0.01, DENSITY_MAX)
        heckel_left = torch.log(1.0 / (1.0 - density_clamped))
        heckel_right = k * pressure / 2.7 + a
        heckel_loss = torch.mean((heckel_left - heckel_right) ** 2)

        efrf = elastic_recovery / (tensile + 1e-8)
        efrf_loss = torch.mean(torch.relu(efrf - EFRF_MAX) ** 2)

        if compute_grad:
            if not x_scaled.requires_grad:
                x_scaled.requires_grad_(True)

            density_for_grad = self.forward(x_scaled)[:, 0]

            grad_density = torch.autograd.grad(
                outputs=density_for_grad,
                inputs=x_scaled,
                grad_outputs=torch.ones_like(density_for_grad),
                create_graph=True,
                retain_graph=True,
            )[0]

            pressure_gradient = grad_density[:, 5]
            monotonic_loss = torch.mean(torch.relu(-pressure_gradient) ** 2)
        else:
            monotonic_loss = torch.tensor(0.0, device=x_scaled.device)

        physics_weight = 0.03 + 0.12 * min(epoch / 500.0, 1.0)
        physics_loss = heckel_loss + efrf_loss + monotonic_loss
        total_loss = data_loss + physics_weight * physics_loss

        if record_loss:
            self.loss_history["train"].append(float(total_loss.detach().cpu()))
            self.loss_history["data"].append(float(data_loss.detach().cpu()))
            self.loss_history["physics"].append(float(physics_loss.detach().cpu()))

        return total_loss


@st.cache_resource(show_spinner=False)
def load_pinn_model():
    torch.manual_seed(42)
    np.random.seed(42)

    df = generate_pinn_data()

    x_raw = df[FEATURE_NAMES].values.astype(np.float32)
    y = df[
        ["Density", "Tensile_Strength_MPa", "Elastic_Recovery"]
    ].values.astype(np.float32)

    input_scaler = StandardScaler()
    x_scaled = input_scaler.fit_transform(
        add_interaction_features(x_raw)
    ).astype(np.float32)

    x_train, x_val, raw_train, raw_val, y_train, y_val = train_test_split(
        x_scaled,
        x_raw,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = MultiTaskPINN(input_dim=x_scaled.shape[1])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=7e-4,
        weight_decay=1e-5,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=80,
        factor=0.6,
    )

    x_train_t = torch.tensor(x_train, dtype=torch.float32, requires_grad=True)
    raw_train_t = torch.tensor(raw_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    raw_val_t = torch.tensor(raw_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    progress = st.progress(0, text="Training PINN...")

    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    max_epochs = 1500

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        train_loss = model.compute_loss(
            x_train_t,
            raw_train_t,
            y_train_t,
            epoch=epoch,
            compute_grad=True,
            record_loss=True,
        )

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()

        with torch.no_grad():
            val_prediction = model.forward(x_val_t)[:, :3]
            val_data_loss = nn.MSELoss()(val_prediction, y_val_t)

        val_loss = model.compute_loss(
            x_val_t,
            raw_val_t,
            y_val_t,
            epoch=epoch,
            compute_grad=False,
            record_loss=False,
        )

        val_value = float(val_loss.detach().cpu())
        model.loss_history["val"].append(val_value)

        scheduler.step(val_data_loss)

        if val_value < best_val:
            best_val = val_value
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch % 30 == 0:
            progress.progress(
                min((epoch + 1) / max_epochs, 1.0),
                text=f"Training PINN... epoch {epoch + 1}",
            )

        if stale_epochs >= 180:
            break

    progress.empty()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    val_prediction = model.predict(x_val)

    metrics = {
        "r2_density": r2_score(y_val[:, 0], val_prediction[:, 0]),
        "r2_tensile": r2_score(y_val[:, 1], val_prediction[:, 1]),
        "r2_er": r2_score(y_val[:, 2], val_prediction[:, 2]),
        "rmse_tensile": float(
            np.sqrt(mean_squared_error(y_val[:, 1], val_prediction[:, 1]))
        ),
        "mae_tensile": float(
            mean_absolute_error(y_val[:, 1], val_prediction[:, 1])
        ),
    }

    return model, input_scaler, df, metrics


def predict_formulation(model, input_scaler, inputs):
    x_augmented = add_interaction_features(
        np.array([inputs], dtype=np.float32)
    )

    x_scaled = input_scaler.transform(x_augmented).astype(np.float32)

    density, tensile, elastic_recovery = model.predict(x_scaled)[0]

    efrf = elastic_recovery / max(tensile, 1e-8)

    return float(density), float(tensile), float(elastic_recovery), float(efrf)


def compare_models(df, input_scaler, model):
    x = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["Tensile_Strength_MPa"].values.astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=7,
    )

    x_train_scaled = input_scaler.transform(
        add_interaction_features(x_train)
    )

    x_test_scaled = input_scaler.transform(
        add_interaction_features(x_test)
    )

    rows = []

    pinn_prediction = model.predict(x_test_scaled)[:, 1]

    rows.append(
        {
            "Model": "PINN",
            "R2": r2_score(y_test, pinn_prediction),
            "RMSE": np.sqrt(mean_squared_error(y_test, pinn_prediction)),
            "MAE": mean_absolute_error(y_test, pinn_prediction),
            "Physics": "Enforced",
        }
    )

    baselines = {
        "MLP": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=1200,
            random_state=7,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            random_state=7,
            min_samples_leaf=2,
        ),
    }

    for name, baseline in baselines.items():
        baseline.fit(x_train_scaled, y_train)
        prediction = baseline.predict(x_test_scaled)

        rows.append(
            {
                "Model": name,
                "R2": r2_score(y_test, prediction),
                "RMSE": np.sqrt(mean_squared_error(y_test, prediction)),
                "MAE": mean_absolute_error(y_test, prediction),
                "Physics": "Not enforced",
            }
        )

    return pd.DataFrame(rows)


def plot_training_history(model):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=model.loss_history["train"],
            name="Training loss",
        )
    )

    fig.add_trace(
        go.Scatter(
            y=model.loss_history["val"],
            name="Validation loss",
        )
    )

    fig.add_trace(
        go.Scatter(
            y=model.loss_history["data"],
            name="Data loss",
        )
    )

    fig.add_trace(
        go.Scatter(
            y=model.loss_history["physics"],
            name="Physics loss",
        )
    )

    fig.update_layout(
        height=420,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log",
    )

    return fig


def main():
    st.set_page_config(
        page_title="Improved PINN Framework",
        layout="wide",
    )

    st.title("Improved Physics-Informed Neural Network Framework")
    st.caption("High-R2 corrected version using physical-unit targets.")

    with st.spinner("Preparing model..."):
        model, input_scaler, df, metrics = load_pinn_model()

    with st.sidebar:
        st.header("Model R2")
        st.metric("Density R2", f"{metrics['r2_density']:.3f}")
        st.metric("Tensile R2", f"{metrics['r2_tensile']:.3f}")
        st.metric("Elastic recovery R2", f"{metrics['r2_er']:.3f}")
        st.metric("Tensile RMSE", f"{metrics['rmse_tensile']:.3f} MPa")

        st.header("Constraints")
        st.write(f"Tensile >= {TENSILE_MIN:.2f} MPa")
        st.write(f"EFRF < {EFRF_MAX:.2f}")
        st.write(f"MCC <= {MCC_MAX:.1f}%")

    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.subheader("Formulation")

        api = st.slider(
            "API loading (%)",
            85.0,
            95.0,
            DEFAULTS["api"],
            0.1,
        )

        mcc = st.slider(
            "MCC (%)",
            0.0,
            MCC_MAX,
            DEFAULTS["mcc"],
            0.1,
        )

        pvpp = st.slider(
            "PVPP (%)",
            0.5,
            6.0,
            DEFAULTS["pvpp"],
            0.1,
        )

        mgst = st.slider(
            "Mg-St (%)",
            0.01,
            1.2,
            DEFAULTS["mgst"],
            0.01,
        )

        binder = st.slider(
            "Binder (%)",
            BINDER_MIN,
            BINDER_MAX,
            DEFAULTS["binder"],
            0.1,
        )

        total = api + mcc + pvpp + mgst + binder

        if abs(total - 100.0) <= 0.1:
            st.success(f"Total = {total:.2f}%")
        else:
            st.warning(f"Total = {total:.2f}%. Adjust formulation toward 100%.")

        st.subheader("Process")

        pressure = st.slider(
            "Pressure (MPa)",
            80.0,
            PRESSURE_MAX,
            DEFAULTS["pressure"],
            1.0,
        )

        speed = st.slider(
            "Speed (rpm)",
            1.0,
            50.0,
            DEFAULTS["speed"],
            0.5,
        )

        granule = st.slider(
            "Granule size (um)",
            30.0,
            250.0,
            DEFAULTS["granule"],
            1.0,
        )

    inputs = [
        api,
        mcc,
        pvpp,
        mgst,
        binder,
        pressure,
        speed,
        granule,
    ]

    density, tensile, elastic_recovery, efrf = predict_formulation(
        model,
        input_scaler,
        inputs,
    )

    passed = tensile >= TENSILE_MIN and efrf < EFRF_MAX

    with right:
        st.subheader("Prediction")

        cols = st.columns(4)

        cols[0].metric(
            "Density",
            f"{density:.3f}",
        )

        cols[1].metric(
            "Tensile",
            f"{tensile:.3f} MPa",
        )

        cols[2].metric(
            "Elastic recovery",
            f"{elastic_recovery:.3f}",
        )

        cols[3].metric(
            "EFRF",
            f"{efrf:.4f}",
        )

        if passed:
            st.success(
                f"PASS: tensile >= {TENSILE_MIN:.2f} MPa and "
                f"EFRF < {EFRF_MAX:.2f}."
            )
        else:
            st.error(
                f"FAIL: check tensile >= {TENSILE_MIN:.2f} MPa and "
                f"EFRF < {EFRF_MAX:.2f}."
            )

        tabs = st.tabs(
            [
                "Physics outputs",
                "Model comparison",
                "Training curves",
                "Dataset preview",
            ]
        )

        with tabs[0]:
            x_augmented = add_interaction_features(
                np.array([inputs], dtype=np.float32)
            )

            x_scaled = input_scaler.transform(x_augmented).astype(np.float32)

            with torch.no_grad():
                full_output = model.forward(
                    torch.tensor(x_scaled, dtype=torch.float32)
                ).numpy()[0]

            physics_df = pd.DataFrame(
                [
                    {
                        "k": full_output[3],
                        "A": full_output[4],
                        "Heckel_left": np.log(1.0 / (1.0 - density)),
                        "Heckel_right": full_output[3] * pressure / 2.7
                        + full_output[4],
                    }
                ]
            )

            st.dataframe(
                physics_df,
                use_container_width=True,
                hide_index=True,
            )

        with tabs[1]:
            comparison = compare_models(df, input_scaler, model)

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=comparison["Model"],
                    y=comparison["R2"],
                    name="R2",
                )
            )

            fig.update_layout(
                height=340,
                yaxis_title="R2",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with tabs[2]:
            st.plotly_chart(
                plot_training_history(model),
                use_container_width=True,
            )

        with tabs[3]:
            st.dataframe(
                df.head(30),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
