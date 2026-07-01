import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)

TENSILE_MIN = 1.90
EFRF_MAX = 0.40
MCC_MAX = 8.0
DENSITY_MAX = 0.99
BINDER_MIN = 0.5
BINDER_MAX = 5.0
N_SAMPLES = 6000
EPOCHS = 500
PATIENCE = 60
DATA_WEIGHT = 1.0
PHYSICS_WEIGHT = 0.05

FEATURE_NAMES = [
    "API_%", "MCC_%", "PVPP_%", "MgSt_%",
    "Binder_%", "Pressure_MPa", "Speed_rpm", "Granule_Size_µm"
]

OUTPUT_NAMES = ["Density", "Tensile_Strength_MPa", "Elastic_Recovery_%"]

def add_interaction_features(X_raw):
    pressure = X_raw[:, 5:6]
    binder = X_raw[:, 4:5]
    api = X_raw[:, 0:1]
    speed = X_raw[:, 6:7]
    mcc = X_raw[:, 1:2]
    pressure_speed = np.clip(pressure / (speed + 0.1), 0, 1000)
    api_mcc = np.clip(api / (mcc + 0.1), 0, 1000)
    binder_speed = np.clip(binder / (speed + 0.1), 0, 100)
    pressure_binder = pressure * binder
    pressure_api = pressure * api
    return np.concatenate([
        X_raw, pressure_binder, pressure_api, pressure_speed, api_mcc, binder_speed
    ], axis=1)

def generate_pinn_data(n_samples=N_SAMPLES, random_state=42):
    np.random.seed(random_state)
    X = np.zeros((n_samples, 8))
    y = np.zeros((n_samples, 3))

    for i in range(n_samples):
        if i < n_samples // 2:
            api = np.random.uniform(85, 95)
            binder = np.random.uniform(BINDER_MIN, BINDER_MAX)
            mgst = np.random.uniform(0.01, 1.2)
            pvpp = np.random.uniform(0.5, 6.0)
            mcc = np.random.uniform(0, MCC_MAX)
            pressure = np.random.uniform(80, 300)
            speed = np.random.uniform(1, 50)
            granule = np.random.uniform(30, 250)
        else:
            api = np.clip(np.random.normal(90.5, 1.5), 85, 95)
            binder = np.clip(np.random.normal(2.8, 0.4), BINDER_MIN, BINDER_MAX)
            mgst = np.clip(np.random.normal(0.15, 0.06), 0.01, 1.2)
            pvpp = np.clip(np.random.normal(3.0, 0.5), 0.5, 6.0)
            mcc = np.clip(np.random.normal(5.0, 1.0), 0, MCC_MAX)
            pressure = np.clip(np.random.normal(230, 15), 80, 300)
            speed = np.clip(np.random.normal(10, 3), 1, 50)
            granule = np.clip(np.random.normal(125, 20), 30, 250)

        total = api + binder + mgst + pvpp + mcc
        if total > 100:
            scale = 100 / total
            api *= scale
            binder *= scale
            mgst *= scale
            pvpp *= scale
            mcc *= scale
        else:
            remainder = 100 - total
            if mcc + remainder <= MCC_MAX:
                mcc += remainder
            else:
                excess = (mcc + remainder) - MCC_MAX
                mcc = MCC_MAX
                api -= excess

        api = np.clip(api, 85, 95)
        binder = np.clip(binder, BINDER_MIN, BINDER_MAX)
        mgst = np.clip(mgst, 0.01, 1.2)
        pvpp = np.clip(pvpp, 0.5, 6.0)
        mcc = np.clip(mcc, 0, MCC_MAX)

        X[i] = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]

        k_true = 0.03 * (1 - 0.25 * (api - 85) / 10) * (1 - 0.12 * (speed - 10) / 30)
        k_true = max(k_true, 0.008)
        A_true = 1.0 + 0.08 * (binder - 1.5) - 0.10 * (mgst - 0.5)

        noise_d = np.random.normal(0, 0.005)
        noise_t = np.random.normal(0, 0.04)
        noise_er = np.random.normal(0, 0.03)

        D = np.clip(1 - np.exp(-(k_true * pressure + A_true)) + noise_d, 0.35, 0.99)

        strength = np.clip(
            4.0 - 0.10 * (api - 85) + 0.20 * binder + 0.006 * (pressure - 100)
            - 1.0 * mgst - 0.010 * (speed - 10) + noise_t,
            0.4, 6.5
        )

        er = np.clip(
            1.6 + 0.18 * (api - 85) / 10 + 0.05 * (speed - 10) / 30
            - 0.06 * (pressure - 100) / 150 + noise_er,
            0.4, 4.0
        )

        y[i] = [D, strength, er]

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["Density"] = y[:, 0]
    df["Tensile_Strength_MPa"] = y[:, 1]
    df["Elastic_Recovery_%"] = y[:, 2]
    return df

class PINN(nn.Module):
    def __init__(self, input_dim=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.head_density = nn.Linear(64, 1)
        self.head_tensile = nn.Linear(64, 1)
        self.head_er = nn.Linear(64, 1)
        self.head_k = nn.Linear(64, 1)
        self.head_A = nn.Linear(64, 1)

    def forward(self, x):
        f = self.net(x)
        density = torch.sigmoid(self.head_density(f))
        tensile = torch.nn.functional.softplus(self.head_tensile(f))
        er = torch.nn.functional.softplus(self.head_er(f))
        k = torch.nn.functional.softplus(self.head_k(f))
        A = self.head_A(f)
        return density, tensile, er, k, A

    def predict_primary(self, x):
        self.eval()
        with torch.no_grad():
            d, t, e, _, _ = self.forward(x)
            return torch.cat([d, t, e], dim=1)

class PINNTrainer:
    def __init__(self):
        self.model = PINN(input_dim=13)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.best_state = None
        self.loss_history = {"train": [], "val": []}

    def compute_loss(self, X_scaled, X_raw, y_true):
        pressure_real = X_raw[:, 5]
        mcc_real = X_raw[:, 1]
        density_pred, tensile_pred, er_pred, k_pred, A_pred = self.model(X_scaled)

        data_loss = nn.MSELoss()(torch.cat([density_pred, tensile_pred, er_pred], dim=1), y_true)

        D_clamped = torch.clamp(density_pred, 0.01, 0.99)
        heckel_pred = torch.log(1.0 / (1.0 - D_clamped))
        heckel_target = k_pred * pressure_real + A_pred
        heckel_loss = torch.mean((heckel_pred - heckel_target) ** 2)

        efrf_pred = er_pred / (tensile_pred + 1e-8)
        efrf_loss = torch.mean(torch.relu(efrf_pred - EFRF_MAX) ** 2)

        mcc_loss = torch.mean(torch.relu(mcc_real - MCC_MAX) ** 2)
        density_penalty = torch.mean(torch.relu(density_pred - DENSITY_MAX) ** 2)

        physics_loss = heckel_loss + efrf_loss
        total_loss = (
            DATA_WEIGHT * data_loss +
            PHYSICS_WEIGHT * physics_loss +
            0.02 * mcc_loss +
            0.02 * density_penalty
        )

        loss_dict = {
            "data_loss": float(data_loss.item()),
            "heckel_loss": float(heckel_loss.item()),
            "efrf_loss": float(efrf_loss.item()),
            "physics_loss": float(physics_loss.item()),
            "total_loss": float(total_loss.item())
        }
        return total_loss, loss_dict

    def fit(self, df, epochs=EPOCHS, patience=PATIENCE, lr=1e-3):
        X_raw = df[FEATURE_NAMES].values.astype(float)
        y = df[OUTPUT_NAMES].values.astype(float)

        X_feat = add_interaction_features(X_raw)
        Xs = self.x_scaler.fit_transform(X_feat)
        ys = self.y_scaler.fit_transform(y)

        X_train, X_val, y_train, y_val, X_raw_train, X_raw_val = train_test_split(
            Xs, ys, X_raw, test_size=0.2, random_state=42
        )

        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_raw_train_t = torch.FloatTensor(X_raw_train)

        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)

        opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        best = float("inf")
        wait = 0

        for epoch in range(epochs):
            self.model.train()
            opt.zero_grad()
            loss, _ = self.compute_loss(X_train_t, X_raw_train_t, y_train_t)
            loss.backward()
            opt.step()

            self.model.eval()
            with torch.no_grad():
                val_pred_scaled = self.model.predict_primary(X_val_t).numpy()
                val_pred = self.y_scaler.inverse_transform(val_pred_scaled)
                val_true = self.y_scaler.inverse_transform(y_val)
                val_loss = mean_squared_error(val_true, val_pred)

            self.loss_history["train"].append(float(loss.item()))
            self.loss_history["val"].append(float(val_loss))

            if val_loss < best - 1e-5:
                best = val_loss
                wait = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self

    def predict(self, X_raw):
        X_feat = add_interaction_features(X_raw)
        Xs = self.x_scaler.transform(X_feat)
        Xs_t = torch.FloatTensor(Xs)
        pred_scaled = self.model.predict_primary(Xs_t).numpy()
        return self.y_scaler.inverse_transform(pred_scaled)

    def evaluate(self, df):
        X_raw = df[FEATURE_NAMES].values.astype(float)
        y_true = df[OUTPUT_NAMES].values.astype(float)
        y_pred = self.predict(X_raw)

        rows = []
        for i, out in enumerate(OUTPUT_NAMES):
            rows.append({
                "Output": out,
                "R2": r2_score(y_true[:, i], y_pred[:, i]),
                "RMSE": mean_squared_error(y_true[:, i], y_pred[:, i], squared=False),
                "MAE": mean_absolute_error(y_true[:, i], y_pred[:, i]),
                "PearsonR": pearsonr(y_true[:, i], y_pred[:, i])[0]
            })
        return pd.DataFrame(rows)
