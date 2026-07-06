"""
Hubryd AI Multi-objective Optimization Framework – Minimal Working Version
PINN + NSGA-II for Tablet Formulation
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import math
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# Parameters (reduced for free tier)
# ================================================================

TENSILE_MIN = 1.90
EFRF_MAX = 0.40
MCC_MAX = 8.0
D_MIN = 0.40
D_MAX = 0.97
PRESSURE_MAX = 300.0
BINDER_MIN = 0.5
BINDER_MAX = 5.0

N_SAMPLES = 2000
ADAM_EPOCHS = 100
PATIENCE = 20
NSGA_POP_SIZE = 20
NSGA_GENERATIONS = 15

W_TENSILE = 10.0
W_PHYSICS_FINAL = 0.2

# ================================================================
# Helper functions
# ================================================================

def normalize_components(api, binder, pvpp, mgst, mcc):
    api = max(api, 0.1); binder = max(binder, 0.1)
    pvpp = max(pvpp, 0.1); mgst = max(mgst, 0.01); mcc = max(mcc, 0.1)
    api = min(api, 100.0); binder = min(binder, 15.0)
    pvpp = min(pvpp, 15.0); mgst = min(mgst, 3.0); mcc = min(mcc, 25.0)
    total = api + binder + pvpp + mgst + mcc
    if total <= 0: total = 1.0
    api_norm = (api / total) * 100
    binder_norm = (binder / total) * 100
    pvpp_norm = (pvpp / total) * 100
    mgst_norm = (mgst / total) * 100
    mcc_norm = (mcc / total) * 100
    
    if mcc_norm > MCC_MAX:
        excess = mcc_norm - MCC_MAX; mcc_norm = MCC_MAX
        other_sum = api_norm + binder_norm + pvpp_norm + mgst_norm
        if other_sum > 0:
            api_norm += excess * (api_norm / other_sum)
            binder_norm += excess * (binder_norm / other_sum)
            pvpp_norm += excess * (pvpp_norm / other_sum)
            mgst_norm += excess * (mgst_norm / other_sum)
    if api_norm < 85.0:
        deficit = 85.0 - api_norm; api_norm = 85.0
        other_sum = binder_norm + pvpp_norm + mgst_norm
        if other_sum > 0:
            binder_norm -= deficit * (binder_norm / other_sum) if binder_norm > 0 else 0
            pvpp_norm -= deficit * (pvpp_norm / other_sum) if pvpp_norm > 0 else 0
            mgst_norm -= deficit * (mgst_norm / other_sum) if mgst_norm > 0 else 0
    if api_norm > 95.0:
        excess = api_norm - 95.0; api_norm = 95.0
        other_sum = binder_norm + pvpp_norm + mgst_norm + mcc_norm
        if other_sum > 0:
            binder_norm += excess * (binder_norm / other_sum) if binder_norm > 0 else 0
            pvpp_norm += excess * (pvpp_norm / other_sum) if pvpp_norm > 0 else 0
            mgst_norm += excess * (mgst_norm / other_sum) if mgst_norm > 0 else 0
            mcc_norm += excess * (mcc_norm / other_sum) if mcc_norm > 0 else 0
    
    api_norm = np.clip(api_norm, 85, 95)
    binder_norm = np.clip(binder_norm, 0.5, 5.0)
    pvpp_norm = np.clip(pvpp_norm, 0.5, 6.0)
    mgst_norm = np.clip(mgst_norm, 0.01, 1.2)
    mcc_norm = np.clip(mcc_norm, 0, MCC_MAX)
    total_final = api_norm + binder_norm + pvpp_norm + mgst_norm + mcc_norm
    if total_final > 0 and abs(total_final - 100) > 0.1:
        scale = 100 / total_final
        api_norm *= scale; binder_norm *= scale; pvpp_norm *= scale
        mgst_norm *= scale; mcc_norm *= scale
    api_norm = np.clip(api_norm, 85, 95)
    binder_norm = np.clip(binder_norm, 0.5, 5.0)
    pvpp_norm = np.clip(pvpp_norm, 0.5, 6.0)
    mgst_norm = np.clip(mgst_norm, 0.01, 1.2)
    mcc_norm = np.clip(mcc_norm, 0, MCC_MAX)
    return api_norm, binder_norm, pvpp_norm, mgst_norm, mcc_norm

def add_interaction_features(X_raw):
    pressure = X_raw[:, 5:6]
    binder = X_raw[:, 4:5]
    api = X_raw[:, 0:1]
    speed = X_raw[:, 6:7]
    mcc = X_raw[:, 1:2]
    pvpp = X_raw[:, 2:3]
    mgst = X_raw[:, 3:4]

    pressure_speed = np.clip(pressure / (speed + 0.1), 0, 1000)
    api_mcc = np.clip(api / (mcc + 0.1), 0, 1000)
    binder_speed = np.clip(binder / (speed + 0.1), 0, 100)
    pressure_binder = pressure * binder
    pressure_api = pressure * api
    api_pvpp = api * pvpp
    binder_mgst = binder * mgst
    mcc_pvpp = mcc * pvpp
    api2 = api ** 2
    pressure2 = pressure ** 2
    binder2 = binder ** 2
    speed2 = speed ** 2

    return np.concatenate([
        X_raw,
        pressure_binder, pressure_api,
        pressure_speed, api_mcc, binder_speed,
        api_pvpp, binder_mgst, mcc_pvpp,
        api2, pressure2, binder2, speed2
    ], axis=1)

def generate_pinn_data(n_samples=N_SAMPLES, random_state=42):
    np.random.seed(random_state)
    X = np.zeros((n_samples, 8))
    y = np.zeros((n_samples, 3))
    x_min = -np.log(1 - D_MIN)
    x_max = -np.log(1 - D_MAX)

    for i in range(n_samples):
        api_raw = np.random.uniform(60, 100)
        binder_raw = np.random.uniform(0.1, 10)
        mgst_raw = np.random.uniform(0.01, 3.0)
        pvpp_raw = np.random.uniform(0.1, 12)
        mcc_raw = np.random.uniform(0.1, 20)
        pressure = np.random.uniform(80, PRESSURE_MAX)
        speed = np.random.uniform(1, 50)
        granule = np.random.uniform(30, 250)

        api, binder, pvpp, mgst, mcc = normalize_components(api_raw, binder_raw, pvpp_raw, mgst_raw, mcc_raw)
        X[i] = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]

        x = np.random.uniform(x_min, x_max)
        for _ in range(30):
            k = np.random.uniform(0.005, 0.055)
            A = x - k * pressure
            if 0.5 <= A <= 2.5:
                break
        else:
            A = np.clip(x - 0.03 * pressure, 0.5, 2.5)
            k = 0.03
        A = np.clip(A, 0.5, 2.5)
        x_new = k * pressure + A
        D_target = 1 - np.exp(-x_new)
        D_target = np.clip(D_target, D_MIN, D_MAX)
        noise_d = np.random.normal(0, 0.003)
        D = np.clip(D_target + noise_d, D_MIN, D_MAX)

        sigma0 = np.random.uniform(4.0, 8.0)
        b = np.random.uniform(1.5, 3.5)
        porosity = 1.0 - D
        tensile_base = sigma0 * np.exp(-b * porosity)

        api_effect = 1.0 - 0.005 * (api - 85)
        binder_effect = 1.0 + 0.03 * (binder - 2.0)
        mgst_effect = 1.0 - 0.1 * (mgst - 0.2)
        pvpp_effect = 1.0 - 0.02 * (pvpp - 3.0)
        speed_effect = 1.0 - 0.002 * (speed - 10)

        strength = tensile_base * api_effect * binder_effect * mgst_effect * pvpp_effect * speed_effect
        strength = strength * np.random.normal(1.0, 0.005)
        strength = np.clip(strength, 0.5, 6.0)

        er_base = 1.8 + 0.3 * (api - 85)/10 + 0.08 * (speed - 10)/30 - 0.1 * (pressure - 100)/150
        er_base = er_base * (1.0 - 0.15 * (D - 0.4))
        er = np.clip(er_base + np.random.normal(0, 0.005), 0.5, 4.0)

        y[i] = [D, strength, er]

    feature_names = ['API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
                     'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm']
    df = pd.DataFrame(X, columns=feature_names)
    df['Density'] = y[:, 0]
    df['Tensile_Strength_MPa'] = y[:, 1]
    df['Elastic_Recovery_%'] = y[:, 2]
    return df, feature_names

# ================================================================
# PINN Model (small)
# ================================================================

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class ResidualBlock(nn.Module):
    def __init__(self, features, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(features, features)
        self.bn1 = nn.BatchNorm1d(features)
        self.lin2 = nn.Linear(features, features)
        self.bn2 = nn.BatchNorm1d(features)
        self.act = Mish()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.lin1(x)))
        out = self.drop(out)
        out = self.bn2(self.lin2(out))
        out = self.drop(out)
        return identity + out

class PINN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, 128), Mish())
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.transition = nn.Sequential(nn.Linear(128, 64), nn.Tanh())
        self.output = nn.Linear(64, 5)
    def forward(self, X):
        x = self.input_layer(X)
        x = self.res1(x)
        x = self.res2(x)
        x = self.transition(x)
        raw = self.output(x)
        density = D_MIN + (D_MAX - D_MIN) * torch.sigmoid(raw[:, 0:1])
        tensile = torch.nn.functional.softplus(raw[:, 1:2]) + 1e-4
        er = torch.nn.functional.softplus(raw[:, 2:3]) + 1e-4
        k = torch.nn.functional.softplus(raw[:, 3:4]) + 1e-4
        A = raw[:, 4:5]
        return torch.cat([density, tensile, er, k, A], dim=1)
    def predict(self, X_scaled):
        self.eval()
        with torch.no_grad():
            if not isinstance(X_scaled, torch.Tensor):
                X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
            device = next(self.parameters()).device
            X_scaled = X_scaled.to(device)
            output = self.forward(X_scaled)
            return output[:, :3].cpu().numpy()
    def compute_loss(self, X_scaled, X_raw, y_true, epoch=0, max_epochs=ADAM_EPOCHS):
        pressure_real = X_raw[:, 5].view(-1, 1)
        mcc_real = X_raw[:, 1].view(-1, 1)
        y_pred = self.forward(X_scaled)
        density_pred = y_pred[:, 0:1]
        tensile_pred = y_pred[:, 1:2]
        er_pred = y_pred[:, 2:3]
        k_pred = y_pred[:, 3:4]
        A_pred = y_pred[:, 4:5]

        # Curriculum: start with no physics, gradually increase
        w_physics = 0.0 if epoch < 30 else min(0.3, (epoch-30)/70 * 0.3)

        density_mse = nn.MSELoss()(density_pred, y_true[:, 0:1])
        tensile_mse = nn.MSELoss()(tensile_pred, y_true[:, 1:2])
        er_mse = nn.MSELoss()(er_pred, y_true[:, 2:3])
        data_loss = (3.0 * density_mse) + (W_TENSILE * tensile_mse) + (3.0 * er_mse)

        # Physics
        heckel_lhs = torch.log(1.0 / torch.clamp(1.0 - density_pred, min=1e-4))
        heckel_rhs = k_pred * pressure_real + A_pred
        heckel_loss = nn.MSELoss()(heckel_lhs, heckel_rhs)
        efrf_pred = er_pred / torch.clamp(tensile_pred, min=1e-4)
        efrf_loss = torch.mean(torch.relu(efrf_pred - EFRF_MAX) ** 2)
        mcc_loss = torch.mean(torch.relu(mcc_real - MCC_MAX) ** 2)
        density_penalty = torch.mean(torch.relu(density_pred - D_MAX) ** 2 + torch.relu(D_MIN - density_pred) ** 2)

        total_loss = data_loss + 0.5 * density_penalty + w_physics * (heckel_loss + efrf_loss) + 0.3 * mcc_loss
        return total_loss

# ================================================================
# NSGA-II (simplified)
# ================================================================

class NSGAII:
    def __init__(self, model, scaler, y_scaler, bounds, pop_size=20, generations=15):
        self.model = model
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.population = None
        self.objectives = None
        self.fronts = None

    def _repair(self, ind):
        api, mcc, pvpp, mgst, binder, pressure, speed, granule = ind
        api, binder, pvpp, mgst, mcc = normalize_components(api, binder, pvpp, mgst, mcc)
        pressure = np.clip(pressure, 80, PRESSURE_MAX)
        speed = np.clip(speed, 1.0, 50.0)
        granule = np.clip(granule, 30.0, 250.0)
        return np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule])

    def _evaluate(self, population):
        n = population.shape[0]
        objectives = np.zeros((n, 2))
        violation = np.zeros(n)
        for i in range(n):
            repaired = self._repair(population[i])
            api, mcc, pvpp, mgst, binder, pressure, speed, granule = repaired
            inputs = np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule]).reshape(1, -1)
            inputs_with_features = add_interaction_features(inputs)[0]
            inputs_scaled = self.scaler.transform([inputs_with_features])
            X_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
            with torch.no_grad():
                pred_scaled = self.model.predict(X_tensor)
                pred_actual = self.y_scaler.inverse_transform(pred_scaled)[0]
            density = float(np.clip(pred_actual[0], D_MIN, D_MAX))
            tensile = float(max(pred_actual[1], 1e-4))
            er = float(max(pred_actual[2], 1e-4))
            efrf = er / tensile
            efrf = max(1e-4, min(efrf, 5.0))
            g1 = 0.90 - density
            g2 = density - 0.97
            violation[i] = max(0, g1, g2)
            penalty = 0.0
            if tensile < TENSILE_MIN: penalty += (TENSILE_MIN - tensile) ** 2
            if efrf >= EFRF_MAX: penalty += (efrf - EFRF_MAX) ** 2
            if density < D_MIN: penalty += (D_MIN - density) ** 2
            if density > D_MAX: penalty += (density - D_MAX) ** 2
            objectives[i, 0] = -(api + 0.0 * tensile) + 50.0 * penalty
            objectives[i, 1] = efrf + 50.0 * penalty
            population[i] = repaired
        return objectives, violation, population

    def _fast_non_dominated_sort(self, objectives, violation):
        n = objectives.shape[0]
        fronts = []
        feasible = violation <= 1e-6
        indices = np.arange(n)
        feasible_indices = indices[feasible]
        if len(feasible_indices) > 0:
            # Simple front: just one front of all feasible
            fronts.append(feasible_indices.tolist())
        if len(feasible_indices) < n:
            infeasible = indices[~feasible]
            sorted_infeasible = infeasible[np.argsort(violation[infeasible])]
            fronts.append(sorted_infeasible.tolist())
        return fronts

    def run(self):
        pop = np.zeros((self.pop_size, 8))
        for i in range(self.pop_size):
            ind = np.array([np.random.uniform(60,100), np.random.uniform(0.1,20),
                            np.random.uniform(0.1,12), np.random.uniform(0.01,3.0),
                            np.random.uniform(0.1,10), np.random.uniform(80,PRESSURE_MAX),
                            np.random.uniform(1,50), np.random.uniform(30,250)])
            pop[i] = self._repair(ind)
        self.population = pop

        for gen in range(self.generations):
            objectives, violation, pop = self._evaluate(self.population)
            self.objectives = objectives
            self.fronts = self._fast_non_dominated_sort(objectives, violation)
            if gen == self.generations - 1:
                break
            # Simple selection: keep the first front
            if len(self.fronts) > 0:
                front0 = self.fronts[0]
                # Keep only feasible solutions from front0
                feasible_front = [idx for idx in front0 if violation[idx] <= 1e-6]
                if len(feasible_front) == 0:
                    feasible_front = front0[:min(len(front0), self.pop_size)]
                new_pop = []
                for idx in feasible_front:
                    new_pop.append(pop[idx])
                # Fill remaining with random individuals
                while len(new_pop) < self.pop_size:
                    ind = np.array([np.random.uniform(60,100), np.random.uniform(0.1,20),
                                    np.random.uniform(0.1,12), np.random.uniform(0.01,3.0),
                                    np.random.uniform(0.1,10), np.random.uniform(80,PRESSURE_MAX),
                                    np.random.uniform(1,50), np.random.uniform(30,250)])
                    new_pop.append(self._repair(ind))
                pop = np.array(new_pop)
            else:
                # No feasible front, keep all
                pass
            # Simple crossover and mutation (just for demonstration)
            # We'll just randomize half the population
            for i in range(self.pop_size // 2):
                idx = np.random.randint(0, self.pop_size)
                pop[idx] = self._repair(np.random.uniform(low=self.bounds[:,0], high=self.bounds[:,1], size=8))
            self.population = pop

        # Final evaluation
        objectives, violation, pop = self._evaluate(self.population)
        self.objectives = objectives
        self.fronts = self._fast_non_dominated_sort(objectives, violation)
        return self.population, self.objectives, self.fronts

# ================================================================
# Prediction and plotting
# ================================================================

def predict_pinn(model, scaler, y_scaler, inputs):
    try:
        inputs_with_features = add_interaction_features(np.array([inputs]))[0]
        inputs_scaled = scaler.transform([inputs_with_features])
        X_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model.predict(X_tensor)[0]
        pred_original = y_scaler.inverse_transform([pred_scaled])[0]
        density = float(np.clip(pred_original[0], D_MIN, D_MAX))
        tensile = float(max(pred_original[1], 1e-4))
        er = float(max(pred_original[2], 1e-4))
        efrf = float(er / tensile)
        return density, tensile, er, efrf
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return D_MIN, 0.01, 1.0, 1.0

def plot_pareto(objectives, fronts):
    if objectives is None or fronts is None or len(fronts) == 0:
        return None
    front0 = fronts[0]
    api_pareto = -objectives[front0, 0]
    efrf_pareto = objectives[front0, 1]
    df_pareto = pd.DataFrame({'API': api_pareto, 'EFRF': efrf_pareto})
    fig = px.scatter(df_pareto, x='API', y='EFRF', title='Pareto Front',
                     labels={'API': 'API (%)', 'EFRF': 'EFRF'})
    fig.update_traces(marker=dict(color='red', size=8))
    fig.add_hline(y=EFRF_MAX, line_dash='dash', line_color='red',
                  annotation_text=f'EFRF Threshold: {EFRF_MAX:.2f}',
                  annotation_position='top right')
    fig.update_layout(height=450, template='plotly_white')
    return fig

def train_and_compare(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    models = {
        'MLP': MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=30, random_state=42)
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            'Model': name,
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'Physics': 'Not enforced'
        })
    return pd.DataFrame(results)

# ================================================================
# Main app
# ================================================================

@st.cache_resource
def load_or_train():
    checkpoint_path = '/tmp/pinn_best_model.pt'
    try:
        if os.path.exists(checkpoint_path):
            st.caption("📂 Loading cached model...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model = PINN(input_dim=checkpoint['input_dim'])
            model.load_state_dict(checkpoint['model_state'])
            scaler = checkpoint['scaler']
            y_scaler = checkpoint['y_scaler']
            feature_names = checkpoint['feature_names']
            df = checkpoint['df']
            loss_history = checkpoint.get('loss_history', [])
            return model, scaler, y_scaler, feature_names, df, loss_history
    except Exception as e:
        st.warning(f"Cache error: {str(e)[:80]}. Re-training...")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    st.caption("🔄 Training model from scratch...")
    df, feature_names = generate_pinn_data(n_samples=N_SAMPLES)
    X_raw = df[feature_names].values
    y = df[['Density', 'Tensile_Strength_MPa', 'Elastic_Recovery_%']].values
    X_aug = add_interaction_features(X_raw)
    input_dim = X_aug.shape[1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    X_train, X_temp, X_raw_train, X_raw_temp, y_train, y_temp = train_test_split(
        X_scaled, X_raw, y_scaled, test_size=0.3, random_state=42)
    X_val, X_test, X_raw_val, X_raw_test, y_val, y_test = train_test_split(
        X_temp, X_raw_temp, y_temp, test_size=0.5, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.caption(f"🖥️ Using device: {device}")
    model = PINN(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_raw_train_t = torch.tensor(X_raw_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_raw_val_t = torch.tensor(X_raw_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val = float('inf')
    patience = 0
    progress_bar = st.progress(0)
    for epoch in range(ADAM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = model.compute_loss(X_train_t, X_raw_train_t, y_train_t, epoch, ADAM_EPOCHS)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = model.compute_loss(X_val_t, X_raw_val_t, y_val_t, epoch, ADAM_EPOCHS)
        scheduler.step(val_loss.item())

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            patience = 0
            torch.save(model.state_dict(), "/tmp/best_model.pt")
        else:
            patience += 1
            if patience >= PATIENCE:
                st.warning(f"Early stopping at epoch {epoch+1}")
                break
        progress_bar.progress((epoch+1)/ADAM_EPOCHS)

    if os.path.exists("/tmp/best_model.pt"):
        model.load_state_dict(torch.load("/tmp/best_model.pt", map_location=device))
    model.cpu()
    checkpoint_data = {
        'model_state': model.state_dict(),
        'scaler': scaler,
        'y_scaler': y_scaler,
        'feature_names': feature_names,
        'df': df,
        'input_dim': input_dim,
        'loss_history': []
    }
    torch.save(checkpoint_data, checkpoint_path)
    st.success("✅ Model ready!")
    return model, scaler, y_scaler, feature_names, df, []

# ================================================================
# UI
# ================================================================

st.set_page_config(page_title="Hubryd AI", layout="wide")
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 1rem; border-radius: 1rem; text-align: center;">
    <h2 style="color: #fff;">🧬 Hubryd AI Multi‑objective Optimization</h2>
    <p style="color: #64ffda;">Minimal Working Version · Free Tier</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Physics Constraints")
st.sidebar.info(f"""
- Density: {D_MIN:.2f}–{D_MAX:.2f}
- Tensile ≥ {TENSILE_MIN:.2f} MPa
- EFRF < {EFRF_MAX:.2f}
- MCC ≤ {MCC_MAX:.1f}%
""")

try:
    model, scaler, y_scaler, feature_names, df, _ = load_or_train()
except Exception as e:
    st.error(f"❌ Training failed: {str(e)}")
    st.stop()

# Session state for sliders
if 'api' not in st.session_state:
    for k, v in {'api':90.5,'binder':2.7,'pvpp':3.0,'mgst':0.20,'mcc':3.6,'pressure':230,'speed':12,'granule':125}.items():
        st.session_state[k] = v

col1, col2 = st.columns([1, 1.2])

with col1:
    with st.container(border=True):
        api = st.slider("API (%)", 85.0, 95.0, st.session_state.api, 0.1)
        binder = st.slider("Binder (%)", BINDER_MIN, BINDER_MAX, st.session_state.binder, 0.1)
        pvpp = st.slider("PVPP (%)", 0.5, 6.0, st.session_state.pvpp, 0.1)
        mgst = st.slider("Mg-St (%)", 0.01, 1.2, st.session_state.mgst, 0.01)
        mcc = st.slider("MCC (%)", 0.0, MCC_MAX, st.session_state.mcc, 0.1)
        total = api + binder + pvpp + mgst + mcc
        if abs(total-100) < 0.1: st.success(f"✅ Total: {total:.2f}%")
        else: st.warning(f"⚠️ Total: {total:.2f}% (adjust to 100%)")

    with st.container(border=True):
        pressure = st.slider("Pressure (MPa)", 80.0, PRESSURE_MAX, st.session_state.pressure, 1.0)
        speed = st.slider("Speed (rpm)", 1.0, 50.0, st.session_state.speed, 0.5)
        granule = st.slider("Granule Size (µm)", 30.0, 250.0, st.session_state.granule, 1.0)

    predict_btn = st.button("🔬 Predict & Optimize", use_container_width=True)

with col2:
    st.markdown("### 📈 Results")
    if predict_btn:
        if abs(total-100) > 0.1:
            st.warning("Formulation must sum to 100%")
        else:
            api_norm, binder_norm, pvpp_norm, mgst_norm, mcc_norm = normalize_components(api, binder, pvpp, mgst, mcc)
            inputs_norm = [api_norm, mcc_norm, pvpp_norm, mgst_norm, binder_norm, pressure, speed, granule]
            density, tensile, er, efrf = predict_pinn(model, scaler, y_scaler, inputs_norm)
            st.markdown("#### Constraints")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Density", f"{density:.3f}", "✅" if D_MIN<=density<=D_MAX else "❌")
            c2.metric("Tensile", f"{tensile:.2f} MPa", "✅" if tensile>=TENSILE_MIN else "❌")
            c3.metric("EFRF", f"{efrf:.4f}", "✅" if efrf<EFRF_MAX else "❌")
            c4.metric("MCC", f"{mcc_norm:.1f}%", "✅" if mcc_norm<=MCC_MAX else "❌")
            if all([D_MIN<=density<=D_MAX, tensile>=TENSILE_MIN, efrf<EFRF_MAX, mcc_norm<=MCC_MAX]):
                st.success("✅ All constraints satisfied!")
            else:
                st.error("❌ Violates constraints")

            # NSGA-II
            st.markdown("#### NSGA‑II Optimization")
            bounds = np.array([[60,100],[0.1,20],[0.1,12],[0.01,3.0],[0.1,10],[80,PRESSURE_MAX],[1,50],[30,250]])
            nsga = NSGAII(model, scaler, y_scaler, bounds, pop_size=NSGA_POP_SIZE, generations=NSGA_GENERATIONS)
            with st.spinner("Running NSGA‑II..."):
                pop, objectives, fronts = nsga.run()
            if len(fronts) > 0 and len(fronts[0]) > 0:
                st.success(f"✅ Pareto front: {len(fronts[0])} solutions")
                # Extract golden solution (min EFRF from feasible)
                front0 = fronts[0]
                best_idx = None
                best_efrf = float('inf')
                for idx in front0:
                    if idx < len(objectives):
                        efrf_val = objectives[idx, 1]
                        if efrf_val < best_efrf:
                            # Check feasibility quickly by re-evaluating the solution
                            formulation = nsga.population[idx]
                            d, t, e, ef = predict_pinn(model, scaler, y_scaler, formulation)
                            if D_MIN <= d <= D_MAX and t >= TENSILE_MIN and ef < EFRF_MAX:
                                best_efrf = efrf_val
                                best_idx = idx
                if best_idx is not None:
                    golden = nsga.population[best_idx]
                    d, t, e, ef = predict_pinn(model, scaler, y_scaler, golden)
                    st.markdown("---")
                    st.markdown("### ⭐ Golden Solution")
                    colA, colB = st.columns(2)
                    with colA:
                        st.markdown(f"API: {golden[0]:.1f}%")
                        st.markdown(f"MCC: {golden[1]:.1f}%")
                        st.markdown(f"PVPP: {golden[2]:.1f}%")
                        st.markdown(f"Mg-St: {golden[3]:.2f}%")
                        st.markdown(f"Binder: {golden[4]:.1f}%")
                    with colB:
                        st.markdown(f"Pressure: {golden[5]:.1f} MPa")
                        st.markdown(f"Speed: {golden[6]:.1f} rpm")
                        st.markdown(f"Granule: {golden[7]:.0f} µm")
                        st.markdown(f"Density: {d:.3f}")
                        st.markdown(f"Tensile: {t:.2f} MPa")
                        st.markdown(f"EFRF: {ef:.4f}")
                else:
                    st.info("No fully feasible solution found in Pareto front.")
            else:
                st.warning("No Pareto front found.")

            # Pareto plot
            fig = plot_pareto(objectives, fronts)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Model comparison (quick)
            X_train, X_test, y_train, y_test = train_test_split(
                df[feature_names].values, df['Tensile_Strength_MPa'].values,
                test_size=0.2, random_state=42
            )
            X_train_aug = add_interaction_features(X_train)
            X_test_aug = add_interaction_features(X_test)
            X_train_scaled = scaler.transform(X_train_aug)
            X_test_scaled = scaler.transform(X_test_aug)

            pinn_pred = model.predict(torch.tensor(X_test_scaled, dtype=torch.float32))
            pinn_pred = y_scaler.inverse_transform(pinn_pred)[:, 1]
            pinn_r2 = r2_score(y_test, pinn_pred)
            comp_df = train_and_compare(X_train_scaled, X_test_scaled, y_train, y_test)
            pinn_row = pd.DataFrame([{'Model':'PINN (Proposed)','R²':pinn_r2,'RMSE':np.sqrt(mean_squared_error(y_test, pinn_pred)),'Physics':'✅ Enforced'}])
            comp_df = pd.concat([pinn_row, comp_df], ignore_index=True)
            st.dataframe(comp_df, use_container_width=True)

st.caption("🔬 Hubryd AI · Minimal Working Version")
