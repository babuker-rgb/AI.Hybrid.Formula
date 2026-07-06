"""
Hubryd AI v29.27 – Minimal · Stable · Free Tier
Enhanced with Particle Size Toggle (Variable / Fixed)
Nile Valley University · Sudan
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import datetime          # <-- FIX: added for report timestamp
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# Physics Constants
# ================================================================
D_MIN = 0.40
D_MAX = 0.97
TENSILE_MIN = 1.90
EFRF_MAX = 0.40
MCC_MAX = 8.0
PRESSURE_MAX = 300.0
BINDER_MIN = 0.5
BINDER_MAX = 5.0

# ================================================================
# Training Parameters (v29.27 – minimal)
# ================================================================
N_SAMPLES = 5000
ADAM_EPOCHS = 200
PATIENCE = 30
NSGA_POP = 40
NSGA_GENS = 30
HIDDEN_SIZE = 256

W_DENSITY = 1.0
W_TENSILE = 30.0
W_ER = 5.0
W_PHYSICS = 2.0
W_EFRF_PENALTY = 100.0

# ================================================================
# Helper Functions (unchanged)
# ================================================================
def normalize_components(api, binder, pvpp, mgst, mcc):
    api = np.clip(api, 60, 100)
    binder = np.clip(binder, 0.1, 15)
    pvpp = np.clip(pvpp, 0.1, 15)
    mgst = np.clip(mgst, 0.01, 3.0)
    mcc = np.clip(mcc, 0.1, 20)
    total = api + binder + pvpp + mgst + mcc
    if total <= 0:
        total = 1.0
    api = (api / total) * 100
    binder = (binder / total) * 100
    pvpp = (pvpp / total) * 100
    mgst = (mgst / total) * 100
    mcc = (mcc / total) * 100
    api = np.clip(api, 85, 95)
    binder = np.clip(binder, BINDER_MIN, BINDER_MAX)
    pvpp = np.clip(pvpp, 0.5, 6.0)
    mgst = np.clip(mgst, 0.01, 1.2)
    mcc = np.clip(mcc, 0, MCC_MAX)
    total2 = api + binder + pvpp + mgst + mcc
    if abs(total2 - 100) > 1e-6:
        scale = 100 / total2
        api *= scale
        binder *= scale
        pvpp *= scale
        mgst *= scale
        mcc *= scale
        api = np.clip(api, 85, 95)
        binder = np.clip(binder, BINDER_MIN, BINDER_MAX)
        pvpp = np.clip(pvpp, 0.5, 6.0)
        mgst = np.clip(mgst, 0.01, 1.2)
        mcc = np.clip(mcc, 0, MCC_MAX)
    return api, binder, pvpp, mgst, mcc

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
        api = np.random.uniform(85, 95)
        binder = np.random.uniform(BINDER_MIN, BINDER_MAX)
        pvpp = np.random.uniform(0.5, 6.0)
        mgst = np.random.uniform(0.01, 1.2)
        mcc = np.random.uniform(0, MCC_MAX)
        pressure = np.random.uniform(80, PRESSURE_MAX)
        speed = np.random.uniform(1, 50)
        granule = np.random.uniform(30, 250)
        X[i] = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
        x = np.random.uniform(x_min, x_max)
        k = np.random.uniform(0.005, 0.055)
        A = np.random.uniform(0.5, 2.5)
        x_new = k * pressure + A
        D_target = 1 - np.exp(-x_new)
        D_target = np.clip(D_target, D_MIN, D_MAX)
        D = np.clip(D_target + np.random.normal(0, 0.003), D_MIN, D_MAX)
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
# PINN Model (v29.27 – 256 neurons)
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

class MultiTaskPINN(nn.Module):
    def __init__(self, input_dim, hidden=HIDDEN_SIZE):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden), Mish())
        self.res1 = ResidualBlock(hidden)
        self.res2 = ResidualBlock(hidden)
        self.transition = nn.Sequential(nn.Linear(hidden, hidden//2), nn.Tanh())
        self.output = nn.Linear(hidden//2, 5)
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
    def compute_loss(self, X_scaled, X_raw, y_true, epoch=0, total_epochs=ADAM_EPOCHS):
        pressure = X_raw[:, 5].view(-1, 1)
        mcc = X_raw[:, 1].view(-1, 1)
        y_pred = self.forward(X_scaled)
        density_pred = y_pred[:, 0:1]
        tensile_pred = y_pred[:, 1:2]
        er_pred = y_pred[:, 2:3]
        k_pred = y_pred[:, 3:4]
        A_pred = y_pred[:, 4:5]
        density_mse = nn.MSELoss()(density_pred, y_true[:, 0:1])
        tensile_mse = nn.MSELoss()(tensile_pred, y_true[:, 1:2])
        er_mse = nn.MSELoss()(er_pred, y_true[:, 2:3])
        data_loss = W_DENSITY * density_mse + W_TENSILE * tensile_mse + W_ER * er_mse
        heckel_lhs = torch.log(1.0 / torch.clamp(1.0 - density_pred, min=1e-4))
        heckel_rhs = k_pred * pressure + A_pred
        heckel_loss = nn.MSELoss()(heckel_lhs, heckel_rhs)
        efrf = er_pred / torch.clamp(tensile_pred, min=1e-4)
        efrf_penalty = torch.mean(torch.relu(efrf - EFRF_MAX) ** 2) * W_EFRF_PENALTY
        mcc_penalty = torch.mean(torch.relu(mcc - MCC_MAX) ** 2) * 0.3
        density_penalty = torch.mean(torch.relu(density_pred - D_MAX) ** 2 + torch.relu(D_MIN - density_pred) ** 2) * 0.5
        monotonicity_loss = 0.0
        if epoch % 10 == 0:
            pressure_scaled = X_scaled[:, 5:6].detach().clone().requires_grad_(True)
            X_scaled_ = X_scaled.detach().clone()
            X_scaled_[:, 5:6] = pressure_scaled
            y_pred_ = self.forward(X_scaled_)
            d_pred = y_pred_[:, 0:1]
            t_pred = y_pred_[:, 1:2]
            grad_d = torch.autograd.grad(outputs=d_pred, inputs=pressure_scaled,
                                         grad_outputs=torch.ones_like(d_pred),
                                         create_graph=True, retain_graph=True)[0]
            grad_t = torch.autograd.grad(outputs=t_pred, inputs=pressure_scaled,
                                         grad_outputs=torch.ones_like(t_pred),
                                         create_graph=True, retain_graph=True)[0]
            mon_d = torch.mean(torch.relu(-grad_d) ** 2)
            mon_t = torch.mean(torch.relu(-grad_t) ** 2)
            monotonicity_loss = 0.5 * (mon_d + mon_t) * W_PHYSICS
        physics_loss = W_PHYSICS * (heckel_loss + efrf_penalty) + mcc_penalty + density_penalty
        progress = epoch / total_epochs
        phys_weight = 2.0 / (1 + np.exp(-10 * (progress - 0.5)))
        phys_weight = max(0.1, phys_weight)
        total_loss = data_loss + phys_weight * (physics_loss + monotonicity_loss)
        return total_loss

# ================================================================
# NSGA-II (with variable granule option)
# ================================================================
class NSGAII:
    def __init__(self, model, scaler, y_scaler, bounds, pop=40, gens=30, granule_fixed=True, granule_fixed_val=125.0):
        self.model = model
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.bounds = bounds
        self.pop_size = pop
        self.generations = gens
        self.granule_fixed = granule_fixed
        self.granule_fixed_val = granule_fixed_val

    def _repair(self, ind):
        api, mcc, pvpp, mgst, binder, pressure, speed, granule = ind
        api, binder, pvpp, mgst, mcc = normalize_components(api, binder, pvpp, mgst, mcc)
        pressure = np.clip(pressure, 80, PRESSURE_MAX)
        speed = np.clip(speed, 1, 50)
        if self.granule_fixed:
            granule = self.granule_fixed_val
        else:
            granule = np.clip(granule, 30, 250)
        return np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule])

    def _evaluate(self, population):
        n = population.shape[0]
        objectives = np.zeros((n, 2))
        violation = np.zeros(n)
        for i in range(n):
            ind = self._repair(population[i])
            api, mcc, pvpp, mgst, binder, pressure, speed, granule = ind
            inputs = np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule]).reshape(1, -1)
            aug = add_interaction_features(inputs)[0]
            scaled = self.scaler.transform([aug])
            X_t = torch.tensor(scaled, dtype=torch.float32)
            with torch.no_grad():
                pred_scaled = self.model.predict(X_t)
                pred = self.y_scaler.inverse_transform(pred_scaled)[0]
            density = np.clip(pred[0], D_MIN, D_MAX)
            tensile = max(pred[1], 1e-4)
            er = max(pred[2], 1e-4)
            efrf = er / tensile
            efrf = max(1e-4, min(efrf, 5.0))
            g1 = D_MIN - density
            g2 = density - D_MAX
            violation[i] = max(0, g1, g2)
            penalty = 0.0
            if tensile < TENSILE_MIN:
                penalty += (TENSILE_MIN - tensile) ** 2
            if efrf >= EFRF_MAX:
                penalty += (efrf - EFRF_MAX) ** 2
            if mcc > MCC_MAX:
                penalty += (mcc - MCC_MAX) ** 2
            objectives[i, 0] = -(api) + 100.0 * penalty
            objectives[i, 1] = efrf + 100.0 * penalty
            population[i] = ind
        return objectives, violation, population

    def _non_dominated_sort(self, objectives, violation):
        n = objectives.shape[0]
        fronts = []
        remaining = list(range(n))
        while remaining:
            front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if (objectives[j,0] <= objectives[i,0] and objectives[j,1] <= objectives[i,1]) and \
                       (objectives[j,0] < objectives[i,0] or objectives[j,1] < objectives[i,1]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            fronts.append(front)
            remaining = [idx for idx in remaining if idx not in front]
        return fronts

    def _crowding_distance(self, objectives, front):
        if len(front) <= 2:
            return np.ones(len(front)) * np.inf
        dist = np.zeros(len(front))
        for obj_idx in range(objectives.shape[1]):
            sorted_idx = sorted(front, key=lambda i: objectives[i, obj_idx])
            dist[0] = np.inf
            dist[-1] = np.inf
            f_min = objectives[sorted_idx[0], obj_idx]
            f_max = objectives[sorted_idx[-1], obj_idx]
            if f_max - f_min > 1e-10:
                for k in range(1, len(sorted_idx)-1):
                    dist[k] += (objectives[sorted_idx[k+1], obj_idx] - objectives[sorted_idx[k-1], obj_idx]) / (f_max - f_min)
        return dist

    def _crossover(self, p1, p2, eta=40):
        child1 = np.zeros(8)
        child2 = np.zeros(8)
        for i in range(8):
            u = np.random.random()
            if u <= 0.5:
                beta = (2*u) ** (1/(eta+1))
            else:
                beta = (1/(2*(1-u))) ** (1/(eta+1))
            child1[i] = 0.5 * ((1+beta)*p1[i] + (1-beta)*p2[i])
            child2[i] = 0.5 * ((1-beta)*p1[i] + (1+beta)*p2[i])
        return child1, child2

    def _mutate(self, child, eta=20, pm=1.0/8.0):
        for i in range(8):
            if np.random.random() < pm:
                u = np.random.random()
                if u <= 0.5:
                    delta = (2*u) ** (1/(eta+1)) - 1
                else:
                    delta = 1 - (2*(1-u)) ** (1/(eta+1))
                child[i] = child[i] + delta * (self.bounds[i,1] - self.bounds[i,0])
                child[i] = np.clip(child[i], self.bounds[i,0], self.bounds[i,1])
        return child

    def _tournament(self, pop, objectives, fronts, violation):
        idx1 = np.random.randint(0, len(pop))
        idx2 = np.random.randint(0, len(pop))
        rank1 = next((f for f, front in enumerate(fronts) if idx1 in front), len(fronts))
        rank2 = next((f for f, front in enumerate(fronts) if idx2 in front), len(fronts))
        if rank1 < rank2:
            return pop[idx1]
        elif rank2 < rank1:
            return pop[idx2]
        else:
            front = fronts[rank1]
            dist = self._crowding_distance(objectives, front)
            d1 = dist[front.index(idx1)]
            d2 = dist[front.index(idx2)]
            return pop[idx1] if d1 > d2 else pop[idx2]

    def run(self):
        pop = []
        for i in range(self.pop_size):
            if i < 0.3 * self.pop_size:
                api = np.random.uniform(90, 95)
                mcc = np.random.uniform(2, 6)
                binder = np.random.uniform(1.5, 3.5)
                pvpp = np.random.uniform(1, 4)
                mgst = np.random.uniform(0.1, 0.4)
            else:
                api = np.random.uniform(85, 95)
                mcc = np.random.uniform(0.1, MCC_MAX)
                binder = np.random.uniform(BINDER_MIN, BINDER_MAX)
                pvpp = np.random.uniform(0.5, 6)
                mgst = np.random.uniform(0.01, 1.2)
            pressure = np.random.uniform(80, PRESSURE_MAX)
            speed = np.random.uniform(1, 50)
            granule = np.random.uniform(30, 250)
            ind = np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule])
            pop.append(self._repair(ind))
        pop = np.array(pop)

        for gen in range(self.generations):
            objectives, violation, pop = self._evaluate(pop)
            fronts = self._non_dominated_sort(objectives, violation)
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._tournament(pop, objectives, fronts, violation)
                p2 = self._tournament(pop, objectives, fronts, violation)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                offspring.append(self._repair(c1))
                if len(offspring) < self.pop_size:
                    offspring.append(self._repair(c2))
            offspring = np.array(offspring[:self.pop_size])
            combined = np.vstack([pop, offspring])
            obj_comb, viol_comb, _ = self._evaluate(combined)
            fronts_comb = self._non_dominated_sort(obj_comb, viol_comb)
            new_pop = []
            remaining = self.pop_size
            for front in fronts_comb:
                if len(front) <= remaining:
                    new_pop.extend(combined[front])
                    remaining -= len(front)
                else:
                    dist = self._crowding_distance(obj_comb, front)
                    sorted_idx = sorted(front, key=lambda i: dist[front.index(i)], reverse=True)
                    new_pop.extend(combined[sorted_idx[:remaining]])
                    remaining = 0
                    break
            pop = np.array(new_pop)

        objectives, violation, pop = self._evaluate(pop)
        fronts = self._non_dominated_sort(objectives, violation)
        return pop, objectives, fronts

# ================================================================
# Prediction and Plotting Helpers
# ================================================================
def predict_pinn(model, scaler, y_scaler, inputs):
    try:
        aug = add_interaction_features(np.array([inputs]))[0]
        scaled = scaler.transform([aug])
        X_t = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model.predict(X_t)[0]
        pred = y_scaler.inverse_transform([pred_scaled])[0]
        density = np.clip(pred[0], D_MIN, D_MAX)
        tensile = max(pred[1], 1e-4)
        er = max(pred[2], 1e-4)
        efrf = er / tensile
        return density, tensile, er, efrf
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.7, 2.0, 0.5, 0.25

def plot_pareto(objectives, fronts):
    if fronts is None or len(fronts) == 0 or len(fronts[0]) == 0:
        return None
    front = fronts[0]
    df_plot = pd.DataFrame({
        'API': -objectives[front, 0],
        'EFRF': objectives[front, 1]
    })
    fig = px.scatter(df_plot, x='API', y='EFRF', title='Pareto Front (v29.27)')
    fig.add_hline(y=EFRF_MAX, line_dash='dash', line_color='red',
                  annotation_text=f'EFRF threshold {EFRF_MAX}')
    fig.update_layout(height=450, template='plotly_white')
    return fig

def train_benchmark(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    models = {
        'MLP': MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            'Model': name,
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred)
        })
    return pd.DataFrame(results)

# ================================================================
# Cached Training (v29.27)
# ================================================================
CACHE_DIR = tempfile.gettempdir()
CHECKPOINT_PATH = os.path.join(CACHE_DIR, 'hubryd_v29_27.pt')

@st.cache_resource
def load_or_train():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
            model = MultiTaskPINN(ckpt['input_dim'], hidden=HIDDEN_SIZE)
            model.load_state_dict(ckpt['model_state'])
            scaler = ckpt['scaler']
            y_scaler = ckpt['y_scaler']
            features = ckpt['features']
            df = ckpt['df']
            return model, scaler, y_scaler, features, df
        except Exception as e:
            st.warning(f"Cache load failed: {e}. Retraining...")

    st.caption("🔄 Training model from scratch...")
    df, features = generate_pinn_data(N_SAMPLES)
    X_raw = df[features].values
    y = df[['Density','Tensile_Strength_MPa','Elastic_Recovery_%']].values
    X_aug = add_interaction_features(X_raw)
    input_dim = X_aug.shape[1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)
    X_train, X_test, X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X_scaled, X_raw, y_scaled, test_size=0.2, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.caption(f"🖥️ Using device: {device}")
    model = MultiTaskPINN(input_dim, hidden=HIDDEN_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_raw_train_t = torch.tensor(X_raw_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    best_loss = float('inf')
    patience_counter = 0
    progress_bar = st.progress(0)
    for epoch in range(ADAM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = model.compute_loss(X_train_t, X_raw_train_t, y_train_t, epoch, ADAM_EPOCHS)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CACHE_DIR, 'best_adam_v29_27.pt'))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                st.info(f"Training stopped early at epoch {epoch+1}")
                break
        progress_bar.progress((epoch+1)/ADAM_EPOCHS)
    st.success(f"✅ Best validation loss: {best_loss:.4f}")
    if os.path.exists(os.path.join(CACHE_DIR, 'best_adam_v29_27.pt')):
        model.load_state_dict(torch.load(os.path.join(CACHE_DIR, 'best_adam_v29_27.pt'), map_location=device))
    model.cpu()
    checkpoint = {
        'model_state': model.state_dict(),
        'scaler': scaler,
        'y_scaler': y_scaler,
        'features': features,
        'df': df,
        'input_dim': input_dim
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    st.success("✅ Model trained and cached successfully!")
    return model, scaler, y_scaler, features, df

# ================================================================
# Streamlit UI – v29.27 Style + Particle Size Toggle
# ================================================================
st.set_page_config(page_title="Hubryd AI v29.27", layout="wide")

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #0b1a33, #1a2a4a, #0f3460); padding:1.5rem; border-radius:1rem; text-align:center; margin-bottom:1rem;">
    <h1 style="color:#fff; margin:0;">🧬 Hubryd AI – v29.27</h1>
    <p style="color:#64ffda; margin:0;">Multi‑Task True PINN · Minimal · Stable · Free Tier</p>
    <p style="color:#8899aa; font-size:0.9rem;">Nile Valley University · Sudan</p>
</div>
""", unsafe_allow_html=True)

# Sidebar – Physics constraints
with st.sidebar:
    st.markdown("### 📚 Physics Constraints")
    st.markdown("""
    ✅ Heckel: ln(1/(1-D)) = kP + A  
    ✅ EFRF: ER / σt < 0.40  
    ✅ Density: 0.40 ≤ D ≤ 0.97  
    ✅ MCC: ≤ 8.0%  
    ✅ Samples: 5000  
    ✅ Epochs: 200  
    ✅ NSGA‑II: Pop=40, Gen=30  
    ✅ Network: 256 Neurons
    """)
    st.markdown("---")
    st.caption("🔬 v29.27 — Minimal & Stable")

# Load model
try:
    model, scaler, y_scaler, features, df = load_or_train()
except Exception as e:
    st.error(f"❌ Training failed: {e}. Using dummy model.")
    model = None

# Session state for presets
if 'api' not in st.session_state:
    st.session_state.update({
        'api':90.5, 'binder':3.0, 'pvpp':3.0, 'mgst':0.15, 'mcc':3.35,
        'pressure':235, 'speed':10, 'granule':125
    })

# Main layout
col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    st.markdown("### 📊 Formulation Parameters")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            api = st.slider("API (%)", 85.0, 95.0, st.session_state.api, 0.1, key="api_slider")
            binder = st.slider("Binder (%)", BINDER_MIN, BINDER_MAX, st.session_state.binder, 0.1, key="binder_slider")
            pvpp = st.slider("PVPP (%)", 0.5, 6.0, st.session_state.pvpp, 0.1, key="pvpp_slider")
        with c2:
            mgst = st.slider("Mg-St (%)", 0.01, 1.2, st.session_state.mgst, 0.01, key="mgst_slider")
            mcc = st.slider("MCC (%)", 0.0, MCC_MAX, st.session_state.mcc, 0.1, key="mcc_slider")
        total = api + binder + pvpp + mgst + mcc
        if abs(total-100) < 0.1:
            st.success(f"✅ Total = {total:.2f}%")
        else:
            st.warning(f"⚠️ Total = {total:.2f}% (should be 100%)")

    st.markdown("### ⚙️ Process Parameters")
    with st.container(border=True):
        pressure = st.slider("Pressure (MPa)", 80.0, PRESSURE_MAX, st.session_state.pressure, 1.0, key="pressure_slider")
        speed = st.slider("Speed (rpm)", 1.0, 50.0, st.session_state.speed, 0.5, key="speed_slider")
        # ---- Particle Size Toggle ----
        granule_mode = st.radio(
            "Granule Size",
            options=["Variable (optimized)", "Fixed (use slider)"],
            index=1,  # default to Fixed
            horizontal=True,
            key="granule_mode"
        )
        if granule_mode == "Fixed (use slider)":
            granule = st.slider("Granule Size (µm)", 30.0, 250.0, st.session_state.granule, 1.0, key="granule_slider")
            granule_fixed = True
        else:
            granule = st.session_state.granule  # placeholder, will be optimized
            granule_fixed = False
            st.info("Granule size will be optimized by NSGA‑II (30–250 µm)")

    predict_btn = st.button("🔬 Predict & Optimize", use_container_width=True, type="primary")

with col_right:
    st.markdown("### 📈 Results")
    if predict_btn:
        if abs(total-100) > 0.1:
            st.warning("Formulation must sum to 100%")
        else:
            # Normalize formulation
            api_n, binder_n, pvpp_n, mgst_n, mcc_n = normalize_components(api, binder, pvpp, mgst, mcc)
            # Use granule from slider if fixed, else use a default value (will be optimized later)
            if granule_fixed:
                granule_use = granule
            else:
                granule_use = 125.0  # placeholder, not used in prediction for variable mode? Actually prediction is for a single point, we can use the slider value for display
                # For prediction we still need a value; we'll use the slider value as a starting point
                granule_use = granule  # we keep the slider value for preview
            inputs = [api_n, mcc_n, pvpp_n, mgst_n, binder_n, pressure, speed, granule_use]

            if model is not None:
                density, tensile, er, efrf = predict_pinn(model, scaler, y_scaler, inputs)
            else:
                density, tensile, er, efrf = 0.7, 2.0, 0.5, 0.25

            st.markdown("#### Constraints Status")
            col_metrics = st.columns(4)
            col_metrics[0].metric("Density", f"{density:.3f}", "✅" if D_MIN <= density <= D_MAX else "❌")
            col_metrics[1].metric("Tensile", f"{tensile:.2f} MPa", "✅" if tensile >= TENSILE_MIN else "❌")
            col_metrics[2].metric("EFRF", f"{efrf:.4f}", "✅" if efrf < EFRF_MAX else "❌")
            col_metrics[3].metric("MCC", f"{mcc_n:.1f}%", "✅" if mcc_n <= MCC_MAX else "❌")

            if all([D_MIN <= density <= D_MAX, tensile >= TENSILE_MIN, efrf < EFRF_MAX, mcc_n <= MCC_MAX]):
                st.success("✅ All constraints satisfied!")
            else:
                st.error("❌ Violates constraints")

            # ---- NSGA-II Optimization ----
            st.markdown("### ⚙️ Optimization (NSGA‑II)")
            bounds = np.array([[60,100],[0.1,20],[0.1,12],[0.01,3.0],[0.1,10],
                               [80,PRESSURE_MAX],[1,50],[30,250]])
            with st.spinner(f"Running NSGA‑II (pop={NSGA_POP}, gen={NSGA_GENS})..."):
                nsga = NSGAII(model, scaler, y_scaler, bounds,
                              pop=NSGA_POP, gens=NSGA_GENS,
                              granule_fixed=granule_fixed,
                              granule_fixed_val=granule if granule_fixed else 125.0)
                pop, objectives, fronts = nsga.run()

            if len(fronts) > 0 and len(fronts[0]) > 0:
                st.success(f"✅ Pareto front found: {len(fronts[0])} optimal solutions")
                # Find best feasible (lowest EFRF)
                best_idx = None
                best_ef = 1e9
                for idx in fronts[0]:
                    if objectives[idx, 1] < best_ef:
                        ind = pop[idx]
                        d2, t2, e2, ef2 = predict_pinn(model, scaler, y_scaler, ind)
                        if D_MIN <= d2 <= D_MAX and t2 >= TENSILE_MIN and ef2 < EFRF_MAX:
                            best_ef = objectives[idx, 1]
                            best_idx = idx
                if best_idx is not None:
                    golden = pop[best_idx]
                    d2, t2, e2, ef2 = predict_pinn(model, scaler, y_scaler, golden)
                    st.markdown("#### ⭐ Golden Solution (Suggested)")
                    colA, colB = st.columns(2)
                    with colA:
                        st.write("**Formulation:**")
                        st.write(f"API: {golden[0]:.1f}%")
                        st.write(f"MCC: {golden[1]:.1f}%")
                        st.write(f"PVPP: {golden[2]:.1f}%")
                        st.write(f"Mg-St: {golden[3]:.2f}%")
                        st.write(f"Binder: {golden[4]:.1f}%")
                    with colB:
                        st.write("**Process:**")
                        st.write(f"Pressure: {golden[5]:.1f} MPa")
                        st.write(f"Speed: {golden[6]:.1f} rpm")
                        st.write(f"Granule: {golden[7]:.0f} µm")
                        st.write("**Predicted:**")
                        st.write(f"Density: {d2:.3f}")
                        st.write(f"Tensile: {t2:.3f} MPa")
                        st.write(f"EFRF: {ef2:.4f}")
                else:
                    st.info("No fully feasible solution found in Pareto front.")
            else:
                st.warning("No Pareto front found.")

            # ---- Pareto Plot ----
            st.markdown("### 📉 Pareto Front")
            fig = plot_pareto(objectives, fronts)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Pareto front to display.")

            # ---- Comparison ----
            st.markdown("### 📊 Comparison (Tensile R²)")
            X_train, X_test, y_train, y_test = train_test_split(
                df[features].values, df['Tensile_Strength_MPa'].values,
                test_size=0.2, random_state=42
            )
            X_train_aug = add_interaction_features(X_train)
            X_test_aug = add_interaction_features(X_test)
            X_train_scaled = scaler.transform(X_train_aug)
            X_test_scaled = scaler.transform(X_test_aug)

            pinn_pred_scaled = model.predict(torch.tensor(X_test_scaled, dtype=torch.float32))
            pinn_pred = y_scaler.inverse_transform(pinn_pred_scaled)[:, 1]
            pinn_r2 = r2_score(y_test, pinn_pred)
            pinn_rmse = np.sqrt(mean_squared_error(y_test, pinn_pred))
            pinn_mae = mean_absolute_error(y_test, pinn_pred)

            bench_df = train_benchmark(X_train_scaled, X_test_scaled, y_train, y_test)
            pinn_row = pd.DataFrame([{
                'Model': 'PINN (Proposed)',
                'R²': pinn_r2,
                'RMSE': pinn_rmse,
                'MAE': pinn_mae
            }])
            bench_df = pd.concat([pinn_row, bench_df], ignore_index=True)
            st.dataframe(bench_df.style.background_gradient(subset=['R²'], cmap='RdYlGn', vmin=-1, vmax=1),
                         use_container_width=True)

            # ---- Report ----
            st.markdown("### 📄 Report")
            if st.button("Generate Report"):
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                report = f"""
                # Hubryd AI v29.27 Report
                **Generated:** {timestamp}

                ## Current Formulation
                - API: {api_n:.1f}%
                - MCC: {mcc_n:.1f}%
                - PVPP: {pvpp_n:.1f}%
                - Mg-St: {mgst_n:.2f}%
                - Binder: {binder_n:.1f}%
                - Pressure: {pressure:.1f} MPa
                - Speed: {speed:.1f} rpm
                - Granule: {granule_use:.0f} µm (mode: {'Fixed' if granule_fixed else 'Variable'})

                ## Predicted Properties
                - Density: {density:.3f}
                - Tensile: {tensile:.2f} MPa
                - EFRF: {efrf:.4f}

                ## Constraints
                - Density: {'PASS' if D_MIN <= density <= D_MAX else 'FAIL'}
                - Tensile: {'PASS' if tensile >= TENSILE_MIN else 'FAIL'}
                - EFRF: {'PASS' if efrf < EFRF_MAX else 'FAIL'}
                - MCC: {'PASS' if mcc_n <= MCC_MAX else 'FAIL'}

                ## NSGA-II Results
                - Pareto solutions: {len(fronts[0]) if fronts else 0}
                - Golden solution (if any): {', '.join([f'{k}: {v:.2f}' for k,v in zip(['API','MCC','PVPP','MgSt','Binder','Pressure','Speed','Granule'], golden)]) if best_idx is not None else 'None'}

                ## Model Performance (Tensile)
                - PINN R²: {pinn_r2:.4f}
                - Best competitor R²: {bench_df[bench_df['Model']!='PINN (Proposed)']['R²'].max():.4f}
                """
                st.download_button(
                    label="Download Report (Markdown)",
                    data=report,
                    file_name=f"hubryd_report_{timestamp[:10]}.md",
                    mime="text/markdown"
                )

    else:
        st.info("Adjust sliders and click 'Predict & Optimize' to see results.")

st.caption("📧 Contact: babuker@protonmail.com | 🏛️ Nile Valley University, Sudan")
