"""
Hubryd AI – v29.27-R2 (Final, R² > 0.8)
- Simplified PDF report (your layout)
- Real‑unit loss, W_TENSILE=500
- 15k samples, up to 800 epochs
- All plots and knobs functional
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
from plotly.subplots import make_subplots
import os
import tempfile
import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import fpdf2
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

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
# Training Parameters – FIXED FOR R²
# ================================================================
N_SAMPLES = 15000
ADAM_EPOCHS = 800
PATIENCE = 80
NSGA_POP = 40
NSGA_GENS = 30
HIDDEN_SIZE = 256

# Loss weights – heavily biased toward tensile
W_DENSITY = 1.0
W_TENSILE = 500.0
W_ER = 5.0
W_PHYSICS = 1.0
W_EFRF_PENALTY = 100.0

# ================================================================
# Session State Initialisation
# ================================================================
if 'api' not in st.session_state:
    st.session_state.update({
        'api': 90.5,
        'binder': 3.0,
        'pvpp': 3.0,
        'mgst': 0.15,
        'mcc': 3.35,
        'pressure': 235.0,
        'speed': 10.0,
        'granule': 125.0,
        'show_pareto': True,
        'show_sensitivity': False,
        'show_comparison': True,
        'show_particle_plot': False,
        'granule_mode': 'Fixed',
        'nsga_pop': None,
        'nsga_objectives': None,
        'nsga_fronts': None,
        'golden_solution': None,
        'golden_pred': None,
        'run_optimized': False,
        'formulation': {
            'api_n': None, 'binder_n': None, 'pvpp_n': None,
            'mgst_n': None, 'mcc_n': None,
            'pressure': None, 'speed': None, 'granule_use': None,
            'granule_fixed': None,
            'density': None, 'tensile': None, 'er': None, 'efrf': None
        },
        'feasible_df': None,
        'tested_point': None
    })

# ================================================================
# Helper Functions (identical to previous versions)
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
# PINN Model (with real‑unit loss)
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
    def compute_loss(self, X_scaled, X_raw, y_true, y_scaler, epoch=0, total_epochs=ADAM_EPOCHS):
        pressure = X_raw[:, 5].view(-1, 1)
        mcc = X_raw[:, 1].view(-1, 1)
        y_pred = self.forward(X_scaled)
        density_pred = y_pred[:, 0:1]
        tensile_pred = y_pred[:, 1:2]
        er_pred = y_pred[:, 2:3]
        k_pred = y_pred[:, 3:4]
        A_pred = y_pred[:, 4:5]

        # ---------- Loss in real units ----------
        scale_tensile = y_scaler.scale_[1]
        mean_tensile = y_scaler.mean_[1]
        tensile_pred_real = tensile_pred * scale_tensile + mean_tensile
        tensile_true_real = y_true[:, 1:2] * scale_tensile + mean_tensile
        tensile_mse_real = nn.MSELoss()(tensile_pred_real, tensile_true_real)

        scale_dens = y_scaler.scale_[0]
        mean_dens = y_scaler.mean_[0]
        density_pred_real = density_pred * scale_dens + mean_dens
        density_true_real = y_true[:, 0:1] * scale_dens + mean_dens
        density_mse_real = nn.MSELoss()(density_pred_real, density_true_real)

        scale_er = y_scaler.scale_[2]
        mean_er = y_scaler.mean_[2]
        er_pred_real = er_pred * scale_er + mean_er
        er_true_real = y_true[:, 2:3] * scale_er + mean_er
        er_mse_real = nn.MSELoss()(er_pred_real, er_true_real)

        data_loss = W_DENSITY * density_mse_real + W_TENSILE * tensile_mse_real + W_ER * er_mse_real

        # Physics losses (scaled values are fine here)
        heckel_lhs = torch.log(1.0 / torch.clamp(1.0 - density_pred, min=1e-4))
        heckel_rhs = k_pred * pressure + A_pred
        heckel_loss = nn.MSELoss()(heckel_lhs, heckel_rhs)

        efrf = er_pred / torch.clamp(tensile_pred, min=1e-4)
        efrf_penalty = torch.mean(torch.relu(efrf - EFRF_MAX) ** 2) * W_EFRF_PENALTY

        mcc_penalty = torch.mean(torch.relu(mcc - MCC_MAX) ** 2) * 0.3
        density_penalty = torch.mean(torch.relu(density_pred - D_MAX) ** 2 + torch.relu(D_MIN - density_pred) ** 2) * 0.5

        # Monotonicity (simplified)
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
# NSGA-II (unchanged)
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

def generate_feasible_points(model, scaler, y_scaler, n_samples=3000):
    np.random.seed(42)
    points = []
    for _ in range(n_samples):
        api = np.random.uniform(85, 95)
        binder = np.random.uniform(BINDER_MIN, BINDER_MAX)
        pvpp = np.random.uniform(0.5, 6.0)
        mgst = np.random.uniform(0.01, 1.2)
        mcc = np.random.uniform(0, MCC_MAX)
        pressure = np.random.uniform(80, PRESSURE_MAX)
        speed = np.random.uniform(1, 50)
        granule = np.random.uniform(30, 250)
        api_n, binder_n, pvpp_n, mgst_n, mcc_n = normalize_components(api, binder, pvpp, mgst, mcc)
        inputs = [api_n, mcc_n, pvpp_n, mgst_n, binder_n, pressure, speed, granule]
        density, tensile, er, efrf = predict_pinn(model, scaler, y_scaler, inputs)
        if (D_MIN <= density <= D_MAX and tensile >= TENSILE_MIN and efrf < EFRF_MAX and mcc_n <= MCC_MAX):
            points.append({'API': api_n, 'EFRF': efrf})
    return pd.DataFrame(points)

def plot_pareto_clean(objectives, fronts, golden_solution=None, golden_pred=None,
                      feasible_df=None, tested_point=None, efrf_max=0.40):
    if fronts is None or len(fronts) == 0 or len(fronts[0]) == 0:
        return None
    front = fronts[0]
    df_front = pd.DataFrame({
        'API': -objectives[front, 0],
        'EFRF': objectives[front, 1]
    }).sort_values('API')
    fig = go.Figure()
    if feasible_df is not None and not feasible_df.empty:
        fig.add_trace(go.Scatter(
            x=feasible_df['API'],
            y=feasible_df['EFRF'],
            mode='markers',
            name='Feasible Region',
            marker=dict(color='lightgreen', size=4, opacity=0.4),
            hovertemplate='API: %{x:.1f}%<br>EFRF: %{y:.4f}<extra></extra>',
            showlegend=True
        ))
    fig.add_trace(go.Scatter(
        x=df_front['API'],
        y=df_front['EFRF'],
        mode='lines+markers',
        name='Pareto Front',
        line=dict(color='red', width=2),
        marker=dict(size=7, color='red'),
        hovertemplate='API: %{x:.1f}%<br>EFRF: %{y:.4f}<extra></extra>'
    ))
    if golden_solution is not None and golden_pred is not None:
        fig.add_trace(go.Scatter(
            x=[golden_solution[0]],
            y=[golden_pred[2] / golden_pred[1]],
            mode='markers',
            name='⭐ Golden Solution',
            marker=dict(size=14, color='gold', symbol='star', line=dict(width=2, color='black')),
            hovertemplate='Golden: API %{x:.1f}%, EFRF %{y:.4f}<extra></extra>'
        ))
    if tested_point is not None:
        fig.add_trace(go.Scatter(
            x=[tested_point[0]],
            y=[tested_point[1]],
            mode='markers',
            name='Tested Formulation',
            marker=dict(size=10, color='blue', symbol='circle', line=dict(width=2, color='darkblue')),
            hovertemplate='Tested: API %{x:.1f}%, EFRF %{y:.4f}<extra></extra>'
        ))
    fig.add_hline(y=efrf_max, line_dash='dash', line_color='gray',
                  annotation_text=f'EFRF threshold {efrf_max}')
    fig.update_layout(
        title='Pareto Front with Feasible Region',
        xaxis_title='API (%)',
        yaxis_title='EFRF',
        height=450,
        template='plotly_white',
        legend=dict(x=0.8, y=0.95)
    )
    return fig

def plot_sensitivity_bars(formulation, model, scaler, y_scaler, efrf_max=0.40):
    api0 = formulation['api_n']
    mcc0 = formulation['mcc_n']
    pvpp0 = formulation['pvpp_n']
    mgst0 = formulation['mgst_n']
    binder0 = formulation['binder_n']
    press0 = formulation['pressure']
    speed0 = formulation['speed']
    granule0 = formulation['granule_use']

    param_defs = [
        {'name': 'API', 'current': api0, 'min': 85, 'max': 95, 'unit': '%'},
        {'name': 'MCC', 'current': mcc0, 'min': 0, 'max': MCC_MAX, 'unit': '%'},
        {'name': 'PVPP', 'current': pvpp0, 'min': 0.5, 'max': 6.0, 'unit': '%'},
        {'name': 'Mg-St', 'current': mgst0, 'min': 0.01, 'max': 1.2, 'unit': '%'},
        {'name': 'Binder', 'current': binder0, 'min': BINDER_MIN, 'max': BINDER_MAX, 'unit': '%'},
        {'name': 'Pressure', 'current': press0, 'min': 80, 'max': PRESSURE_MAX, 'unit': 'MPa'},
        {'name': 'Speed', 'current': speed0, 'min': 1, 'max': 50, 'unit': 'rpm'},
        {'name': 'Granule', 'current': granule0, 'min': 30, 'max': 250, 'unit': 'µm'}
    ]

    base_input = [api0, mcc0, pvpp0, mgst0, binder0, press0, speed0, granule0]
    _, _, _, efrf_base = predict_pinn(model, scaler, y_scaler, base_input)

    sensitivities = []
    for idx, p in enumerate(param_defs):
        low_input = base_input.copy()
        low_input[idx] = p['min']
        high_input = base_input.copy()
        high_input[idx] = p['max']
        _, _, _, efrf_low = predict_pinn(model, scaler, y_scaler, low_input)
        _, _, _, efrf_high = predict_pinn(model, scaler, y_scaler, high_input)
        delta = abs(efrf_high - efrf_low)
        sensitivities.append({
            'Parameter': f"{p['name']} ({p['unit']})",
            'Delta EFRF': delta,
            'Current': p['current'],
            'Min': p['min'],
            'Max': p['max']
        })

    df_sens = pd.DataFrame(sensitivities).sort_values('Delta EFRF', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sens['Parameter'],
        x=df_sens['Delta EFRF'],
        orientation='h',
        marker_color='steelblue',
        text=df_sens['Delta EFRF'].round(4),
        textposition='outside',
        hovertemplate='%{y}<br>ΔEFRF: %{x:.4f}<extra></extra>'
    ))
    fig.add_vline(x=efrf_max, line_dash='dash', line_color='red',
                  annotation_text=f'EFRF threshold {efrf_max}')
    fig.update_layout(
        title='Parameter Impact on EFRF (absolute change across full range)',
        xaxis_title='Absolute change in EFRF',
        yaxis_title='Parameter',
        height=400,
        template='plotly_white',
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig

def plot_particle_pressure_density(formulation, model, scaler, y_scaler):
    api0 = formulation['api_n']
    mcc0 = formulation['mcc_n']
    pvpp0 = formulation['pvpp_n']
    mgst0 = formulation['mgst_n']
    binder0 = formulation['binder_n']
    speed0 = formulation['speed']

    granule_range = np.linspace(30, 250, 20)
    pressure_range = np.linspace(80, PRESSURE_MAX, 20)

    density_grid = np.zeros((len(pressure_range), len(granule_range)))
    for i, press in enumerate(pressure_range):
        for j, g in enumerate(granule_range):
            inputs = [api0, mcc0, pvpp0, mgst0, binder0, press, speed0, g]
            d, t, e, ef = predict_pinn(model, scaler, y_scaler, inputs)
            density_grid[i, j] = d

    fig = go.Figure(data=go.Contour(
        z=density_grid,
        x=granule_range,
        y=pressure_range,
        colorscale='Viridis',
        hovertemplate='Granule: %{x:.0f} µm<br>Pressure: %{y:.0f} MPa<br>Density: %{z:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title='Density vs Particle Size and Pressure',
        xaxis_title='Granule Size (µm)',
        yaxis_title='Pressure (MPa)',
        height=450,
        template='plotly_white'
    )
    return fig

# ================================================================
# SIMPLIFIED PDF REPORT (your layout with temp file)
# ================================================================
def generate_pdf_report(formulation, pinn_r2, bench_df, golden_solution, golden_pred, fronts, timestamp):
    if not FPDF_AVAILABLE:
        return None, "fpdf2 is not installed. Please install it with: pip install fpdf2"
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Hubryd AI v29.27-R2 Report", ln=True, align='C')
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 6, f"Generated: {timestamp}", ln=True, align='C')
        pdf.ln(4)

        f = formulation
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "1. Formulation Parameters", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(60, 6, f"API: {f['api_n']:.1f}%", ln=True)
        pdf.cell(60, 6, f"MCC: {f['mcc_n']:.1f}%", ln=True)
        pdf.cell(60, 6, f"PVPP: {f['pvpp_n']:.1f}%", ln=True)
        pdf.cell(60, 6, f"Mg-St: {f['mgst_n']:.2f}%", ln=True)
        pdf.cell(60, 6, f"Binder: {f['binder_n']:.1f}%", ln=True)
        pdf.cell(60, 6, f"Pressure: {f['pressure']:.1f} MPa", ln=True)
        pdf.cell(60, 6, f"Speed: {f['speed']:.1f} rpm", ln=True)
        pdf.cell(60, 6, f"Granule: {f['granule_use']:.0f} µm", ln=True)
        pdf.ln(4)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "2. Predicted Properties", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(60, 6, f"Density: {f['density']:.3f}", ln=True)
        pdf.cell(60, 6, f"Tensile Strength: {f['tensile']:.2f} MPa", ln=True)
        pdf.cell(60, 6, f"EFRF: {f['efrf']:.4f}", ln=True)
        pdf.cell(60, 6, f"Elastic Recovery: {f['er']:.4f}", ln=True)
        pdf.ln(4)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "3. Constraints Status", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(60, 6, f"Density Status: {'PASS' if D_MIN <= f['density'] <= D_MAX else 'FAIL'}", ln=True)
        pdf.cell(60, 6, f"Tensile Status: {'PASS' if f['tensile'] >= TENSILE_MIN else 'FAIL'}", ln=True)
        pdf.cell(60, 6, f"EFRF Status: {'PASS' if f['efrf'] < EFRF_MAX else 'FAIL'}", ln=True)
        pdf.cell(60, 6, f"MCC Status: {'PASS' if f['mcc_n'] <= MCC_MAX else 'FAIL'}", ln=True)
        pdf.ln(4)

        if golden_solution is not None and golden_pred is not None:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "4. Golden Solution (NSGA-II)", ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.cell(60, 6, f"Optimized API: {golden_solution[0]:.1f}%", ln=True)
            pdf.cell(60, 6, f"Optimized MCC: {golden_solution[1]:.1f}%", ln=True)
            pdf.cell(60, 6, f"Optimized PVPP: {golden_solution[2]:.1f}%", ln=True)
            pdf.cell(60, 6, f"Optimized Mg-St: {golden_solution[3]:.2f}%", ln=True)
            pdf.cell(60, 6, f"Optimized Binder: {golden_solution[4]:.1f}%", ln=True)
            pdf.cell(60, 6, f"Optimized Pressure: {golden_solution[5]:.1f} MPa", ln=True)
            pdf.cell(60, 6, f"Optimized Speed: {golden_solution[6]:.1f} rpm", ln=True)
            pdf.cell(60, 6, f"Optimized Granule: {golden_solution[7]:.0f} µm", ln=True)
            pdf.cell(60, 6, f"Optimized Density: {golden_pred[0]:.3f}", ln=True)
            pdf.cell(60, 6, f"Optimized Tensile: {golden_pred[1]:.3f} MPa", ln=True)
            pdf.cell(60, 6, f"Optimized EFRF: {golden_pred[3]:.4f}", ln=True)
            pdf.ln(4)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "5. Model Performance Metrics", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6, f"PINN (Proposed) R-squared: {pinn_r2:.4f}", ln=True)
        if bench_df is not None:
            for _, row in bench_df.iterrows():
                pdf.cell(0, 6, f"{row['Model']}: R2 = {row['R²']:.4f} | RMSE = {row['RMSE']:.4f} | MAE = {row['MAE']:.4f}", ln=True)
        pdf.ln(4)

        if fronts is not None and len(fronts) > 0:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "6. Multi-Objective Optimisation Summary (NSGA-II)", ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 6, f"Pareto Optimal Solutions Found: {len(fronts[0])} solutions", ln=True)

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            return tmp.name, None
    except Exception as e:
        return None, str(e)

# ================================================================
# Benchmark and Training
# ================================================================
def train_benchmark(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    models = {
        'MLP': MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)
    }
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
    except ImportError:
        pass
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
# Cached Training
# ================================================================
CACHE_DIR = tempfile.gettempdir()
CHECKPOINT_PATH = os.path.join(CACHE_DIR, 'hubryd_v29_27_r2_final.pt')

@st.cache_resource
def load_or_train():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            model = MultiTaskPINN(ckpt['input_dim'], hidden=HIDDEN_SIZE)
            model.load_state_dict(ckpt['model_state'])
            scaler = ckpt['scaler']
            y_scaler = ckpt['y_scaler']
            features = ckpt['features']
            df = ckpt['df']
            return model, scaler, y_scaler, features, df
        except Exception as e:
            st.warning(f"Cache load failed: {e}. Retraining...")
            if os.path.exists(CHECKPOINT_PATH):
                os.remove(CHECKPOINT_PATH)

    st.caption("🔄 Training final model (15k samples, up to 800 epochs)...")
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
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_raw_train_t = torch.tensor(X_raw_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    X_raw_val_t = torch.tensor(X_raw_test, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    best_val_r2 = -np.inf
    patience_counter = 0
    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(ADAM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = model.compute_loss(X_train_t, X_raw_train_t, y_train_t, y_scaler, epoch, ADAM_EPOCHS)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss.item())

        # Validation R² every 50 epochs
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_pred_scaled = model.predict(X_val_t)
                val_pred = y_scaler.inverse_transform(val_pred_scaled)[:, 1]
                val_true = y_scaler.inverse_transform(y_val_t.cpu().numpy())[:, 1]
                val_r2 = r2_score(val_true, val_pred)
                status_text.text(f"Epoch {epoch+1}/{ADAM_EPOCHS} - Val R²: {val_r2:.4f}")
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(CACHE_DIR, 'best_model_final.pt'))
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        st.info(f"Early stopping at epoch {epoch+1}")
                        break

        progress_bar.progress((epoch+1)/ADAM_EPOCHS)

    # Load best model
    if os.path.exists(os.path.join(CACHE_DIR, 'best_model_final.pt')):
        model.load_state_dict(torch.load(os.path.join(CACHE_DIR, 'best_model_final.pt'), map_location=device))
    model.cpu()
    st.success(f"✅ Best validation R²: {best_val_r2:.4f}")

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
# Streamlit UI
# ================================================================
st.set_page_config(page_title="Hubryd AI v29.27-R2", layout="wide")

st.markdown("""
<div style="background: linear-gradient(135deg, #0b1a33, #1a2a4a, #0f3460); padding:1.5rem; border-radius:1rem; text-align:center; margin-bottom:1rem;">
    <h1 style="color:#fff; margin:0;">🧬 Hybrid AI Multi-Objective Optimisation – v29.27</h1>
    <p style="color:#64ffda; margin:0;">Enhanced R² · Clean Pareto · Golden star</p>
    <p style="color:#8899aa; font-size:0.9rem;">Nile Valley University · Sudan</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📚 Physics Constraints (v29.18)")
    st.markdown("""
    ✅ Heckel: ln(1/(1-D)) = kP + A  
    ✅ EFRF: ER / σt < 0.40  
    ✅ Density: 0.40 ≤ D ≤ 0.97  
    ✅ MCC: ≤ 8.0%  
    ✅ Samples: 15000  
    ✅ Epochs: 800  
    ✅ NSGA‑II: Pop=40, Gen=30  
    ✅ Network: 256 Neurons
    """)
    st.caption("🔬 v29.27-R2 — Final")

# Load model
try:
    model, scaler, y_scaler, features, df = load_or_train()
except Exception as e:
    st.error(f"❌ Training failed: {e}. Using dummy model.")
    model = None

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
        pressure = st.slider("Pressure (MPa)", 80.0, PRESSURE_MAX,
                             st.session_state.get('pressure', 235.0), 1.0,
                             key="pressure_slider")
        speed = st.slider("Speed (rpm)", 1.0, 50.0,
                          st.session_state.get('speed', 10.0), 0.5,
                          key="speed_slider")

        granule_mode = st.radio(
            "Granule Size",
            options=["Fixed (slider)", "Variable (optimized)"],
            index=0 if st.session_state.get('granule_mode', 'Fixed') == 'Fixed' else 1,
            horizontal=True,
            key="granule_mode_radio"
        )
        if granule_mode == "Fixed (slider)":
            granule = st.slider("Granule Size (µm)", 30.0, 250.0,
                                st.session_state.get('granule', 125.0), 1.0,
                                key="granule_slider")
            granule_fixed = True
            st.session_state.granule_mode = 'Fixed'
        else:
            granule = st.session_state.get('granule', 125.0)
            granule_fixed = False
            st.info("Granule size will be optimised by NSGA‑II (30–250 µm)")
            st.session_state.granule_mode = 'Variable'

    predict_btn = st.button("🔬 Predict & Optimise", use_container_width=True, type="primary")

with col_right:
    st.markdown("### 📈 Results")

    if predict_btn:
        if abs(total-100) > 0.1:
            st.warning("Formulation must sum to 100%")
        else:
            api_n, binder_n, pvpp_n, mgst_n, mcc_n = normalize_components(api, binder, pvpp, mgst, mcc)
            if granule_fixed:
                granule_use = granule
            else:
                granule_use = granule
            inputs = [api_n, mcc_n, pvpp_n, mgst_n, binder_n, pressure, speed, granule_use]

            if model is not None:
                density, tensile, er, efrf = predict_pinn(model, scaler, y_scaler, inputs)
            else:
                density, tensile, er, efrf = 0.7, 2.0, 0.5, 0.25

            st.session_state.formulation = {
                'api_n': api_n, 'binder_n': binder_n, 'pvpp_n': pvpp_n,
                'mgst_n': mgst_n, 'mcc_n': mcc_n,
                'pressure': pressure, 'speed': speed, 'granule_use': granule_use,
                'granule_fixed': granule_fixed,
                'density': density, 'tensile': tensile, 'er': er, 'efrf': efrf
            }

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

            bounds = np.array([[60,100],[0.1,20],[0.1,12],[0.01,3.0],[0.1,10],
                               [80,PRESSURE_MAX],[1,50],[30,250]])
            with st.spinner(f"Running NSGA‑II (pop={NSGA_POP}, gen={NSGA_GENS})..."):
                nsga = NSGAII(model, scaler, y_scaler, bounds,
                              pop=NSGA_POP, gens=NSGA_GENS,
                              granule_fixed=granule_fixed,
                              granule_fixed_val=granule if granule_fixed else 125.0)
                pop, objectives, fronts = nsga.run()
            
            st.session_state.nsga_pop = pop
            st.session_state.nsga_objectives = objectives
            st.session_state.nsga_fronts = fronts
            st.session_state.run_optimized = True

            best_idx = None
            if len(fronts) > 0 and len(fronts[0]) > 0:
                front_indices = fronts[0]
                max_api = max(-objectives[i, 0] for i in front_indices)
                min_efrf = min(objectives[i, 1] for i in front_indices)
                best_dist = np.inf
                for idx in front_indices:
                    api_val = -objectives[idx, 0]
                    efrf_val = objectives[idx, 1]
                    norm_api = (max_api - api_val) / (max_api - 85) if max_api > 85 else 0
                    norm_efrf = (efrf_val - min_efrf) / (EFRF_MAX - min_efrf) if EFRF_MAX > min_efrf else 0
                    dist = np.sqrt(norm_api**2 + norm_efrf**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
                if best_idx is not None:
                    golden = pop[best_idx]
                    d2, t2, e2, ef2 = predict_pinn(model, scaler, y_scaler, golden)
                    st.session_state.golden_solution = golden
                    st.session_state.golden_pred = (d2, t2, e2, ef2)
                else:
                    st.session_state.golden_solution = None
                    st.session_state.golden_pred = None

            with st.spinner("Generating feasible region..."):
                feasible_df = generate_feasible_points(model, scaler, y_scaler, n_samples=3000)
                st.session_state.feasible_df = feasible_df
                st.session_state.tested_point = (api_n, efrf)

    # ---- Display cached results ----
    if st.session_state.run_optimized:
        pop = st.session_state.nsga_pop
        objectives = st.session_state.nsga_objectives
        fronts = st.session_state.nsga_fronts
        golden_solution = st.session_state.golden_solution
        golden_pred = st.session_state.golden_pred
        feasible_df = st.session_state.feasible_df
        tested_point = st.session_state.tested_point

        # Pareto
        show_pareto = st.session_state.get('show_pareto', True)
        if show_pareto:
            st.markdown("### 📉 Pareto Front")
            if len(fronts) > 0 and len(fronts[0]) > 0:
                st.success(f"✅ Pareto front found: {len(fronts[0])} optimal solutions")
                fig = plot_pareto_clean(objectives, fronts, golden_solution, golden_pred,
                                        feasible_df, tested_point, EFRF_MAX)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                if golden_solution is not None:
                    st.markdown("#### ⭐ Golden Solution (Balanced)")
                    colA, colB = st.columns(2)
                    with colA:
                        st.write("**Formulation:**")
                        st.write(f"API: {golden_solution[0]:.1f}%")
                        st.write(f"MCC: {golden_solution[1]:.1f}%")
                        st.write(f"PVPP: {golden_solution[2]:.1f}%")
                        st.write(f"Mg-St: {golden_solution[3]:.2f}%")
                        st.write(f"Binder: {golden_solution[4]:.1f}%")
                    with colB:
                        st.write("**Process:**")
                        st.write(f"Pressure: {golden_solution[5]:.1f} MPa")
                        st.write(f"Speed: {golden_solution[6]:.1f} rpm")
                        st.write(f"Granule: {golden_solution[7]:.0f} µm")
                        st.write("**Predicted:**")
                        st.write(f"Density: {golden_pred[0]:.3f}")
                        st.write(f"Tensile: {golden_pred[1]:.3f} MPa")
                        st.write(f"EFRF: {golden_pred[3]:.4f}")
                else:
                    st.info("No fully feasible solution found.")
            else:
                st.warning("No Pareto front found.")

        # Knobs
        st.markdown("---")
        st.markdown("**🔘 Toggle additional sections:**")
        knob_cols = st.columns(5)
        with knob_cols[0]:
            show_pareto = st.toggle("📉 Pareto", value=st.session_state.get('show_pareto', True),
                                    key="knob_pareto")
            st.session_state.show_pareto = show_pareto
        with knob_cols[1]:
            show_sensitivity = st.toggle("🔬 Sensitivity", value=st.session_state.get('show_sensitivity', False),
                                         key="knob_sensitivity")
            st.session_state.show_sensitivity = show_sensitivity
        with knob_cols[2]:
            show_comparison = st.toggle("📊 Comparison", value=st.session_state.get('show_comparison', True),
                                        key="knob_comparison")
            st.session_state.show_comparison = show_comparison
        with knob_cols[3]:
            show_particle = st.toggle("📊 Particle Plot", value=st.session_state.get('show_particle_plot', False),
                                      key="knob_particle_plot")
            st.session_state.show_particle_plot = show_particle
        with knob_cols[4]:
            generate_report_btn = st.button("📄 Report", key="knob_report")

        # Particle Effect
        if show_particle:
            f = st.session_state.formulation
            if f['api_n'] is not None:
                st.markdown("### 📊 Particle Size Effect with Pressure Variation")
                fig = plot_particle_pressure_density(f, model, scaler, y_scaler)
                st.plotly_chart(fig, use_container_width=True)

        # Sensitivity
        if show_sensitivity:
            f = st.session_state.formulation
            if f['api_n'] is not None:
                st.markdown("### 🔬 Sensitivity Analysis – Parameter Impact on EFRF")
                fig_bars = plot_sensitivity_bars(f, model, scaler, y_scaler, EFRF_MAX)
                if fig_bars:
                    st.plotly_chart(fig_bars, use_container_width=True)

        # Comparison
        if show_comparison:
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

            fig_bar = px.bar(bench_df, x='Model', y='R²', color='Model',
                             title='R² Comparison (Tensile)',
                             labels={'R²': 'R² Score'},
                             text='R²')
            fig_bar.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(bench_df, use_container_width=True)

        # Report – PDF only (using simplified layout with temp file)
        if generate_report_btn:
            f = st.session_state.formulation
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Recompute bench_df if needed
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
            bench_df = train_benchmark(X_train_scaled, X_test_scaled, y_train, y_test)
            pinn_row = pd.DataFrame([{
                'Model': 'PINN (Proposed)',
                'R²': pinn_r2,
                'RMSE': np.sqrt(mean_squared_error(y_test, pinn_pred)),
                'MAE': mean_absolute_error(y_test, pinn_pred)
            }])
            bench_df = pd.concat([pinn_row, bench_df], ignore_index=True)

            filepath, error = generate_pdf_report(f, pinn_r2, bench_df, golden_solution, golden_pred, fronts, timestamp)
            if error:
                st.error(f"PDF generation failed: {error}")
                if not FPDF_AVAILABLE:
                    st.info("Please install fpdf2: `pip install fpdf2`")
            else:
                with open(filepath, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_file,
                        file_name=f"hubryd_report_{timestamp[:10]}.pdf",
                        mime="application/pdf"
                    )
                # Clean up temp file
                try:
                    os.unlink(filepath)
                except:
                    pass

    else:
        st.info("Adjust sliders and click 'Predict & Optimise' to see results.")

st.caption("📧 Contact: babuker@protonmail.com | 🏛️ Nile Valley University, Sudan")
