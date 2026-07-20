"""
Hubryd AI – v29.27-R20 (Simplified Header)
Hybrid AI for Multi-Objective Optimization of Tablet Formulation
- Slider minimums: MCC=1.5%, PVPP=1.0%, MgSt=0.10%, Binder=1.4% (for manual tests up to ~96% API)
- Optimiser bounds: MCC≥2.0%, PVPP≥1.5%, MgSt≥0.3%, Binder≥3.0% (practical Golden Solution)
- Default formulation sums to 100% (API 90.5, MCC 3.5, PVPP 2.0, MgSt 0.5, Binder 3.5)
- All toggles fixed, early stopping fixed, vectorised, XGBoost fallback handled.
Nile Valley University · Sudan
"""

import streamlit as st

# ================================================================
# MUST BE FIRST STREAMLIT COMMAND
# ================================================================
st.set_page_config(page_title="Hybrid AI for Multi-Objective Optimization", layout="wide")

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
import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# ================================================================
# SLIDER RANGES (Wide for manual testing)
# ================================================================
D_MIN = 0.70
D_MAX = 0.99
TENSILE_MIN = 1.50
EFRF_MAX = 0.50

SLIDER_API_MIN = 80.0
SLIDER_API_MAX = 98.0
SLIDER_MCC_MIN = 1.5
SLIDER_MCC_MAX = 8.0
SLIDER_PVPP_MIN = 1.0
SLIDER_PVPP_MAX = 6.0
SLIDER_MGST_MIN = 0.10
SLIDER_MGST_MAX = 1.2
SLIDER_BINDER_MIN = 1.4
SLIDER_BINDER_MAX = 6.0

SLIDER_PRESSURE_MIN = 150.0
SLIDER_PRESSURE_MAX = 250.0
SLIDER_SPEED_MIN = 15.0
SLIDER_SPEED_MAX = 30.0
SLIDER_GRANULE_MIN = 30.0
SLIDER_GRANULE_MAX = 250.0

# ================================================================
# NSGA-II BOUNDS (Practical – for realistic optimisation)
# ================================================================
BOUND_MCC_MIN = 2.0
BOUND_MCC_MAX = 8.0
BOUND_PVPP_MIN = 1.5
BOUND_PVPP_MAX = 6.0
BOUND_MGST_MIN = 0.3
BOUND_MGST_MAX = 1.2
BOUND_BINDER_MIN = 3.0
BOUND_BINDER_MAX = 6.0
BOUND_PRESSURE_MIN = 150.0
BOUND_PRESSURE_MAX = 250.0
BOUND_SPEED_MIN = 15.0
BOUND_SPEED_MAX = 30.0
BOUND_GRANULE_MIN = 30.0
BOUND_GRANULE_MAX = 250.0

# ================================================================
# Training Parameters
# ================================================================
N_SAMPLES = 15000
ADAM_EPOCHS = 800
PATIENCE = 80
NSGA_POP = 80
NSGA_GENS = 50
HIDDEN_SIZE = 256

W_DENSITY = 1.0
W_TENSILE = 500.0
W_ER = 5.0
W_PHYSICS = 1.0
W_EFRF_PENALTY = 100.0

# ================================================================
# Session State – DEFAULT FORMULATION SUMS TO 100%
# ================================================================
if 'api' not in st.session_state:
    st.session_state.update({
        'api': 90.5,
        'binder': 3.5,
        'pvpp': 2.0,
        'mgst': 0.5,
        'mcc': 3.5,
        'pressure': 200.0,
        'speed': 20.0,
        'granule': 125.0,
        'show_cost_solution': False,
        'show_quality_solution': False,
        'show_comparison': True,
        'show_sensitivity': False,
        'show_particle_plot': False,
        'granule_mode': 'Fixed',
        'nsga_pop': None,
        'nsga_objectives': None,
        'nsga_fronts': None,
        'balanced_solution': None,
        'quality_solution': None,
        'cost_solution': None,
        'run_optimized': False,
        'formulation': {
            'api_n': None, 'binder_n': None, 'pvpp_n': None,
            'mgst_n': None, 'mcc_n': None,
            'pressure': None, 'speed': None, 'granule_use': None,
            'granule_fixed': None,
            'density': None, 'tensile': None, 'er': None, 'efrf': None
        },
        'feasible_df': None,
        'tested_point': None,
        'benchmark_df': None
    })

# ================================================================
# Helper Functions
# ================================================================
def normalize_components(api, binder, pvpp, mgst, mcc):
    api = np.asarray(api, dtype=float)
    binder = np.asarray(binder, dtype=float)
    pvpp = np.asarray(pvpp, dtype=float)
    mgst = np.asarray(mgst, dtype=float)
    mcc = np.asarray(mcc, dtype=float)

    api = np.clip(api, SLIDER_API_MIN, SLIDER_API_MAX)
    binder = np.clip(binder, SLIDER_BINDER_MIN, SLIDER_BINDER_MAX)
    pvpp = np.clip(pvpp, SLIDER_PVPP_MIN, SLIDER_PVPP_MAX)
    mgst = np.clip(mgst, SLIDER_MGST_MIN, SLIDER_MGST_MAX)
    mcc = np.clip(mcc, SLIDER_MCC_MIN, SLIDER_MCC_MAX)

    total = api + binder + pvpp + mgst + mcc
    total = np.where(total <= 0, 1.0, total)

    api = (api / total) * 100.0
    binder = (binder / total) * 100.0
    pvpp = (pvpp / total) * 100.0
    mgst = (mgst / total) * 100.0
    mcc = (mcc / total) * 100.0

    api = np.clip(api, SLIDER_API_MIN, SLIDER_API_MAX)
    binder = np.clip(binder, SLIDER_BINDER_MIN, SLIDER_BINDER_MAX)
    pvpp = np.clip(pvpp, SLIDER_PVPP_MIN, SLIDER_PVPP_MAX)
    mgst = np.clip(mgst, SLIDER_MGST_MIN, SLIDER_MGST_MAX)
    mcc = np.clip(mcc, SLIDER_MCC_MIN, SLIDER_MCC_MAX)

    total2 = api + binder + pvpp + mgst + mcc
    total2 = np.where(total2 <= 0, 1.0, total2)
    scale = 100.0 / total2
    api = api * scale
    binder = binder * scale
    pvpp = pvpp * scale
    mgst = mgst * scale
    mcc = mcc * scale

    api = np.clip(api, SLIDER_API_MIN, SLIDER_API_MAX)
    binder = np.clip(binder, SLIDER_BINDER_MIN, SLIDER_BINDER_MAX)
    pvpp = np.clip(pvpp, SLIDER_PVPP_MIN, SLIDER_PVPP_MAX)
    mgst = np.clip(mgst, SLIDER_MGST_MIN, SLIDER_MGST_MAX)
    mcc = np.clip(mcc, SLIDER_MCC_MIN, SLIDER_MCC_MAX)

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
    rng = np.random.default_rng(random_state)
    api_raw = rng.uniform(SLIDER_API_MIN, SLIDER_API_MAX, n_samples)
    binder_raw = rng.uniform(SLIDER_BINDER_MIN, SLIDER_BINDER_MAX, n_samples)
    pvpp_raw = rng.uniform(SLIDER_PVPP_MIN, SLIDER_PVPP_MAX, n_samples)
    mgst_raw = rng.uniform(SLIDER_MGST_MIN, SLIDER_MGST_MAX, n_samples)
    mcc_raw = rng.uniform(SLIDER_MCC_MIN, SLIDER_MCC_MAX, n_samples)
    pressure_raw = rng.uniform(SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX, n_samples)
    speed_raw = rng.uniform(SLIDER_SPEED_MIN, SLIDER_SPEED_MAX, n_samples)
    granule_raw = rng.uniform(SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX, n_samples)

    api_n, binder_n, pvpp_n, mgst_n, mcc_n = normalize_components(
        api_raw, binder_raw, pvpp_raw, mgst_raw, mcc_raw
    )

    X = np.column_stack([api_n, mcc_n, pvpp_n, mgst_n, binder_n,
                         pressure_raw, speed_raw, granule_raw])

    k = 0.025 + 0.0001 * pressure_raw
    A = 1.0 + 0.01 * (api_n - 85.0) - 0.05 * binder_n
    x_val = k * pressure_raw + A
    D = 1.0 - np.exp(-x_val)
    D = np.clip(D, D_MIN, D_MAX)
    D += rng.normal(0, 0.002, n_samples)
    D = np.clip(D, D_MIN, D_MAX)

    porosity = 1.0 - D
    sigma0 = 5.0 + 0.1 * (api_n - 85.0) + 0.2 * binder_n - 0.5 * mgst_n
    sigma0 = np.clip(sigma0, 2.0, 8.0)
    b = 2.5 - 0.005 * (pressure_raw - 80.0)
    b = np.clip(b, 1.5, 3.5)

    tensile_base = sigma0 * np.exp(-b * porosity)
    api_effect = 1.0 - 0.005 * (api_n - 85.0)
    binder_effect = 1.0 + 0.03 * (binder_n - 2.0)
    mgst_effect = 1.0 - 0.1 * (mgst_n - 0.2)
    pvpp_effect = 1.0 - 0.02 * (pvpp_n - 3.0)
    speed_effect = 1.0 - 0.002 * (speed_raw - 10.0)

    strength = (tensile_base * api_effect * binder_effect *
                mgst_effect * pvpp_effect * speed_effect)
    strength *= rng.normal(1.0, 0.01, n_samples)
    strength = np.clip(strength, 0.5, 6.0)

    er_base = (1.8 + 0.3 * (api_n - 85.0)/10.0 +
               0.08 * (speed_raw - 10.0)/30.0 -
               0.1 * (pressure_raw - 100.0)/150.0)
    er_base = er_base * (1.0 - 0.15 * (D - 0.4))
    er = er_base + rng.normal(0, 0.01, n_samples)
    er = np.clip(er, 0.5, 4.0)

    feature_names = ['API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
                     'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm']
    df = pd.DataFrame(X, columns=feature_names)
    df['Density'] = D
    df['Tensile_Strength_MPa'] = strength
    df['Elastic_Recovery_%'] = er
    return df, feature_names

# ================================================================
# PINN Model
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
        density = raw[:, 0:1]
        tensile = raw[:, 1:2]
        er = raw[:, 2:3]
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

        loss_dens = nn.MSELoss()(density_pred, y_true[:, 0:1])
        loss_tensile = nn.MSELoss()(tensile_pred, y_true[:, 1:2])
        loss_er = nn.MSELoss()(er_pred, y_true[:, 2:3])
        data_loss = W_DENSITY * loss_dens + W_TENSILE * loss_tensile + W_ER * loss_er

        scale_dens, mean_dens = y_scaler.scale_[0], y_scaler.mean_[0]
        scale_tensile, mean_tensile = y_scaler.scale_[1], y_scaler.mean_[1]
        scale_er, mean_er = y_scaler.scale_[2], y_scaler.mean_[2]

        density_real = density_pred * scale_dens + mean_dens
        tensile_real = tensile_pred * scale_tensile + mean_tensile
        er_real = er_pred * scale_er + mean_er

        heckel_lhs = torch.log(1.0 / torch.clamp(1.0 - density_real, min=1e-4))
        heckel_rhs = k_pred * pressure + A_pred
        heckel_loss = nn.MSELoss()(heckel_lhs, heckel_rhs)

        efrf_real = er_real / torch.clamp(tensile_real, min=1e-4)
        efrf_penalty = torch.mean(torch.relu(efrf_real - 0.50) ** 2) * W_EFRF_PENALTY

        mcc_penalty = torch.mean(torch.relu(mcc - SLIDER_MCC_MAX) ** 2) * 0.3
        density_penalty = torch.mean(torch.relu(density_real - D_MAX) ** 2 +
                                     torch.relu(D_MIN - density_real) ** 2) * 0.5

        physics_loss = W_PHYSICS * (heckel_loss + efrf_penalty) + mcc_penalty + density_penalty
        return data_loss + physics_loss

# ================================================================
# NSGA-II – PRACTICAL BOUNDS
# ================================================================
class NSGAII:
    def __init__(self, model, scaler, y_scaler, bounds, pop=NSGA_POP, gens=NSGA_GENS,
                 granule_fixed=True, granule_fixed_val=125.0):
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
        api = np.clip(api, self.bounds[0,0], self.bounds[0,1])
        mcc = np.clip(mcc, self.bounds[1,0], self.bounds[1,1])
        pvpp = np.clip(pvpp, self.bounds[2,0], self.bounds[2,1])
        mgst = np.clip(mgst, self.bounds[3,0], self.bounds[3,1])
        binder = np.clip(binder, self.bounds[4,0], self.bounds[4,1])
        pressure = np.clip(pressure, self.bounds[5,0], self.bounds[5,1])
        speed = np.clip(speed, self.bounds[6,0], self.bounds[6,1])
        if self.granule_fixed:
            granule = self.granule_fixed_val
        else:
            granule = np.clip(granule, self.bounds[7,0], self.bounds[7,1])
        return np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule])

    def _repair_batch(self, pop):
        api = pop[:, 0]; mcc = pop[:, 1]; pvpp = pop[:, 2]
        mgst = pop[:, 3]; binder = pop[:, 4]
        pressure = pop[:, 5]; speed = pop[:, 6]; granule = pop[:, 7]

        api, binder, pvpp, mgst, mcc = normalize_components(api, binder, pvpp, mgst, mcc)
        api = np.clip(api, self.bounds[0,0], self.bounds[0,1])
        mcc = np.clip(mcc, self.bounds[1,0], self.bounds[1,1])
        pvpp = np.clip(pvpp, self.bounds[2,0], self.bounds[2,1])
        mgst = np.clip(mgst, self.bounds[3,0], self.bounds[3,1])
        binder = np.clip(binder, self.bounds[4,0], self.bounds[4,1])
        pressure = np.clip(pressure, self.bounds[5,0], self.bounds[5,1])
        speed = np.clip(speed, self.bounds[6,0], self.bounds[6,1])
        if self.granule_fixed:
            granule = np.full_like(granule, self.granule_fixed_val)
        else:
            granule = np.clip(granule, self.bounds[7,0], self.bounds[7,1])
        return np.column_stack([api, mcc, pvpp, mgst, binder, pressure, speed, granule])

    def _evaluate(self, population):
        n = population.shape[0]
        repaired = self._repair_batch(population)
        inputs = repaired
        aug = add_interaction_features(inputs)
        scaled = self.scaler.transform(aug)
        X_t = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = self.model.predict(X_t)
            pred = self.y_scaler.inverse_transform(pred_scaled)

        density = np.clip(pred[:, 0], D_MIN, D_MAX)
        tensile = np.maximum(pred[:, 1], 1e-4)
        er = np.maximum(pred[:, 2], 1e-4)
        efrf = er / tensile
        efrf = np.clip(efrf, 1e-4, 5.0)

        g1 = D_MIN - density
        g2 = density - D_MAX
        violation = np.maximum(0, np.maximum(g1, g2))

        penalty = np.zeros(n)
        penalty += np.where(tensile < TENSILE_MIN, (TENSILE_MIN - tensile)**2, 0.0)
        penalty += np.where(efrf >= 0.40, (efrf - 0.40)**2, 0.0)
        mcc_val = repaired[:, 1]
        penalty += np.where(mcc_val > self.bounds[1,1], (mcc_val - self.bounds[1,1])**2, 0.0)

        objectives = np.zeros((n, 2))
        objectives[:, 0] = -(repaired[:, 0]) + 100.0 * penalty
        objectives[:, 1] = efrf + 100.0 * penalty

        return objectives, violation, repaired

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
                    dist[k] += (objectives[sorted_idx[k+1], obj_idx] -
                                objectives[sorted_idx[k-1], obj_idx]) / (f_max - f_min)
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
        rng = np.random.default_rng()
        pop = []
        for i in range(self.pop_size):
            if i < 0.3 * self.pop_size:
                api = rng.uniform(90, 95)
                mcc = rng.uniform(2.5, 4.0)
                binder = rng.uniform(3.5, 5.0)
                pvpp = rng.uniform(2, 4)
                mgst = rng.uniform(0.4, 0.8)
            else:
                api = rng.uniform(SLIDER_API_MIN, SLIDER_API_MAX)
                mcc = rng.uniform(BOUND_MCC_MIN, BOUND_MCC_MAX)
                binder = rng.uniform(BOUND_BINDER_MIN, BOUND_BINDER_MAX)
                pvpp = rng.uniform(BOUND_PVPP_MIN, BOUND_PVPP_MAX)
                mgst = rng.uniform(BOUND_MGST_MIN, BOUND_MGST_MAX)
            pressure = rng.uniform(BOUND_PRESSURE_MIN, BOUND_PRESSURE_MAX)
            speed = rng.uniform(BOUND_SPEED_MIN, BOUND_SPEED_MAX)
            granule = rng.uniform(BOUND_GRANULE_MIN, BOUND_GRANULE_MAX)
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
    rng = np.random.default_rng(42)
    api = rng.uniform(SLIDER_API_MIN, SLIDER_API_MAX, n_samples)
    binder = rng.uniform(SLIDER_BINDER_MIN, SLIDER_BINDER_MAX, n_samples)
    pvpp = rng.uniform(SLIDER_PVPP_MIN, SLIDER_PVPP_MAX, n_samples)
    mgst = rng.uniform(SLIDER_MGST_MIN, SLIDER_MGST_MAX, n_samples)
    mcc = rng.uniform(SLIDER_MCC_MIN, SLIDER_MCC_MAX, n_samples)
    pressure = rng.uniform(BOUND_PRESSURE_MIN, BOUND_PRESSURE_MAX, n_samples)
    speed = rng.uniform(BOUND_SPEED_MIN, BOUND_SPEED_MAX, n_samples)
    granule = rng.uniform(BOUND_GRANULE_MIN, BOUND_GRANULE_MAX, n_samples)

    api_n, binder_n, pvpp_n, mgst_n, mcc_n = normalize_components(
        api, binder, pvpp, mgst, mcc
    )
    inputs = np.column_stack([api_n, mcc_n, pvpp_n, mgst_n, binder_n,
                              pressure, speed, granule])

    aug = add_interaction_features(inputs)
    scaled = scaler.transform(aug)
    X_t = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model.predict(X_t)
        pred = y_scaler.inverse_transform(pred_scaled)
    density = np.clip(pred[:, 0], D_MIN, D_MAX)
    tensile = np.maximum(pred[:, 1], 1e-4)
    er = np.maximum(pred[:, 2], 1e-4)
    efrf = er / tensile
    efrf = np.clip(efrf, 1e-4, 5.0)

    mask = ((D_MIN <= density) & (density <= D_MAX) &
            (tensile >= TENSILE_MIN) & (efrf < 0.40) &
            (mcc_n <= BOUND_MCC_MAX) & (mcc_n >= BOUND_MCC_MIN))
    feasible_api = api_n[mask]
    feasible_efrf = efrf[mask]
    return pd.DataFrame({'API': feasible_api, 'EFRF': feasible_efrf})

def plot_pareto_clean(objectives, fronts, balanced_solution=None, feasible_df=None,
                      tested_point=None, efrf_max=0.40):
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
            name='Feasible Region (EFRF<0.40)',
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
    if tested_point is not None:
        fig.add_trace(go.Scatter(
            x=[tested_point[0]],
            y=[tested_point[1]],
            mode='markers',
            name='Tested Formulation',
            marker=dict(size=10, color='blue', symbol='circle',
                        line=dict(width=2, color='darkblue')),
            hovertemplate='Tested: API %{x:.1f}%, EFRF %{y:.4f}<extra></extra>'
        ))
    fig.add_hline(y=0.40, line_dash='dash', line_color='gray',
                  annotation_text='EFRF threshold (0.40)')
    fig.update_layout(
        title='Pareto Front with Feasible Region (D: 0.70–0.99, EFRF<0.40)',
        xaxis_title='API (%)',
        yaxis_title='EFRF',
        height=450,
        template='plotly_white',
        legend=dict(x=0.8, y=0.95)
    )
    return fig

def plot_sensitivity_bars(formulation, model, scaler, y_scaler, efrf_max=0.40):
    api0 = formulation['api_n']; mcc0 = formulation['mcc_n']
    pvpp0 = formulation['pvpp_n']; mgst0 = formulation['mgst_n']
    binder0 = formulation['binder_n']; press0 = formulation['pressure']
    speed0 = formulation['speed']; granule0 = formulation['granule_use']

    param_defs = [
        {'name': 'API', 'current': api0, 'min': SLIDER_API_MIN, 'max': SLIDER_API_MAX, 'unit': '%'},
        {'name': 'MCC', 'current': mcc0, 'min': SLIDER_MCC_MIN, 'max': SLIDER_MCC_MAX, 'unit': '%'},
        {'name': 'PVPP', 'current': pvpp0, 'min': SLIDER_PVPP_MIN, 'max': SLIDER_PVPP_MAX, 'unit': '%'},
        {'name': 'Mg-St', 'current': mgst0, 'min': SLIDER_MGST_MIN, 'max': SLIDER_MGST_MAX, 'unit': '%'},
        {'name': 'Binder', 'current': binder0, 'min': SLIDER_BINDER_MIN, 'max': SLIDER_BINDER_MAX, 'unit': '%'},
        {'name': 'Pressure', 'current': press0, 'min': BOUND_PRESSURE_MIN, 'max': BOUND_PRESSURE_MAX, 'unit': 'MPa'},
        {'name': 'Speed', 'current': speed0, 'min': BOUND_SPEED_MIN, 'max': BOUND_SPEED_MAX, 'unit': 'rpm'},
        {'name': 'Granule', 'current': granule0, 'min': BOUND_GRANULE_MIN, 'max': BOUND_GRANULE_MAX, 'unit': 'µm'}
    ]

    base_input = [api0, mcc0, pvpp0, mgst0, binder0, press0, speed0, granule0]
    _, _, _, efrf_base = predict_pinn(model, scaler, y_scaler, base_input)

    sensitivities = []
    for idx, p in enumerate(param_defs):
        low_input = base_input.copy(); low_input[idx] = p['min']
        high_input = base_input.copy(); high_input[idx] = p['max']
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
    fig.add_vline(x=0.40, line_dash='dash', line_color='red',
                  annotation_text='EFRF threshold 0.40')
    fig.update_layout(
        title='Parameter Impact on EFRF (absolute change across full range)',
        xaxis_title='Absolute change in EFRF',
        yaxis_title='Parameter',
        height=400,
        template='plotly_white'
    )
    return fig

def plot_particle_pressure_density(formulation, model, scaler, y_scaler):
    api0 = formulation['api_n']; mcc0 = formulation['mcc_n']
    pvpp0 = formulation['pvpp_n']; mgst0 = formulation['mgst_n']
    binder0 = formulation['binder_n']; speed0 = formulation['speed']
    granule_range = np.linspace(BOUND_GRANULE_MIN, BOUND_GRANULE_MAX, 20)
    pressure_range = np.linspace(BOUND_PRESSURE_MIN, BOUND_PRESSURE_MAX, 20)
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
        title='Density vs Particle Size and Pressure (D: 0.70–0.99)',
        xaxis_title='Granule Size (µm)',
        yaxis_title='Pressure (MPa)',
        height=450,
        template='plotly_white'
    )
    return fig

# ================================================================
# PDF Report (unchanged)
# ================================================================
def generate_pdf_report(formulation, bench_df, balanced_solution, quality_solution, cost_solution,
                        balanced_pred, quality_pred, cost_pred, fronts, timestamp):
    if not FPDF_AVAILABLE:
        return None, "fpdf2 is not installed. Please install it with: pip install fpdf2"
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Hybrid AI for Multi-Objective Optimization of Tablet Formulation", ln=True, align='C')
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
        pdf.cell(0, 8, "3. Constraints Status (D: 0.70-0.99, Tensile >= 1.50, EFRF < 0.40)", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(60, 6, f"Density Status: {'PASS' if D_MIN <= f['density'] <= D_MAX else 'FAIL'}", ln=True)
        pdf.cell(60, 6, f"Tensile Status: {'PASS' if f['tensile'] >= TENSILE_MIN else 'FAIL'}", ln=True)
        pdf.cell(60, 6, f"EFRF Status: {'PASS' if f['efrf'] < 0.40 else 'FAIL'}", ln=True)
        pdf.cell(60, 6, f"MCC Status: {'PASS' if f['mcc_n'] <= BOUND_MCC_MAX else 'FAIL'}", ln=True)
        pdf.ln(4)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "4. Optimised Solutions (Pareto Front)", ln=True)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 6, "Golden Solution (Balanced)", ln=True)
        pdf.set_font("Arial", "", 10)
        if balanced_solution is not None and balanced_pred is not None:
            pdf.cell(60, 6, f"API: {balanced_solution[0]:.1f}%", ln=True)
            pdf.cell(60, 6, f"MCC: {balanced_solution[1]:.1f}%", ln=True)
            pdf.cell(60, 6, f"PVPP: {balanced_solution[2]:.1f}%", ln=True)
            pdf.cell(60, 6, f"Mg-St: {balanced_solution[3]:.2f}%", ln=True)
            pdf.cell(60, 6, f"Binder: {balanced_solution[4]:.1f}%", ln=True)
            pdf.cell(60, 6, f"Pressure: {balanced_solution[5]:.1f} MPa", ln=True)
            pdf.cell(60, 6, f"Speed: {balanced_solution[6]:.1f} rpm", ln=True)
            pdf.cell(60, 6, f"Granule: {balanced_solution[7]:.0f} µm", ln=True)
            pdf.cell(60, 6, f"Density: {balanced_pred[0]:.3f}", ln=True)
            pdf.cell(60, 6, f"Tensile: {balanced_pred[1]:.3f} MPa", ln=True)
            pdf.cell(60, 6, f"EFRF: {balanced_pred[3]:.4f}", ln=True)
            pdf.ln(4)

        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 6, "Quality-Optimised Solution (Max Tensile)", ln=True)
        pdf.set_font("Arial", "", 10)
        if quality_solution is not None and quality_pred is not None:
            pdf.cell(60, 6, f"API: {quality_solution[0]:.1f}%", ln=True)
            pdf.cell(60, 6, f"MCC: {quality_solution[1]:.1f}%", ln=True)
            pdf.cell(60, 6, f"PVPP: {quality_solution[2]:.1f}%", ln=True)
            pdf.cell(60, 6, f"Mg-St: {quality_solution[3]:.2f}%", ln=True)
            pdf.cell(60, 6, f"Binder: {quality_solution[4]:.1f}%", ln=True)
            pdf.cell(60, 6, f"Pressure: {quality_solution[5]:.1f} MPa", ln=True)
            pdf.cell(60, 6, f"Speed: {quality_solution[6]:.1f} rpm", ln=True)
            pdf.cell(60, 6, f"Granule: {quality_solution[7]:.0f} µm", ln=True)
            pdf.cell(60, 6, f"Density: {quality_pred[0]:.3f}", ln=True)
            pdf.cell(60, 6, f"Tensile: {quality_pred[1]:.3f} MPa", ln=True)
            pdf.cell(60, 6, f"EFRF: {quality_pred[3]:.4f}", ln=True)
            pdf.ln(4)

        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 6, "Cost-Optimised Solution (Max API, Min Pressure)", ln=True)
        pdf.set_font("Arial", "", 10)
        if cost_solution is not None and cost_pred is not None:
            pdf.cell(60, 6, f"API: {cost_solution[0]:.1f}%", ln=True)
            pdf.cell(60, 6, f"MCC: {cost_solution[1]:.1f}%", ln=True)
            pdf.cell(60, 6, f"PVPP: {cost_solution[2]:.1f}%", ln=True)
            pdf.cell(60, 6, f"Mg-St: {cost_solution[3]:.2f}%", ln=True)
            pdf.cell(60, 6, f"Binder: {cost_solution[4]:.1f}%", ln=True)
            pdf.cell(60, 6, f"Pressure: {cost_solution[5]:.1f} MPa", ln=True)
            pdf.cell(60, 6, f"Speed: {cost_solution[6]:.1f} rpm", ln=True)
            pdf.cell(60, 6, f"Granule: {cost_solution[7]:.0f} µm", ln=True)
            pdf.cell(60, 6, f"Density: {cost_pred[0]:.3f}", ln=True)
            pdf.cell(60, 6, f"Tensile: {cost_pred[1]:.3f} MPa", ln=True)
            pdf.cell(60, 6, f"EFRF: {cost_pred[3]:.4f}", ln=True)
            pdf.ln(4)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "5. Model Performance Comparison", ln=True)
        pdf.set_font("Arial", "", 10)
        if bench_df is not None:
            for _, row in bench_df.iterrows():
                model_name = row['Model']
                r2_str = row['R2 (Test)']
                rmse_str = row['RMSE (MPa)']
                mae_str = row['MAE (MPa)']
                r2_str = r2_str.replace('–', '-')
                rmse_str = rmse_str.replace('–', '-')
                mae_str = mae_str.replace('–', '-')
                pdf.cell(0, 6, f"{model_name}: R2 = {r2_str} | RMSE = {rmse_str} | MAE = {mae_str}", ln=True)
        pdf.ln(4)

        if fronts is not None and len(fronts) > 0:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "6. Multi-Objective Optimisation Summary (NSGA-II)", ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 6, f"Pareto Optimal Solutions Found: {len(fronts[0])} solutions", ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            return tmp.name, None
    except Exception as e:
        return None, str(e)

# ================================================================
# Cached Training
# ================================================================
CACHE_DIR = tempfile.gettempdir()
CHECKPOINT_PATH = os.path.join(CACHE_DIR, 'hubryd_v29_27_r20_eng.pt')

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

    st.caption("🔄 Training specified-range model (15k samples, up to 800 epochs)...")
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

        model.eval()
        with torch.no_grad():
            val_pred_scaled = model.predict(X_val_t)
            val_pred = y_scaler.inverse_transform(val_pred_scaled)[:, 1]
            val_true = y_scaler.inverse_transform(y_val_t.cpu().numpy())[:, 1]
            val_r2 = r2_score(val_true, val_pred)

        if epoch % 50 == 0:
            status_text.text(f"Epoch {epoch+1}/{ADAM_EPOCHS} - Val R²: {val_r2:.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CACHE_DIR, 'best_model_final_eng.pt'))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                st.info(f"Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break

        progress_bar.progress((epoch+1)/ADAM_EPOCHS)

    if os.path.exists(os.path.join(CACHE_DIR, 'best_model_final_eng.pt')):
        model.load_state_dict(torch.load(os.path.join(CACHE_DIR, 'best_model_final_eng.pt'), map_location=device))
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
# Model Comparison (without caching – safe for Streamlit)
# ================================================================
def run_model_comparison(model, scaler, y_scaler, features, df, device):
    X_raw_all = df[features].values
    y_raw_all = df[['Density','Tensile_Strength_MPa','Elastic_Recovery_%']].values
    X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
        X_raw_all, y_raw_all, test_size=0.2, random_state=42
    )
    X_b_train_scaled = scaler.transform(add_interaction_features(X_b_train))
    X_b_test_scaled = scaler.transform(add_interaction_features(X_b_test))
    y_train_target = y_b_train[:, 1]
    y_test_target = y_b_test[:, 1]

    model.eval()
    with torch.no_grad():
        pinn_input = torch.tensor(X_b_test_scaled, dtype=torch.float32).to(device)
        pinn_pred_scaled = model.predict(pinn_input)
        pinn_pred = y_scaler.inverse_transform(pinn_pred_scaled)[:, 1]

    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    mlp_mod = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42)
    mlp_mod.fit(X_b_train_scaled, y_train_target)
    mlp_pred = mlp_mod.predict(X_b_test_scaled)

    rf_mod = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_mod.fit(X_b_train_scaled, y_train_target)
    rf_pred = rf_mod.predict(X_b_test_scaled)

    models_registry = {
        'PINN (Proposed)': (pinn_pred, 'Enforced'),
        'MLP (Baseline)': (mlp_pred, 'Not enforced'),
        'Random Forest': (rf_pred, 'Not enforced')
    }

    try:
        from xgboost import XGBRegressor
        xgb_mod = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)
        xgb_mod.fit(X_b_train_scaled, y_train_target)
        xgb_pred = xgb_mod.predict(X_b_test_scaled)
        models_registry['XGBoost'] = (xgb_pred, 'Not enforced')
    except ImportError:
        pass

    def compute_metrics_with_variance(y_true, y_pred, n_bootstraps=15):
        rng = np.random.default_rng(42)
        r2_scores, rmse_scores, mae_scores = [], [], []
        for _ in range(n_bootstraps):
            indices = rng.choice(len(y_true), len(y_true), replace=True)
            r2_scores.append(r2_score(y_true[indices], y_pred[indices]))
            rmse_scores.append(np.sqrt(mean_squared_error(y_true[indices], y_pred[indices])))
            mae_scores.append(mean_absolute_error(y_true[indices], y_pred[indices]))
        return (np.mean(r2_scores), np.std(r2_scores),
                np.mean(rmse_scores), np.std(rmse_scores),
                np.mean(mae_scores), np.std(mae_scores))

    table_rows = []
    chart_data = []
    for name, (preds, consistency) in models_registry.items():
        r2_m, r2_s, rmse_m, rmse_s, mae_m, mae_s = compute_metrics_with_variance(y_test_target, preds)
        table_rows.append({
            'Model': name,
            'R2 (Test)': f"{r2_m:.2f} +/- {r2_s:.2f}",
            'RMSE (MPa)': f"{rmse_m:.2f} +/- {rmse_s:.2f}",
            'MAE (MPa)': f"{mae_m:.2f} +/- {mae_s:.2f}",
            'Physical Consistency': consistency
        })
        chart_data.append({'Model': name, 'R² Score': r2_m})

    bench_df = pd.DataFrame(table_rows)
    return bench_df, chart_data

# ================================================================
# MAIN UI – SIMPLIFIED HEADER
# ================================================================
st.markdown("""
<div style="background: #0b1a33; padding:1rem; border-radius:0.5rem; text-align:center; margin-bottom:1rem;">
    <h2 style="color:#fff; margin:0;">🧬 Hybrid AI · Multi‑Objective Tablet Optimization</h2>
    <p style="color:#64ffda; margin:0; font-size:0.9rem;">v29.27-R20</p>
    <p style="color:#aabbcc; margin:0; font-size:0.85rem;">Nile Valley University, Sudan</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📚 Physics Constraints")
    st.markdown(f"""
    ✅ **API:** {SLIDER_API_MIN:.0f}–{SLIDER_API_MAX:.0f}%  
    ✅ **Density:** {D_MIN:.2f}–{D_MAX:.2f}  
    ✅ **Tensile:** ≥ {TENSILE_MIN:.2f} MPa  
    ✅ **EFRF:** &lt; 0.40 (feasible)  
    ✅ **Slider MCC:** {SLIDER_MCC_MIN:.1f}–{SLIDER_MCC_MAX:.1f}%  
    ✅ **Slider PVPP:** {SLIDER_PVPP_MIN:.1f}–{SLIDER_PVPP_MAX:.1f}%  
    ✅ **Slider MgSt:** {SLIDER_MGST_MIN:.2f}–{SLIDER_MGST_MAX:.2f}%  
    ✅ **Slider Binder:** {SLIDER_BINDER_MIN:.1f}–{SLIDER_BINDER_MAX:.1f}%  
    ✅ **Optimiser Bounds (practical):** MCC≥2.0%, PVPP≥1.5%, MgSt≥0.3%, Binder≥3.0%  
    ✅ **Pressure:** {BOUND_PRESSURE_MIN:.0f}–{BOUND_PRESSURE_MAX:.0f} MPa  
    ✅ **Speed:** {BOUND_SPEED_MIN:.0f}–{BOUND_SPEED_MAX:.0f} RPM  
    ✅ **NSGA‑II:** Pop=80, Gen=50
    """)
    st.caption("🔬 v29.27-R20 — Complete Final Version")

# Load model (cached)
try:
    model, scaler, y_scaler, features, df = load_or_train()
except Exception as e:
    st.error(f"❌ Training failed: {e}. Using dummy model.")
    model = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if model is not None:
    device = next(model.parameters()).device

# Main layout
col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    st.markdown("### 📊 Formulation Parameters")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            api = st.slider("API (%)", SLIDER_API_MIN, SLIDER_API_MAX, st.session_state.api, 0.1, key="api_slider")
            binder = st.slider("Binder (%)", SLIDER_BINDER_MIN, SLIDER_BINDER_MAX, st.session_state.binder, 0.1, key="binder_slider")
            pvpp = st.slider("PVPP (%)", SLIDER_PVPP_MIN, SLIDER_PVPP_MAX, st.session_state.pvpp, 0.1, key="pvpp_slider")
        with c2:
            mgst = st.slider("Mg-St (%)", SLIDER_MGST_MIN, SLIDER_MGST_MAX, st.session_state.mgst, 0.01, key="mgst_slider")
            mcc = st.slider("MCC (%)", SLIDER_MCC_MIN, SLIDER_MCC_MAX, st.session_state.mcc, 0.1, key="mcc_slider")
        total = api + binder + pvpp + mgst + mcc
        if abs(total-100) < 0.1:
            st.success(f"✅ Total = {total:.2f}%")
        else:
            st.warning(f"⚠️ Total = {total:.2f}% (should be 100%)")

    st.markdown("### ⚙️ Process Parameters (Practical Ranges)")
    with st.container(border=True):
        pressure = st.slider("Pressure (MPa)", BOUND_PRESSURE_MIN, BOUND_PRESSURE_MAX,
                             st.session_state.get('pressure', 200.0), 1.0,
                             key="pressure_slider")
        speed = st.slider("Speed (rpm)", BOUND_SPEED_MIN, BOUND_SPEED_MAX,
                          st.session_state.get('speed', 20.0), 0.5,
                          key="speed_slider")
        granule_mode = st.radio(
            "Granule Size",
            options=["Fixed (slider)", "Variable (optimized)"],
            index=0 if st.session_state.get('granule_mode', 'Fixed') == 'Fixed' else 1,
            horizontal=True,
            key="granule_mode_radio"
        )
        if granule_mode == "Fixed (slider)":
            granule = st.slider("Granule Size (µm)", BOUND_GRANULE_MIN, BOUND_GRANULE_MAX,
                                st.session_state.get('granule', 125.0), 1.0,
                                key="granule_slider")
            granule_fixed = True
            st.session_state.granule_mode = 'Fixed'
        else:
            granule = st.session_state.get('granule', 125.0)
            granule_fixed = False
            st.info(f"Granule size optimised by NSGA‑II ({BOUND_GRANULE_MIN:.0f}–{BOUND_GRANULE_MAX:.0f} µm)")
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

            st.markdown("**Constraints Status** (D: 0.70–0.99, Tensile ≥ 1.50, EFRF < 0.40)")
            col_metrics = st.columns(4)
            col_metrics[0].metric("Density", f"{density:.3f}", f"[{D_MIN:.2f}, {D_MAX:.2f}]")
            col_metrics[1].metric("Tensile", f"{tensile:.2f} MPa", f"≥ {TENSILE_MIN:.2f}")
            col_metrics[2].metric("EFRF", f"{efrf:.4f}", f"< 0.40")
            col_metrics[3].metric("MCC", f"{mcc_n:.1f}%", f"≤ {BOUND_MCC_MAX:.1f}%")

            if all([D_MIN <= density <= D_MAX, tensile >= TENSILE_MIN, efrf < 0.40, mcc_n <= BOUND_MCC_MAX]):
                st.success("✅ All constraints satisfied")
            else:
                st.error("❌ Violates constraints")

            # ---- NSGA-II bounds (practical) ----
            bounds = np.array([
                [SLIDER_API_MIN, SLIDER_API_MAX],
                [BOUND_MCC_MIN, BOUND_MCC_MAX],
                [BOUND_PVPP_MIN, BOUND_PVPP_MAX],
                [BOUND_MGST_MIN, BOUND_MGST_MAX],
                [BOUND_BINDER_MIN, BOUND_BINDER_MAX],
                [BOUND_PRESSURE_MIN, BOUND_PRESSURE_MAX],
                [BOUND_SPEED_MIN, BOUND_SPEED_MAX],
                [BOUND_GRANULE_MIN, BOUND_GRANULE_MAX]
            ])
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

            # ---- Extract 3 Solutions ----
            balanced_idx = None
            quality_idx = None
            cost_idx = None

            if len(fronts) > 0 and len(fronts[0]) > 0:
                front_indices = fronts[0]
                max_api = max(-objectives[i, 0] for i in front_indices)
                min_efrf = min(objectives[i, 1] for i in front_indices)
                best_dist = np.inf
                api_range = SLIDER_API_MAX - SLIDER_API_MIN
                efrf_range = 0.40 - min_efrf

                for idx in front_indices:
                    api_val = -objectives[idx, 0]
                    efrf_val = objectives[idx, 1]
                    norm_api = (SLIDER_API_MAX - api_val) / api_range if api_range > 0 else 0
                    norm_efrf = (efrf_val - min_efrf) / efrf_range if efrf_range > 0 else 0
                    dist = np.sqrt(norm_api**2 + norm_efrf**2)
                    if dist < best_dist:
                        best_dist = dist
                        balanced_idx = idx

                # Quality: max tensile
                best_tensile = -np.inf
                for idx in front_indices:
                    ind = pop[idx]
                    d2, t2, e2, ef2 = predict_pinn(model, scaler, y_scaler, ind)
                    if t2 > best_tensile:
                        best_tensile = t2
                        quality_idx = idx

                # Cost: max API, min pressure
                best_cost_score = -np.inf
                for idx in front_indices:
                    ind = pop[idx]
                    api_val = ind[0]
                    pressure_val = ind[5]
                    cost_score = api_val - 0.05 * pressure_val
                    if cost_score > best_cost_score:
                        best_cost_score = cost_score
                        cost_idx = idx

                st.session_state.balanced_solution = pop[balanced_idx] if balanced_idx is not None else None
                st.session_state.quality_solution = pop[quality_idx] if quality_idx is not None else None
                st.session_state.cost_solution = pop[cost_idx] if cost_idx is not None else None

            with st.spinner("Generating feasible region..."):
                feasible_df = generate_feasible_points(model, scaler, y_scaler, n_samples=3000)
                st.session_state.feasible_df = feasible_df
                st.session_state.tested_point = (api_n, efrf)

    if st.session_state.run_optimized:
        pop = st.session_state.nsga_pop
        objectives = st.session_state.nsga_objectives
        fronts = st.session_state.nsga_fronts
        balanced_solution = st.session_state.balanced_solution
        quality_solution = st.session_state.quality_solution
        cost_solution = st.session_state.cost_solution
        feasible_df = st.session_state.feasible_df
        tested_point = st.session_state.tested_point

        # ---- 1. Pareto Front ----
        st.markdown("### 📉 Pareto Front")
        if len(fronts) > 0 and len(fronts[0]) > 0:
            num_solutions = len(fronts[0])
            st.success(f"✅ Pareto front: {num_solutions} optimal solutions")
            fig = plot_pareto_clean(objectives, fronts, None, feasible_df, tested_point, efrf_max=0.40)
            if fig is not None:
                if balanced_solution is not None:
                    d, t, e, ef = predict_pinn(model, scaler, y_scaler, balanced_solution)
                    fig.add_trace(go.Scatter(
                        x=[balanced_solution[0]],
                        y=[ef],
                        mode='markers',
                        name='⭐ Golden (Balanced)',
                        marker=dict(size=12, color='gold', symbol='star', line=dict(width=2, color='black')),
                        hovertemplate='Golden (Balanced)<br>API: %{x:.1f}%<br>EFRF: %{y:.4f}<extra></extra>'
                    ))
                st.plotly_chart(fig, use_container_width=True)

        # ---- 2. Golden Solution ----
        st.markdown("### ⭐ Golden Solution (Balanced)")
        if balanced_solution is not None:
            d, t, e, ef = predict_pinn(model, scaler, y_scaler, balanced_solution)
            with st.container(border=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Formulation**")
                    st.write(f"API: {balanced_solution[0]:.1f}%")
                    st.write(f"MCC: {balanced_solution[1]:.1f}%")
                    st.write(f"PVPP: {balanced_solution[2]:.1f}%")
                    st.write(f"Mg-St: {balanced_solution[3]:.2f}%")
                    st.write(f"Binder: {balanced_solution[4]:.1f}%")
                with col2:
                    st.write("**Process & CQAs**")
                    st.write(f"Pressure: {balanced_solution[5]:.1f} MPa")
                    st.write(f"Speed: {balanced_solution[6]:.1f} rpm")
                    st.write(f"Granule: {balanced_solution[7]:.0f} µm")
                    st.write(f"Density: {d:.3f}")
                    st.write(f"Tensile: {t:.3f} MPa")
                    st.write(f"EFRF: {ef:.4f}")
            st.session_state.balanced_pred = (d, t, e, ef)
        else:
            st.info("No balanced solution found.")

        st.markdown("---")

        # ---- Toggles (with consistent keys) ----
        st.toggle(
            "💰 Cost-wise Solution (Max API, Min Pressure)",
            value=st.session_state.get("show_cost_solution", False),
            key="show_cost_solution"
        )
        if st.session_state.show_cost_solution and cost_solution is not None:
            st.markdown("#### 💰 Cost-Optimised Formulation (Max API, Min Pressure)")
            d, t, e, ef = predict_pinn(model, scaler, y_scaler, cost_solution)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Formulation:**")
                st.write(f"API: {cost_solution[0]:.1f}%")
                st.write(f"MCC: {cost_solution[1]:.1f}%")
                st.write(f"PVPP: {cost_solution[2]:.1f}%")
                st.write(f"Mg-St: {cost_solution[3]:.2f}%")
                st.write(f"Binder: {cost_solution[4]:.1f}%")
            with col2:
                st.write("**Process & CQAs:**")
                st.write(f"Pressure: {cost_solution[5]:.1f} MPa")
                st.write(f"Speed: {cost_solution[6]:.1f} rpm")
                st.write(f"Granule: {cost_solution[7]:.0f} µm")
                st.write(f"Density: {d:.3f}")
                st.write(f"Tensile: {t:.3f} MPa")
                st.write(f"EFRF: {ef:.4f}")

        st.toggle(
            "🏆 Quality-wise Solution (Max Tensile Strength)",
            value=st.session_state.get("show_quality_solution", False),
            key="show_quality_solution"
        )
        if st.session_state.show_quality_solution and quality_solution is not None:
            st.markdown("#### 🏆 Quality-Optimised Formulation (Max Tensile Strength)")
            d, t, e, ef = predict_pinn(model, scaler, y_scaler, quality_solution)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Formulation:**")
                st.write(f"API: {quality_solution[0]:.1f}%")
                st.write(f"MCC: {quality_solution[1]:.1f}%")
                st.write(f"PVPP: {quality_solution[2]:.1f}%")
                st.write(f"Mg-St: {quality_solution[3]:.2f}%")
                st.write(f"Binder: {quality_solution[4]:.1f}%")
            with col2:
                st.write("**Process & CQAs:**")
                st.write(f"Pressure: {quality_solution[5]:.1f} MPa")
                st.write(f"Speed: {quality_solution[6]:.1f} rpm")
                st.write(f"Granule: {quality_solution[7]:.0f} µm")
                st.write(f"Density: {d:.3f}")
                st.write(f"Tensile: {t:.3f} MPa")
                st.write(f"EFRF: {ef:.4f}")

        st.toggle(
            "📊 Model Comparison",
            value=st.session_state.get("show_comparison", True),
            key="show_comparison"
        )
        if st.session_state.show_comparison:
            st.markdown("### 📊 Model Comparison (Tensile R²)")
            bench_df, chart_data = run_model_comparison(model, scaler, y_scaler, features, df, device)
            st.session_state.benchmark_df = bench_df
            fig_bar = px.bar(pd.DataFrame(chart_data), x='Model', y='R² Score', color='Model',
                             title='Real R² Comparison (Tensile Strength)',
                             text=pd.DataFrame(chart_data)['R² Score'].round(3))
            fig_bar.update_layout(height=380, template='plotly_white')
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(bench_df, use_container_width=True)

        st.toggle(
            "🔬 Sensitivity Analysis",
            value=st.session_state.get("show_sensitivity", False),
            key="show_sensitivity"
        )
        if st.session_state.show_sensitivity:
            f = st.session_state.formulation
            if f['api_n'] is not None:
                st.markdown("### 🔬 Sensitivity Analysis – Parameter Impact on EFRF")
                fig_bars = plot_sensitivity_bars(f, model, scaler, y_scaler, efrf_max=0.40)
                if fig_bars:
                    st.plotly_chart(fig_bars, use_container_width=True)

        st.toggle(
            "📊 Particle Size Effect",
            value=st.session_state.get("show_particle_plot", False),
            key="show_particle_plot"
        )
        if st.session_state.show_particle_plot:
            f = st.session_state.formulation
            if f['api_n'] is not None:
                st.markdown("### 📊 Particle Size Effect with Pressure Variation")
                fig = plot_particle_pressure_density(f, model, scaler, y_scaler)
                st.plotly_chart(fig, use_container_width=True)

        # ---- Report button ----
        generate_report_btn = st.button("📄 Generate Report (PDF)", key="knob_report")
        if generate_report_btn and st.session_state.benchmark_df is not None:
            f = st.session_state.formulation
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            bench_df = st.session_state.benchmark_df
            balanced_sol = st.session_state.balanced_solution
            quality_sol = st.session_state.quality_solution
            cost_sol = st.session_state.cost_solution
            balanced_pred = st.session_state.get('balanced_pred', None)
            quality_pred = st.session_state.get('quality_pred', None)
            cost_pred = st.session_state.get('cost_pred', None)
            filepath, error = generate_pdf_report(
                f, bench_df, balanced_sol, quality_sol, cost_sol,
                balanced_pred, quality_pred, cost_pred, fronts, timestamp
            )
            if error:
                st.error(f"Failed to generate report: {error}")
                if not FPDF_AVAILABLE:
                    st.info("Please install fpdf2: `pip install fpdf2`")
            else:
                with open(filepath, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download Report (PDF)",
                        data=pdf_file,
                        file_name=f"hubryd_report_all_{timestamp[:10]}.pdf",
                        mime="application/pdf"
                    )
                try:
                    os.unlink(filepath)
                except Exception:
                    pass

    else:
        st.info("Adjust sliders and click '🔬 Predict & Optimise' to see results.")

st.caption("📧 Contact: babuker@protonmail.com | 🏛️ Nile Valley University, Sudan")
