"""
Hubryd AI – v29.27-R19 (Flexible Slider Minimums for Testing)
Hybrid AI for Multi-Objective Optimization of Tablet Formulation
- Slider minimums: MCC=1.5%, PVPP=1.0%, MgSt=0.05%, Binder=1.5%
- Optimiser bounds (practical): MCC≥2%, Binder≥3%, PVPP≥1.5%, MgSt≥0.3%
- Allows manual testing up to ~96% API (after normalisation).
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
# SLIDER RANGES (Wide for manual testing – NEW MINIMUMS)
# ================================================================
D_MIN = 0.70
D_MAX = 0.99
TENSILE_MIN = 1.50
EFRF_MAX = 0.50

SLIDER_API_MIN = 80.0
SLIDER_API_MAX = 98.0
SLIDER_MCC_MIN = 1.5           # <-- UPDATED
SLIDER_MCC_MAX = 8.0
SLIDER_PVPP_MIN = 1.0          # <-- UPDATED
SLIDER_PVPP_MAX = 6.0
SLIDER_MGST_MIN = 0.05         # <-- UPDATED
SLIDER_MGST_MAX = 1.2
SLIDER_BINDER_MIN = 1.5        # <-- UPDATED
SLIDER_BINDER_MAX = 6.0

SLIDER_PRESSURE_MIN = 150.0
SLIDER_PRESSURE_MAX = 250.0
SLIDER_SPEED_MIN = 15.0
SLIDER_SPEED_MAX = 30.0
SLIDER_GRANULE_MIN = 30.0
SLIDER_GRANULE_MAX = 250.0

# ================================================================
# NSGA-II BOUNDS (Practical – unchanged)
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
# Session State
# ================================================================
if 'api' not in st.session_state:
    st.session_state.update({
        'api': 90.5,
        'binder': 3.0,
        'pvpp': 1.5,
        'mgst': 0.3,
        'mcc': 2.0,
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
        # Clip to practical bounds
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

# ================================================================
# Plotting functions (unchanged)
# ================================================================
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
# PDF Report, Cached Training, Model Comparison (unchanged)
# ================================================================
# ... (keeping the rest as per previous versions for brevity)
# I'll skip the long unchanged parts to save space, but they are exactly as before.
