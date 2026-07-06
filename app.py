"""
Hubryd AI Multi-objective Optimization Framework v29.27
Minimal Working Version – for Streamlit Cloud Free Tier
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
import plotly.graph_objects as go
import plotly.express as px
import math
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# PARAMETERS (reduced for free tier)
# ================================================================

TENSILE_MIN = 1.90
EFRF_MAX = 0.40
MCC_MAX = 8.0
D_MIN = 0.40
D_MAX = 0.97
PRESSURE_MAX = 300.0
BINDER_MIN = 0.5
BINDER_MAX = 5.0

NOISE_DENSITY = 0.002
NOISE_STRENGTH = 0.005
NOISE_ER = 0.005

N_SAMPLES = 5000              # reduced
ADAM_EPOCHS = 200             # reduced
MONOTONICITY_FREQUENCY = 10
PATIENCE = 50                 # reduced

W_DENSITY = 2.0
W_TENSILE = 15.0
W_TENSILE_PHYSICS = 0.5
W_PHYSICS_BASE = 0.0
W_PHYSICS_FINAL = 0.3
W_DENSITY_PENALTY = 8.0
W_MCC = 0.5

NSGA_POP_SIZE = 40            # reduced
NSGA_GENERATIONS = 30         # reduced

# ================================================================
# SESSION STATE
# ================================================================

if 'api' not in st.session_state:       st.session_state.api = 90.5
if 'binder' not in st.session_state:    st.session_state.binder = 2.7
if 'pvpp' not in st.session_state:      st.session_state.pvpp = 3.0
if 'mgst' not in st.session_state:      st.session_state.mgst = 0.20
if 'mcc' not in st.session_state:       st.session_state.mcc = 3.6
if 'pressure' not in st.session_state:  st.session_state.pressure = 230.0
if 'speed' not in st.session_state:     st.session_state.speed = 12.0
if 'granule' not in st.session_state:   st.session_state.granule = 125.0

def get_val(key):
    try:
        return float(st.session_state[key])
    except:
        return 90.5 if key == 'api' else 2.7 if key == 'binder' else 3.0 if key == 'pvpp' else 0.2 if key == 'mgst' else 3.6 if key == 'mcc' else 230.0 if key == 'pressure' else 12.0 if key == 'speed' else 125.0

# ================================================================
# HELPER FUNCTIONS
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
    api_mgst = api * mgst
    pressure_speed2 = pressure / (speed + 0.1) ** 2
    api2 = api ** 2
    pressure2 = pressure ** 2
    binder2 = binder ** 2
    speed2 = speed ** 2

    return np.concatenate([
        X_raw,
        pressure_binder, pressure_api,
        pressure_speed, api_mcc, binder_speed,
        api_pvpp, binder_mgst, mcc_pvpp,
        api_mgst, pressure_speed2,
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
        noise_d = np.random.normal(0, NOISE_DENSITY)
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
        strength = strength * np.random.normal(1.0, NOISE_STRENGTH)
        strength = np.clip(strength, 0.5, 6.0)

        er_base = 1.8 + 0.3 * (api - 85)/10 + 0.08 * (speed - 10)/30 - 0.1 * (pressure - 100)/150
        er_base = er_base * (1.0 - 0.15 * (D - 0.4))
        er = np.clip(er_base + np.random.normal(0, NOISE_ER), 0.5, 4.0)

        y[i] = [D, strength, er]

    feature_names = ['API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
                     'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm']
    df = pd.DataFrame(X, columns=feature_names)
    df['Density'] = y[:, 0]
    df['Tensile_Strength_MPa'] = y[:, 1]
    df['Elastic_Recovery_%'] = y[:, 2]
    return df, feature_names

# ================================================================
# PINN MODEL (reduced size: 256 neurons)
# ================================================================

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class ResidualBlock(nn.Module):
    def __init__(self, features, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(features, features)
        self.bn1 = nn.BatchNorm1d(features)
        self.linear2 = nn.Linear(features, features)
        self.bn2 = nn.BatchNorm1d(features)
        self.activation = Mish()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        out = self.activation(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        out = self.dropout(out)
        return identity + out

class MultiTaskTruePINN(nn.Module):
    def __init__(self, input_dim, output_dim=5):
        super(MultiTaskTruePINN, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),   # reduced
            Mish()
        )
        self.res_block1 = ResidualBlock(256)
        self.res_block2 = ResidualBlock(256)
        self.res_block3 = ResidualBlock(256)  # 3 blocks instead of 4
        self.transition = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh()
        )
        self.output_layer = nn.Linear(128, output_dim)

    def forward(self, X):
        x = self.input_layer(X)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.transition(x)
        raw = self.output_layer(x)

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

    def compute_loss(self, X_scaled, X_raw, y_true, epoch=0, max_epochs=ADAM_EPOCHS,
                     w_density=W_DENSITY, w_tensile=W_TENSILE,
                     w_tensile_physics=W_TENSILE_PHYSICS,
                     w_physics_base=W_PHYSICS_BASE, w_physics_final=W_PHYSICS_FINAL,
                     w_mcc=W_MCC, w_density_penalty=W_DENSITY_PENALTY,
                     efrf_target=EFRF_MAX, mcc_max=MCC_MAX,
                     compute_grad=True):
        pressure_real = X_raw[:, 5].view(-1, 1)
        mcc_real = X_raw[:, 1].view(-1, 1)

        y_pred = self.forward(X_scaled)
        density_pred = y_pred[:, 0:1]
        tensile_pred = y_pred[:, 1:2]
        er_pred = y_pred[:, 2:3]
        k_pred = y_pred[:, 3:4]
        A_pred = y_pred[:, 4:5]

        if epoch < 100:
            w_physics = 0.0
        else:
            w_physics = min(w_physics_final, (epoch - 100) / 200 * w_physics_final)

        density_mse = nn.MSELoss()(density_pred, y_true[:, 0:1])
        tensile_mse = nn.MSELoss()(tensile_pred, y_true[:, 1:2])
        er_mse = nn.MSELoss()(er_pred, y_true[:, 2:3])

        data_loss = (3.5 * density_mse) + (w_tensile * tensile_mse) + (3.5 * er_mse)

        if data_loss.item() > 1.0 and w_physics > 0.01:
            w_physics = w_physics * 0.5

        tensile_physics_loss = torch.mean(torch.relu(0.3 - (tensile_pred * density_pred)) ** 2) * w_tensile_physics
        heckel_lhs = torch.log(1.0 / torch.clamp(1.0 - density_pred, min=1e-4))
        heckel_rhs = k_pred * pressure_real + A_pred
        heckel_loss = torch.mean((heckel_lhs - heckel_rhs) ** 2)
        efrf_pred = er_pred / torch.clamp(tensile_pred, min=1e-4)
        efrf_loss = torch.mean(torch.relu(efrf_pred - efrf_target) ** 2)

        mcc_loss = torch.mean(torch.relu(mcc_real - mcc_max) ** 2)
        density_penalty = torch.mean(
            torch.relu(density_pred - D_MAX) ** 2 + torch.relu(D_MIN - density_pred) ** 2
        )

        total_loss = (
            data_loss +
            (0.7 * w_density_penalty) * density_penalty +
            w_physics * (heckel_loss + efrf_loss + tensile_physics_loss) +
            w_mcc * mcc_loss
        )
        return total_loss, {'total_loss': total_loss.item()}

# ================================================================
# NSGA-II (minimal, no granule mode)
# ================================================================

class NSGAII:
    def __init__(self, model, scaler, y_scaler, bounds,
                 pop_size=NSGA_POP_SIZE, n_generations=NSGA_GENERATIONS, w_tensile=0.0):
        self.model = model
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.bounds = bounds
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.w_tensile = w_tensile
        self.population = None
        self.objectives = None
        self.constraints = None
        self.fronts = None

    def _repair(self, individual):
        api, mcc, pvpp, mgst, binder, pressure, speed, granule = individual
        api, binder, pvpp, mgst, mcc = normalize_components(api, binder, pvpp, mgst, mcc)
        pressure = np.clip(pressure, 80, PRESSURE_MAX)
        speed = np.clip(speed, 1.0, 50.0)
        granule = np.clip(granule, 30.0, 250.0)
        return np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule], dtype=float)

    def _evaluate(self, population):
        n = population.shape[0]
        objectives = np.zeros((n, 2))
        constraints = np.zeros((n, 2))
        constraint_violation = np.zeros(n)
        device = next(self.model.parameters()).device
        for i in range(n):
            try:
                repaired = self._repair(population[i])
                api, mcc, pvpp, mgst, binder, pressure, speed, granule = repaired
                inputs = np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule]).reshape(1, -1)
                inputs_with_features = add_interaction_features(inputs)[0]
                inputs_scaled = self.scaler.transform([inputs_with_features])
                X_tensor = torch.tensor(inputs_scaled, dtype=torch.float32).to(device)
                with torch.no_grad():
                    pred_scaled = self.model.predict(X_tensor)
                    pred_actual = self.y_scaler.inverse_transform(pred_scaled)[0]
                density = float(np.clip(pred_actual[0], D_MIN, D_MAX))
                tensile = float(max(pred_actual[1], 1e-4))
                er = float(max(pred_actual[2], 1e-4))
                efrf = float(er / tensile)
                efrf = max(1e-4, min(efrf, 5.0))
                g1 = 0.90 - density
                g2 = density - 0.97
                constraints[i, 0] = g1
                constraints[i, 1] = g2
                constraint_violation[i] = max(0, g1, g2)
                penalty = 0.0
                if tensile < TENSILE_MIN: penalty += (TENSILE_MIN - tensile) ** 2
                if efrf >= EFRF_MAX: penalty += (efrf - EFRF_MAX) ** 2
                if density < D_MIN: penalty += (D_MIN - density) ** 2
                if density > D_MAX: penalty += (density - D_MAX) ** 2
                objectives[i, 0] = -(api + self.w_tensile * tensile) + 30.0 * penalty
                objectives[i, 1] = efrf + 30.0 * penalty
                population[i] = repaired
            except Exception:
                objectives[i, 0] = 100.0
                objectives[i, 1] = 100.0
                constraints[i, 0] = 10.0
                constraints[i, 1] = 10.0
                constraint_violation[i] = 10.0
        return objectives, constraints, constraint_violation, population

    def _fast_non_dominated_sort(self, objectives, constraints, constraint_violation):
        n = objectives.shape[0]
        fronts = []
        rank = np.zeros(n, dtype=int)
        feasible_mask = constraint_violation <= 1e-6
        feasible_indices = np.where(feasible_mask)[0]
        if len(feasible_indices) > 0:
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n)
            current_front = []
            for i in feasible_indices:
                for j in feasible_indices:
                    if i == j:
                        continue
                    if (objectives[i, 0] <= objectives[j, 0] and objectives[i, 1] <= objectives[j, 1]) and \
                       (objectives[i, 0] < objectives[j, 0] or objectives[i, 1] < objectives[j, 1]):
                        S[i].append(j)
                    elif (objectives[j, 0] <= objectives[i, 0] and objectives[j, 1] <= objectives[i, 1]) and \
                         (objectives[j, 0] < objectives[i, 0] or objectives[j, 1] < objectives[i, 1]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            if current_front:
                fronts.append(current_front)
            i = 0
            while i < len(fronts) and fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                if next_front:
                    fronts.append(next_front)
                i += 1
        if len(feasible_indices) < n:
            infeasible = np.where(~feasible_mask)[0]
            sorted_infeasible = sorted(infeasible, key=lambda idx: constraint_violation[idx])
            fronts.append(sorted_infeasible)
        return fronts, rank

    def _crowding_distance(self, objectives, front):
        n = len(front)
        if n <= 2:
            return np.ones(n) * np.inf
        distance = np.zeros(n)
        obj_range = objectives[front].max(axis=0) - objectives[front].min(axis=0)
        obj_range[obj_range == 0] = 1.0
        for m in range(2):
            sorted_idx = sorted(range(n), key=lambda i: objectives[front[i], m])
            distance[sorted_idx[0]] = np.inf
            distance[sorted_idx[-1]] = np.inf
            for i in range(1, n - 1):
                prev_obj = objectives[front[sorted_idx[i - 1]], m]
                next_obj = objectives[front[sorted_idx[i + 1]], m]
                distance[sorted_idx[i]] += (next_obj - prev_obj) / obj_range[m]
        return distance

    def _tournament_selection(self, pop_indices, objectives, ranks, crowding, constraint_violation):
        selected = []
        for _ in range(len(pop_indices)):
            i1, i2 = np.random.choice(pop_indices, 2, replace=False)
            v1 = constraint_violation[i1]
            v2 = constraint_violation[i2]
            if v1 <= 0 and v2 > 0:
                selected.append(i1)
            elif v2 <= 0 and v1 > 0:
                selected.append(i2)
            else:
                if ranks[i1] < ranks[i2]:
                    selected.append(i1)
                elif ranks[i1] > ranks[i2]:
                    selected.append(i2)
                else:
                    selected.append(i1 if crowding[i1] >= crowding[i2] else i2)
        return selected

    def _simulated_binary_crossover(self, p1, p2):
        if np.random.random() > 0.90:
            return p1.copy(), p2.copy()
        c1 = np.zeros(8)
        c2 = np.zeros(8)
        for i in range(8):
            if np.random.random() < 0.5:
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / 41)
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / 41)
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]
        return self._repair(c1), self._repair(c2)

    def _polynomial_mutation(self, ind):
        mutated = ind.copy()
        for i in range(8):
            if np.random.random() < 0.125:
                u = np.random.random()
                delta = min(u, 1 - u) ** (1 / 21)
                if u < 0.5:
                    mutated[i] = ind[i] + delta * (self.bounds[i, 1] - self.bounds[i, 0])
                else:
                    mutated[i] = ind[i] - delta * (self.bounds[i, 1] - self.bounds[i, 0])
        return self._repair(mutated)

    def run(self):
        pop = np.zeros((self.pop_size, 8))
        for i in range(self.pop_size):
            ind = np.array([np.random.uniform(60,100), np.random.uniform(0.1,20),
                            np.random.uniform(0.1,12), np.random.uniform(0.01,3.0),
                            np.random.uniform(0.1,10), np.random.uniform(80,PRESSURE_MAX),
                            np.random.uniform(1,50), np.random.uniform(30,250)])
            pop[i] = self._repair(ind)
        self.population = pop

        for gen in range(self.n_generations):
            objectives, constraints, violation, pop = self._evaluate(self.population)
            self.population = pop
            self.objectives = objectives
            self.constraints = constraints
            fronts, ranks = self._fast_non_dominated_sort(objectives, constraints, violation)
            self.fronts = fronts
            if gen == self.n_generations - 1:
                break

            crowding = np.zeros(self.pop_size)
            for front in fronts:
                dist = self._crowding_distance(objectives, front)
                for idx, d in zip(front, dist):
                    crowding[idx] = d

            selected = self._tournament_selection(range(self.pop_size), objectives, ranks, crowding, violation)
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    c1, c2 = self._simulated_binary_crossover(self.population[selected[i]], self.population[selected[i+1]])
                    offspring.append(self._polynomial_mutation(c1))
                    offspring.append(self._polynomial_mutation(c2))
                else:
                    offspring.append(self._polynomial_mutation(self.population[selected[i]]))
            offspring = np.array(offspring[:self.pop_size])
            obj_off, cons_off, vio_off, off = self._evaluate(offspring)

            combined_pop = np.vstack([self.population, off])
            combined_obj = np.vstack([self.objectives, obj_off])
            combined_cons = np.vstack([self.constraints, cons_off])
            combined_vio = np.concatenate([violation, vio_off])

            combined_fronts, _ = self._fast_non_dominated_sort(combined_obj, combined_cons, combined_vio)
            combined_crowding = np.zeros(len(combined_pop))
            for front in combined_fronts:
                dist = self._crowding_distance(combined_obj, front)
                for idx, d in zip(front, dist):
                    combined_crowding[idx] = d

            new_pop, new_obj, new_cons, new_vio = [], [], [], []
            for front in combined_fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    for idx in front:
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])
                        new_cons.append(combined_cons[idx])
                        new_vio.append(combined_vio[idx])
                else:
                    front_sorted = sorted(front, key=lambda i: combined_crowding[i], reverse=True)
                    remain = self.pop_size - len(new_pop)
                    for idx in front_sorted[:remain]:
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])
                        new_cons.append(combined_cons[idx])
                        new_vio.append(combined_vio[idx])
                    break
            self.population = np.array(new_pop)
            self.objectives = np.array(new_obj)
            self.constraints = np.array(new_cons)

        objectives, constraints, violation, pop = self._evaluate(self.population)
        self.population = pop
        self.objectives = objectives
        self.constraints = constraints
        self.fronts, _ = self._fast_non_dominated_sort(objectives, constraints, violation)
        return self.population, self.objectives, self.constraints, self.fronts

# ================================================================
# PREDICTION & PLOTTING
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

def plot_pareto_clean(objectives, fronts):
    if objectives is None or fronts is None or len(fronts) == 0 or len(fronts[0]) == 0:
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
    fig.update_layout(height=500, template='plotly_white')
    return fig

def train_and_compare(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    models = {
        'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            'Model': name,
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'Physics': 'Not enforced'
        })
    return pd.DataFrame(results)

# ================================================================
# MODEL LOADING / TRAINING (with error catching inside)
# ================================================================

@st.cache_resource
def load_or_train_model():
    # This function will catch its own errors and display them
    try:
        checkpoint_path = '/tmp/pinn_best_model.pt'
        try:
            if os.path.exists(checkpoint_path):
                st.caption("📂 Loading cached model...")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                if all(k in checkpoint for k in ['model_state', 'scaler', 'y_scaler', 'feature_names', 'df', 'loss_history']):
                    input_dim = checkpoint['scaler'].mean_.shape[0]
                    model = MultiTaskTruePINN(input_dim=input_dim)
                    model.load_state_dict(checkpoint['model_state'])
                    scaler = checkpoint['scaler']
                    y_scaler = checkpoint['y_scaler']
                    feature_names = checkpoint['feature_names']
                    df = checkpoint['df']
                    loss_history = checkpoint['loss_history']
                    return model, scaler, y_scaler, feature_names, df, loss_history
                else:
                    st.warning("⚠️ Cached file missing keys. Re-training...")
                    os.remove(checkpoint_path)
        except Exception as e:
            st.warning(f"⚠️ Cache error: {str(e)[:80]}. Re-training...")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

        st.caption("🔄 Training model from scratch...")
        df, feature_names = generate_pinn_data(n_samples=N_SAMPLES)
        X_raw = df[feature_names].values
        y = df[['Density', 'Tensile_Strength_MPa', 'Elastic_Recovery_%']].values
        X_augmented = add_interaction_features(X_raw)
        input_dim = X_augmented.shape[1]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_augmented)
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y)

        X_train, X_temp, X_raw_train, X_raw_temp, y_train, y_temp = train_test_split(
            X_scaled, X_raw, y_scaled, test_size=0.3, random_state=42
        )
        X_val, X_test, X_raw_val, X_raw_test, y_val, y_test = train_test_split(
            X_temp, X_raw_temp, y_temp, test_size=0.5, random_state=42
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.caption(f"🖥️ Using device: {device}")

        model = MultiTaskTruePINN(input_dim=input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_raw_train_t = torch.tensor(X_raw_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        X_raw_val_t = torch.tensor(X_raw_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

        best_val_loss = float("inf")
        patience_counter = 0
        patience = PATIENCE

        progress_bar = st.progress(0)
        train_losses = []
        val_losses = []

        for epoch in range(ADAM_EPOCHS):
            model.train()
            optimizer.zero_grad()
            compute_grad = (epoch % MONOTONICITY_FREQUENCY == 0)

            loss, _ = model.compute_loss(
                X_train_t, X_raw_train_t, y_train_t,
                epoch=epoch, max_epochs=ADAM_EPOCHS,
                compute_grad=compute_grad
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss, _ = model.compute_loss(
                    X_val_t, X_raw_val_t, y_val_t,
                    epoch=epoch, max_epochs=ADAM_EPOCHS,
                    compute_grad=False
                )

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            scheduler.step(val_loss.item())

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                torch.save(model.state_dict(), "/tmp/best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    st.warning(f"⏹️ Training stopped early at epoch {epoch+1}")
                    break

            progress_bar.progress((epoch + 1) / ADAM_EPOCHS)

        if os.path.exists("/tmp/best_model.pt"):
            model.load_state_dict(torch.load("/tmp/best_model.pt", map_location=device))
            st.caption(f"✅ Best validation loss: {best_val_loss:.4f}")

        model.cpu()

        checkpoint_data = {
            'model_state': model.state_dict(),
            'scaler': scaler,
            'y_scaler': y_scaler,
            'feature_names': feature_names,
            'df': df,
            'loss_history': {'train': train_losses, 'val': val_losses}
        }
        torch.save(checkpoint_data, checkpoint_path)
        st.success("✅ Model trained and cached successfully!")
        return model, scaler, y_scaler, feature_names, df, {'train': train_losses, 'val': val_losses}

    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        st.error(f"**Full traceback:**")
        st.code(traceback.format_exc(), language="python")
        st.stop()

# ================================================================
# MAIN UI
# ================================================================

st.set_page_config(page_title="Hubryd AI v29.27", page_icon="🧬", layout="wide")

st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 1.2rem; border-radius: 1rem; margin-bottom: 1rem; text-align: center;">
    <h2 style="color: #ffffff; margin: 0;">🧬 Hubryd AI Multi‑objective Optimization Framework</h2>
    <p style="color: #64ffda; font-size: 1rem; margin: 0.2rem 0 0 0;">v29.27 — Minimal · Stable · Free Tier</p>
    <p style="color: #a8b2d1; font-size: 0.85rem; margin: 0.2rem 0 0 0;">Nile Valley University · Sudan</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

with st.sidebar:
    st.markdown("### 📚 Physics Constraints")
    st.markdown(f"""
    - ✅ Heckel: ln(1/(1-D)) = kP + A
    - ✅ EFRF: ER / σt < {EFRF_MAX:.2f}
    - ✅ Density: {D_MIN:.2f} ≤ D ≤ {D_MAX:.2f}
    - ✅ MCC: ≤ {MCC_MAX:.1f}%
    - ✅ Samples: {N_SAMPLES}
    - ✅ Epochs: {ADAM_EPOCHS}
    - ✅ NSGA‑II: Pop={NSGA_POP_SIZE}, Gen={NSGA_GENERATIONS}
    - ✅ Network: 256 Neurons
    """)
    st.info("🔬 **v29.27** — Minimal & Stable")

# Load model
try:
    with st.spinner("📂 Loading/Training model..."):
        model, scaler, y_scaler, feature_names, df, loss_history = load_or_train_model()
    st.success("✅ Model ready!")
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.stop()

st.markdown("---")

# Quick experiments
st.markdown("### 🧪 Quick Experiments")
exp_cols = st.columns(4)
experiments = {
    "Baseline": {'api': 90.5, 'binder': 2.7, 'pvpp': 3.0, 'mgst': 0.20, 'mcc': 3.6, 'pressure': 230, 'speed': 12, 'granule': 125},
    "High Binder": {'api': 90.5, 'binder': 3.0, 'pvpp': 3.0, 'mgst': 0.15, 'mcc': 3.35, 'pressure': 235, 'speed': 10, 'granule': 125},
    "High Pressure": {'api': 90.5, 'binder': 2.8, 'pvpp': 3.0, 'mgst': 0.12, 'mcc': 3.58, 'pressure': 250, 'speed': 9, 'granule': 125},
    "Low Mg-St": {'api': 90.5, 'binder': 3.0, 'pvpp': 3.0, 'mgst': 0.10, 'mcc': 3.4, 'pressure': 245, 'speed': 8, 'granule': 125}
}
for i, (name, params) in enumerate(experiments.items()):
    with exp_cols[i]:
        if st.button(f"📌 {name}", key=f"exp_{i}", use_container_width=True):
            for key in params:
                st.session_state[key] = params[key]
            st.rerun()

st.markdown("---")

col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    st.markdown("### 📊 Formulation Parameters")
    with st.container(border=True):
        api = st.slider("🧪 API (%)", 85.0, 95.0, get_val('api'), 0.1, key="api")
        binder = st.slider("🔗 Binder (%)", BINDER_MIN, BINDER_MAX, get_val('binder'), 0.1, key="binder")
        pvpp = st.slider("💊 PVPP (%)", 0.5, 6.0, get_val('pvpp'), 0.1, key="pvpp")
        mgst = st.slider("🧴 Mg-St (%)", 0.01, 1.2, get_val('mgst'), 0.01, key="mgst")
        mcc = st.slider("📦 MCC (%)", 0.0, MCC_MAX, get_val('mcc'), 0.1, key="mcc")
        total = api + binder + pvpp + mgst + mcc
        if abs(total - 100) < 0.1:
            st.success(f"✅ Total = {total:.2f}%")
        else:
            st.warning(f"⚠️ Total = {total:.2f}% (adjust to 100%)")
    st.markdown("### ⚙️ Process Parameters")
    with st.container(border=True):
        pressure = st.slider("⚙️ Pressure (MPa)", 80.0, PRESSURE_MAX, get_val('pressure'), 1.0, key="pressure")
        speed = st.slider("🔄 Speed (rpm)", 1.0, 50.0, get_val('speed'), 0.5, key="speed")
        granule = st.slider("🔬 Granule Size (µm)", 30.0, 250.0, get_val('granule'), 1.0, key="granule")
    predict_btn = st.button("🔬 Predict & Optimize", use_container_width=True)

with col_right:
    st.markdown("### 📈 Results")

    objectives = None; constraints = None; fronts = None; nsga = None
    api_norm = None; efrf = None; comp_df = pd.DataFrame()
    density = 0.0; tensile = 0.0; er = 0.0
    density_ok = False; tensile_ok = False; efrf_ok = False; mcc_ok = False
    api_use = 0.0; mcc_use = 0.0; pvpp_use = 0.0; mgst_use = 0.0; binder_use = 0.0
    golden_info = None

    if predict_btn:
        if abs(total - 100) > 0.1:
            st.warning("⚠️ Formulation must sum to 100%")
        else:
            api_norm, binder_norm, pvpp_norm, mgst_norm, mcc_norm = normalize_components(api, binder, pvpp, mgst, mcc)
            inputs_norm = [api_norm, mcc_norm, pvpp_norm, mgst_norm, binder_norm, pressure, speed, granule]
            api_use, mcc_use, pvpp_use, mgst_use, binder_use = api_norm, mcc_norm, pvpp_norm, mgst_norm, binder_norm
            with st.spinner("🧠 Predicting..."):
                density, tensile, er, efrf = predict_pinn(model, scaler, y_scaler, inputs_norm)

            st.markdown("#### Constraints Status")
            cols = st.columns(4)
            density_ok = (D_MIN <= density <= D_MAX)
            tensile_ok = (tensile >= TENSILE_MIN)
            efrf_ok = (efrf < EFRF_MAX)
            mcc_ok = (mcc_norm <= MCC_MAX)

            cols[0].metric("Density", f"{density:.3f}", delta="✅" if density_ok else "❌")
            cols[1].metric("Tensile", f"{tensile:.2f} MPa", delta="✅" if tensile_ok else "❌")
            cols[2].metric("EFRF", f"{efrf:.4f}", delta="✅" if efrf_ok else "❌")
            cols[3].metric("MCC", f"{mcc_norm:.1f}%", delta="✅" if mcc_ok else "❌")

            if density_ok and tensile_ok and efrf_ok and mcc_ok:
                st.success("✅ All constraints satisfied!")
            else:
                st.error("❌ One or more constraints violated.")

            # NSGA‑II
            st.markdown("#### ⚙️ Optimization (NSGA‑II)")
            bounds = np.array([[60,100],[0.1,20],[0.1,12],[0.01,3.0],[0.1,10],[80,PRESSURE_MAX],[1,50],[30,250]])
            with st.spinner(f"🔄 NSGA‑II (pop={NSGA_POP_SIZE}, gen={NSGA_GENERATIONS})..."):
                nsga = NSGAII(model, scaler, y_scaler, bounds)
                pop, objectives, constraints, fronts = nsga.run()

                if len(fronts) > 0 and len(fronts[0]) > 0:
                    pareto_count = len(fronts[0])
                    st.success(f"📊 Pareto front found: **{pareto_count}** optimal solutions")
                else:
                    st.warning("No feasible Pareto solutions found.")

                if pareto_count > 0:
                    front0 = fronts[0]
                    golden_candidates = []
                    for idx in front0:
                        formulation = nsga.population[idx]
                        d, t, e, ef = predict_pinn(model, scaler, y_scaler, formulation)
                        if D_MIN <= d <= D_MAX and t >= TENSILE_MIN and ef < EFRF_MAX:
                            golden_candidates.append({
                                'api': formulation[0],
                                'mcc': formulation[1],
                                'pvpp': formulation[2],
                                'mgst': formulation[3],
                                'binder': formulation[4],
                                'pressure': formulation[5],
                                'speed': formulation[6],
                                'granule': formulation[7],
                                'density': d,
                                'tensile': t,
                                'er': e,
                                'efrf': ef
                            })
                    if golden_candidates:
                        best = min(golden_candidates, key=lambda x: (x['efrf'], -x['tensile']))
                        golden_info = best
                        st.markdown("---")
                        st.markdown("### ⭐ Golden Solution (Suggested)")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            **Formulation:**
                            - API: `{golden_info['api']:.1f}%`
                            - MCC: `{golden_info['mcc']:.1f}%`
                            - PVPP: `{golden_info['pvpp']:.1f}%`
                            - Mg‑St: `{golden_info['mgst']:.2f}%`
                            - Binder: `{golden_info['binder']:.1f}%`
                            """)
                        with col2:
                            st.markdown(f"""
                            **Process:**
                            - Pressure: `{golden_info['pressure']:.1f} MPa`
                            - Speed: `{golden_info['speed']:.1f} rpm`
                            - Granule: `{golden_info['granule']:.0f} µm`

                            **Predicted:**
                            - Density: `{golden_info['density']:.3f}`
                            - Tensile: `{golden_info['tensile']:.3f} MPa`
                            - EFRF: `{golden_info['efrf']:.4f}`
                            """)

            # Model Comparison
            X_train, X_test, y_train, y_test = train_test_split(
                df[feature_names].values, df['Tensile_Strength_MPa'].values,
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

            comp_df = train_and_compare(X_train_scaled, X_test_scaled, y_train, y_test)
            pinn_row = pd.DataFrame([{
                'Model': 'PINN (Proposed)',
                'R²': pinn_r2,
                'RMSE': pinn_rmse,
                'MAE': pinn_mae,
                'Physics': '✅ Enforced'
            }])
            comp_df = pd.concat([pinn_row, comp_df], ignore_index=True)
            comp_df_display = comp_df.copy()
            for col in ['R²', 'RMSE', 'MAE']:
                comp_df_display[col] = comp_df_display[col].map(lambda x: f"{x:.4f}")
            comp_df = comp_df_display

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📉 Pareto", "📊 Comparison", "📄 Report"])

    with tab1:
        if predict_btn and objectives is not None:
            fig = plot_pareto_clean(objectives, fronts)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.caption("🔴 Red markers = Pareto‑optimal solutions · Dashed red line = EFRF threshold")
            else:
                st.info("No Pareto front data available.")
        else:
            st.info("👆 Click 'Predict & Optimize' to run NSGA‑II.")

    with tab2:
        if predict_btn and not comp_df.empty:
            st.markdown("### Model Performance Comparison")
            df_plot = comp_df.copy()
            df_plot['R²'] = df_plot['R²'].astype(float)
            df_plot = df_plot.sort_values('R²', ascending=True)
            fig = go.Figure()
            colors = ['#2ecc71' if m == 'PINN (Proposed)' else '#3498db' for m in df_plot['Model']]
            fig.add_trace(go.Bar(
                y=df_plot['Model'],
                x=df_plot['R²'],
                orientation='h',
                marker_color=colors,
                text=df_plot['R²'].round(4),
                textposition='outside'
            ))
            fig.update_layout(
                title='R² Score Comparison (v29.27)',
                xaxis=dict(title='R² Score', range=[-0.2, 1.05]),
                yaxis=dict(title='Model'),
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                comp_df.style
                .apply(lambda x: ['background-color: #e6f7e6' if i == 0 else '' for i in range(len(x))], axis=0)
                .set_properties(**{'text-align': 'center'})
                .set_table_styles([{'selector': 'thead th', 'props': [('text-align', 'center')]}]),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("👆 Click 'Predict & Optimize' to see model comparison.")

    with tab3:
        if predict_btn and not comp_df.empty:
            status = "PASS" if (density_ok and tensile_ok and efrf_ok and mcc_ok) else "FAIL"
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.download_button(
                "📥 Download Results (CSV)",
                data=comp_df.to_csv(index=False),
                file_name=f"results_{timestamp[:10]}.csv",
                mime="text/csv"
            )
            st.info("PDF report is simplified to CSV download in this minimal version.")
        else:
            st.info("👆 Click 'Predict & Optimize' to generate results.")

st.markdown("---")
st.caption("🔬 **Hubryd AI v29.27** — Minimal & Stable | Nile Valley University")
