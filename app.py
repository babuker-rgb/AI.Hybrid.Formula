"""
True Physics-Informed Neural Network (PINN) - Version v29.42
Multi-Objective Tablet Manufacturing Optimization

Author: Babuker A. Abdalla
Version: 29.42 (Simplified adaptive loss + PyMOO NSGA-II)
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
from fpdf import FPDF
import datetime
import warnings
import plotly.graph_objects as go
import plotly.express as px
import time
import math
import os
import pickle
import re
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

try:
    from scipy.interpolate import UnivariateSpline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

warnings.filterwarnings('ignore')

# ================================================================
# 0. ENHANCED PARAMETERS (v29.42)
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

N_SAMPLES = 12000
ADAM_EPOCHS = 500
MONOTONICITY_FREQUENCY = 10
PATIENCE = 50

W_DENSITY = 2.0
W_TENSILE = 10.0
W_TENSILE_PHYSICS = 0.6
W_PHYSICS_BASE = 0.2
W_PHYSICS_FINAL = 0.8
W_EFRF = 2.0
W_DENSITY_PENALTY = 8.0
W_MCC = 0.5

NSGA_POP_SIZE = 80
NSGA_GENERATIONS = 60

# ================================================================
# 1. SESSION STATE & HELPERS
# ================================================================

DEFAULTS = {
    'api': 90.5, 'binder': 2.7, 'pvpp': 3.0, 'mgst': 0.20,
    'mcc': 3.6,
    'pressure': 230.0, 'speed': 12.0, 'granule': 125.0
}

RANGES = {
    'api': (85.0, 95.0), 'binder': (BINDER_MIN, BINDER_MAX),
    'pvpp': (0.5, 6.0), 'mgst': (0.01, 1.2), 'mcc': (0.0, MCC_MAX),
    'pressure': (80.0, PRESSURE_MAX), 'speed': (1.0, 50.0), 'granule': (30.0, 250.0)
}

def safe_initialize():
    for key in DEFAULTS:
        if key not in st.session_state:
            st.session_state[key] = DEFAULTS[key]
        else:
            try:
                float(st.session_state[key])
            except:
                st.session_state[key] = DEFAULTS[key]

def clamp_session_state():
    for key in DEFAULTS:
        if key in st.session_state:
            try:
                val = float(st.session_state[key])
                min_val, max_val = RANGES[key]
                if val < min_val: st.session_state[key] = min_val
                elif val > max_val: st.session_state[key] = max_val
            except:
                st.session_state[key] = DEFAULTS[key]

def get_safe_value(key):
    if key not in st.session_state:
        st.session_state[key] = DEFAULTS[key]
    try:
        val = float(st.session_state[key])
        min_val, max_val = RANGES[key]
        return max(min_val, min(val, max_val))
    except:
        return DEFAULTS[key]

safe_initialize()
clamp_session_state()

# ================================================================
# 2. HELPER FUNCTIONS (with granule mode support)
# ================================================================

def sanitize_text(text):
    replacements = {
        '🌟': '[GOLDEN]',
        '✅': '[PASS]',
        '❌': '[FAIL]',
        '⚠️': '[WARNING]',
        'σ': 'sigma',
        'µ': 'um',
        '≥': '>=',
        '≤': '<=',
        '•': '-',
        '—': '-',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

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

def generate_pinn_data(n_samples=N_SAMPLES, random_state=42,
                       granule_mode='Variable', fixed_granule=125.0):
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
        if granule_mode == 'Variable':
            granule = np.random.uniform(30, 250)
        else:
            granule = fixed_granule

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
# 3. PINN MODEL (with simplified adaptive loss)
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
            nn.Linear(input_dim, 512),
            Mish()
        )
        self.res_block1 = ResidualBlock(512)
        self.res_block2 = ResidualBlock(512)
        self.res_block3 = ResidualBlock(512)
        self.transition = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh()
        )
        self.output_layer = nn.Linear(256, output_dim)

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

    # --- SIMPLIFIED ADAPTIVE LOSS (NEW) ---
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

        # Physics weight scheduling (unchanged)
        progress = epoch / max_epochs
        schedule_factor = 1 / (1 + math.exp(-12 * (progress - 0.5)))
        w_physics = w_physics_base + (w_physics_final - w_physics_base) * schedule_factor
        w_physics = max(w_physics, 0.1)

        # Individual losses
        density_loss = nn.MSELoss()(density_pred, y_true[:, 0:1])
        tensile_loss = nn.MSELoss()(tensile_pred, y_true[:, 1:2])
        er_loss = nn.MSELoss()(er_pred, y_true[:, 2:3])

        # --- Simplified adaptive weights ---
        losses_dict = {
            "density": density_loss.item(),
            "tensile": tensile_loss.item(),
            "er": er_loss.item()
        }
        max_target = max(losses_dict, key=losses_dict.get)
        w_den, w_ten, w_er = 1.0, 1.0, 1.0
        if max_target == "density":
            w_den = 3.0
        elif max_target == "tensile":
            w_ten = 3.0
        elif max_target == "er":
            w_er = 3.0

        # Data loss with adaptive weights (but we still have base weights for scaling)
        data_loss = (w_den * density_loss) + (w_ten * tensile_loss) + (w_er * er_loss)

        # Physics constraints (unchanged)
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
# 4. PyMOO NSGA-II (replaces custom class)
# ================================================================

class TabletProblem(Problem):
    """
    Multi-objective problem for tablet formulation.
    Objectives:
        1. Maximize (API + w_tensile * tensile) -> we minimize negative of that.
        2. Minimize EFRF.
    Constraints:
        g1: 0.40 - density <= 0  (density >= 0.40)
        g2: density - 0.97 <= 0   (density <= 0.97)
        g3: 1.90 - tensile <= 0   (tensile >= 1.90)
        g4: (er / tensile) - 0.40 <= 0  (EFRF <= 0.40)
    """
    def __init__(self, model, scaler, y_scaler, bounds,
                 granule_mode='Variable', fixed_granule=125.0,
                 w_tensile=0.0):
        self.model = model
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.granule_mode = granule_mode
        self.fixed_granule = fixed_granule
        self.w_tensile = w_tensile
        # bounds: 8 variables (API, MCC, PVPP, MgSt, Binder, Pressure, Speed, Granule)
        # We'll use the provided bounds array (shape 8x2)
        self.bounds = bounds
        super().__init__(n_var=8, n_obj=2, n_constr=4,
                         xl=bounds[:, 0], xu=bounds[:, 1])

    def _evaluate(self, x, out, *args, **kwargs):
        n = x.shape[0]
        F = np.zeros((n, 2))
        G = np.zeros((n, 4))

        for i in range(n):
            sol = x[i, :].copy()
            # If fixed granule, override the granule value
            if self.granule_mode == "Fixed":
                sol[7] = self.fixed_granule

            # Normalize components and convert to list
            api, mcc, pvpp, mgst, binder, pressure, speed, granule = sol
            api, binder, pvpp, mgst, mcc = normalize_components(api, binder, pvpp, mgst, mcc)
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

            # Objectives: maximize (api + w_tensile * tensile), minimize efrf
            F[i, 0] = -(api + self.w_tensile * tensile)   # negative for minimization
            F[i, 1] = efrf

            # Constraints (must be <= 0 for feasible)
            G[i, 0] = 0.40 - density
            G[i, 1] = density - 0.97
            G[i, 2] = 1.90 - tensile
            G[i, 3] = efrf - 0.40

        out["F"] = F
        out["G"] = G

def run_pymoo_nsga2(model, scaler, y_scaler, bounds,
                    pop_size=NSGA_POP_SIZE, generations=NSGA_GENERATIONS,
                    granule_mode='Variable', fixed_granule=125.0, w_tensile=0.0):
    """
    Run NSGA-II using pymoo and return results in a format compatible with the rest of the app.
    Returns: (population, objectives, constraints, fronts)
    where:
        population: numpy array of decision variables (all evaluated)
        objectives: numpy array of objective values
        constraints: numpy array of constraint violations (or values)
        fronts: list of lists of indices in the Pareto front (first front only)
    """
    problem = TabletProblem(model, scaler, y_scaler, bounds,
                            granule_mode=granule_mode,
                            fixed_granule=fixed_granule,
                            w_tensile=w_tensile)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', generations),
                   verbose=False,
                   save_history=False)

    # Extract all evaluated solutions from the final population
    pop = res.pop
    if pop is None or len(pop) == 0:
        # Fallback: use the final X and F from res
        X_all = res.X
        F_all = res.F
    else:
        X_all = np.array([ind.X for ind in pop])
        F_all = np.array([ind.F for ind in pop])

    # Compute constraint violations (we'll just use the original G values)
    # For front extraction, we need the Pareto front from pymoo.
    # We can get the front from res (non-dominated solutions)
    pareto_front = res.F  # this is the objective matrix of the Pareto front
    # We also need the corresponding decision variables for the Pareto front
    # In pymoo, res.X is the Pareto set (decision variables for the Pareto front)
    # However, res.X contains only the non-dominated solutions, not all.
    # But we also need the full population to plot all solutions.
    # We'll keep X_all, F_all as all solutions (from final population), and then use the Pareto indices from res.
    # To get the Pareto indices, we can compute the non-dominated front from F_all.
    # Or we can use res.opt.get("X") and res.opt.get("F") which give Pareto set and front.
    if hasattr(res, 'opt') and res.opt is not None:
        pareto_X = res.opt.get('X')
        pareto_F = res.opt.get('F')
        # Build a list of indices of the Pareto solutions in the full population
        # Since we might not have all, we'll just use the Pareto set as the "fronts"
        # For consistency, we'll create fronts as a list of lists of indices (but we don't have indices easily)
        # We'll just return the Pareto set and objectives separately.
        # We'll still return fronts as a list where front0 are the Pareto objective values.
        # The plotting function expects fronts as a list where fronts[0] is indices of the first front.
        # We'll adapt: we'll store the Pareto set and front as separate arrays, and modify the plotting function to accept them.
        # To keep compatibility, we'll store the Pareto front objectives as an array and the indices as all zeros (placeholder).
        # We'll handle this in the plotting function.
        # Actually, the original plotting function uses fronts[0] to get indices and then extracts pareto_api and pareto_efrf.
        # We'll adjust the plotting function to also accept direct arrays.
        # For now, we'll store a dummy fronts list of indices (range(len(pareto_F))) and also pass the pareto_F.
        # The plotting function will use the provided arrays.
        # So we'll return the pareto_X, pareto_F, and the full X_all, F_all.
        # We'll adapt the plotting function to accept two new arguments: pareto_api and pareto_efrf directly.
        # That's simpler.
        # We'll just return the full population and objectives, and separately the Pareto objectives and decision vars.
        return X_all, F_all, None, pareto_X, pareto_F  # We'll handle this in the UI.
    else:
        # No Pareto front found
        return X_all, F_all, None, None, None

# ================================================================
# 5. PREDICTION, PLOTTING, AND COMPARISON FUNCTIONS
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

def plot_training_curves(loss_history):
    if not loss_history or len(loss_history['train']) == 0:
        return None
    epochs = list(range(1, len(loss_history['train']) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=loss_history['train'], mode='lines', name='Training Loss'))
    if len(loss_history['val']) > 0:
        fig.add_trace(go.Scatter(x=epochs[:len(loss_history['val'])], y=loss_history['val'], mode='lines', name='Validation Loss'))
    fig.update_layout(title='Training Curves (v29.42)', xaxis_title='Epoch', yaxis_title='Loss', height=400)
    return fig

def smooth_pareto_curve(api_points, efrf_points, num_points=200):
    if len(api_points) < 3:
        return api_points, efrf_points
    sorted_idx = np.argsort(api_points)
    api_sorted = np.array(api_points)[sorted_idx]
    efrf_sorted = np.array(efrf_points)[sorted_idx]
    _, unique_idx = np.unique(api_sorted, return_index=True)
    api_unique = api_sorted[unique_idx]
    efrf_unique = efrf_sorted[unique_idx]
    if len(api_unique) < 3:
        return api_unique, efrf_unique
    x_new = np.linspace(api_unique.min(), api_unique.max(), num_points)
    try:
        if SCIPY_AVAILABLE:
            spline = UnivariateSpline(api_unique, efrf_unique, s=0.001, k=3)
            y_new = spline(x_new)
        else:
            degree = min(3, len(api_unique) - 1)
            coeffs = np.polyfit(api_unique, efrf_unique, degree)
            y_new = np.polyval(coeffs, x_new)
        return x_new, y_new
    except:
        return api_unique, efrf_unique

def plot_pareto_with_stars(objectives, fronts,
                           user_api=None, user_efrf=None,
                           golden_api=None, golden_efrf=None,
                           smooth=True,
                           pareto_api=None, pareto_efrf=None):
    """
    Plot Pareto front with two stars.
    If pareto_api and pareto_efrf are provided, use them directly;
    otherwise use fronts to extract them.
    """
    fig = go.Figure()
    fig.data = []

    # Determine Pareto front data
    if pareto_api is not None and pareto_efrf is not None:
        # Use direct arrays
        api_pareto = pareto_api
        efrf_pareto = pareto_efrf
    else:
        # Use fronts (old method)
        if objectives is None or fronts is None or len(fronts) == 0 or len(fronts[0]) == 0:
            return None
        front0 = fronts[0]
        api_pareto = -objectives[front0, 0]
        efrf_pareto = objectives[front0, 1]

    sorted_idx = np.argsort(api_pareto)
    api_pareto_sorted = api_pareto[sorted_idx]
    efrf_pareto_sorted = efrf_pareto[sorted_idx]

    fig.add_trace(go.Scatter(
        x=api_pareto_sorted, y=efrf_pareto_sorted,
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Pareto Solutions (discrete)'
    ))

    if smooth and len(api_pareto_sorted) >= 3:
        x_s, y_s = smooth_pareto_curve(api_pareto_sorted, efrf_pareto_sorted)
        fig.add_trace(go.Scatter(
            x=x_s, y=y_s,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Pareto Front (smooth)'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=api_pareto_sorted, y=efrf_pareto_sorted,
            mode='lines',
            line=dict(color='red', width=2),
            name='Pareto Front (line)'
        ))

    if golden_api is not None and golden_efrf is not None:
        fig.add_trace(go.Scatter(
            x=[golden_api], y=[golden_efrf],
            mode='markers+text',
            marker=dict(size=18, color='gold', symbol='star',
                        line=dict(color='darkgoldenrod', width=2)),
            text=['⭐ Golden'], textposition='top center',
            name='Golden Solution'
        ))

    if user_api is not None and user_efrf is not None:
        fig.add_trace(go.Scatter(
            x=[user_api], y=[user_efrf],
            mode='markers+text',
            marker=dict(size=14, color='blue', symbol='star',
                        line=dict(color='darkblue', width=2)),
            text=['🔵 Tested'], textposition='top center',
            name='Tested Solution'
        ))

    fig.add_hline(y=EFRF_MAX, line_dash='dash', line_color='red',
                  annotation_text=f'EFRF Threshold: {EFRF_MAX:.2f}',
                  annotation_position='top right')
    fig.update_layout(
        title='Pareto Front with Two Stars (v29.42)',
        xaxis_title='API (%)',
        yaxis_title='EFRF',
        height=500,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig

def plot_sensitivity_plotly(inputs, model, scaler, y_scaler):
    try:
        features = ['API%', 'MCC%', 'PVPP%', 'Mg-St%', 'Binder%', 'Pressure', 'Speed', 'Granule']
        _, _, _, base_efrf = predict_pinn(model, scaler, y_scaler, inputs)
        sensitivities = []
        for i in range(8):
            test = inputs.copy()
            test[i] += 0.05 * (inputs[i] + 0.1)
            _, _, _, efrf_pos = predict_pinn(model, scaler, y_scaler, test)
            test[i] = inputs[i] - 0.05 * (inputs[i] + 0.1)
            _, _, _, efrf_neg = predict_pinn(model, scaler, y_scaler, test)
            sensitivities.append(max(abs(efrf_pos - base_efrf), abs(efrf_neg - base_efrf)))
        sorted_idx = np.argsort(sensitivities)[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[features[i] for i in sorted_idx],
            x=[sensitivities[i] for i in sorted_idx],
            orientation='h',
            marker_color='#1f77b4'
        ))
        fig.update_layout(title='Sensitivity Analysis (EFRF)', xaxis_title='Sensitivity', height=400)
        return fig
    except:
        return None

def train_and_compare(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    try:
        from xgboost import XGBRegressor
        xgb_available = True
    except:
        xgb_available = False

    models = {
        'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)
    }
    if xgb_available:
        models['XGBoost'] = XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)

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

def generate_full_pdf_report(api, mcc, pvpp, mgst, binder, pressure, speed, granule,
                             density, tensile, er, efrf, status, timestamp,
                             model_comparison_df=None, golden_info=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, sanitize_text("Formulation Optimization Report (v29.42)"), ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, sanitize_text(f"Date: {timestamp}"), ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, sanitize_text("Formulation Summary"), ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(50, 6, sanitize_text("API"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{api:.1f}%"), 1, 1)
    pdf.cell(50, 6, sanitize_text("MCC"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{mcc:.1f}%"), 1, 1)
    pdf.cell(50, 6, sanitize_text("PVPP"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{pvpp:.1f}%"), 1, 1)
    pdf.cell(50, 6, sanitize_text("Mg-St"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{mgst:.2f}%"), 1, 1)
    pdf.cell(50, 6, sanitize_text("Binder"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{binder:.1f}%"), 1, 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, sanitize_text("Predicted Quality Attributes"), ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(50, 6, sanitize_text("Density"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{density:.3f}"), 1, 1)
    pdf.cell(50, 6, sanitize_text("Tensile Strength"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{tensile:.3f} MPa"), 1, 1)
    pdf.cell(50, 6, sanitize_text("Elastic Recovery"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{er:.3f} %"), 1, 1)
    pdf.cell(50, 6, sanitize_text("EFRF"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{efrf:.4f}"), 1, 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, sanitize_text(f"Status: {status}"), ln=True)

    if golden_info is not None:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, sanitize_text("Golden Solution (Optimal Trade-off)"), ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(50, 6, sanitize_text("API"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{golden_info['api']:.1f}%"), 1, 1)
        pdf.cell(50, 6, sanitize_text("Binder"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{golden_info['binder']:.1f}%"), 1, 1)
        pdf.cell(50, 6, sanitize_text("PVPP"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{golden_info['pvpp']:.1f}%"), 1, 1)
        pdf.cell(50, 6, sanitize_text("Mg-St"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{golden_info['mgst']:.2f}%"), 1, 1)
        pdf.cell(50, 6, sanitize_text("MCC"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{golden_info['mcc']:.1f}%"), 1, 1)
        pdf.cell(50, 6, sanitize_text("Pressure"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{golden_info['pressure']:.1f} MPa"), 1, 1)
        pdf.cell(50, 6, sanitize_text("Speed"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{golden_info['speed']:.1f} rpm"), 1, 1)
        pdf.cell(50, 6, sanitize_text("Granule"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{golden_info['granule']:.0f} um"), 1, 1)
        pdf.set_font("Arial", "", 10)
        pdf.cell(50, 6, sanitize_text("Density"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{golden_info['density']:.3f}"), 1, 1)
        pdf.cell(50, 6, sanitize_text("Tensile"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{golden_info['tensile']:.3f} MPa"), 1, 1)
        pdf.cell(50, 6, sanitize_text("EFRF"), 1, 0); pdf.cell(30, 6, sanitize_text(f"{golden_info['efrf']:.4f}"), 1, 1)

    if model_comparison_df is not None:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, sanitize_text("Model Comparison"), ln=True)
        pdf.set_font("Arial", "", 8)
        pdf.cell(40, 6, sanitize_text("Model"), 1, 0)
        pdf.cell(30, 6, sanitize_text("R2"), 1, 0)
        pdf.cell(30, 6, sanitize_text("RMSE"), 1, 0)
        pdf.cell(30, 6, sanitize_text("MAE"), 1, 0)
        pdf.cell(40, 6, sanitize_text("Physics"), 1, 1)
        for _, row in model_comparison_df.iterrows():
            pdf.cell(40, 6, sanitize_text(str(row['Model'])[:10]), 1, 0)
            pdf.cell(30, 6, sanitize_text(f"{row['R²']:.4f}"), 1, 0)
            pdf.cell(30, 6, sanitize_text(f"{row['RMSE']:.4f}"), 1, 0)
            pdf.cell(30, 6, sanitize_text(f"{row['MAE']:.4f}"), 1, 0)
            pdf.cell(40, 6, sanitize_text(str(row['Physics'])), 1, 1)
    
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, bytearray):
        return bytes(pdf_bytes)
    return pdf_bytes

# ================================================================
# 6. MODEL LOADING / TRAINING (AUTO-REPAIR)
# ================================================================

@st.cache_resource
def load_or_train_model():
    checkpoint_path = '/tmp/pinn_best_model.pt'
    
    try:
        if os.path.exists(checkpoint_path):
            st.caption("📂 Loading cached model from /tmp...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            required_keys = ['model_state', 'scaler', 'y_scaler', 'feature_names', 'df', 'loss_history']
            if all(k in checkpoint for k in required_keys):
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
                st.warning("⚠️ Cached file is missing some keys. Re-training...")
                os.remove(checkpoint_path)
    except Exception as e:
        st.warning(f"⚠️ Cached model file is corrupted or incompatible. Re-training... (Error: {str(e)[:80]})")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    st.caption("🔄 Training model from scratch (v29.42 improved settings)...")

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

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
            w_density=W_DENSITY, w_tensile=W_TENSILE,
            w_tensile_physics=W_TENSILE_PHYSICS,
            w_physics_base=W_PHYSICS_BASE, w_physics_final=W_PHYSICS_FINAL,
            w_mcc=W_MCC, w_density_penalty=W_DENSITY_PENALTY,
            efrf_target=EFRF_MAX, mcc_max=MCC_MAX,
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
                w_density=W_DENSITY, w_tensile=W_TENSILE,
                w_tensile_physics=W_TENSILE_PHYSICS,
                w_physics_base=W_PHYSICS_BASE, w_physics_final=W_PHYSICS_FINAL,
                w_mcc=W_MCC, w_density_penalty=W_DENSITY_PENALTY,
                efrf_target=EFRF_MAX, mcc_max=MCC_MAX,
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
                st.warning(f"⏹️ Training stopped early at epoch {epoch+1} (no improvement for {patience} epochs).")
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

    temp_path = checkpoint_path + ".tmp"
    torch.save(checkpoint_data, temp_path)

    try:
        test_load = torch.load(temp_path, map_location='cpu', weights_only=False)
        os.rename(temp_path, checkpoint_path)
        st.success("✅ Model trained and cached successfully (verified).")
    except Exception as e:
        st.error(f"❌ Failed to verify saved checkpoint: {e}. The model will not be cached for this session.")
        pass

    return model, scaler, y_scaler, feature_names, df, {'train': train_losses, 'val': val_losses}

# ================================================================
# 7. MAIN USER INTERFACE (Streamlit UI)
# ================================================================

st.set_page_config(page_title="PINN Cloud v29.42", page_icon="🧬", layout="wide")

st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 1.5rem; border-radius: 1rem; margin-bottom: 1.5rem; text-align: center;">
    <h1 style="color: #ffffff; font-size: 2rem; margin: 0;">🧬 Hybrid AI Framework v29.42</h1>
    <p style="color: #64ffda; font-size: 0.9rem; margin: 0.5rem 0 0 0;">⚡ Simplified Adaptive Loss · PyMOO NSGA-II · Granule Analysis</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

with st.sidebar:
    st.markdown("### 📚 Physics Constraints (v29.42)")
    st.markdown(f"""
    - ✅ **Heckel:** ln(1/(1-D)) = kP + A
    - ✅ **EFRF:** ER / σt < {EFRF_MAX:.2f}
    - ✅ **Density:** {D_MIN:.2f} ≤ D ≤ {D_MAX:.2f}
    - ✅ **MCC:** ≤ {MCC_MAX:.1f}%
    - ✅ **Samples:** {N_SAMPLES} (Enhanced)
    - ✅ **Epochs:** {ADAM_EPOCHS} (Early Stopping at {PATIENCE})
    - ✅ **Device:** GPU (if available)
    - ✅ **Loss:** Simplified adaptive (boost largest error)
    - ✅ **Noise:** Ultra-low (σ = 0.002, 0.005, 0.005)
    - ✅ **Cache:** Auto-repair if corrupted (with verification)
    - ✅ **NSGA-II:** PyMOO, Pop={NSGA_POP_SIZE}, Gen={NSGA_GENERATIONS}
    - ✅ **Network:** BatchNorm + Dropout (0.1)
    - ✅ **Granule Analysis:** Toggle Fixed/Variable
    """)
    show_smooth = st.checkbox("Show smooth Pareto curve", value=True)
    st.info("🔬 **v29.42** — PyMOO NSGA-II & Simplified Adaptive Loss")

# Load or train model (cached)
with st.spinner("📂 Loading/Training model (v29.42)..."):
    model, scaler, y_scaler, feature_names, df, loss_history = load_or_train_model()
st.success("✅ Model ready!")

# ================================================================
# Granule Analysis Section
# ================================================================
st.markdown("---")
st.markdown("### 🔬 Granule Size Analysis")
with st.expander("Granule Size Toggle & Plots", expanded=True):
    granule_mode = st.radio(
        "Granule Size Mode:",
        ["Variable", "Fixed"],
        horizontal=True,
        key="granule_mode"
    )
    fixed_granule = 125.0
    if granule_mode == "Fixed":
        fixed_granule = st.number_input(
            "Enter fixed granule size (µm):",
            min_value=30.0,
            max_value=250.0,
            value=125.0,
            step=1.0,
            key="fixed_granule"
        )
    with st.spinner("Generating data for granule analysis..."):
        df_granule, _ = generate_pinn_data(
            n_samples=2000,
            granule_mode=granule_mode,
            fixed_granule=fixed_granule
        )
    fig1 = px.scatter(
        df_granule, x="Granule_Size_µm", y="Tensile_Strength_MPa",
        color="Density", title="Granule Size vs Tensile Strength",
        color_continuous_scale="viridis"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        df_granule, x="Granule_Size_µm", y="Density",
        color="Tensile_Strength_MPa", title="Granule Size vs Density",
        color_continuous_scale="plasma"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(f"Dataset size: {len(df_granule)} samples. Mode: {granule_mode}" + 
               (f" (Fixed at {fixed_granule} µm)" if granule_mode=="Fixed" else ""))

st.markdown("---")

# Quick experiments (sum to 100%)
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
            clamp_session_state()
            st.rerun()

st.markdown("---")

col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    st.markdown("### 📊 Formulation Parameters")
    with st.container(border=True):
        api = st.slider("🧪 API (%)", 85.0, 95.0, get_safe_value('api'), 0.1, key="api")
        binder = st.slider("🔗 Binder (%)", BINDER_MIN, BINDER_MAX, get_safe_value('binder'), 0.1, key="binder")
        pvpp = st.slider("💊 PVPP (%)", 0.5, 6.0, get_safe_value('pvpp'), 0.1, key="pvpp")
        mgst = st.slider("🧴 Mg-St (%)", 0.01, 1.2, get_safe_value('mgst'), 0.01, key="mgst")
        mcc = st.slider("📦 MCC (%)", 0.0, MCC_MAX, get_safe_value('mcc'), 0.1, key="mcc")
        total = api + binder + pvpp + mgst + mcc
        if abs(total - 100) < 0.1:
            st.success(f"✅ Total = {total:.2f}%")
        else:
            st.warning(f"⚠️ Total = {total:.2f}% (adjust to 100%)")
    st.markdown("### ⚙️ Process Parameters")
    with st.container(border=True):
        pressure = st.slider("⚙️ Pressure (MPa)", 80.0, PRESSURE_MAX, get_safe_value('pressure'), 1.0, key="pressure")
        speed = st.slider("🔄 Speed (rpm)", 1.0, 50.0, get_safe_value('speed'), 0.5, key="speed")
        granule = st.slider("🔬 Granule Size (µm)", 30.0, 250.0, get_safe_value('granule'), 1.0, key="granule")
    predict_btn = st.button("🔬 Predict & Optimize (v29.42)", use_container_width=True)

with col_right:
    st.markdown("### 📈 Results")
    objectives = None; constraints = None; fronts = None; nsga = None
    api_norm = None; efrf = None; comp_df = pd.DataFrame()
    density = 0.0; tensile = 0.0; er = 0.0
    density_ok = False; tensile_ok = False; efrf_ok = False; mcc_ok = False
    api_use = 0.0; mcc_use = 0.0; pvpp_use = 0.0; mgst_use = 0.0; binder_use = 0.0
    golden_info = None
    pareto_api_arr = None
    pareto_efrf_arr = None

    if predict_btn:
        if abs(total - 100) > 0.1:
            st.warning("⚠️ Formulation must sum to 100%")
        else:
            api_norm, binder_norm, pvpp_norm, mgst_norm, mcc_norm = normalize_components(api, binder, pvpp, mgst, mcc)
            inputs_norm = [api_norm, mcc_norm, pvpp_norm, mgst_norm, binder_norm, pressure, speed, granule]
            api_use, mcc_use, pvpp_use, mgst_use, binder_use = api_norm, mcc_norm, pvpp_norm, mgst_norm, binder_norm
            with st.spinner("🧠 Predicting (v29.42)..."):
                density, tensile, er, efrf = predict_pinn(model, scaler, y_scaler, inputs_norm)
            kpi_cols = st.columns(3)
            kpi_cols[0].metric("Density", f"{density:.3f}", delta=f"Target: {D_MIN:.2f}–{D_MAX:.2f}")
            kpi_cols[1].metric("Tensile", f"{tensile:.3f} MPa", delta=f"Min: {TENSILE_MIN:.2f} MPa")
            kpi_cols[2].metric("EFRF", f"{efrf:.4f}", delta=f"Max: {EFRF_MAX:.2f}")
            st.markdown("---")
            density_ok = (density >= D_MIN and density <= D_MAX)
            tensile_ok = (tensile >= TENSILE_MIN)
            efrf_ok = (efrf < EFRF_MAX)
            mcc_ok = (mcc_norm <= MCC_MAX)
            if density_ok and tensile_ok and efrf_ok and mcc_ok:
                st.success("✅ All constraints satisfied!")
            else:
                st.error("❌ Violates constraints")
            st.markdown("### ✅ Feasibility")
            pass_cols = st.columns(4)
            pass_cols[0].metric("Density", "✅" if density_ok else "❌")
            pass_cols[1].metric("Tensile", "✅" if tensile_ok else "❌")
            pass_cols[2].metric("EFRF", "✅" if efrf_ok else "❌")
            pass_cols[3].metric("MCC", "✅" if mcc_ok else "❌")

            # --- NSGA‑II with PyMOO ---
            st.markdown("### ⚙️ NSGA‑II (v29.42, PyMOO)")
            bounds = np.array([[60,100],[0.1,20],[0.1,12],[0.01,3.0],[0.1,10],[80,PRESSURE_MAX],[1,50],[30,250]])
            with st.spinner(f"🔄 PyMOO NSGA‑II (pop={NSGA_POP_SIZE}, gen={NSGA_GENERATIONS})..."):
                # Run PyMOO NSGA-II
                X_all, F_all, _, pareto_X, pareto_F = run_pymoo_nsga2(
                    model, scaler, y_scaler, bounds,
                    pop_size=NSGA_POP_SIZE,
                    generations=NSGA_GENERATIONS,
                    granule_mode='Variable',  # or use the user's granule mode? Keep it flexible
                    fixed_granule=125.0,
                    w_tensile=0.0
                )

                if pareto_F is not None and len(pareto_F) > 0:
                    st.success(f"📊 Pareto front found: **{len(pareto_F)}** optimal solutions")
                    # Extract API and EFRF from Pareto front
                    # We need to convert F back to original objectives: F[0] = -(api + w_tensile*tensile)
                    # So we need to compute api and efrf from the decision variables.
                    # But we can also compute them from the model predictions.
                    # We'll compute them by evaluating each Pareto solution.
                    api_list = []
                    efrf_list = []
                    density_list = []
                    tensile_list = []
                    er_list = []
                    for sol in pareto_X:
                        inputs = sol.copy()
                        api, mcc, pvpp, mgst, binder, pressure, speed, granule = inputs
                        api, binder, pvpp, mgst, mcc = normalize_components(api, binder, pvpp, mgst, mcc)
                        inputs_norm = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
                        d, t, e, ef = predict_pinn(model, scaler, y_scaler, inputs_norm)
                        api_list.append(api)  # we use the normalized API
                        efrf_list.append(ef)
                        density_list.append(d)
                        tensile_list.append(t)
                        er_list.append(e)

                    pareto_api_arr = np.array(api_list)
                    pareto_efrf_arr = np.array(efrf_list)

                    # Golden solution: minimize EFRF, then maximize tensile (or minimize -tensile)
                    # We'll find the solution with minimum EFRF, and if tie, max tensile
                    # We can use the same logic as before.
                    # Create a list of candidates with their data
                    candidates = []
                    for idx in range(len(pareto_X)):
                        candidates.append({
                            'formulation': pareto_X[idx],
                            'density': density_list[idx],
                            'tensile': tensile_list[idx],
                            'er': er_list[idx],
                            'efrf': efrf_list[idx],
                            'api': api_list[idx]
                        })
                    # Filter feasible (already feasible if they are in Pareto front, but double-check)
                    feasible_candidates = [c for c in candidates if D_MIN <= c['density'] <= D_MAX and c['tensile'] >= TENSILE_MIN and c['efrf'] < EFRF_MAX]
                    if feasible_candidates:
                        best = min(feasible_candidates, key=lambda x: (x['efrf'], -x['tensile']))
                        golden_info = {
                            'api': best['api'],
                            'mcc': best['formulation'][1],
                            'pvpp': best['formulation'][2],
                            'mgst': best['formulation'][3],
                            'binder': best['formulation'][4],
                            'pressure': best['formulation'][5],
                            'speed': best['formulation'][6],
                            'granule': best['formulation'][7],
                            'density': best['density'],
                            'tensile': best['tensile'],
                            'er': best['er'],
                            'efrf': best['efrf']
                        }
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
                    else:
                        st.info("No fully feasible solution found in Pareto front.")
                else:
                    st.warning("No Pareto front found. Try adjusting NSGA-II parameters.")

                # Also store objectives and fronts for the plotting function
                # We'll set objectives as F_all, and fronts as the Pareto front indices? We'll use the direct arrays.
                objectives = F_all
                # For compatibility, we'll create a dummy fronts list (not used if we pass pareto_api_arr)
                fronts = None

            # --- Model Comparison ---
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📉 Pareto", "🔍 Sensitivity", "📊 Comparison", "📄 Report", "🔬 Granule"])

    with tab1:
        if predict_btn and objectives is not None and pareto_api_arr is not None:
            golden_api = golden_info['api'] if golden_info else None
            golden_efrf = golden_info['efrf'] if golden_info else None

            fig = plot_pareto_with_stars(
                objectives=objectives,
                fronts=fronts,
                user_api=api_norm,
                user_efrf=efrf,
                golden_api=golden_api,
                golden_efrf=golden_efrf,
                smooth=show_smooth,
                pareto_api=pareto_api_arr,
                pareto_efrf=pareto_efrf_arr
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.caption("🔵 Blue star = Your formulation · ⭐ Gold star = Optimal (golden) solution · Red dashed line = EFRF threshold")
                if show_smooth:
                    st.caption("The dashed red curve is a smooth interpolation of the Pareto points.")
            else:
                st.info("No Pareto front data available.")
        else:
            st.info("👆 Click 'Predict & Optimize' to run NSGA‑II and see the Pareto front.")

    with tab2:
        if predict_btn and api_norm is not None:
            fig = plot_sensitivity_plotly(inputs_norm, model, scaler, y_scaler)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sensitivity analysis not available.")
        else:
            st.info("👆 Click 'Predict & Optimize' to run sensitivity analysis.")

    with tab3:
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
                textposition='outside',
                hovertemplate='%{y}<br>R² = %{x:.4f}<extra></extra>'
            ))
            fig.update_layout(
                title='R² Score Comparison (v29.42)',
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

    with tab4:
        if predict_btn and not comp_df.empty:
            status = "PASS" if (density_ok and tensile_ok and efrf_ok and mcc_ok) else "FAIL"
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            pdf_comp_df = comp_df.copy()
            for col in ['R²', 'RMSE', 'MAE']:
                pdf_comp_df[col] = pdf_comp_df[col].astype(float)

            pdf_data = generate_full_pdf_report(
                api_use, mcc_use, pvpp_use, mgst_use, binder_use,
                pressure, speed, granule, density, tensile, er, efrf,
                status, timestamp, pdf_comp_df, golden_info
            )
            st.download_button(
                "📥 Download PDF Report (v29.42)",
                data=pdf_data,
                file_name=f"report_v29.42_{timestamp[:10]}.pdf",
                mime="application/pdf"
            )
        else:
            st.info("👆 Click 'Predict & Optimize' to generate the report.")

    with tab5:
        st.markdown("### 🔬 Granule Size Effect")
        st.info("Use the toggle at the top of the page to switch between Fixed and Variable granule modes.")
        st.markdown("The plots are shown in the expander at the top of the page.")

st.markdown("---")
st.caption("🔬 **PINN v29.42** — Simplified Adaptive Loss · PyMOO NSGA-II · Granule Analysis | Nile Valley University")
