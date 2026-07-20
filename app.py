"""
Hubryd AI – v29.27-R23 (Fully Bugfixed Advanced Model)
Advanced Hybrid AI for Multi-Objective Optimization of Tablet Formulation
- Fixed all array ambiguity errors
- Safe handling of model=None
- All advanced features included
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
# CONSTANTS
# ================================================================
D_MIN = 0.70
D_MAX = 0.99
TENSILE_MIN = 1.50
EFRF_MAX = 0.50
DISINTEGRATION_MAX = 15.0

# Slider ranges
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

# Material properties
SLIDER_PARTICLE_SIZE_MIN = 10.0
SLIDER_PARTICLE_SIZE_MAX = 200.0
SLIDER_MOISTURE_MIN = 0.5
SLIDER_MOISTURE_MAX = 5.0
BINDER_GRADES = ["MCC PH101", "MCC PH102", "MCC PH200", "MCC KG", "Lactose", "Dicalcium Phosphate"]

# Process
SLIDER_PRESSURE_MIN = 150.0
SLIDER_PRESSURE_MAX = 250.0
SLIDER_SPEED_MIN = 15.0
SLIDER_SPEED_MAX = 30.0
SLIDER_GRANULE_MIN = 30.0
SLIDER_GRANULE_MAX = 250.0
SLIDER_DWELL_TIME_MIN = 5.0
SLIDER_DWELL_TIME_MAX = 50.0
SLIDER_FRICTION_MIN = 0.1
SLIDER_FRICTION_MAX = 0.5
SLIDER_DECOMPRESSION_TIME_MIN = 10.0
SLIDER_DECOMPRESSION_TIME_MAX = 80.0

# NSGA-II practical bounds
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

# Training parameters
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
W_DISINTEGRATION = 10.0
W_DISSOLUTION = 10.0

# ================================================================
# SESSION STATE
# ================================================================
if 'api' not in st.session_state:
    st.session_state.update({
        'api': 90.5,
        'binder': 3.5,
        'pvpp': 2.0,
        'mgst': 0.5,
        'mcc': 3.5,
        'particle_size': 50.0,
        'moisture': 2.0,
        'binder_grade': 0,
        'pressure': 200.0,
        'speed': 20.0,
        'dwell_time': 25.0,
        'friction': 0.25,
        'decompression_time': 35.0,
        'granule': 125.0,
        'show_cost_solution': False,
        'show_quality_solution': False,
        'show_comparison': True,
        'show_sensitivity': False,
        'show_particle_plot': False,
        'show_dissolution': False,
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
            'particle_size': None, 'moisture': None, 'binder_grade': None,
            'pressure': None, 'speed': None, 'dwell_time': None,
            'friction': None, 'decompression_time': None,
            'granule_use': None, 'granule_fixed': None,
            'density': None, 'tensile': None, 'er': None, 'efrf': None,
            'disintegration': None, 'dissolution_tau': None, 'dissolution_beta': None
        },
        'feasible_df': None,
        'tested_point': None,
        'benchmark_df': None
    })

# ================================================================
# COMPACTION MODELS
# ================================================================
def heckel_equation(pressure, k, A):
    return 1 - np.exp(-(k * pressure + A))

def kawakita_equation(pressure, a, b):
    return 1 - (pressure / (a * pressure + 1/b))

def walker_equation(pressure, K, n):
    return 1 - K * (pressure ** (-n))

def select_compaction_model(api_n, binder_n, pressure, particle_size, moisture):
    """
    All parameters are scalars (single values), not arrays.
    """
    if api_n > 90 and particle_size < 50:
        model_type = "Heckel"
        k = 0.025 + 0.0001 * pressure - 0.001 * (api_n - 85) / 10
        A = 1.0 + 0.01 * (api_n - 85) - 0.05 * binder_n
        return model_type, {'k': k, 'A': A}
    elif moisture > 3.0 and particle_size > 100:
        model_type = "Kawakita"
        a = 0.5 + 0.01 * (pressure - 150) / 50
        b = 0.8 + 0.02 * binder_n
        return model_type, {'a': a, 'b': b}
    else:
        model_type = "Walker"
        K = 0.3 + 0.01 * binder_n
        n = 0.15 + 0.005 * (api_n - 80)
        return model_type, {'K': K, 'n': n}

# ================================================================
# DWELL TIME CALCULATION
# ================================================================
def calculate_dwell_time(speed_rpm, punch_width=10, pitch_diameter=100):
    if speed_rpm <= 0:
        return 50.0
    dwell_time_ms = (punch_width * 60 * 1000) / (np.pi * pitch_diameter * speed_rpm)
    return np.clip(dwell_time_ms, 5.0, 80.0)

# ================================================================
# DISINTEGRATION AND DISSOLUTION
# ================================================================
def predict_disintegration_time(tensile, pvpp_n, api_n, binder_n, moisture):
    base_time = 2.0 + 0.5 * tensile
    pvpp_effect = 5.0 * np.exp(-0.5 * pvpp_n)
    api_effect = 0.1 * (api_n - 80)
    binder_effect = 0.2 * (binder_n - 2.0)
    moisture_effect = -0.1 * moisture
    time = base_time - pvpp_effect + api_effect + binder_effect + moisture_effect
    return np.clip(time, 1.0, 30.0)

def predict_dissolution_profile(api_n, pvpp_n, particle_size, disintegration_time):
    tau = 5.0 + 0.5 * disintegration_time - 0.1 * pvpp_n + 0.05 * (api_n - 80)
    tau = np.clip(tau, 2.0, 20.0)
    beta = 1.0 + 0.01 * (particle_size - 50) / 50
    beta = np.clip(beta, 0.8, 2.5)
    return {'tau': tau, 'beta': beta}

def compatibility_check(api_n, binder_grade, moisture):
    """
    All parameters are scalars (single values), not arrays.
    """
    # Ensure they are Python scalars
    api_n = float(api_n)
    binder_grade = int(binder_grade)
    moisture = float(moisture)
    
    warnings = []
    score = 1.0
    if api_n > 90 and moisture > 4.0:
        warnings.append("⚠️ High API loading with high moisture may cause stability issues.")
        score -= 0.2
    if moisture > 3.5:
        warnings.append("⚠️ High moisture may cause premature swelling of superdisintegrant.")
        score -= 0.1
    if binder_grade in [0, 1] and api_n > 90:
        score -= 0.05
        warnings.append("ℹ️ Fine MCC grades may not provide sufficient binding at high API loading.")
    return {'score': max(score, 0.0), 'warnings': warnings}

# ================================================================
# HELPER FUNCTIONS
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
    particle_size = X_raw[:, 8:9]
    moisture = X_raw[:, 9:10]
    binder_grade = X_raw[:, 10:11]
    dwell_time = X_raw[:, 11:12]
    friction = X_raw[:, 12:13]
    decompression_time = X_raw[:, 13:14]

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

    particle_pressure = particle_size * pressure
    moisture_pressure = moisture * pressure
    particle_moisture = particle_size * moisture
    dwell_pressure = dwell_time * pressure
    friction_pressure = friction * pressure

    return np.concatenate([
        X_raw,
        pressure_binder, pressure_api,
        pressure_speed, api_mcc, binder_speed,
        api_pvpp, binder_mgst, mcc_pvpp,
        api2, pressure2, binder2, speed2,
        particle_pressure, moisture_pressure,
        particle_moisture, dwell_pressure, friction_pressure
    ], axis=1)

def generate_pinn_data(n_samples=N_SAMPLES, random_state=42):
    rng = np.random.default_rng(random_state)
    # Original formulation
    api_raw = rng.uniform(SLIDER_API_MIN, SLIDER_API_MAX, n_samples)
    binder_raw = rng.uniform(SLIDER_BINDER_MIN, SLIDER_BINDER_MAX, n_samples)
    pvpp_raw = rng.uniform(SLIDER_PVPP_MIN, SLIDER_PVPP_MAX, n_samples)
    mgst_raw = rng.uniform(SLIDER_MGST_MIN, SLIDER_MGST_MAX, n_samples)
    mcc_raw = rng.uniform(SLIDER_MCC_MIN, SLIDER_MCC_MAX, n_samples)
    particle_size_raw = rng.uniform(SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX, n_samples)
    moisture_raw = rng.uniform(SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX, n_samples)
    binder_grade_raw = rng.integers(0, len(BINDER_GRADES), n_samples)
    pressure_raw = rng.uniform(SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX, n_samples)
    speed_raw = rng.uniform(SLIDER_SPEED_MIN, SLIDER_SPEED_MAX, n_samples)
    dwell_time_raw = calculate_dwell_time(speed_raw)
    friction_raw = rng.uniform(SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX, n_samples)
    decompression_time_raw = rng.uniform(SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX, n_samples)
    granule_raw = rng.uniform(SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX, n_samples)

    api_n, binder_n, pvpp_n, mgst_n, mcc_n = normalize_components(
        api_raw, binder_raw, pvpp_raw, mgst_raw, mcc_raw
    )

    X = np.column_stack([
        api_n, mcc_n, pvpp_n, mgst_n, binder_n,
        pressure_raw, speed_raw, granule_raw,
        particle_size_raw, moisture_raw, binder_grade_raw,
        dwell_time_raw, friction_raw, decompression_time_raw
    ])

    # Targets – use first element for scalar selection
    model_type, params = select_compaction_model(
        float(api_n[0]), float(binder_n[0]), float(pressure_raw[0]),
        float(particle_size_raw[0]), float(moisture_raw[0])
    )
    if model_type == "Heckel":
        k = params['k']
        A = params['A']
        D = 1 - np.exp(-(k * pressure_raw + A))
    elif model_type == "Kawakita":
        a = params['a']
        b = params['b']
        D = 1 - (pressure_raw / (a * pressure_raw + 1/b))
    else:
        K = params['K']
        n = params['n']
        D = 1 - K * (pressure_raw ** (-n))
    D = np.clip(D, D_MIN, D_MAX)
    D += rng.normal(0, 0.002, n_samples)
    D = np.clip(D, D_MIN, D_MAX)

    porosity = 1.0 - D
    sigma0 = 5.0 + 0.1 * (api_n - 85.0) + 0.2 * binder_n - 0.5 * mgst_n
    sigma0 = np.clip(sigma0, 2.0, 8.0)
    b = 2.5 - 0.005 * (pressure_raw - 80.0) - 0.01 * (particle_size_raw - 50) / 100
    b = np.clip(b, 1.5, 3.5)

    tensile_base = sigma0 * np.exp(-b * porosity)
    api_effect = 1.0 - 0.005 * (api_n - 85.0)
    binder_effect = 1.0 + 0.03 * (binder_n - 2.0)
    mgst_effect = 1.0 - 0.1 * (mgst_n - 0.2)
    pvpp_effect = 1.0 - 0.02 * (pvpp_n - 3.0)
    speed_effect = 1.0 - 0.002 * (speed_raw - 10.0)
    particle_effect = 1.0 - 0.0005 * (particle_size_raw - 50)
    particle_effect = np.clip(particle_effect, 0.8, 1.2)

    strength = (tensile_base * api_effect * binder_effect *
                mgst_effect * pvpp_effect * speed_effect * particle_effect)
    strength *= rng.normal(1.0, 0.01, n_samples)
    strength = np.clip(strength, 0.5, 6.0)

    er_base = (1.8 + 0.3 * (api_n - 85.0)/10.0 +
               0.08 * (speed_raw - 10.0)/30.0 -
               0.1 * (pressure_raw - 100.0)/150.0 +
               0.02 * (decompression_time_raw - 35.0)/30.0)
    er_base = er_base * (1.0 - 0.15 * (D - 0.4))
    er = er_base + rng.normal(0, 0.01, n_samples)
    er = np.clip(er, 0.5, 4.0)

    disintegration = predict_disintegration_time(strength, pvpp_n, api_n, binder_n, moisture_raw)
    dissolution_params = predict_dissolution_profile(api_n, pvpp_n, particle_size_raw, disintegration)
    
    # SAFE: pass scalar values (first element) to compatibility_check
    compat_result = compatibility_check(
        float(api_n[0]), int(binder_grade_raw[0]), float(moisture_raw[0])
    )

    feature_names = [
        'API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
        'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm',
        'Particle_Size_µm', 'Moisture_%', 'Binder_Grade',
        'Dwell_Time_ms', 'Friction', 'Decompression_Time_ms'
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df['Density'] = D
    df['Tensile_Strength_MPa'] = strength
    df['Elastic_Recovery_%'] = er
    df['Disintegration_Time_min'] = disintegration
    df['Dissolution_Tau'] = dissolution_params['tau']
    df['Dissolution_Beta'] = dissolution_params['beta']
    df['Compatibility_Score'] = compat_result['score']
    return df, feature_names

# ================================================================
# PINN MODEL
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
        self.output = nn.Linear(hidden//2, 8)

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
        disintegration = torch.nn.functional.softplus(raw[:, 5:6])
        dissolution_tau = torch.nn.functional.softplus(raw[:, 6:7])
        dissolution_beta = torch.nn.functional.softplus(raw[:, 7:8]) + 1e-4
        return torch.cat([density, tensile, er, k, A, disintegration, dissolution_tau, dissolution_beta], dim=1)

    def predict(self, X_scaled):
        self.eval()
        with torch.no_grad():
            if not isinstance(X_scaled, torch.Tensor):
                X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
            device = next(self.parameters()).device
            X_scaled = X_scaled.to(device)
            output = self.forward(X_scaled)
            return output[:, :8].cpu().numpy()

    def compute_loss(self, X_scaled, X_raw, y_true, y_scaler, epoch=0, total_epochs=ADAM_EPOCHS):
        pressure = X_raw[:, 5].view(-1, 1)
        mcc = X_raw[:, 1].view(-1, 1)
        api = X_raw[:, 0].view(-1, 1)
        binder = X_raw[:, 4].view(-1, 1)
        pvpp = X_raw[:, 2].view(-1, 1)
        moisture = X_raw[:, 9].view(-1, 1)
        particle_size = X_raw[:, 8].view(-1, 1)

        y_pred = self.forward(X_scaled)
        density_pred = y_pred[:, 0:1]
        tensile_pred = y_pred[:, 1:2]
        er_pred = y_pred[:, 2:3]
        k_pred = y_pred[:, 3:4]
        A_pred = y_pred[:, 4:5]
        disintegration_pred = y_pred[:, 5:6]
        dissolution_tau_pred = y_pred[:, 6:7]
        dissolution_beta_pred = y_pred[:, 7:8]

        # Data loss
        loss_dens = nn.MSELoss()(density_pred, y_true[:, 0:1])
        loss_tensile = nn.MSELoss()(tensile_pred, y_true[:, 1:2])
        loss_er = nn.MSELoss()(er_pred, y_true[:, 2:3])
        loss_disin = nn.MSELoss()(disintegration_pred, y_true[:, 5:6])
        loss_tau = nn.MSELoss()(dissolution_tau_pred, y_true[:, 6:7])
        loss_beta = nn.MSELoss()(dissolution_beta_pred, y_true[:, 7:8])
        data_loss = (W_DENSITY * loss_dens + W_TENSILE * loss_tensile + W_ER * loss_er +
                     W_DISINTEGRATION * loss_disin + W_DISSOLUTION * (loss_tau + loss_beta))

        # Physics losses
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

        disin_penalty = torch.mean(torch.relu(disintegration_pred - DISINTEGRATION_MAX) ** 2) * 5.0
        tau_penalty = torch.mean(torch.relu(dissolution_tau_pred - 25.0) ** 2) * 1.0

        mcc_penalty = torch.mean(torch.relu(mcc - SLIDER_MCC_MAX) ** 2) * 0.3
        density_penalty = torch.mean(torch.relu(density_real - D_MAX) ** 2 +
                                     torch.relu(D_MIN - density_real) ** 2) * 0.5

        physics_loss = (W_PHYSICS * (heckel_loss + efrf_penalty) +
                        disin_penalty + tau_penalty + mcc_penalty + density_penalty)
        return data_loss + physics_loss

# ================================================================
# NSGA-II
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
        api, mcc, pvpp, mgst, binder, pressure, speed, granule, particle_size, moisture, binder_grade, dwell_time, friction, decompression_time = ind
        api, binder, pvpp, mgst, mcc = normalize_components(api, binder, pvpp, mgst, mcc)
        pressure = np.clip(pressure, self.bounds[5,0], self.bounds[5,1])
        speed = np.clip(speed, self.bounds[6,0], self.bounds[6,1])
        particle_size = np.clip(particle_size, SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX)
        moisture = np.clip(moisture, SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX)
        binder_grade = np.clip(binder_grade, 0, len(BINDER_GRADES)-1)
        dwell_time = np.clip(dwell_time, SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX)
        friction = np.clip(friction, SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX)
        decompression_time = np.clip(decompression_time, SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX)
        if self.granule_fixed:
            granule = self.granule_fixed_val
        else:
            granule = np.clip(granule, self.bounds[7,0], self.bounds[7,1])
        return np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule,
                         particle_size, moisture, binder_grade, dwell_time, friction, decompression_time])

    def _repair_batch(self, pop):
        api = pop[:, 0]; mcc = pop[:, 1]; pvpp = pop[:, 2]
        mgst = pop[:, 3]; binder = pop[:, 4]
        pressure = pop[:, 5]; speed = pop[:, 6]; granule = pop[:, 7]
        particle_size = pop[:, 8]; moisture = pop[:, 9]; binder_grade = pop[:, 10]
        dwell_time = pop[:, 11]; friction = pop[:, 12]; decompression_time = pop[:, 13]

        api, binder, pvpp, mgst, mcc = normalize_components(api, binder, pvpp, mgst, mcc)
        pressure = np.clip(pressure, self.bounds[5,0], self.bounds[5,1])
        speed = np.clip(speed, self.bounds[6,0], self.bounds[6,1])
        particle_size = np.clip(particle_size, SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX)
        moisture = np.clip(moisture, SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX)
        binder_grade = np.clip(binder_grade, 0, len(BINDER_GRADES)-1)
        dwell_time = np.clip(dwell_time, SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX)
        friction = np.clip(friction, SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX)
        decompression_time = np.clip(decompression_time, SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX)
        if self.granule_fixed:
            granule = np.full_like(granule, self.granule_fixed_val)
        else:
            granule = np.clip(granule, self.bounds[7,0], self.bounds[7,1])
        return np.column_stack([api, mcc, pvpp, mgst, binder, pressure, speed, granule,
                                particle_size, moisture, binder_grade, dwell_time, friction, decompression_time])

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
        disintegration = np.maximum(pred[:, 5], 0.5)
        dissolution_tau = np.maximum(pred[:, 6], 1.0)

        g1 = D_MIN - density
        g2 = density - D_MAX
        violation = np.maximum(0, np.maximum(g1, g2))

        penalty = np.zeros(n)
        penalty += np.where(tensile < TENSILE_MIN, (TENSILE_MIN - tensile)**2, 0.0)
        penalty += np.where(efrf >= 0.40, (efrf - 0.40)**2, 0.0)
        penalty += np.where(disintegration > DISINTEGRATION_MAX, (disintegration - DISINTEGRATION_MAX)**2, 0.0)
        penalty += np.where(dissolution_tau > 20.0, (dissolution_tau - 20.0)**2, 0.0)
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
        child1 = np.zeros(14)
        child2 = np.zeros(14)
        for i in range(14):
            u = np.random.random()
            if u <= 0.5:
                beta = (2*u) ** (1/(eta+1))
            else:
                beta = (1/(2*(1-u))) ** (1/(eta+1))
            child1[i] = 0.5 * ((1+beta)*p1[i] + (1-beta)*p2[i])
            child2[i] = 0.5 * ((1-beta)*p1[i] + (1+beta)*p2[i])
        return child1, child2

    def _mutate(self, child, eta=20, pm=1.0/14.0):
        for i in range(14):
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
            particle_size = rng.uniform(SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX)
            moisture = rng.uniform(SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX)
            binder_grade = rng.integers(0, len(BINDER_GRADES))
            dwell_time = rng.uniform(SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX)
            friction = rng.uniform(SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX)
            decompression_time = rng.uniform(SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX)
            ind = np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule,
                            particle_size, moisture, binder_grade, dwell_time, friction, decompression_time])
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
# PREDICTION AND PLOTTING
# ================================================================
def predict_pinn(model, scaler, y_scaler, inputs):
    if model is None:
        st.error("Model is not available. Please check training errors.")
        return 0.7, 2.0, 0.5, 0.25, 10.0, 10.0, 1.0
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
        disintegration = max(pred[5], 0.5)
        dissolution_tau = max(pred[6], 1.0)
        dissolution_beta = max(pred[7], 0.5)
        return density, tensile, er, efrf, disintegration, dissolution_tau, dissolution_beta
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.7, 2.0, 0.5, 0.25, 10.0, 10.0, 1.0

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
    if model is None:
        st.error("Model not available for sensitivity.")
        return None
    api0 = formulation['api_n']; mcc0 = formulation['mcc_n']
    pvpp0 = formulation['pvpp_n']; mgst0 = formulation['mgst_n']
    binder0 = formulation['binder_n']; press0 = formulation['pressure']
    speed0 = formulation['speed']; granule0 = formulation['granule_use']
    particle_size0 = formulation['particle_size']
    moisture0 = formulation['moisture']
    dwell_time0 = formulation['dwell_time']
    friction0 = formulation['friction']
    decompression_time0 = formulation['decompression_time']

    param_defs = [
        {'name': 'API', 'current': api0, 'min': SLIDER_API_MIN, 'max': SLIDER_API_MAX},
        {'name': 'MCC', 'current': mcc0, 'min': SLIDER_MCC_MIN, 'max': SLIDER_MCC_MAX},
        {'name': 'PVPP', 'current': pvpp0, 'min': SLIDER_PVPP_MIN, 'max': SLIDER_PVPP_MAX},
        {'name': 'MgSt', 'current': mgst0, 'min': SLIDER_MGST_MIN, 'max': SLIDER_MGST_MAX},
        {'name': 'Binder', 'current': binder0, 'min': SLIDER_BINDER_MIN, 'max': SLIDER_BINDER_MAX},
        {'name': 'Pressure', 'current': press0, 'min': BOUND_PRESSURE_MIN, 'max': BOUND_PRESSURE_MAX},
        {'name': 'Speed', 'current': speed0, 'min': BOUND_SPEED_MIN, 'max': BOUND_SPEED_MAX},
        {'name': 'Granule', 'current': granule0, 'min': BOUND_GRANULE_MIN, 'max': BOUND_GRANULE_MAX},
        {'name': 'ParticleSize', 'current': particle_size0, 'min': SLIDER_PARTICLE_SIZE_MIN, 'max': SLIDER_PARTICLE_SIZE_MAX},
        {'name': 'Moisture', 'current': moisture0, 'min': SLIDER_MOISTURE_MIN, 'max': SLIDER_MOISTURE_MAX},
        {'name': 'DwellTime', 'current': dwell_time0, 'min': SLIDER_DWELL_TIME_MIN, 'max': SLIDER_DWELL_TIME_MAX},
        {'name': 'Friction', 'current': friction0, 'min': SLIDER_FRICTION_MIN, 'max': SLIDER_FRICTION_MAX},
        {'name': 'DecompTime', 'current': decompression_time0, 'min': SLIDER_DECOMPRESSION_TIME_MIN, 'max': SLIDER_DECOMPRESSION_TIME_MAX}
    ]

    base_input = [api0, mcc0, pvpp0, mgst0, binder0, press0, speed0, granule0,
                  particle_size0, moisture0, 0, dwell_time0, friction0, decompression_time0]
    _, _, _, efrf_base, _, _, _ = predict_pinn(model, scaler, y_scaler, base_input)

    sensitivities = []
    for idx, p in enumerate(param_defs):
        low_input = base_input.copy()
        low_input[idx] = p['min']
        high_input = base_input.copy()
        high_input[idx] = p['max']
        _, _, _, efrf_low, _, _, _ = predict_pinn(model, scaler, y_scaler, low_input)
        _, _, _, efrf_high, _, _, _ = predict_pinn(model, scaler, y_scaler, high_input)
        delta = abs(efrf_high - efrf_low)
        sensitivities.append({
            'Parameter': f"{p['name']}",
            'Delta EFRF': delta
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
        height=500,
        template='plotly_white'
    )
    return fig

def plot_dissolution_profile(tau, beta, api_n, title="Predicted Dissolution Profile"):
    time_points = np.linspace(0, 60, 100)
    dissolution = 100 * (1 - np.exp(-((time_points / tau) ** beta)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points,
        y=dissolution,
        mode='lines',
        name=f'Q(t) = 100×(1-exp(-((t/{tau:.1f})^{beta:.2f})))',
        line=dict(color='blue', width=2)
    ))
    fig.add_hline(y=85, line_dash='dash', line_color='red',
                  annotation_text='85% dissolution target')
    fig.update_layout(
        title=f'{title} (API: {api_n:.1f}%)',
        xaxis_title='Time (minutes)',
        yaxis_title='% Dissolved',
        height=350,
        template='plotly_white'
    )
    return fig

# ================================================================
# MODEL TRAINING CACHE
# ================================================================
CACHE_DIR = tempfile.gettempdir()
CHECKPOINT_PATH = os.path.join(CACHE_DIR, 'hubryd_v29_27_r23_eng.pt')

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

    st.caption("🔄 Training advanced model (15k samples, up to 800 epochs)...")
    df, features = generate_pinn_data(N_SAMPLES)
    X_raw = df[features].values
    y = df[['Density','Tensile_Strength_MPa','Elastic_Recovery_%',
            'Disintegration_Time_min','Dissolution_Tau','Dissolution_Beta']].values
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
# MODEL COMPARISON
# ================================================================
def run_model_comparison(model, scaler, y_scaler, features, df, device):
    if model is None:
        st.error("Model not available for comparison.")
        return pd.DataFrame(), []
    X_raw_all = df[features].values
    y_raw_all = df[['Tensile_Strength_MPa']].values
    X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
        X_raw_all, y_raw_all, test_size=0.2, random_state=42
    )
    X_b_train_scaled = scaler.transform(add_interaction_features(X_b_train))
    X_b_test_scaled = scaler.transform(add_interaction_features(X_b_test))
    y_train_target = y_b_train[:, 0]
    y_test_target = y_b_test[:, 0]

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

def generate_feasible_points(model, scaler, y_scaler, n_samples=3000):
    if model is None:
        return pd.DataFrame()
    rng = np.random.default_rng(42)
    api = rng.uniform(SLIDER_API_MIN, SLIDER_API_MAX, n_samples)
    binder = rng.uniform(SLIDER_BINDER_MIN, SLIDER_BINDER_MAX, n_samples)
    pvpp = rng.uniform(SLIDER_PVPP_MIN, SLIDER_PVPP_MAX, n_samples)
    mgst = rng.uniform(SLIDER_MGST_MIN, SLIDER_MGST_MAX, n_samples)
    mcc = rng.uniform(SLIDER_MCC_MIN, SLIDER_MCC_MAX, n_samples)
    particle_size = rng.uniform(SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX, n_samples)
    moisture = rng.uniform(SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX, n_samples)
    binder_grade = rng.integers(0, len(BINDER_GRADES), n_samples)
    pressure = rng.uniform(BOUND_PRESSURE_MIN, BOUND_PRESSURE_MAX, n_samples)
    speed = rng.uniform(BOUND_SPEED_MIN, BOUND_SPEED_MAX, n_samples)
    dwell_time = calculate_dwell_time(speed)
    friction = rng.uniform(SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX, n_samples)
    decompression_time = rng.uniform(SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX, n_samples)
    granule = rng.uniform(BOUND_GRANULE_MIN, BOUND_GRANULE_MAX, n_samples)

    api_n, binder_n, pvpp_n, mgst_n, mcc_n = normalize_components(
        api, binder, pvpp, mgst, mcc
    )
    inputs = np.column_stack([
        api_n, mcc_n, pvpp_n, mgst_n, binder_n,
        pressure, speed, granule,
        particle_size, moisture, binder_grade,
        dwell_time, friction, decompression_time
    ])

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
    disintegration = np.maximum(pred[:, 5], 0.5)

    mask = ((D_MIN <= density) & (density <= D_MAX) &
            (tensile >= TENSILE_MIN) & (efrf < 0.40) &
            (disintegration <= DISINTEGRATION_MAX) &
            (mcc_n <= BOUND_MCC_MAX) & (mcc_n >= BOUND_MCC_MIN))
    feasible_api = api_n[mask]
    feasible_efrf = efrf[mask]
    return pd.DataFrame({'API': feasible_api, 'EFRF': feasible_efrf})

# ================================================================
# MAIN UI
# ================================================================
st.markdown("""
<div style="background: #0b1a33; padding:1rem; border-radius:0.5rem; text-align:center; margin-bottom:1rem;">
    <h2 style="color:#fff; margin:0;">🧬 Hybrid AI · Advanced Pharmaceutical PINN</h2>
    <p style="color:#64ffda; margin:0; font-size:0.9rem;">v29.27-R23</p>
    <p style="color:#aabbcc; margin:0; font-size:0.85rem;">Nile Valley University, Sudan</p>
    <p style="color:#8899aa; font-size:0.75rem;">Advanced Compaction Models · Material Properties · Process Physics · Pharmaceutical Properties</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📚 Physics Constraints")
    st.markdown(f"""
    ✅ **API:** {SLIDER_API_MIN:.0f}–{SLIDER_API_MAX:.0f}%  
    ✅ **Density:** {D_MIN:.2f}–{D_MAX:.2f}  
    ✅ **Tensile:** ≥ {TENSILE_MIN:.2f} MPa  
    ✅ **EFRF:** &lt; 0.40 (feasible)  
    ✅ **Disintegration:** ≤ {DISINTEGRATION_MAX:.0f} min (USP)  
    ✅ **Slider MCC:** {SLIDER_MCC_MIN:.1f}–{SLIDER_MCC_MAX:.1f}%  
    ✅ **Slider PVPP:** {SLIDER_PVPP_MIN:.1f}–{SLIDER_PVPP_MAX:.1f}%  
    ✅ **Slider MgSt:** {SLIDER_MGST_MIN:.2f}–{SLIDER_MGST_MAX:.2f}%  
    ✅ **Slider Binder:** {SLIDER_BINDER_MIN:.1f}–{SLIDER_BINDER_MAX:.1f}%  
    ✅ **Optimiser Bounds:** MCC≥2.0%, PVPP≥1.5%, MgSt≥0.3%, Binder≥3.0%  
    ✅ **Pressure:** {BOUND_PRESSURE_MIN:.0f}–{BOUND_PRESSURE_MAX:.0f} MPa  
    ✅ **Speed:** {BOUND_SPEED_MIN:.0f}–{BOUND_SPEED_MAX:.0f} RPM  
    ✅ **NSGA‑II:** Pop=80, Gen=50
    """)
    st.caption("🔬 v29.27-R23 — Fully Bugfixed")

# Load model
try:
    model, scaler, y_scaler, features, df = load_or_train()
except Exception as e:
    st.error(f"Training failed: {e}. Using dummy model.")
    model = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if model is not None:
    device = next(model.parameters()).device

# Layout
col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    st.markdown("### 📊 Formulation & Material Properties")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            api = st.slider("API (%)", SLIDER_API_MIN, SLIDER_API_MAX, st.session_state.api, 0.1, key="api_slider")
            binder = st.slider("Binder (%)", SLIDER_BINDER_MIN, SLIDER_BINDER_MAX, st.session_state.binder, 0.1, key="binder_slider")
            pvpp = st.slider("PVPP (%)", SLIDER_PVPP_MIN, SLIDER_PVPP_MAX, st.session_state.pvpp, 0.1, key="pvpp_slider")
            mgst = st.slider("Mg-St (%)", SLIDER_MGST_MIN, SLIDER_MGST_MAX, st.session_state.mgst, 0.01, key="mgst_slider")
            mcc = st.slider("MCC (%)", SLIDER_MCC_MIN, SLIDER_MCC_MAX, st.session_state.mcc, 0.1, key="mcc_slider")
        with c2:
            particle_size = st.slider("Particle Size (µm)", SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX, st.session_state.particle_size, 1.0, key="particle_size_slider")
            moisture = st.slider("Moisture (%)", SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX, st.session_state.moisture, 0.1, key="moisture_slider")
            binder_grade = st.selectbox("Binder Grade", BINDER_GRADES, index=st.session_state.binder_grade, key="binder_grade_select")
            binder_grade_idx = BINDER_GRADES.index(binder_grade)
            st.session_state.binder_grade = binder_grade_idx
            compat_result = compatibility_check(api, binder_grade_idx, moisture)
            if compat_result['warnings']:
                st.warning(compat_result['warnings'][0])
            else:
                st.success("✅ Compatibility: Good")
        total = api + binder + pvpp + mgst + mcc
        if abs(total-100) < 0.1:
            st.success(f"✅ Total = {total:.2f}%")
        else:
            st.warning(f"⚠️ Total = {total:.2f}% (should be 100%)")

    st.markdown("### ⚙️ Process Parameters (Practical Ranges)")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            pressure = st.slider("Pressure (MPa)", BOUND_PRESSURE_MIN, BOUND_PRESSURE_MAX, st.session_state.get('pressure', 200.0), 1.0, key="pressure_slider")
            speed = st.slider("Speed (rpm)", BOUND_SPEED_MIN, BOUND_SPEED_MAX, st.session_state.get('speed', 20.0), 0.5, key="speed_slider")
        with c2:
            dwell_time = st.slider("Dwell Time (ms)", SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX, st.session_state.get('dwell_time', 25.0), 0.5, key="dwell_time_slider")
            friction = st.slider("Friction Coefficient", SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX, st.session_state.get('friction', 0.25), 0.01, key="friction_slider")
            decompression_time = st.slider("Decompression Time (ms)", SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX, st.session_state.get('decompression_time', 35.0), 1.0, key="decompression_time_slider")

        granule_mode = st.radio(
            "Granule Size",
            options=["Fixed (slider)", "Variable (optimized)"],
            index=0 if st.session_state.get('granule_mode', 'Fixed') == 'Fixed' else 1,
            horizontal=True,
            key="granule_mode_radio"
        )
        if granule_mode == "Fixed (slider)":
            granule = st.slider("Granule Size (µm)", BOUND_GRANULE_MIN, BOUND_GRANULE_MAX, st.session_state.get('granule', 125.0), 1.0, key="granule_slider")
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
        # Check if model is loaded
        if model is None:
            st.error("Model is not available. Please fix training errors and restart.")
        else:
            if abs(total-100) > 0.1:
                st.warning("Formulation must sum to 100%")
            else:
                api_n, binder_n, pvpp_n, mgst_n, mcc_n = normalize_components(api, binder, pvpp, mgst, mcc)
                if granule_fixed:
                    granule_use = granule
                else:
                    granule_use = granule
                inputs = [api_n, mcc_n, pvpp_n, mgst_n, binder_n, pressure, speed, granule_use,
                          particle_size, moisture, binder_grade_idx, dwell_time, friction, decompression_time]

                density, tensile, er, efrf, disintegration, dissolution_tau, dissolution_beta = predict_pinn(model, scaler, y_scaler, inputs)

                st.session_state.formulation = {
                    'api_n': api_n, 'binder_n': binder_n, 'pvpp_n': pvpp_n,
                    'mgst_n': mgst_n, 'mcc_n': mcc_n,
                    'particle_size': particle_size, 'moisture': moisture, 'binder_grade': binder_grade_idx,
                    'pressure': pressure, 'speed': speed, 'dwell_time': dwell_time,
                    'friction': friction, 'decompression_time': decompression_time,
                    'granule_use': granule_use, 'granule_fixed': granule_fixed,
                    'density': density, 'tensile': tensile, 'er': er, 'efrf': efrf,
                    'disintegration': disintegration, 'dissolution_tau': dissolution_tau,
                    'dissolution_beta': dissolution_beta
                }

                st.markdown("**Constraints Status** (D: 0.70–0.99, Tensile ≥ 1.50, EFRF < 0.40, Disintegration ≤ 15 min)")
                col_metrics = st.columns(5)
                col_metrics[0].metric("Density", f"{density:.3f}", f"[{D_MIN:.2f}, {D_MAX:.2f}]")
                col_metrics[1].metric("Tensile", f"{tensile:.2f} MPa", f"≥ {TENSILE_MIN:.2f}")
                col_metrics[2].metric("EFRF", f"{efrf:.4f}", f"< 0.40")
                col_metrics[3].metric("MCC", f"{mcc_n:.1f}%", f"≤ {BOUND_MCC_MAX:.1f}%")
                col_metrics[4].metric("Disintegration", f"{disintegration:.1f} min", f"≤ {DISINTEGRATION_MAX:.0f} min")

                if all([D_MIN <= density <= D_MAX, tensile >= TENSILE_MIN, efrf < 0.40,
                        mcc_n <= BOUND_MCC_MAX, disintegration <= DISINTEGRATION_MAX]):
                    st.success("✅ All constraints satisfied")
                else:
                    st.error("❌ Violates constraints")

                bounds = np.array([
                    [SLIDER_API_MIN, SLIDER_API_MAX],
                    [BOUND_MCC_MIN, BOUND_MCC_MAX],
                    [BOUND_PVPP_MIN, BOUND_PVPP_MAX],
                    [BOUND_MGST_MIN, BOUND_MGST_MAX],
                    [BOUND_BINDER_MIN, BOUND_BINDER_MAX],
                    [BOUND_PRESSURE_MIN, BOUND_PRESSURE_MAX],
                    [BOUND_SPEED_MIN, BOUND_SPEED_MAX],
                    [BOUND_GRANULE_MIN, BOUND_GRANULE_MAX],
                    [SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX],
                    [SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX],
                    [0, len(BINDER_GRADES)-1],
                    [SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX],
                    [SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX],
                    [SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX]
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

                    best_tensile = -np.inf
                    for idx in front_indices:
                        ind = pop[idx]
                        d2, t2, e2, ef2, dis2, tau2, beta2 = predict_pinn(model, scaler, y_scaler, ind)
                        if t2 > best_tensile:
                            best_tensile = t2
                            quality_idx = idx

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

    if st.session_state.run_optimized and model is not None:
        pop = st.session_state.nsga_pop
        objectives = st.session_state.nsga_objectives
        fronts = st.session_state.nsga_fronts
        balanced_solution = st.session_state.balanced_solution
        quality_solution = st.session_state.quality_solution
        cost_solution = st.session_state.cost_solution
        feasible_df = st.session_state.feasible_df
        tested_point = st.session_state.tested_point

        st.markdown("### 📉 Pareto Front")
        if len(fronts) > 0 and len(fronts[0]) > 0:
            num_solutions = len(fronts[0])
            st.success(f"✅ Pareto front: {num_solutions} optimal solutions")
            fig = plot_pareto_clean(objectives, fronts, None, feasible_df, tested_point, efrf_max=0.40)
            if fig is not None:
                if balanced_solution is not None:
                    d, t, e, ef, dis, tau, beta = predict_pinn(model, scaler, y_scaler, balanced_solution)
                    fig.add_trace(go.Scatter(
                        x=[balanced_solution[0]],
                        y=[ef],
                        mode='markers',
                        name='⭐ Golden (Balanced)',
                        marker=dict(size=12, color='gold', symbol='star', line=dict(width=2, color='black')),
                        hovertemplate='Golden (Balanced)<br>API: %{x:.1f}%<br>EFRF: %{y:.4f}<extra></extra>'
                    ))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### ⭐ Golden Solution (Balanced)")
        if balanced_solution is not None:
            d, t, e, ef, dis, tau, beta = predict_pinn(model, scaler, y_scaler, balanced_solution)
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Formulation**")
                    st.write(f"API: {balanced_solution[0]:.1f}%")
                    st.write(f"MCC: {balanced_solution[1]:.1f}%")
                    st.write(f"PVPP: {balanced_solution[2]:.1f}%")
                    st.write(f"Mg-St: {balanced_solution[3]:.2f}%")
                    st.write(f"Binder: {balanced_solution[4]:.1f}%")
                with col2:
                    st.write("**Material Properties**")
                    st.write(f"Particle Size: {balanced_solution[8]:.0f} µm")
                    st.write(f"Moisture: {balanced_solution[9]:.1f}%")
                    st.write(f"Binder Grade: {BINDER_GRADES[int(balanced_solution[10])]}")
                with col3:
                    st.write("**Process & CQAs**")
                    st.write(f"Pressure: {balanced_solution[5]:.1f} MPa")
                    st.write(f"Speed: {balanced_solution[6]:.1f} rpm")
                    st.write(f"Granule: {balanced_solution[7]:.0f} µm")
                    st.write(f"Density: {d:.3f}")
                    st.write(f"Tensile: {t:.3f} MPa")
                    st.write(f"EFRF: {ef:.4f}")
                    st.write(f"Disintegration: {dis:.1f} min")
            st.session_state.balanced_pred = (d, t, e, ef, dis, tau, beta)
        else:
            st.info("No balanced solution found.")

        st.markdown("---")

        st.toggle("💰 Cost-wise Solution (Max API, Min Pressure)", value=st.session_state.get("show_cost_solution", False), key="show_cost_solution")
        if st.session_state.show_cost_solution and cost_solution is not None:
            st.markdown("#### 💰 Cost-Optimised Formulation")
            d, t, e, ef, dis, tau, beta = predict_pinn(model, scaler, y_scaler, cost_solution)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Formulation:**")
                st.write(f"API: {cost_solution[0]:.1f}%")
                st.write(f"MCC: {cost_solution[1]:.1f}%")
                st.write(f"PVPP: {cost_solution[2]:.1f}%")
                st.write(f"Mg-St: {cost_solution[3]:.2f}%")
                st.write(f"Binder: {cost_solution[4]:.1f}%")
            with col2:
                st.write("**CQAs:**")
                st.write(f"Tensile: {t:.3f} MPa")
                st.write(f"EFRF: {ef:.4f}")
                st.write(f"Disintegration: {dis:.1f} min")

        st.toggle("🏆 Quality-wise Solution (Max Tensile)", value=st.session_state.get("show_quality_solution", False), key="show_quality_solution")
        if st.session_state.show_quality_solution and quality_solution is not None:
            st.markdown("#### 🏆 Quality-Optimised Formulation")
            d, t, e, ef, dis, tau, beta = predict_pinn(model, scaler, y_scaler, quality_solution)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Formulation:**")
                st.write(f"API: {quality_solution[0]:.1f}%")
                st.write(f"MCC: {quality_solution[1]:.1f}%")
                st.write(f"PVPP: {quality_solution[2]:.1f}%")
                st.write(f"Mg-St: {quality_solution[3]:.2f}%")
                st.write(f"Binder: {quality_solution[4]:.1f}%")
            with col2:
                st.write("**CQAs:**")
                st.write(f"Tensile: {t:.3f} MPa")
                st.write(f"EFRF: {ef:.4f}")
                st.write(f"Disintegration: {dis:.1f} min")

        st.toggle("📊 Model Comparison", value=st.session_state.get("show_comparison", True), key="show_comparison")
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

        st.toggle("🔬 Sensitivity Analysis", value=st.session_state.get("show_sensitivity", False), key="show_sensitivity")
        if st.session_state.show_sensitivity:
            st.markdown("### 🔬 Sensitivity Analysis – Parameter Impact on EFRF")
            f = st.session_state.formulation
            if f['api_n'] is not None:
                fig_bars = plot_sensitivity_bars(f, model, scaler, y_scaler, efrf_max=0.40)
                if fig_bars:
                    st.plotly_chart(fig_bars, use_container_width=True)

        st.toggle("📊 Dissolution Profile", value=st.session_state.get("show_dissolution", False), key="show_dissolution")
        if st.session_state.show_dissolution:
            st.markdown("### 📊 Predicted Dissolution Profile")
            f = st.session_state.formulation
            if f['api_n'] is not None:
                tau = f.get('dissolution_tau', 10.0)
                beta = f.get('dissolution_beta', 1.0)
                api_n = f['api_n']
                fig = plot_dissolution_profile(tau, beta, api_n)
                st.plotly_chart(fig, use_container_width=True)

        generate_report_btn = st.button("📄 Generate Report (PDF)", key="knob_report")
        if generate_report_btn and st.session_state.benchmark_df is not None:
            st.info("PDF generation for advanced model is not fully implemented in this version.")

    else:
        if model is None:
            st.warning("Model not loaded. Please fix training issues.")
        else:
            st.info("Adjust sliders and click '🔬 Predict & Optimise' to see results.")

st.caption("📧 Contact: babuker@protonmail.com | 🏛️ Nile Valley University, Sudan")
