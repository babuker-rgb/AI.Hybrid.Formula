"""
True Physics-Informed Neural Network (PINN) - Complete Professional Framework
Multi-Objective Tablet Manufacturing Optimization with Full Analytics

Author: Babuker A. Abdalla
Affiliation: Nile Valley University, Postgraduate College, Sudan
Version: 29.11 (Enhanced with Ryshkewitch-Duckworth Physics + Soft Sigmoid Scheduling)
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from fpdf import FPDF
import datetime
import warnings
import plotly.graph_objects as go
import plotly.express as px
import time
import math
warnings.filterwarnings('ignore')

# ================================================================
# 0. USER-CONFIGURABLE PARAMETERS (v29.11 ENHANCED)
# ================================================================

TENSILE_MIN = 1.90          # MPa
EFRF_MAX = 0.40             # dimensionless
MCC_MAX = 8.0               # %
D_MIN = 0.40                # Physical lower bound
D_MAX = 0.97                # Physical upper bound
DENSITY_TOL = 1e-6          # Tolerance for floating-point errors
PRESSURE_MAX = 300.0        # MPa

BINDER_MIN = 0.5
BINDER_MAX = 5.0

# --- Enhanced Hyperparameters ---
N_SAMPLES = 5000            # More samples for complex physics
ADAM_EPOCHS = 900           # Increased for better convergence with new physics
LBFGS_STEPS = 3             
MONOTONICITY_FREQUENCY = 10 

# --- NSGA-II Toggle ---
USE_FINAL_NSGA = False      
if USE_FINAL_NSGA:
    NSGA_POP_SIZE = 120
    NSGA_GENERATIONS = 120
else:
    NSGA_POP_SIZE = 80      
    NSGA_GENERATIONS = 60   

# --- Loss Weights ---
W_DENSITY = 12.0            
TENSILE_DATA_WEIGHT = 2.5   
TENSILE_PHYSICS_WEIGHT = 2.0 
TENSILE_BINDER_WEIGHT = 0.5 

# --- Physics / Data Balance (Initial/Final) ---
W_DATA_INIT = 1.0          
W_PHYSICS_INIT = 0.3       
W_DATA_FINAL = 1.0         
W_PHYSICS_FINAL = 0.7      

# --- Ryshkewitch-Duckworth Constants ---
RYSK_SIGMA0_LOG = 1.2       # ln(σ0) ~ 1.2
RYSK_B = 2.0                # b ~ 2.0

# --- Safety Penalty (Soft) ---
SAFETY_PENALTY_WEIGHT = 5.0     
SAFETY_DENSITY_LIMIT = 0.970    
SAFETY_EFRF_LIMIT = 0.390       
EFRF_PENALTY_WEIGHT = 2.0       
DENSITY_PREFERENCE_WEIGHT = 0.5 
DENSITY_PREFERENCE_LIMIT = 0.95 

PHYSICS_WEIGHT_INIT = 0.3  

# ================================================================
# 1. SAFE IMPORTS
# ================================================================

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ================================================================
# 2. SESSION STATE (UNCHANGED)
# ================================================================

DEFAULTS = {
    'api': 90.5,
    'binder': 2.7,
    'pvpp': 3.0,
    'mgst': 0.20,
    'mcc': 5.0,
    'pressure': 230.0,
    'speed': 12.0,
    'granule': 125.0
}

RANGES = {
    'api': (85.0, 95.0),
    'binder': (BINDER_MIN, BINDER_MAX),
    'pvpp': (0.5, 6.0),
    'mgst': (0.01, 1.2),
    'mcc': (0.0, MCC_MAX),
    'pressure': (80.0, PRESSURE_MAX),
    'speed': (1.0, 50.0),
    'granule': (30.0, 250.0)
}

def safe_initialize():
    for key in DEFAULTS:
        if key not in st.session_state:
            st.session_state[key] = DEFAULTS[key]
        else:
            try:
                float(st.session_state[key])
            except (ValueError, TypeError):
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
# 3. DYNAMIC NORMALIZATION & FEATURE ENGINEERING
# ================================================================

def normalize_components(api, binder, pvpp, mgst, mcc):
    api = max(api, 0.1)
    binder = max(binder, 0.1)
    pvpp = max(pvpp, 0.1)
    mgst = max(mgst, 0.01)
    mcc = max(mcc, 0.1)
    
    api = min(api, 100.0)
    binder = min(binder, 15.0)
    pvpp = min(pvpp, 15.0)
    mgst = min(mgst, 3.0)
    mcc = min(mcc, 25.0)
    
    total = api + binder + pvpp + mgst + mcc
    if total <= 0:
        total = 1.0
    
    api_norm = (api / total) * 100
    binder_norm = (binder / total) * 100
    pvpp_norm = (pvpp / total) * 100
    mgst_norm = (mgst / total) * 100
    mcc_norm = (mcc / total) * 100
    
    if mcc_norm > MCC_MAX:
        excess = mcc_norm - MCC_MAX
        mcc_norm = MCC_MAX
        other_sum = api_norm + binder_norm + pvpp_norm + mgst_norm
        if other_sum > 0:
            api_norm += excess * (api_norm / other_sum)
            binder_norm += excess * (binder_norm / other_sum)
            pvpp_norm += excess * (pvpp_norm / other_sum)
            mgst_norm += excess * (mgst_norm / other_sum)
    
    if api_norm < 85.0:
        deficit = 85.0 - api_norm
        api_norm = 85.0
        other_sum = binder_norm + pvpp_norm + mgst_norm
        if other_sum > 0:
            binder_norm -= deficit * (binder_norm / other_sum) if binder_norm > 0 else 0
            pvpp_norm -= deficit * (pvpp_norm / other_sum) if pvpp_norm > 0 else 0
            mgst_norm -= deficit * (mgst_norm / other_sum) if mgst_norm > 0 else 0
    
    if api_norm > 95.0:
        excess = api_norm - 95.0
        api_norm = 95.0
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
        api_norm *= scale
        binder_norm *= scale
        pvpp_norm *= scale
        mgst_norm *= scale
        mcc_norm *= scale
    
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
    
    pressure_speed = np.clip(pressure / (speed + 0.1), 0, 1000)
    api_mcc = np.clip(api / (mcc + 0.1), 0, 1000)
    binder_speed = np.clip(binder / (speed + 0.1), 0, 100)
    
    pressure_binder = pressure * binder
    pressure_api = pressure * api
    
    return np.concatenate([
        X_raw, pressure_binder, pressure_api, 
        pressure_speed, api_mcc, binder_speed
    ], axis=1)

# ================================================================
# 4. MULTI-TASK TRUE PINN MODEL (ENHANCED PHYSICS v29.11)
# ================================================================

def bounded_density(raw):
    return D_MIN + (D_MAX - D_MIN) * torch.sigmoid(raw)

class MultiTaskTruePINN(nn.Module):
    def __init__(self, input_dim=13, output_dim=5):
        super(MultiTaskTruePINN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 384), nn.LayerNorm(384), nn.Tanh(), nn.Dropout(0.05),
            nn.Linear(384, 384), nn.LayerNorm(384), nn.Tanh(), nn.Dropout(0.05),
            nn.Linear(384, 192), nn.LayerNorm(192), nn.Tanh(),
            nn.Linear(192, output_dim)
        )

    def forward(self, X):
        raw = self.network(X)
        density = bounded_density(raw[:, 0:1])
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
            output = self.forward(X_scaled)
            return output[:, :3].cpu().numpy()

    def compute_loss(self, X_scaled, X_raw, y_true, epoch=0, max_epochs=ADAM_EPOCHS,
                     w_data_init=W_DATA_INIT, w_physics_init=W_PHYSICS_INIT,
                     w_data_final=W_DATA_FINAL, w_physics_final=W_PHYSICS_FINAL,
                     w_mcc=0.5, w_density=W_DENSITY,
                     efrf_target=EFRF_MAX, mcc_max=MCC_MAX,
                     compute_grad=True):
        pressure_real = X_raw[:, 5].view(-1, 1)
        mcc_real = X_raw[:, 1].view(-1, 1)
        binder_real = X_raw[:, 4].view(-1, 1)

        y_pred = self.forward(X_scaled)
        density_pred = y_pred[:, 0:1]
        tensile_pred = y_pred[:, 1:2]
        er_pred = y_pred[:, 2:3]
        k_pred = y_pred[:, 3:4]
        A_pred = y_pred[:, 4:5]

        # --- SOFT SIGMOID SCHEDULING (FIXED) ---
        t = epoch / max_epochs
        sigmoid_val = 1 / (1 + math.exp(-10 * (t - 0.4)))
        w_data = w_data_init + (w_data_final - w_data_init) * sigmoid_val
        w_physics = w_physics_init + (w_physics_final - w_physics_init) * sigmoid_val
        w_data = max(w_data, 0.1)
        w_physics = max(w_physics, 0.1)

        # --- Data Loss (Weighted Tensile) ---
        density_data_loss = nn.MSELoss()(density_pred, y_true[:, 0:1])
        tensile_data_loss = nn.MSELoss()(tensile_pred, y_true[:, 1:2])
        er_data_loss = nn.MSELoss()(er_pred, y_true[:, 2:3])
        data_loss = density_data_loss + TENSILE_DATA_WEIGHT * tensile_data_loss + er_data_loss

        # --- HECKEL EQUATION ---
        heckel_lhs = torch.log(1.0 / torch.clamp(1.0 - density_pred, min=1e-4))
        heckel_rhs = k_pred * pressure_real + A_pred
        heckel_loss = torch.mean((heckel_lhs - heckel_rhs) ** 2)

        # --- EFRF BOUND ---
        efrf_pred = er_pred / torch.clamp(tensile_pred, min=1e-4)
        efrf_loss = torch.mean(torch.relu(efrf_pred - efrf_target) ** 2)
        efrf_safe_loss = torch.mean(torch.relu(efrf_pred - 0.36) ** 2) * EFRF_PENALTY_WEIGHT

        # --- RYSZHKEWITCH-DUCKWORTH PHYSICS ---
        log_tensile_pred = torch.log(tensile_pred + 1e-6)
        porosity = 1.0 - density_pred
        target_log_tensile = RYSK_SIGMA0_LOG - RYSK_B * porosity
        ryshkewitch_loss = torch.mean((log_tensile_pred - target_log_tensile) ** 2) * 0.5

        # --- TENSILE-BINDER RELATION ---
        tensile_binder_loss = torch.mean(torch.relu(0.05 - (tensile_pred * binder_real)) ** 2) * TENSILE_BINDER_WEIGHT

        # --- MCC BOUND & DENSITY PREFERENCES ---
        mcc_loss = torch.mean(torch.relu(mcc_real - mcc_max) ** 2)
        density_penalty = torch.mean(
            torch.relu(density_pred - D_MAX) ** 2 + torch.relu(D_MIN - density_pred) ** 2
        )
        density_preference_loss = torch.mean(torch.relu(density_pred - DENSITY_PREFERENCE_LIMIT) ** 2) * DENSITY_PREFERENCE_WEIGHT

        # --- MONOTONICITY (∂D/∂P > 0) ---
        if compute_grad:
            Xg = X_scaled.clone().detach().requires_grad_(True)
            yg = self.forward(Xg)
            dg = yg[:, 0:1]
            grad = torch.autograd.grad(
                outputs=dg,
                inputs=Xg,
                grad_outputs=torch.ones_like(dg),
                create_graph=True,
                retain_graph=True
            )[0]
            monotonic_loss = torch.mean(torch.relu(-grad[:, 5:6]) ** 2)
        else:
            monotonic_loss = torch.tensor(0.0, device=X_scaled.device)

        # --- BOUNDARY SMOOTHNESS ---
        boundary_loss = (
            torch.mean(torch.relu((D_MIN + 0.1) - density_pred) ** 2) +
            torch.mean(torch.relu(density_pred - D_MAX) ** 2)
        )

        # --- Regularization for k and A ---
        k_reg = torch.mean(torch.relu(k_pred - 0.1) ** 2) + torch.mean(torch.relu(0.005 - k_pred) ** 2)
        A_reg = torch.mean(torch.relu(A_pred - 2.0) ** 2) + torch.mean(torch.relu(0.5 - A_pred) ** 2)

        # --- TOTAL LOSS ---
        total_loss = (
            w_data * data_loss +
            w_physics * (heckel_loss + efrf_loss + monotonic_loss + boundary_loss + 
                         tensile_binder_loss + ryshkewitch_loss) +
            w_mcc * mcc_loss +
            w_density * density_penalty +
            efrf_safe_loss +
            density_preference_loss +
            0.1 * k_reg + 0.1 * A_reg
        )

        # --- Loss Dictionary (FIXED: corrected key spelling) ---
        loss_dict = {
            'data_loss': data_loss.item(),
            'tensile_data_loss': tensile_data_loss.item(),
            'heckel_loss': heckel_loss.item(),
            'efrf_loss': efrf_loss.item(),
            'ryshkewitch_loss': ryshkewitch_loss.item(),  # <--- FIXED
            'tensile_binder_loss': tensile_binder_loss.item(),
            'mcc_loss': mcc_loss.item(),
            'density_penalty': density_penalty.item(),
            'monotonic_loss': monotonic_loss.item() if compute_grad else 0.0,
            'boundary_loss': boundary_loss.item(),
            'w_data': w_data,
            'w_physics': w_physics,
            'total_loss': total_loss.item()
        }
        return total_loss, loss_dict

# ================================================================
# 5. DATA GENERATION (ENHANCED - NONLINEAR TENSILE)
# ================================================================

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

        # Heckel physics for density
        x = np.random.uniform(x_min, x_max)
        max_trials = 30
        for _ in range(max_trials):
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

        noise_d = np.random.normal(0, 0.01)
        D = np.clip(D_target + noise_d, D_MIN, D_MAX)

        # --- ENHANCED TENSILE STRENGTH (NONLINEAR INTERACTIONS) ---
        strength = (
            3.5 
            - 0.15 * (api - 85) 
            + 0.3 * binder 
            + 0.008 * (pressure - 100)
            - 1.5 * mgst 
            - 0.02 * (speed - 10)
            + 0.05 * (api - 85) * binder      
            - 0.03 * (pressure - 100) * mgst   
            + 0.01 * binder * (pressure - 100) / 100
            + np.random.normal(0, 0.04)
        )
        strength = np.clip(strength, 0.5, 6.0)
        
        er = np.clip(
            1.8 + 0.3 * (api - 85)/10 + 0.08 * (speed - 10)/30
            - 0.1 * (pressure - 100)/150 + np.random.normal(0, 0.03),
            0.5, 4.0
        )

        y[i] = [D, strength, er]

    feature_names = ['API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
                     'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm']
    df = pd.DataFrame(X, columns=feature_names)
    df['Density'] = y[:, 0]
    df['Tensile_Strength_MPa'] = y[:, 1]
    df['Elastic_Recovery_%'] = y[:, 2]
    return df, feature_names

# ================================================================
# 6. PDF GENERATION (UNCHANGED)
# ================================================================

def sanitize_text(text):
    replacements = {'σ': 'sigma', 'µ': 'um', '≥': '>=', '≤': '<=',
                    '✅': '[PASS]', '❌': '[FAIL]', '⚠️': '[WARNING]'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

@st.cache_data
def generate_full_pdf_report(api, mcc, pvpp, mgst, binder, pressure, speed, granule,
                             density, tensile, er, efrf, status, timestamp,
                             model_comparison_df=None):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, sanitize_text("Formulation Optimization Report"), ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 6, sanitize_text("Hybrid AI Framework (PINN + NSGA-II) - v29.11"), ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Date: {timestamp}", ln=True, align="C")
    pdf.ln(4)
    
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 0, 150)
    pdf.cell(0, 8, "Chem. Eng. Babuker A. Abdalla, PhD Researcher", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, "Nile Valley University, Postgraduate College, Sudan", ln=True, align="C")
    pdf.ln(5)
    
    # 1. Formulation Summary
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, sanitize_text("1. Formulation Summary"), ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    
    components = [("API", f"{api:.1f}%", "Paracetamol"),
                  ("MCC", f"{mcc:.1f}%", "Filler/Binder"),
                  ("PVPP", f"{pvpp:.1f}%", "Superdisintegrant"),
                  ("Mg-St", f"{mgst:.2f}%", "Lubricant"),
                  ("Binder", f"{binder:.1f}%", "Binding Agent"),
                  ("TOTAL", f"{api+binder+pvpp+mgst+mcc:.1f}%", "100% Complete")]
    
    pdf.set_font("Arial", "B", 10)
    pdf.cell(50, 6, sanitize_text("Component"), 1, 0, "C")
    pdf.cell(30, 6, sanitize_text("Value"), 1, 0, "C")
    pdf.cell(80, 6, sanitize_text("Function"), 1, 1, "C")
    
    pdf.set_font("Arial", "", 10)
    for comp, val, func in components:
        pdf.cell(50, 6, sanitize_text(comp), 1, 0, "L")
        pdf.cell(30, 6, sanitize_text(val), 1, 0, "C")
        pdf.cell(80, 6, sanitize_text(func), 1, 1, "L")
    
    pdf.ln(4)
    
    # 2. Process Parameters
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, sanitize_text("2. Process Parameters"), ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    
    params = [("Compaction Pressure", f"{pressure:.1f} MPa"),
              ("Punch Speed", f"{speed:.1f} rpm"),
              ("Granule Size", f"{granule:.1f} um")]
    
    pdf.set_font("Arial", "B", 10)
    pdf.cell(60, 6, sanitize_text("Parameter"), 1, 0, "C")
    pdf.cell(60, 6, sanitize_text("Value"), 1, 1, "C")
    
    pdf.set_font("Arial", "", 10)
    for p, v in params:
        pdf.cell(60, 6, sanitize_text(p), 1, 0, "L")
        pdf.cell(60, 6, sanitize_text(v), 1, 1, "C")
    
    pdf.ln(4)
    
    # 3. Prediction Results
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, sanitize_text("3. Prediction Results"), ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    
    results = [("Density", f"{density:.3f}", f"[{D_MIN:.2f}, {D_MAX:.2f}]"),
               ("Tensile Strength", f"{tensile:.3f} MPa", f">= {TENSILE_MIN:.2f} MPa"),
               ("Elastic Recovery", f"{er:.3f} %", "-"),
               ("EFRF", f"{efrf:.4f}", f"< {EFRF_MAX:.2f}")]
    
    pdf.set_font("Arial", "B", 10)
    pdf.cell(45, 6, sanitize_text("Metric"), 1, 0, "C")
    pdf.cell(35, 6, sanitize_text("Value"), 1, 0, "C")
    pdf.cell(45, 6, sanitize_text("Threshold"), 1, 1, "C")
    
    pdf.set_font("Arial", "", 10)
    for r in results:
        if len(r) == 2:
            pdf.cell(45, 6, sanitize_text(r[0]), 1, 0, "L")
            pdf.cell(35, 6, sanitize_text(r[1]), 1, 0, "C")
            pdf.cell(45, 6, "-", 1, 1, "C")
        else:
            pdf.cell(45, 6, sanitize_text(r[0]), 1, 0, "L")
            pdf.cell(35, 6, sanitize_text(r[1]), 1, 0, "C")
            pdf.cell(45, 6, sanitize_text(r[2]), 1, 1, "C")
    
    pdf.ln(4)
    
    # 4. Overall Status
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, sanitize_text("4. Overall Status"), ln=True, fill=True)
    pdf.set_font("Arial", "B", 14)
    if status == "PASS":
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 8, sanitize_text("PASS - Formulation Satisfies All Constraints"), ln=True, align="C")
    else:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 8, sanitize_text("FAIL - Formulation Does NOT Satisfy All Constraints"), ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)
    
    # 5. Model Comparison
    if model_comparison_df is not None and not model_comparison_df.empty:
        pdf.set_font("Arial", "B", 13)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(0, 8, sanitize_text("5. Model Performance Comparison"), ln=True, fill=True)
        pdf.set_font("Arial", "", 10)
        
        pdf.set_font("Arial", "B", 10)
        pdf.cell(40, 6, sanitize_text("Model"), 1, 0, "C")
        pdf.cell(30, 6, sanitize_text("R²"), 1, 0, "C")
        pdf.cell(30, 6, sanitize_text("RMSE"), 1, 0, "C")
        pdf.cell(30, 6, sanitize_text("MAE"), 1, 0, "C")
        pdf.cell(40, 6, sanitize_text("Physical Validity"), 1, 1, "C")
        
        pdf.set_font("Arial", "", 10)
        for _, row in model_comparison_df.iterrows():
            pdf.cell(40, 6, sanitize_text(str(row['Model'])), 1, 0, "L")
            pdf.cell(30, 6, f"{row['R²']:.4f}", 1, 0, "C")
            pdf.cell(30, 6, f"{row['RMSE']:.4f}", 1, 0, "C")
            pdf.cell(30, 6, f"{row['MAE']:.4f}", 1, 0, "C")
            pdf.cell(40, 6, sanitize_text(str(row['Physical_Validity'])), 1, 1, "L")
        pdf.ln(4)
    
    # 6. Recommendations
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, sanitize_text("6. Recommendations"), ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    
    if status == "PASS":
        recs = ["1. Proceed with experimental validation.",
                "2. Confirm tensile strength with physical testing.",
                "3. Evaluate disintegration time and dissolution.",
                "4. Assess stability under ICH conditions.",
                "5. Scale-up for process optimization."]
    else:
        recs = ["1. Reduce API or adjust binder concentration.",
                "2. Optimize Mg-St level.",
                "3. Increase compaction pressure.",
                "4. Reduce punch speed.",
                "5. Re-run with adjusted parameters."]
    
    for rec in recs:
        pdf.cell(0, 6, sanitize_text(rec), ln=True)
    
    pdf.ln(4)
    
    # 7. Contact
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, sanitize_text("7. Contact Information"), ln=True, fill=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, "Chem. Eng. Babuker A. Abdalla, PhD Researcher", ln=True)
    pdf.cell(0, 7, "Email: babuker@protonmail.com", ln=True)
    pdf.cell(0, 7, "Phone: +249-123-638-638", ln=True)
    pdf.cell(0, 7, "Nile Valley University, Postgraduate College, Sudan", ln=True)
    
    pdf.ln(3)
    pdf.set_y(270)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 6, "Generated by: Hybrid AI Framework v29.11", ln=True, align="C")
    
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, bytearray):
        return bytes(pdf_bytes)
    elif isinstance(pdf_bytes, bytes):
        return pdf_bytes
    else:
        return str(pdf_bytes).encode('latin1')

# ================================================================
# 7. NSGA-II (OPTIMIZED FOR PERFORMANCE)
# ================================================================

class NSGAII:
    def __init__(self, model, scaler, y_scaler, bounds,
                 pop_size=NSGA_POP_SIZE,
                 n_generations=NSGA_GENERATIONS,
                 w_tensile=0.0):
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
        self.tensile = None
        self.fronts = None

    def _repair(self, individual):
        api, mcc, pvpp, mgst, binder, pressure, speed, granule = individual
        api, binder, pvpp, mgst, mcc = normalize_components(api, binder, pvpp, mgst, mcc)
        pressure = np.clip(pressure, 80, PRESSURE_MAX)
        speed = np.clip(speed, 1.0, 50.0)
        granule = np.clip(granule, 30.0, 250.0)
        return np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule], dtype=float)

    def _initialize_population(self):
        pop = np.zeros((self.pop_size, 8))
        seed = np.array([90.0, 5.0, 3.0, 0.2, 3.0, 225.0, 12.0, 120.0], dtype=float)
        n_seed = int(0.3 * self.pop_size)

        for i in range(n_seed):
            noise = np.array([
                np.random.normal(0, 5.0),
                np.random.normal(0, 3.0),
                np.random.normal(0, 2.0),
                np.random.normal(0, 0.3),
                np.random.normal(0, 2.0),
                np.random.normal(0, 8.0),
                np.random.normal(0, 1.5),
                np.random.normal(0, 12.0)
            ])
            pop[i] = self._repair(seed + noise)

        for i in range(n_seed, self.pop_size):
            individual = np.array([
                np.random.uniform(60, 100),
                np.random.uniform(0.1, 20),
                np.random.uniform(0.1, 12),
                np.random.uniform(0.01, 3.0),
                np.random.uniform(0.1, 10),
                np.random.uniform(80, PRESSURE_MAX),
                np.random.uniform(1, 50),
                np.random.uniform(30, 250)
            ])
            pop[i] = self._repair(individual)

        return pop

    def _evaluate(self, population):
        n = population.shape[0]
        objectives = np.zeros((n, 2))
        constraints = np.zeros(n, dtype=bool)
        tensile_strengths = np.zeros(n)

        for i in range(n):
            try:
                repaired = self._repair(population[i])
                api, mcc, pvpp, mgst, binder, pressure, speed, granule = repaired

                inputs = np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule]).reshape(1, -1)
                inputs_with_features = add_interaction_features(inputs)[0]
                inputs_scaled = self.scaler.transform([inputs_with_features])
                X_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)

                with torch.no_grad():
                    pred_scaled = self.model.predict(X_tensor)
                    pred_actual = self.y_scaler.inverse_transform(pred_scaled)[0]

                density = float(pred_actual[0])
                tensile = float(pred_actual[1])
                er = float(pred_actual[2])

                density = float(np.clip(density, D_MIN, D_MAX))
                tensile = float(max(tensile, 1e-4))
                efrf = float(er / tensile)
                efrf = max(1e-4, min(efrf, 5.0))

                density_ok = (density >= D_MIN - DENSITY_TOL) and (density <= D_MAX + DENSITY_TOL)
                tensile_ok = (tensile >= TENSILE_MIN)
                efrf_ok = (efrf < EFRF_MAX + DENSITY_TOL)
                mcc_ok = (mcc <= MCC_MAX)

                feasible = density_ok and tensile_ok and efrf_ok and mcc_ok
                constraints[i] = feasible

                safety_penalty = 0.0
                if density > SAFETY_DENSITY_LIMIT:
                    safety_penalty += (density - SAFETY_DENSITY_LIMIT) ** 2
                if efrf > SAFETY_EFRF_LIMIT:
                    safety_penalty += (efrf - SAFETY_EFRF_LIMIT) ** 2

                zone_penalty = 0.0
                if density > 0.95:
                    zone_penalty += (density - 0.95) ** 2 * 0.5
                if efrf > 0.36:
                    zone_penalty += (efrf - 0.36) ** 2 * 2.0

                penalty = 0.0
                if tensile < TENSILE_MIN:
                    penalty += (TENSILE_MIN - tensile) ** 2
                if efrf >= EFRF_MAX:
                    penalty += (efrf - EFRF_MAX) ** 2
                if density < D_MIN:
                    penalty += (D_MIN - density) ** 2
                if density > D_MAX:
                    penalty += (density - D_MAX) ** 2

                objectives[i, 0] = -(api + self.w_tensile * tensile) + 30.0 * penalty + SAFETY_PENALTY_WEIGHT * safety_penalty + zone_penalty
                objectives[i, 1] = efrf + 30.0 * penalty + SAFETY_PENALTY_WEIGHT * safety_penalty + zone_penalty
                tensile_strengths[i] = tensile

                population[i] = repaired

            except Exception:
                objectives[i, 0] = 100.0
                objectives[i, 1] = 100.0
                constraints[i] = False
                tensile_strengths[i] = 0.0

        return objectives, constraints, tensile_strengths, population

    def _fast_non_dominated_sort(self, objectives, constraints):
        n = objectives.shape[0]
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n)
        rank = np.zeros(n, dtype=int)
        fronts = []
        violation = ~constraints

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if violation[i] and not violation[j]:
                    n_dom[i] += 1
                elif not violation[i] and violation[j]:
                    S[i].append(j)
                else:
                    better_or_equal = (
                        objectives[i, 0] <= objectives[j, 0] and
                        objectives[i, 1] <= objectives[j, 1]
                    )
                    strictly_better = (
                        objectives[i, 0] < objectives[j, 0] or
                        objectives[i, 1] < objectives[j, 1]
                    )
                    if better_or_equal and strictly_better:
                        S[i].append(j)
                    elif (
                        objectives[j, 0] <= objectives[i, 0] and
                        objectives[j, 1] <= objectives[i, 1] and
                        (objectives[j, 0] < objectives[i, 0] or objectives[j, 1] < objectives[i, 1])
                    ):
                        n_dom[i] += 1
            if n_dom[i] == 0:
                rank[i] = 0
                if not fronts:
                    fronts.append([])
                fronts[0].append(i)

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

    def _tournament_selection(self, pop_indices, objectives, ranks, crowding):
        selected = []
        for _ in range(len(pop_indices)):
            i1, i2 = np.random.choice(pop_indices, 2, replace=False)
            if ranks[i1] < ranks[i2]:
                selected.append(i1)
            elif ranks[i1] > ranks[i2]:
                selected.append(i2)
            else:
                selected.append(i1 if crowding[i1] >= crowding[i2] else i2)
        return selected

    def _simulated_binary_crossover(self, parent1, parent2):
        if np.random.random() > 0.9:
            return parent1.copy(), parent2.copy()

        eta_c = 20
        child1 = np.zeros(8)
        child2 = np.zeros(8)

        for i in range(8):
            if np.random.random() < 0.5:
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta_c + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]

        return self._repair(child1), self._repair(child2)

    def _polynomial_mutation(self, individual):
        eta_m = 20
        mutated = individual.copy()

        for i in range(8):
            if np.random.random() < 0.01:
                u = np.random.random()
                delta = min(u, 1 - u) ** (1 / (eta_m + 1))
                if u < 0.5:
                    mutated[i] = individual[i] + delta * (self.bounds[i, 1] - self.bounds[i, 0])
                else:
                    mutated[i] = individual[i] - delta * (self.bounds[i, 1] - self.bounds[i, 0])

        return self._repair(mutated)

    def run(self):
        self.population = self._initialize_population()

        for gen in range(self.n_generations):
            objectives, constraints, tensile, pop = self._evaluate(self.population)
            self.population = pop
            self.objectives = objectives
            self.constraints = constraints
            self.tensile = tensile

            fronts, ranks = self._fast_non_dominated_sort(objectives, constraints)
            self.fronts = fronts

            if gen == self.n_generations - 1:
                break

            crowding = np.zeros(self.pop_size)
            for front in fronts:
                dist = self._crowding_distance(objectives, front)
                for idx, d in zip(front, dist):
                    crowding[idx] = d

            selected = self._tournament_selection(range(self.pop_size), objectives, ranks, crowding)

            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    c1, c2 = self._simulated_binary_crossover(
                        self.population[selected[i]],
                        self.population[selected[i + 1]]
                    )
                    offspring.append(self._polynomial_mutation(c1))
                    offspring.append(self._polynomial_mutation(c2))
                else:
                    offspring.append(self._polynomial_mutation(self.population[selected[i]]))

            offspring = np.array(offspring[:self.pop_size])
            objectives_off, constraints_off, tensile_off, offspring = self._evaluate(offspring)

            combined_pop = np.vstack([self.population, offspring])
            combined_obj = np.vstack([self.objectives, objectives_off])
            combined_const = np.concatenate([self.constraints, constraints_off])

            combined_fronts, _ = self._fast_non_dominated_sort(combined_obj, combined_const)
            combined_crowding = np.zeros(len(combined_pop))
            for front in combined_fronts:
                dist = self._crowding_distance(combined_obj, front)
                for idx, d in zip(front, dist):
                    combined_crowding[idx] = d

            new_pop, new_obj, new_const = [], [], []
            for front in combined_fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    for idx in front:
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])
                        new_const.append(combined_const[idx])
                else:
                    front_sorted = sorted(front, key=lambda i: combined_crowding[i], reverse=True)
                    remain = self.pop_size - len(new_pop)
                    for idx in front_sorted[:remain]:
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])
                        new_const.append(combined_const[idx])
                    break

            self.population = np.array(new_pop)
            self.objectives = np.array(new_obj)
            self.constraints = np.array(new_const)

        objectives, constraints, tensile, pop = self._evaluate(self.population)
        self.population = pop
        self.objectives = objectives
        self.constraints = constraints
        self.tensile = tensile
        self.fronts, _ = self._fast_non_dominated_sort(objectives, constraints)
        return self.population, self.objectives, self.constraints, self.fronts

# ================================================================
# 8. TRAIN MODEL
# ================================================================

@st.cache_resource
def load_pinn_model():
    df, feature_names = generate_pinn_data(n_samples=N_SAMPLES)
    X_raw = df[feature_names].values
    y = df[['Density', 'Tensile_Strength_MPa', 'Elastic_Recovery_%']].values

    X_augmented = add_interaction_features(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_augmented)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    X_scaled_train, X_scaled_temp, X_raw_train, X_raw_temp, y_train, y_temp = train_test_split(
        X_scaled, X_raw, y_scaled, test_size=0.3, random_state=42
    )
    X_scaled_val, X_scaled_test, X_raw_val, X_raw_test, y_val, y_test = train_test_split(
        X_scaled_temp, X_raw_temp, y_temp, test_size=0.5, random_state=42
    )

    X_scaled_train_t = torch.tensor(X_scaled_train, dtype=torch.float32)
    X_raw_train_t = torch.tensor(X_raw_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    X_scaled_val_t = torch.tensor(X_scaled_val, dtype=torch.float32)
    X_raw_val_t = torch.tensor(X_raw_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    model = MultiTaskTruePINN(input_dim=X_augmented.shape[1])

    optimizer_adam = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, patience=50, factor=0.5)

    best_val_loss = float('inf')
    patience = 120
    counter = 0
    best_state = None

    progress_bar = st.progress(0)

    train_losses = []
    val_losses = []
    val_loss_history = []

    for epoch in range(ADAM_EPOCHS):
        model.train()
        optimizer_adam.zero_grad()

        compute_grad = (epoch % MONOTONICITY_FREQUENCY == 0)

        total_loss, _ = model.compute_loss(
            X_scaled_train_t, X_raw_train_t, y_train_t,
            epoch=epoch, max_epochs=ADAM_EPOCHS,
            w_data_init=W_DATA_INIT, w_physics_init=W_PHYSICS_INIT,
            w_data_final=W_DATA_FINAL, w_physics_final=W_PHYSICS_FINAL,
            w_mcc=0.5, w_density=W_DENSITY,
            efrf_target=EFRF_MAX, mcc_max=MCC_MAX,
            compute_grad=compute_grad
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_adam.step()

        model.eval()
        with torch.no_grad():
            val_loss, _ = model.compute_loss(
                X_scaled_val_t, X_raw_val_t, y_val_t,
                epoch=epoch, max_epochs=ADAM_EPOCHS,
                w_data_init=W_DATA_INIT, w_physics_init=W_PHYSICS_INIT,
                w_data_final=W_DATA_FINAL, w_physics_final=W_PHYSICS_FINAL,
                w_mcc=0.5, w_density=W_DENSITY,
                efrf_target=EFRF_MAX, mcc_max=MCC_MAX,
                compute_grad=False
            )

        val_loss_value = val_loss.item()
        train_losses.append(total_loss.item())
        val_losses.append(val_loss_value)
        
        if epoch % 10 == 0:
            val_loss_history.append((epoch, val_loss_value))
        
        scheduler.step(val_loss_value)

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            counter += 1

        if counter > patience:
            break

        if (epoch + 1) % 100 == 0:
            progress_bar.progress(min((epoch + 1) / ADAM_EPOCHS, 0.6))

    if best_state is not None:
        model.load_state_dict(best_state)

    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=0.05,
        max_iter=200,
        line_search_fn='strong_wolfe'
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        final_epoch = ADAM_EPOCHS
        total_loss, _ = model.compute_loss(
            X_scaled_train_t, X_raw_train_t, y_train_t,
            epoch=final_epoch, max_epochs=ADAM_EPOCHS,
            w_data_init=W_DATA_INIT, w_physics_init=W_PHYSICS_INIT,
            w_data_final=W_DATA_FINAL, w_physics_final=W_PHYSICS_FINAL,
            w_mcc=0.5, w_density=W_DENSITY,
            efrf_target=EFRF_MAX, mcc_max=MCC_MAX,
            compute_grad=True
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        return total_loss

    for i in range(LBFGS_STEPS):
        optimizer_lbfgs.step(closure)
        progress_bar.progress(min(0.6 + 0.08 * (i + 1), 1.0))

    model.eval()
    model.feature_names = feature_names
    model.scaler = scaler
    model.y_scaler = y_scaler

    torch.save(model.state_dict(), 'true_pinn_checkpoint.pt')

    loss_history = {'train': train_losses, 'val': val_losses, 'val_early': val_loss_history}
    return model, scaler, y_scaler, feature_names, df, loss_history

# ================================================================
# 9. PREDICTION & PLOTS
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
    
    fig.add_trace(go.Scatter(
        x=epochs, y=loss_history['train'],
        mode='lines', name='Training Loss',
        line=dict(color='blue', width=2)
    ))
    
    if len(loss_history['val']) > 0:
        fig.add_trace(go.Scatter(
            x=epochs[:len(loss_history['val'])], y=loss_history['val'],
            mode='lines', name='Validation Loss',
            line=dict(color='orange', width=2, dash='dash')
        ))
    
    if 'val_early' in loss_history and loss_history['val_early']:
        early_epochs, early_vals = zip(*loss_history['val_early'])
        fig.add_trace(go.Scatter(
            x=early_epochs, y=early_vals,
            mode='markers', name='Early Validation',
            marker=dict(color='red', size=8, symbol='x')
        ))
    
    fig.update_layout(
        title='Training Curves (v29.11 - Soft Sigmoid Scheduling)',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Loss', type='log'),
        height=400,
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig

def plot_pareto_plotly(objectives, constraints, fronts, nsga, api, efrf):
    try:
        if objectives is None or constraints is None or fronts is None:
            return None
        if len(fronts) == 0 or len(fronts[0]) == 0:
            return None

        front0 = fronts[0]
        pareto_api = -objectives[front0, 0]
        pareto_efrf = objectives[front0, 1]
        feasible = constraints[front0]

        valid_mask = (
            (pareto_api >= 85) & (pareto_api <= 95) &
            (pareto_efrf >= 0.0001) & (pareto_efrf <= 5.0)
        )

        pareto_api = pareto_api[valid_mask]
        pareto_efrf = pareto_efrf[valid_mask]
        feasible = feasible[valid_mask]

        if len(pareto_api) == 0:
            return None

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=-objectives[:, 0],
            y=objectives[:, 1],
            mode='markers',
            marker=dict(size=4, color='gray', opacity=0.35),
            name='All Solutions'
        ))

        fig.add_trace(go.Scatter(
            x=pareto_api,
            y=pareto_efrf,
            mode='lines+markers',
            marker=dict(size=8, color='red'),
            line=dict(color='red', width=2),
            name='Pareto Front'
        ))

        feas_x = pareto_api[feasible]
        feas_y = pareto_efrf[feasible]
        if len(feas_x) > 0:
            fig.add_trace(go.Scatter(
                x=feas_x,
                y=feas_y,
                mode='markers',
                marker=dict(size=10, color='green', symbol='diamond'),
                name='Feasible Solutions'
            ))

        safe_mask = (feas_x >= 0.93) & (feas_x <= 0.96) & (feas_y < 0.36)
        if np.any(safe_mask):
            fig.add_trace(go.Scatter(
                x=feas_x[safe_mask],
                y=feas_y[safe_mask],
                mode='markers',
                marker=dict(size=12, color='gold', symbol='star'),
                name='Preferred Safe Zone'
            ))

        if api and efrf and 85 <= api <= 95 and 0.0001 <= efrf <= 2:
            fig.add_trace(go.Scatter(
                x=[api], y=[efrf],
                mode='markers+text',
                marker=dict(size=14, color='blue', symbol='star'),
                text=[f"Your Formulation<br>API: {api:.1f}%<br>EFRF: {efrf:.4f}"],
                name='Selected Formulation'
            ))

        fig.add_hrect(
            y0=0, y1=EFRF_MAX,
            line_width=0,
            fillcolor="green",
            opacity=0.1,
            annotation_text=f"Safe Zone (EFRF < {EFRF_MAX:.2f})",
            annotation_position="top left"
        )
        fig.add_hline(
            y=EFRF_MAX,
            line_dash='dash',
            line_color='red',
            annotation_text=f'EFRF Threshold: {EFRF_MAX:.2f}'
        )

        fig.update_layout(
            title='Pareto Front: API vs EFRF (Gold = Safe Zone)',
            xaxis_title='API (%)',
            yaxis_title='EFRF',
            template='plotly_white',
            height=500,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        return fig
    except Exception as e:
        st.error(f"Pareto plot error: {e}")
        return None

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
        sorted_names = [features[i] for i in sorted_idx]
        sorted_values = [sensitivities[i] for i in sorted_idx]

        fig = go.Figure()
        colors = ['#e74c3c' if v > np.mean(sensitivities) else '#2ecc71' for v in sorted_values]
        fig.add_trace(go.Bar(
            y=sorted_names, x=sorted_values, orientation='h',
            marker=dict(color=colors),
            text=[f"{v:.4f}" for v in sorted_values],
            textposition='outside'
        ))
        fig.add_vline(x=np.mean(sensitivities), line_dash='dash', line_color='red', 
                      annotation_text=f'Avg: {np.mean(sensitivities):.4f}')
        fig.update_layout(
            title='Sensitivity Analysis (EFRF)',
            xaxis_title='Sensitivity (ΔEFRF)',
            yaxis_title='Parameters',
            height=450
        )
        return fig
    except Exception:
        return None

def train_and_compare(X_train, X_test, y_train, y_test):
    models = {}
    models['MLP'] = MLPRegressor(hidden_layer_sizes=(128, 128, 64, 32), max_iter=1000, random_state=42)
    models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            'Model': name,
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'Physical_Validity': '❌ Not Enforced'
        })
    return pd.DataFrame(results)

# ================================================================
# 10. STREAMLIT UI (UNCHANGED - IDENTICAL TO v29.11)
# ================================================================

# ================================================================
# 10. STREAMLIT UI (UNCHANGED - IDENTICAL TO v29.11)
# ================================================================

# تم تعليق هذا السطر لمنع إعادة التدريب في كل مرة
# st.cache_resource.clear()

st.set_page_config(page_title="PINN Framework v29.11", page_icon="🧬", layout="wide")
clamp_session_state()

st.set_page_config(page_title="PINN Framework v29.11", page_icon="🧬", layout="wide")
clamp_session_state()

st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); 
            padding: 2rem; border-radius: 1rem; margin-bottom: 1.5rem; text-align: center;">
    <h1 style="color: #ffffff; font-size: 2.5rem; margin: 0;">
        🧬 Hybrid AI Framework v29.11
    </h1>
    <p style="color: #a8b2d1; font-size: 1.2rem; margin: 0.5rem 0 0 0;">
        Physics-Informed Neural Network · Multi-Objective Optimization
    </p>
    <p style="color: #64ffda; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
        Nile Valley University · Postgraduate College · Sudan
    </p>
    <p style="color: #ffd700; font-size: 0.85rem; margin: 0.5rem 0 0 0;">
        ⚡ v29.11 — Ryshkewitch-Duckworth Physics
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

with st.sidebar:
    st.markdown("### 📚 Physics Constraints (v29.11)")
    st.markdown(f"""
    - ✅ **Heckel:** ln(1/(1-D)) = kP + A
    - ✅ **EFRF:** ER / σt < {EFRF_MAX:.2f}
    - ✅ **Monotonicity:** ∂D/∂P > 0 (every {MONOTONICITY_FREQUENCY} epochs)
    - ✅ **Density Preference:** Gentle push toward 0.93–0.96
    - ✅ **EFRF Preference:** Extra penalty for > 0.36
    - ✅ **Tensile Physics (NEW):** Ryshkewitch-Duckworth equation
      <br>ln(σt) = ln(σ0) - b·(1-D)
    - ✅ **Soft Scheduling (FIXED):** Sigmoid transition completes during Adam phase
    - ✅ **MCC:** ≤ {MCC_MAX:.1f}%
    - ✅ **Density:** {D_MIN:.2f} ≤ D ≤ {D_MAX:.2f}
    
    **Multi-Task PINN (v29.11):**
    - 5 outputs (D, σt, ER, k, A)
    - Adam ({ADAM_EPOCHS}) → LBFGS ({LBFGS_STEPS})
    - Enhanced Network: 384→384→192 neurons
    - Loss Balance: Sigmoid transition {W_DATA_INIT:.1f}→{W_DATA_FINAL:.1f} / {W_PHYSICS_INIT:.1f}→{W_PHYSICS_FINAL:.1f}
    - NSGA-II: pop={NSGA_POP_SIZE}, gen={NSGA_GENERATIONS}
    - **Backend Normalization:** Dynamic normalization applied internally
    """)
    st.info(f"🔬 **v29.11** — Ryshkewitch-Duckworth + Sigmoid Scheduling (FIXED)")

with st.spinner("🔄 Training Multi-Task PINN (v29.11 Enhanced Physics)..."):
    model, scaler, y_scaler, feature_names, df, loss_history = load_pinn_model()
st.success("✅ Multi-Task True PINN (v29.11) trained successfully")

st.markdown("### 🧪 Quick Experiments")
exp_cols = st.columns(4)
experiments = {
    "Baseline (Stable)": {'api': 90.5, 'binder': 2.7, 'pvpp': 3.0, 'mgst': 0.20, 'mcc': 5.0, 'pressure': 230, 'speed': 12, 'granule': 125},
    "High Binder": {'api': 90.5, 'binder': 3.0, 'pvpp': 3.0, 'mgst': 0.15, 'mcc': 5.0, 'pressure': 235, 'speed': 10, 'granule': 125},
    "High Pressure": {'api': 90.5, 'binder': 2.8, 'pvpp': 3.0, 'mgst': 0.12, 'mcc': 5.0, 'pressure': 250, 'speed': 9, 'granule': 125},
    "Low Mg-St": {'api': 90.5, 'binder': 3.0, 'pvpp': 3.0, 'mgst': 0.10, 'mcc': 5.0, 'pressure': 245, 'speed': 8, 'granule': 125}
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
        api = st.slider("🧪 API Loading (%)", 85.0, 95.0, get_safe_value('api'), 0.1, key="api")
        binder = st.slider("🔗 Binder (%)", BINDER_MIN, BINDER_MAX, get_safe_value('binder'), 0.1, key="binder")
        pvpp = st.slider("💊 PVPP (%)", 0.5, 6.0, get_safe_value('pvpp'), 0.1, key="pvpp")
        mgst = st.slider("🧴 Mg-St (%)", 0.01, 1.2, get_safe_value('mgst'), 0.01, key="mgst")
        mcc = st.slider("📦 MCC (%)", 0.0, MCC_MAX, get_safe_value('mcc'), 0.1, key="mcc")
        
        total = api + binder + pvpp + mgst + mcc
        if abs(total - 100) < 0.1:
            st.success(f"✅ Total = {total:.2f}%")
        elif total > 100.1:
            st.error(f"❌ Total = {total:.2f}% (exceeds 100%)")
        else:
            st.warning(f"⚠️ Total = {total:.2f}% (adjust to 100%)")
    
    st.markdown("### ⚙️ Process Parameters")
    with st.container(border=True):
        pressure = st.slider("⚙️ Pressure (MPa)", 80.0, PRESSURE_MAX, get_safe_value('pressure'), 1.0, key="pressure")
        speed = st.slider("🔄 Speed (rpm)", 1.0, 50.0, get_safe_value('speed'), 0.5, key="speed")
        granule = st.slider("🔬 Granule Size (µm)", 30.0, 250.0, get_safe_value('granule'), 1.0, key="granule")
    
    predict_btn = st.button("🔬 Predict & Optimize (v29.11)", use_container_width=True)

# ================================================================
# RESULTS & TABS
# ================================================================
with col_right:
    st.markdown("### 📈 Results")
    
    # --- Initialize variables ---
    objectives = None
    constraints = None
    fronts = None
    nsga = None
    api_norm = None
    efrf = None
    comp_df = pd.DataFrame()
    density = 0.0
    tensile = 0.0
    er = 0.0
    density_ok = False
    tensile_ok = False
    efrf_ok = False
    mcc_ok = False
    api_use = 0.0
    mcc_use = 0.0
    pvpp_use = 0.0
    mgst_use = 0.0
    binder_use = 0.0
    
    # --- RUN PREDICTION & NSGA-II (BEFORE TABS) ---
    if predict_btn:
        if abs(total - 100) > 0.1:
            st.warning("⚠️ Formulation must sum to 100%")
        else:
            # Apply backend normalization
            api_norm, binder_norm, pvpp_norm, mgst_norm, mcc_norm = normalize_components(api, binder, pvpp, mgst, mcc)
            inputs_norm = [api_norm, mcc_norm, pvpp_norm, mgst_norm, binder_norm, pressure, speed, granule]
            api_use, mcc_use, pvpp_use, mgst_use, binder_use = api_norm, mcc_norm, pvpp_norm, mgst_norm, binder_norm
            
            # Run prediction
            with st.spinner("🧠 Running prediction (v29.11)..."):
                density, tensile, er, efrf = predict_pinn(model, scaler, y_scaler, inputs_norm)
            
            # Display KPIs
            kpi_cols = st.columns(3)
            kpi_cols[0].metric("Density", f"{density:.3f}", delta=f"[{D_MIN:.2f}, {D_MAX:.2f}]")
            kpi_cols[1].metric("Tensile", f"{tensile:.3f} MPa", delta=f">= {TENSILE_MIN:.2f} PASS" if tensile >= TENSILE_MIN else f"< {TENSILE_MIN:.2f} FAIL")
            kpi_cols[2].metric("EFRF", f"{efrf:.4f}", delta=f"< {EFRF_MAX:.2f} PASS" if efrf < EFRF_MAX else f">= {EFRF_MAX:.2f} FAIL")
            
            st.markdown("---")
            
            # Feasibility checks
            density_ok = (density >= D_MIN and density <= D_MAX)
            tensile_ok = (tensile >= TENSILE_MIN)
            efrf_ok = (efrf < EFRF_MAX)
            mcc_ok = (mcc_norm <= MCC_MAX)
            
            if density_ok and tensile_ok and efrf_ok and mcc_ok:
                st.success(f"✅ Formulation satisfies all constraints (Density: {density:.3f}, σt ≥ {TENSILE_MIN:.2f}, EFRF < {EFRF_MAX:.2f})")
            else:
                st.error("❌ Formulation violates one or more constraints")
            
            st.markdown("### ✅ Feasibility")
            pass_cols = st.columns(4)
            pass_cols[0].metric(f"Density [{D_MIN:.2f}, {D_MAX:.2f}]", "✅ PASS" if density_ok else "❌ FAIL")
            pass_cols[1].metric(f"σt ≥ {TENSILE_MIN:.2f}", "✅ PASS" if tensile_ok else "❌ FAIL")
            pass_cols[2].metric(f"EFRF < {EFRF_MAX:.2f}", "✅ PASS" if efrf_ok else "❌ FAIL")
            pass_cols[3].metric(f"MCC ≤ {MCC_MAX:.1f}%", "✅ PASS" if mcc_ok else "❌ FAIL")
            
            # Physics Verification
            with st.expander("🔬 Physics Verification (v29.11 Enhanced)"):
                try:
                    inputs_with_features = add_interaction_features(np.array([inputs_norm]))[0]
                    inputs_scaled = scaler.transform([inputs_with_features])
                    X_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
                    with torch.no_grad():
                        full_output = model.forward(X_tensor).numpy()[0]
                    st.metric("k (Plasticity)", f"{full_output[3]:.4f}")
                    st.metric("A (Rearrangement)", f"{full_output[4]:.4f}")
                    st.caption("Enhanced physics: Ryshkewitch-Duckworth + Binder relation")
                except:
                    pass
            
            # --- RUN NSGA-II ---
            st.markdown("### ⚙️ NSGA-II")
            bounds = np.array([
                [60, 100], [0.1, 20], [0.1, 12], [0.01, 3.0], 
                [0.1, 10], [80, PRESSURE_MAX], [1, 50], [30, 250]
            ])
            with st.spinner(f"🔄 Running NSGA-II (pop={NSGA_POP_SIZE}, gen={NSGA_GENERATIONS})..."):
                start_time = time.time()
                nsga = NSGAII(model, scaler, y_scaler, bounds, pop_size=NSGA_POP_SIZE, n_generations=NSGA_GENERATIONS, w_tensile=0.0)
                pop, objectives, constraints, fronts = nsga.run()
                elapsed = time.time() - start_time
                st.caption(f"⏱️ NSGA-II completed in {elapsed:.1f} seconds")
                
                if len(fronts) > 0 and len(fronts[0]) > 0:
                    front0 = fronts[0]
                    pareto_api = -objectives[front0, 0]
                    feasible = constraints[front0]
                    feasible_indices = [i for i, f in enumerate(feasible) if f]
                    if feasible_indices:
                        best_idx = np.argmax([pareto_api[i] for i in feasible_indices])
                        best_idx = feasible_indices[best_idx]
                        st.success(f"Optimal: API = {pareto_api[best_idx]:.2f}% | EFRF = {objectives[front0, 1][best_idx]:.4f}")
                    else:
                        st.warning("No feasible solutions found")
                else:
                    st.warning("No Pareto front found. Try adjusting NSGA-II parameters.")
            
            # --- Prepare Model Comparison Data ---
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
            
            pinn_valid = "✅ Fully Valid"
            try:
                mono_pass = 0
                efrf_pass = 0
                density_pass = 0
                for _ in range(50):
                    idx = np.random.randint(0, len(X_test))
                    sample = X_test[idx].copy()
                    d1, _, _, e1 = predict_pinn(model, scaler, y_scaler, sample)
                    sample[5] = min(sample[5] * 1.2, PRESSURE_MAX)
                    d2, _, _, e2 = predict_pinn(model, scaler, y_scaler, sample)
                    if d2 > d1: mono_pass += 1
                    if e1 < EFRF_MAX: efrf_pass += 1
                    if D_MIN <= d1 <= D_MAX: density_pass += 1
                if mono_pass > 40 and efrf_pass > 45 and density_pass > 45:
                    pinn_valid = "✅ Fully Valid"
                else:
                    pinn_valid = "⚠️ Partially Valid"
            except:
                pinn_valid = "⚠️ Partially Valid"
            
            comp_df = train_and_compare(X_train_scaled, X_test_scaled, y_train, y_test)
            pinn_row = pd.DataFrame([{
                'Model': 'PINN (Proposed)',
                'R²': pinn_r2,
                'RMSE': pinn_rmse,
                'MAE': pinn_mae,
                'Physical_Validity': pinn_valid
            }])
            comp_df = pd.concat([pinn_row, comp_df], ignore_index=True)
    
    # --- TABS (DEFINED AFTER NSGA-II) ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📉 Pareto Front", "🔍 Sensitivity", "📊 Model Comparison", "📈 Training Curves", "📄 Report"]
    )
    
    # === TAB 1: Pareto Front ===
    with tab1:
        st.markdown("### 📉 Pareto Front")
        if predict_btn and objectives is not None and fronts is not None and len(fronts) > 0:
            fig_p = plot_pareto_plotly(objectives, constraints, fronts, nsga, api_norm, efrf)
            if fig_p:
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.info("Pareto front could not be generated.")
        elif predict_btn and objectives is not None:
            st.info("Pareto front not available (no feasible solutions found).")
        else:
            st.info("👆 Please click 'Predict & Optimize (v29.11)' with a valid formulation (total = 100%) to generate the Pareto Front.")
    
    # === TAB 2: Sensitivity ===
    with tab2:
        st.markdown("### 🔍 Sensitivity Analysis")
        if predict_btn and api_norm is not None:
            fig_s = plot_sensitivity_plotly(inputs_norm, model, scaler, y_scaler)
            if fig_s:
                st.plotly_chart(fig_s, use_container_width=True)
            else:
                st.info("Sensitivity analysis not available.")
        else:
            st.info("👆 Please click 'Predict & Optimize (v29.11)' with a valid formulation to run Sensitivity Analysis.")
    
    # === TAB 3: Model Comparison ===
    with tab3:
        st.markdown("### 📊 Model Performance Comparison (v29.11)")
        st.caption("Hold-out test set (20% of data) — R², RMSE, MAE, and Physical Validity")
        
        if predict_btn and not comp_df.empty:
            pinn_r2_val = comp_df[comp_df['Model'] == 'PINN (Proposed)']['R²'].values[0] if not comp_df.empty else 0
            pinn_valid_val = comp_df[comp_df['Model'] == 'PINN (Proposed)']['Physical_Validity'].values[0] if not comp_df.empty else ""
            
            st.metric("PINN R² (v29.11)", f"{pinn_r2_val:.4f}", delta="Target: ≥ 0.85")
            
            if pinn_r2_val < 0.8 and pinn_valid_val == "✅ Fully Valid":
                st.warning("⚠️ **Trade-off Detected:** R² is moderate (<0.8) but physical validity is fully satisfied. The model prioritises physics constraints over pure data fit to avoid physically impossible solutions. This is expected in True PINN frameworks.")
            
            col1, col2 = st.columns(2)
            with col1:
                colors_r2 = ['#2ecc71' if m == 'PINN (Proposed)' else '#3498db' for m in comp_df['Model']]
                fig_r2 = go.Figure(data=[
                    go.Bar(
                        x=comp_df['Model'], 
                        y=comp_df['R²'],
                        marker_color=colors_r2,
                        text=[f"{v:.4f}" for v in comp_df['R²']],
                        textposition='outside'
                    )
                ])
                fig_r2.update_layout(
                    title='<b>R² Score Comparison (v29.11)</b>',
                    yaxis=dict(title='R² Score', range=[0, 1.05]),
                    height=350,
                    showlegend=False
                )
                st.plotly_chart(fig_r2, use_container_width=True)
            with col2:
                colors_rmse = ['#e74c3c' if m == 'PINN (Proposed)' else '#95a5a6' for m in comp_df['Model']]
                fig_rmse = go.Figure(data=[
                    go.Bar(
                        x=comp_df['Model'], 
                        y=comp_df['RMSE'],
                        marker_color=colors_rmse,
                        text=[f"{v:.4f}" for v in comp_df['RMSE']],
                        textposition='outside'
                    )
                ])
                fig_rmse.update_layout(
                    title='<b>RMSE Comparison (v29.11)</b>',
                    yaxis=dict(title='RMSE (MPa)'),
                    height=350,
                    showlegend=False
                )
                st.plotly_chart(fig_rmse, use_container_width=True)
            
            st.dataframe(
                comp_df.style.highlight_max(subset=['R²'], color='lightgreen')
                          .highlight_min(subset=['RMSE', 'MAE'], color='lightcoral'),
                use_container_width=True,
                hide_index=True
            )
        else:
            if predict_btn:
                st.info("Formulation must sum to 100% to generate comparison data.")
            else:
                st.info("👆 Please click 'Predict & Optimize (v29.11)' with a valid formulation to see model performance comparison.")
    
    # === TAB 4: Training Curves ===
    with tab4:
        st.markdown("### 📈 Training Curves (v29.11 - Soft Sigmoid)")
        fig_loss = plot_training_curves(loss_history)
        if fig_loss:
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("Loss history not available.")
    
    # === TAB 5: Report ===
    with tab5:
        st.markdown("### 📄 Comprehensive Report (PDF)")
        st.caption("Download a complete report with formulation details, predictions, and model comparison.")
        
        if predict_btn and not comp_df.empty:
            status = "PASS" if (density_ok and tensile_ok and efrf_ok and mcc_ok) else "FAIL"
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            pdf_data = generate_full_pdf_report(
                api_use, mcc_use, pvpp_use, mgst_use, binder_use, pressure, speed, granule,
                density, tensile, er, efrf, status, timestamp,
                model_comparison_df=comp_df
            )
            
            st.download_button(
                label="📥 Download Full Report (PDF)",
                data=pdf_data,
                file_name=f"formulation_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.success("✅ One-click download — includes formulation, predictions, and model comparison.")
        else:
            if predict_btn:
                st.info("Formulation must sum to 100% to generate the report.")
            else:
                st.info("👆 Please click 'Predict & Optimize (v29.11)' with a valid formulation to generate the report.")

st.markdown("---")
st.caption(f"🔬 **Multi-Task True PINN — v29.11 (Ryshkewitch-Duckworth + Sigmoid Scheduling - FIXED)**")
st.caption(f"📧 Contact: babuker@protonmail.com | 🏛️ Nile Valley University, Postgraduate College, Sudan")
