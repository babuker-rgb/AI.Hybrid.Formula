"""
True Physics-Informed Neural Network (PINN) - Complete Professional Framework
Multi-Objective Tablet Manufacturing Optimization with Full Analytics

Author: Babuker A. Abdalla
Affiliation: Nile Valley University, Postgraduate College, Sudan
Version: 28.0 (Stable Production - Enhanced Comparison & PDF Report)
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
warnings.filterwarnings('ignore')

# ================================================================
# 0. USER-CONFIGURABLE PARAMETERS
# ================================================================

TENSILE_MIN = 1.90          # MPa
EFRF_MAX = 0.40             # dimensionless
MCC_MAX = 8.0               # %
DENSITY_MAX = 0.99
PRESSURE_MAX = 300.0        # MPa

NSGA_POP_SIZE = 100
NSGA_GENERATIONS = 100

BINDER_MIN = 0.5
BINDER_MAX = 5.0

PHYSICS_WEIGHT_INIT = 0.5

# ================================================================
# 1. SAFE IMPORTS
# ================================================================

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ================================================================
# 2. SESSION STATE
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
# 3. FEATURE ENGINEERING
# ================================================================

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
# 4. MULTI-TASK TRUE PINN MODEL
# ================================================================

class MultiTaskTruePINN(nn.Module):
    def __init__(self, input_dim=13, output_dim=5):
        super(MultiTaskTruePINN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.Tanh(), nn.Dropout(0.05),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.Tanh(),
            nn.Linear(32, output_dim)
        )

    def forward(self, X):
        raw = self.network(X)
        density = torch.sigmoid(raw[:, 0:1])
        tensile = torch.nn.functional.softplus(raw[:, 1:2])
        er = torch.nn.functional.softplus(raw[:, 2:3])
        k = torch.nn.functional.softplus(raw[:, 3:4])
        A = raw[:, 4:5]
        return torch.cat([density, tensile, er, k, A], dim=1)

    def predict(self, X_scaled):
        self.eval()
        with torch.no_grad():
            if not isinstance(X_scaled, torch.Tensor):
                X_scaled = torch.FloatTensor(X_scaled)
            output = self.forward(X_scaled)
            return output[:, :3].numpy()

    def compute_loss(self, X_scaled, X_raw, y_true, epoch=0, max_epochs=4000,
                     w_data_init=2.0, w_physics_init=PHYSICS_WEIGHT_INIT,
                     w_data_final=1.0, w_physics_final=1.0,
                     w_mcc=0.5, w_density=0.5,
                     efrf_target=EFRF_MAX, mcc_max=MCC_MAX,
                     compute_grad=True):
        pressure_real = X_raw[:, 5]
        mcc_real = X_raw[:, 1]

        y_pred = self.forward(X_scaled)
        density_pred = y_pred[:, 0]
        tensile_pred = y_pred[:, 1]
        er_pred = y_pred[:, 2]
        k_pred = y_pred[:, 3]
        A_pred = y_pred[:, 4]

        progress = min(epoch / 500, 1.0)
        w_data = w_data_init + (w_data_final - w_data_init) * progress
        w_physics = w_physics_init + (w_physics_final - w_physics_init) * progress

        data_loss = nn.MSELoss()(y_pred[:, :3], y_true)

        D_clamped = torch.clamp(density_pred, 0.01, 0.99)
        heckel_pred = torch.log(1.0 / (1.0 - D_clamped))
        heckel_target = k_pred * pressure_real + A_pred
        heckel_loss = torch.mean((heckel_pred - heckel_target) ** 2)

        efrf_pred = er_pred / (tensile_pred + 1e-8)
        efrf_loss = torch.mean(torch.relu(efrf_pred - efrf_target) ** 2)

        mcc_loss = torch.mean(torch.relu(mcc_real - mcc_max) ** 2)
        density_penalty = torch.mean(torch.relu(density_pred - DENSITY_MAX) ** 2)

        if compute_grad:
            if not X_scaled.requires_grad:
                X_scaled.requires_grad_(True)
            y_pred_grad = self.forward(X_scaled)
            density_grad = y_pred_grad[:, 0]
            grad_density = torch.autograd.grad(
                outputs=density_grad,
                inputs=X_scaled,
                grad_outputs=torch.ones_like(density_grad),
                create_graph=True, retain_graph=True
            )[0]
            grad_pressure = grad_density[:, 5]
            monotonic_loss = torch.mean(torch.relu(-grad_pressure) ** 2)
        else:
            monotonic_loss = torch.tensor(0.0, device=X_scaled.device)

        mask_low = (pressure_real < 120).float()
        mask_high = (pressure_real > 230).float()
        boundary_loss = (
            torch.mean(mask_low * torch.relu(0.5 - density_pred) ** 2) +
            torch.mean(mask_high * torch.relu(density_pred - 0.98) ** 2)
        )

        k_reg = torch.mean(torch.relu(k_pred - 0.1) ** 2) + torch.mean(torch.relu(0.005 - k_pred) ** 2)
        A_reg = torch.mean(torch.relu(A_pred - 2.0) ** 2) + torch.mean(torch.relu(0.5 - A_pred) ** 2)

        total_loss = (
            w_data * data_loss +
            w_physics * (heckel_loss + efrf_loss + monotonic_loss + boundary_loss) +
            w_mcc * mcc_loss +
            w_density * density_penalty +
            0.1 * k_reg + 0.1 * A_reg
        )

        loss_dict = {
            'data_loss': data_loss.item(),
            'heckel_loss': heckel_loss.item(),
            'efrf_loss': efrf_loss.item(),
            'mcc_loss': mcc_loss.item(),
            'density_penalty': density_penalty.item(),
            'monotonic_loss': monotonic_loss.item() if compute_grad else 0.0,
            'boundary_loss': boundary_loss.item(),
            'k_reg': k_reg.item(),
            'A_reg': A_reg.item(),
            'w_data': w_data,
            'w_physics': w_physics,
            'total_loss': total_loss.item()
        }
        return total_loss, loss_dict

# ================================================================
# 5. DATA GENERATION
# ================================================================

def generate_pinn_data(n_samples=600, random_state=42):
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
            pressure = np.random.uniform(80, PRESSURE_MAX)
            speed = np.random.uniform(1, 50)
            granule = np.random.uniform(30, 250)
        else:
            api = np.clip(np.random.normal(90.5, 1.5), 85, 95)
            binder = np.clip(np.random.normal(2.8, 0.4), BINDER_MIN, BINDER_MAX)
            mgst = np.clip(np.random.normal(0.15, 0.06), 0.01, 1.2)
            pvpp = np.clip(np.random.normal(3.0, 0.5), 0.5, 6.0)
            mcc = np.clip(np.random.normal(5.0, 1.0), 0, MCC_MAX)
            pressure = np.clip(np.random.normal(230, 15), 80, PRESSURE_MAX)
            speed = np.clip(np.random.normal(10, 3), 1, 50)
            granule = np.clip(np.random.normal(125, 20), 30, 250)

        total = api + binder + mgst + pvpp + mcc
        if total > 100:
            scale = 100 / total
            api *= scale; binder *= scale; mgst *= scale; pvpp *= scale; mcc *= scale
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

        k_true = 0.035 * (1 - 0.4 * (api - 85)/10) * (1 - 0.2 * (speed - 10)/30)
        k_true = max(k_true, 0.008)
        A_true = 1.2 + 0.1 * (binder - 1.5) - 0.2 * (mgst - 0.5)
        D = np.clip(1 - np.exp(-(k_true * pressure + A_true)), 0.4, 0.99)

        strength = np.clip(3.5 - 0.15 * (api - 85) + 0.3 * binder + 0.008 * (pressure - 100) - 1.5 * mgst - 0.02 * (speed - 10), 0.5, 6.0)
        er = np.clip(1.8 + 0.3 * (api - 85)/10 + 0.08 * (speed - 10)/30 - 0.1 * (pressure - 100)/150, 0.5, 4.0)

        y[i] = [D, strength, er]

    feature_names = ['API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
                     'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm']
    df = pd.DataFrame(X, columns=feature_names)
    df['Density'] = y[:, 0]
    df['Tensile_Strength_MPa'] = y[:, 1]
    df['Elastic_Recovery_%'] = y[:, 2]
    return df, feature_names

# ================================================================
# 6. PDF GENERATION (Full Report with Comparison Table)
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
    """Generate a detailed PDF report including model comparison."""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, sanitize_text("Formulation Optimization Report"), ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 6, sanitize_text("Hybrid AI Framework (PINN + NSGA-II)"), ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Date: {timestamp}", ln=True, align="C")
    pdf.ln(4)
    
    # Author
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
    
    results = [("Density", f"{density:.3f}", "-"),
               ("Tensile Strength", f"{tensile:.3f} MPa", f">= {TENSILE_MIN:.2f} MPa"),
               ("Elastic Recovery", f"{er:.3f} %", "-"),
               ("EFRF", f"{efrf:.4f}", f"< {EFRF_MAX:.2f}")]
    
    pdf.set_font("Arial", "B", 10)
    pdf.cell(45, 6, sanitize_text("Metric"), 1, 0, "C")
    pdf.cell(35, 6, sanitize_text("Value"), 1, 0, "C")
    pdf.cell(45, 6, sanitize_text("Threshold"), 1, 1, "C")
    
    pdf.set_font("Arial", "", 10)
    for r in results:
        pdf.cell(45, 6, sanitize_text(r[0]), 1, 0, "L")
        pdf.cell(35, 6, sanitize_text(r[1]), 1, 0, "C")
        pdf.cell(45, 6, sanitize_text(r[2]), 1, 1, "C")
    
    pdf.ln(4)
    
    # 4. Status
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
        pdf.cell(40, 6, sanitize_text("Physics"), 1, 1, "C")
        
        pdf.set_font("Arial", "", 10)
        for _, row in model_comparison_df.iterrows():
            pdf.cell(40, 6, sanitize_text(str(row['Model'])), 1, 0, "L")
            pdf.cell(30, 6, f"{row['R²']:.4f}", 1, 0
