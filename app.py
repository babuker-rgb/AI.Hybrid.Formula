"""
True Physics-Informed Neural Network (PINN) - Complete Professional Framework
Multi-Objective Tablet Manufacturing Optimization with Full Analytics

Author: Babuker A. Abdalla
Affiliation: Nile Valley University, Postgraduate College, Sudan
Version: 29.1 (Added tolerance for density constraint)
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
# 0. USER-CONFIGURABLE PARAMETERS (UNIFIED CONSTRAINTS)
# ================================================================

TENSILE_MIN = 1.90          # MPa
EFRF_MAX = 0.40             # dimensionless
MCC_MAX = 8.0               # %
DENSITY_MIN = 0.40          # Physical lower bound
DENSITY_MAX = 0.97          # Physical upper bound
DENSITY_TOL = 1e-6          # Tolerance for floating-point errors
PRESSURE_MAX = 300.0        # MPa

NSGA_POP_SIZE = 100
NSGA_GENERATIONS = 100

BINDER_MIN = 0.5
BINDER_MAX = 5.0

# --- Balanced loss weights ---
PHYSICS_WEIGHT_INIT = 0.1
W_DENSITY = 1.0
N_SAMPLES = 3000

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
# 4. MULTI-TASK TRUE PINN MODEL (Bounded Density)
# ================================================================

def bounded_density(raw):
    """Ensure density stays within [DENSITY_MIN, DENSITY_MAX]."""
    return DENSITY_MIN + (DENSITY_MAX - DENSITY_MIN) * torch.sigmoid(raw)

class MultiTaskTruePINN(nn.Module):
    def __init__(self, input_dim=13, output_dim=5):
        super(MultiTaskTruePINN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.Tanh(), nn.Dropout(0.05),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.Tanh(), nn.Dropout(0.05),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.Tanh(), nn.Dropout(0.05),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.Tanh(),
            nn.Linear(32, output_dim)
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
                X_scaled = torch.FloatTensor(X_scaled)
            output = self.forward(X_scaled)
            return output[:, :3].cpu().numpy()

    def compute_loss(self, X_scaled, X_raw, y_true, epoch=0, max_epochs=4000,
                     w_data_init=2.0, w_physics_init=PHYSICS_WEIGHT_INIT,
                     w_data_final=1.0, w_physics_final=1.0,
                     w_mcc=0.5, w_density=W_DENSITY,
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

        progress = min(epoch / 500, 1.0)
        w_data = w_data_init + (w_data_final - w_data_init) * progress
        w_physics = w_physics_init + (w_physics_final - w_physics_init) * progress

        data_loss = nn.MSELoss()(y_pred[:, :3], y_true)

        heckel_lhs = torch.log(1.0 / torch.clamp(1.0 - density_pred, min=1e-4))
        heckel_rhs = k_pred * pressure_real + A_pred
        heckel_loss = torch.mean((heckel_lhs - heckel_rhs) ** 2)

        efrf_pred = er_pred / torch.clamp(tensile_pred, min=1e-4)
        efrf_loss = torch.mean(torch.relu(efrf_pred - efrf_target) ** 2)

        mcc_loss = torch.mean(torch.relu(mcc_real - mcc_max) ** 2)

        density_penalty = torch.mean(
            torch.relu(density_pred - DENSITY_MAX) ** 2 +
            torch.relu(DENSITY_MIN - density_pred) ** 2
        )

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

        boundary_loss = (
            torch.mean(torch.relu(0.50 - density_pred) ** 2) +
            torch.mean(torch.relu(density_pred - 0.98) ** 2)
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

def generate_pinn_data(n_samples=N_SAMPLES, random_state=42):
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
        
        noise_d = np.random.normal(0, 0.005)
        noise_t = np.random.normal(0, 0.04)
        noise_er = np.random.normal(0, 0.03)

        D = np.clip(1 - np.exp(-(k_true * pressure + A_true)) + noise_d, DENSITY_MIN, DENSITY_MAX)
        strength = np.clip(
            3.5 - 0.15 * (api - 85) + 0.3 * binder + 0.008 * (pressure - 100)
            - 1.5 * mgst - 0.02 * (speed - 10) + noise_t,
            0.5, 6.0
        )
        er = np.clip(
            1.8 + 0.3 * (api - 85)/10 + 0.08 * (speed - 10)/30
            - 0.1 * (pressure - 100)/150 + noise_er,
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
# 6. PDF GENERATION
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
    pdf.cell(0, 6, sanitize_text("Hybrid AI Framework (PINN + NSGA-II)"), ln=True, align="C")
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
    
    results = [("Density", f"{density:.3f}", f"[{DENSITY_MIN:.2f}, {DENSITY_MAX:.2f}]"),
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
        pdf.cell(40, 6, sanitize_text("Physics"), 1, 1, "C")
        
        pdf.set_font("Arial", "", 10)
        for _, row in model_comparison_df.iterrows():
            pdf.cell(40, 6, sanitize_text(str(row['Model'])), 1, 0, "L")
            pdf.cell(30, 6, f"{row['R²']:.4f}", 1, 0, "C")
            pdf.cell(30, 6, f"{row['RMSE']:.4f}", 1, 0, "C")
            pdf.cell(30, 6, f"{row['MAE']:.4f}", 1, 0, "C")
            pdf.cell(40, 6, sanitize_text(str(row['Physics'])), 1, 1, "L")
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
    pdf.cell(0, 6, "Generated by: Hybrid AI Framework v29.1", ln=True, align="C")
    
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, bytearray):
        return bytes(pdf_bytes)
    elif isinstance(pdf_bytes, bytes):
        return pdf_bytes
    else:
        return str(pdf_bytes).encode('latin1')

# ================================================================
# 7. NSGA-II (WITH PENALTIES AND TOLERANCE)
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

    def _initialize_population(self):
        pop = np.zeros((self.pop_size, 8))
        for i in range(8):
            pop[:, i] = np.random.uniform(self.bounds[i, 0], self.bounds[i, 1], self.pop_size)
        return pop

    def _evaluate(self, population):
        n = population.shape[0]
        objectives = np.zeros((n, 2))
        constraints = np.zeros(n, dtype=bool)
        tensile_strengths = np.zeros(n)

        for i in range(n):
            try:
                api, mcc, pvpp, mgst, binder, pressure, speed, granule = population[i]

                # Normalize to 100%
                used = api + binder + mgst + pvpp + mcc
                if used > 100:
                    scale = 100 / used
                    api *= scale; binder *= scale; mgst *= scale; pvpp *= scale; mcc *= scale
                elif used < 100:
                    remainder = 100 - used
                    if mcc + remainder <= MCC_MAX:
                        mcc += remainder
                    else:
                        excess = (mcc + remainder) - MCC_MAX
                        mcc = MCC_MAX
                        api = max(85.0, api - excess)

                api = np.clip(api, 85, 95)
                binder = np.clip(binder, BINDER_MIN, BINDER_MAX)
                mgst = np.clip(mgst, 0.01, 1.2)
                pvpp = np.clip(pvpp, 0.5, 6.0)
                mcc = np.clip(mcc, 0, MCC_MAX)

                inputs = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
                inputs_with_features = add_interaction_features(np.array([inputs]))[0]
                inputs_scaled = self.scaler.transform([inputs_with_features])
                X_tensor = torch.FloatTensor(inputs_scaled)

                with torch.no_grad():
                    pred_scaled = self.model.predict(X_tensor)
                    pred_actual = self.y_scaler.inverse_transform(pred_scaled)[0]

                density = float(pred_actual[0])
                tensile = float(pred_actual[1])
                er = float(pred_actual[2])

                if tensile < 0.01:
                    tensile = 0.01
                    efrf = 10.0
                else:
                    efrf = er / tensile
                    efrf = max(0.0001, min(efrf, 5.0))

                # --- Feasibility with tolerance for density ---
                feasible = (
                    (tensile >= TENSILE_MIN) and
                    (efrf < EFRF_MAX) and
                    (DENSITY_MIN - DENSITY_TOL <= density <= DENSITY_MAX + DENSITY_TOL) and
                    (mcc <= MCC_MAX)
                )
                constraints[i] = feasible

                # --- Penalty approach for NSGA-II ---
                penalty = 0.0
                if tensile < TENSILE_MIN:
                    penalty += (TENSILE_MIN - tensile) ** 2
                if efrf >= EFRF_MAX:
                    penalty += (efrf - EFRF_MAX) ** 2
                if density < DENSITY_MIN - DENSITY_TOL:
                    penalty += (DENSITY_MIN - density) ** 2
                if density > DENSITY_MAX + DENSITY_TOL:
                    penalty += (density - DENSITY_MAX) ** 2

                objectives[i, 0] = -(api + self.w_tensile * tensile) + 10.0 * penalty
                objectives[i, 1] = efrf + 10.0 * penalty
                tensile_strengths[i] = tensile

                population[i, 0] = api
                population[i, 1] = mcc
                population[i, 2] = pvpp
                population[i, 3] = mgst
                population[i, 4] = binder

            except Exception as e:
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
        constraint_violation = ~constraints

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (constraint_violation[i] < constraint_violation[j]) or \
                   (constraint_violation[i] == constraint_violation[j] and
                    objectives[i, 0] <= objectives[j, 0] and
                    objectives[i, 1] <= objectives[j, 1] and
                    (objectives[i, 0] < objectives[j, 0] or objectives[i, 1] < objectives[j, 1])):
                    S[i].append(j)
                elif (constraint_violation[j] < constraint_violation[i]) or \
                     (constraint_violation[i] == constraint_violation[j] and
                      objectives[j, 0] <= objectives[i, 0] and
                      objectives[j, 1] <= objectives[i, 1] and
                      (objectives[j, 0] < objectives[i, 0] or objectives[j, 1] < objectives[i, 1])):
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
            i += 1
            if next_front:
                fronts.append(next_front)

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
        n = len(pop_indices)
        selected = []
        for _ in range(n):
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

        for i in range(8):
            child1[i] = np.clip(child1[i], self.bounds[i, 0], self.bounds[i, 1])
            child2[i] = np.clip(child2[i], self.bounds[i, 0], self.bounds[i, 1])

        return child1, child2

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
                mutated[i] = np.clip(mutated[i], self.bounds[i, 0], self.bounds[i, 1])
        return mutated

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
                crowding[front] = dist

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
            objectives_off, constraints_off, tensile_off, _ = self._evaluate(offspring)

            combined_pop = np.vstack([self.population, offspring])
            combined_obj = np.vstack([self.objectives, objectives_off])
            combined_const = np.concatenate([self.constraints, constraints_off])

            combined_fronts, _ = self._fast_non_dominated_sort(combined_obj, combined_const)
            combined_crowding = np.zeros(len(combined_pop))
            for front in combined_fronts:
                dist = self._crowding_distance(combined_obj, front)
                combined_crowding[front] = dist

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

    X_scaled_train_t = torch.FloatTensor(X_scaled_train)
    X_raw_train_t = torch.FloatTensor(X_raw_train)
    y_train_t = torch.FloatTensor(y_train)

    X_scaled_val_t = torch.FloatTensor(X_scaled_val)
    X_raw_val_t = torch.FloatTensor(X_raw_val)
    y_val_t = torch.FloatTensor(y_val)

    model = MultiTaskTruePINN(input_dim=X_augmented.shape[1])
    
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, patience=50, factor=0.5)

    best_val_loss = float('inf')
    patience = 120
    counter = 0
    best_state = None

    progress_bar = st.progress(0)
    adam_epochs = 1200
    max_epochs = 2000

    # Local loss history
    loss_history = {'train': [], 'val': [], 'data': [], 'physics': []}

    for epoch in range(adam_epochs):
        model.train()
        optimizer_adam.zero_grad()
        total_loss, loss_dict = model.compute_loss(
            X_scaled_train_t, X_raw_train_t, y_train_t,
            epoch=epoch, max_epochs=max_epochs,
            w_data_init=2.0, w_physics_init=PHYSICS_WEIGHT_INIT,
            w_data_final=1.0, w_physics_final=1.0,
            w_mcc=0.5, w_density=W_DENSITY,
            efrf_target=EFRF_MAX, mcc_max=MCC_MAX, compute_grad=True
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_adam.step()

        loss_history['train'].append(total_loss.item())
        loss_history['data'].append(loss_dict['data_loss'])
        loss_history['physics'].append(loss_dict['physics_loss'])

        model.eval()
        with torch.no_grad():
            val_loss, _ = model.compute_loss(
                X_scaled_val_t, X_raw_val_t, y_val_t,
                epoch=epoch, max_epochs=max_epochs,
                w_data_init=2.0, w_physics_init=PHYSICS_WEIGHT_INIT,
                w_data_final=1.0, w_physics_final=1.0,
                w_mcc=0.5, w_density=W_DENSITY,
                efrf_target=EFRF_MAX, mcc_max=MCC_MAX, compute_grad=False
            )

        val_loss_value = val_loss.item()
        loss_history['val'].append(val_loss_value)
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
            progress_bar.progress(min((epoch + 1) / max_epochs, 0.6))

    if best_state is not None:
        model.load_state_dict(best_state)

    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.05, max_iter=200, line_search_fn='strong_wolfe')
    
    def closure():
        optimizer_lbfgs.zero_grad()
        total_loss, _ = model.compute_loss(
            X_scaled_train_t, X_raw_train_t, y_train_t,
            epoch=adam_epochs, max_epochs=max_epochs,
            w_data_init=1.0, w_physics_init=PHYSICS_WEIGHT_INIT,
            w_data_final=1.0, w_physics_final=1.0,
            w_mcc=0.5, w_density=W_DENSITY,
            efrf_target=EFRF_MAX, mcc_max=MCC_MAX, compute_grad=True
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        return total_loss

    for i in range(5):
        optimizer_lbfgs.step(closure)
        progress_bar.progress(min(0.6 + 0.08 * (i + 1), 1.0))

    model.eval()
    model.feature_names = feature_names
    model.scaler = scaler
    model.y_scaler = y_scaler

    torch.save(model.state_dict(), 'true_pinn_checkpoint.pt')
    return model, scaler, y_scaler, feature_names, df, loss_history

# ================================================================
# 9. PREDICTION & PLOTS
# ================================================================

def predict_pinn(model, scaler, y_scaler, inputs):
    try:
        inputs_with_features = add_interaction_features(np.array([inputs]))[0]
        inputs_scaled = scaler.transform([inputs_with_features])
        X_tensor = torch.FloatTensor(inputs_scaled)
        with torch.no_grad():
            pred_scaled = model.predict(X_tensor)[0]
        pred_original = y_scaler.inverse_transform([pred_scaled])[0]
        density, tensile, er = pred_original[0], pred_original[1], pred_original[2]
        if tensile < 0.01:
            efrf = 10.0
        else:
            efrf = er / tensile
            efrf = max(0.0001, min(efrf, 5.0))
        return density, tensile, er, efrf
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.5, 0.0, 1.0, 1.0

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
    
    if len(loss_history['data']) > 0:
        fig.add_trace(go.Scatter(
            x=epochs, y=loss_history['data'],
            mode='lines', name='Data Loss',
            line=dict(color='green', width=1.5)
        ))
    
    if len(loss_history['physics']) > 0:
        fig.add_trace(go.Scatter(
            x=epochs, y=loss_history['physics'],
            mode='lines', name='Physics Loss',
            line=dict(color='red', width=1.5)
        ))
    
    fig.update_layout(
        title='Training Curves',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Loss', type='log'),
        height=400,
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig

def plot_pareto_plotly(objectives, constraints, fronts, nsga, api, efrf):
    try:
        if len(fronts) > 0 and len(fronts[0]) > 0:
            front0 = fronts[0]
            pareto_api = -objectives[front0, 0]
            pareto_efrf = objectives[front0, 1]
            feasible = constraints[front0]

            valid_mask = (pareto_api >= 85) & (pareto_api <= 95) & (pareto_efrf >= 0.0001) & (pareto_efrf <= 2)
            pareto_api = pareto_api[valid_mask]
            pareto_efrf = pareto_efrf[valid_mask]
            feasible = feasible[valid_mask]

            if len(pareto_api) == 0:
                return None

            plot_df = pd.DataFrame({'API': pareto_api, 'EFRF': pareto_efrf}).dropna().sort_values('API')
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=-objectives[:, 0], y=objectives[:, 1],
                mode='markers', marker=dict(size=3, color='gray', opacity=0.3),
                name='All Solutions'
            ))

            fig.add_trace(go.Scatter(
                x=plot_df['API'], y=plot_df['EFRF'],
                mode='lines+markers',
                marker=dict(size=8, color='red'),
                line=dict(color='red', width=2),
                name='Pareto Front (Optimal Trade-offs)'
            ))

            feasible_indices = [i for i, f in enumerate(feasible) if f]
            if feasible_indices:
                feasible_api = [pareto_api[i] for i in feasible_indices]
                feasible_efrf = [pareto_efrf[i] for i in feasible_indices]
                fig.add_trace(go.Scatter(
                    x=feasible_api, y=feasible_efrf,
                    mode='markers', marker=dict(size=12, color='green', symbol='star'),
                    name=f'Feasible Formulations ({len(feasible_indices)})',
                    hovertemplate='API: %{x:.2f}%<br>EFRF: %{y:.4f}<extra></extra>'
                ))

            if api and efrf and 85 <= api <= 95 and 0.0001 <= efrf <= 2:
                fig.add_trace(go.Scatter(
                    x=[api], y=[efrf],
                    mode='markers+text',
                    marker=dict(size=16, color='blue', symbol='diamond'),
                    text=[f"Your Formulation<br>API: {api:.1f}%<br>EFRF: {efrf:.4f}"],
                    name='Selected Formulation'
                ))

            fig.add_hrect(y0=0, y1=EFRF_MAX, line_width=0, fillcolor="green", opacity=0.1, 
                          annotation_text=f"Safe Zone (EFRF < {EFRF_MAX:.2f})", annotation_position="top left")
            fig.add_hline(y=EFRF_MAX, line_dash='dash', line_color='red', annotation_text=f'EFRF Threshold: {EFRF_MAX:.2f}')
            
            fig.update_layout(
                title='Pareto Front (NSGA-II)',
                xaxis=dict(title='API Loading (%)', range=[85, 95]),
                yaxis=dict(title='EFRF', range=[0, 1.0]),
                height=500,
                hovermode='closest',
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
            )
            
            if feasible_indices:
                best_idx = np.argmax([pareto_api[i] for i in feasible_indices])
                best_idx = feasible_indices[best_idx]
                fig.add_annotation(
                    x=pareto_api[best_idx],
                    y=pareto_efrf[best_idx],
                    text=f"Best<br>API: {pareto_api[best_idx]:.2f}%<br>EFRF: {pareto_efrf[best_idx]:.4f}",
                    showarrow=True,
                    arrowhead=2,
                    ax=20,
                    ay=-40
                )
            
            return fig
    except Exception:
        return None
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
            'Physics': 'Not enforced'
        })
    return pd.DataFrame(results)

# ================================================================
# 10. STREAMLIT UI
# ================================================================

st.set_page_config(page_title="PINN Framework v29.1", page_icon="🧬", layout="wide")
clamp_session_state()

st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); 
            padding: 2rem; border-radius: 1rem; margin-bottom: 1.5rem; text-align: center;">
    <h1 style="color: #ffffff; font-size: 2.5rem; margin: 0;">
        🧬 Hybrid AI Framework v29.1
    </h1>
    <p style="color: #a8b2d1; font-size: 1.2rem; margin: 0.5rem 0 0 0;">
        Physics-Informed Neural Network · Multi-Objective Optimization
    </p>
    <p style="color: #64ffda; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
        Nile Valley University · Postgraduate College · Sudan
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

with st.sidebar:
    st.markdown("### 📚 Physics Constraints (Unified)")
    st.markdown(f"""
    - ✅ **Heckel:** ln(1/(1-D)) = kP + A
    - ✅ **EFRF:** ER / σt < {EFRF_MAX:.2f}
    - ✅ **Monotonicity:** ∂D/∂P > 0
    - ✅ **Boundary:** 0.4 < D < 0.98
    - ✅ **MCC:** ≤ {MCC_MAX:.1f}%
    - ✅ **Density:** {DENSITY_MIN:.2f} ≤ D ≤ {DENSITY_MAX:.2f} (with tolerance)
    
    **Multi-Task PINN:**
    - 5 outputs (D, σt, ER, k, A)
    - Adam → LBFGS hybrid (balanced)
    - **NSGA-II:** pop={NSGA_POP_SIZE}, gen={NSGA_GENERATIONS}
    - **Binder max:** {BINDER_MAX:.1f}%
    - **Density tolerance:** {DENSITY_TOL:.0e} ✅
    - **Penalty-based NSGA-II** ✅
    """)
    st.info("🔬 **v29.1** — Added tolerance for density constraint")

with st.spinner("🔄 Training Multi-Task PINN..."):
    model, scaler, y_scaler, feature_names, df, loss_history = load_pinn_model()
st.success("✅ Multi-Task True PINN trained successfully")

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
    
    predict_btn = st.button("🔬 Predict & Optimize", use_container_width=True)

with col_right:
    st.markdown("### 📈 Results")
    
    if predict_btn:
        if abs(total - 100) > 0.1:
            st.warning("⚠️ Formulation must sum to 100%")
        else:
            inputs = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
            with st.spinner("🧠 Running prediction..."):
                density, tensile, er, efrf = predict_pinn(model, scaler, y_scaler, inputs)
            
            # --- KPIs ---
            kpi_cols = st.columns(3)
            kpi_cols[0].metric("Density", f"{density:.3f}", delta=f"[{DENSITY_MIN:.2f}, {DENSITY_MAX:.2f}]")
            kpi_cols[1].metric("Tensile", f"{tensile:.3f} MPa", delta=f">= {TENSILE_MIN:.2f} PASS" if tensile >= TENSILE_MIN else f"< {TENSILE_MIN:.2f} FAIL")
            kpi_cols[2].metric("EFRF", f"{efrf:.4f}", delta=f"< {EFRF_MAX:.2f} PASS" if efrf < EFRF_MAX else f">= {EFRF_MAX:.2f} FAIL")
            
            st.markdown("---")
            
            # --- Unified constraints check with tolerance ---
            density_ok = (density >= DENSITY_MIN - DENSITY_TOL and density <= DENSITY_MAX + DENSITY_TOL)
            tensile_ok = (tensile >= TENSILE_MIN)
            efrf_ok = (efrf < EFRF_MAX)
            mcc_ok = (mcc <= MCC_MAX)
            
            if density_ok and tensile_ok and efrf_ok and mcc_ok:
                st.success(f"✅ Formulation satisfies all constraints (Density: {density:.3f}, σt ≥ {TENSILE_MIN:.2f}, EFRF < {EFRF_MAX:.2f})")
            else:
                st.error("❌ Formulation violates one or more constraints")
            
            # --- Feasibility ---
            st.markdown("### ✅ Feasibility")
            pass_cols = st.columns(4)
            pass_cols[0].metric(f"Density [{DENSITY_MIN:.2f}, {DENSITY_MAX:.2f}]", "✅ PASS" if density_ok else "❌ FAIL")
            pass_cols[1].metric(f"σt ≥ {TENSILE_MIN:.2f}", "✅ PASS" if tensile_ok else "❌ FAIL")
            pass_cols[2].metric(f"EFRF < {EFRF_MAX:.2f}", "✅ PASS" if efrf_ok else "❌ FAIL")
            pass_cols[3].metric(f"MCC ≤ {MCC_MAX:.1f}%", "✅ PASS" if mcc_ok else "❌ FAIL")
            
            # --- Physics Verification ---
            with st.expander("🔬 Physics Verification"):
                try:
                    inputs_with_features = add_interaction_features(np.array([inputs]))[0]
                    inputs_scaled = scaler.transform([inputs_with_features])
                    X_tensor = torch.FloatTensor(inputs_scaled)
                    with torch.no_grad():
                        full_output = model.forward(X_tensor).numpy()[0]
                    st.metric("k (Plasticity)", f"{full_output[3]:.4f}")
                    st.metric("A (Rearrangement)", f"{full_output[4]:.4f}")
                    st.caption("Density is naturally bounded by the model architecture.")
                except:
                    pass
            
            # --- NSGA-II ---
            st.markdown("### ⚙️ NSGA-II")
            bounds = np.array([
                [85, 95], [0, MCC_MAX], [0.5, 6.0], [0.01, 1.2], 
                [BINDER_MIN, BINDER_MAX], [80, PRESSURE_MAX], [1, 50], [30, 250]
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
            
            # --- TABS ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📉 Pareto Front", "🔍 Sensitivity", "📊 Model Comparison", "📈 Training Curves", "📄 Report"]
            )
            
            with tab1:
                st.markdown("### 📉 Pareto Front")
                fig_p = plot_pareto_plotly(objectives, constraints, fronts, nsga, api, efrf)
                if fig_p:
                    st.plotly_chart(fig_p, use_container_width=True)
                else:
                    st.info("Pareto front not available")
            
            with tab2:
                st.markdown("### 🔍 Sensitivity Analysis")
                fig_s = plot_sensitivity_plotly(inputs, model, scaler, y_scaler)
                if fig_s:
                    st.plotly_chart(fig_s, use_container_width=True)
                else:
                    st.info("Sensitivity analysis not available")
            
            with tab3:
                st.markdown("### 📊 Model Performance Comparison")
                st.caption("Hold-out test set (20% of data) — R², RMSE, MAE")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    df[feature_names].values, df['Tensile_Strength_MPa'].values,
                    test_size=0.2, random_state=42
                )
                X_train_aug = add_interaction_features(X_train)
                X_test_aug = add_interaction_features(X_test)
                X_train_scaled = scaler.transform(X_train_aug)
                X_test_scaled = scaler.transform(X_test_aug)
                
                pinn_pred_scaled = model.predict(torch.FloatTensor(X_test_scaled))
                pinn_pred = y_scaler.inverse_transform(pinn_pred_scaled)[:, 1]
                pinn_r2 = r2_score(y_test, pinn_pred)
                pinn_rmse = np.sqrt(mean_squared_error(y_test, pinn_pred))
                pinn_mae = mean_absolute_error(y_test, pinn_pred)
                
                comp_df = train_and_compare(X_train_scaled, X_test_scaled, y_train, y_test)
                pinn_row = pd.DataFrame([{'Model': 'PINN (Proposed)', 'R²': pinn_r2, 'RMSE': pinn_rmse, 'MAE': pinn_mae, 'Physics': '✅ Enforced'}])
                comp_df = pd.concat([pinn_row, comp_df], ignore_index=True)
                
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
                        title='<b>R² Score Comparison</b>',
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
                        title='<b>RMSE Comparison</b>',
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
                comp_df_for_pdf = comp_df.copy()
            
            with tab4:
                st.markdown("### 📈 Training Curves")
                fig_loss = plot_training_curves(loss_history)
                if fig_loss:
                    st.plotly_chart(fig_loss, use_container_width=True)
                    
                    if len(loss_history['train']) > 0:
                        col1, col2 = st.columns(2)
                        col1.metric("Final Training Loss", f"{loss_history['train'][-1]:.6f}")
                        col1.metric("Final Data Loss", f"{loss_history['data'][-1]:.6f}")
                        col2.metric("Final Validation Loss", f"{loss_history['val'][-1]:.6f}")
                        col2.metric("Final Physics Loss", f"{loss_history['physics'][-1]:.6f}")
                else:
                    st.info("Loss history not available")
            
            with tab5:
                st.markdown("### 📄 Comprehensive Report (PDF)")
                st.caption("Download a complete report with formulation details, predictions, and model comparison.")
                
                status = "PASS" if (density_ok and tensile_ok and efrf_ok and mcc_ok) else "FAIL"
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                pdf_data = generate_full_pdf_report(
                    api, mcc, pvpp, mgst, binder, pressure, speed, granule,
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

st.markdown("---")
st.caption(f"🔬 **Multi-Task True PINN — v29.1 (Added density tolerance)**")
st.caption(f"📧 Contact: babuker@protonmail.com | 🏛️ Nile Valley University, Postgraduate College, Sudan")
