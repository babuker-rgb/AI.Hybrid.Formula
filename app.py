"""
True Physics-Informed Neural Network (PINN) - Complete Professional Framework
Multi-Objective Tablet Manufacturing Optimization with Full Analytics

Author: Babuker A. Abdalla
Affiliation: Nile Valley University, Sudan
Version: 24.1 (XGBoost Import Fix)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
warnings.filterwarnings('ignore')

# ================================================================
# 0. SAFE IMPORTS WITH FALLBACKS
# ================================================================

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    st.warning("XGBoost not available. Model comparison will exclude XGBoost.")

# ================================================================
# 1. SESSION STATE INITIALIZATION
# ================================================================

DEFAULTS = {
    'api': 90.5,
    'binder': 2.7,
    'pvpp': 3.0,
    'mgst': 0.20,
    'mcc': 5.0,      # MCC is now a slider
    'pressure': 230.0,
    'speed': 12.0,
    'granule': 125.0
}

RANGES = {
    'api': (85.0, 95.0),
    'binder': (0.5, 4.0),
    'pvpp': (0.5, 6.0),
    'mgst': (0.01, 1.2),
    'mcc': (0.0, 8.0),
    'pressure': (80.0, 280.0),
    'speed': (1.0, 50.0),
    'granule': (30.0, 250.0)
}

def safe_initialize():
    for key in DEFAULTS:
        if key not in st.session_state:
            st.session_state[key] = DEFAULTS[key]
        else:
            try:
                val = float(st.session_state[key])
                min_val, max_val = RANGES[key]
                if val < min_val or val > max_val:
                    st.session_state[key] = DEFAULTS[key]
            except (ValueError, TypeError):
                st.session_state[key] = DEFAULTS[key]

safe_initialize()

# ================================================================
# 2. FEATURE ENGINEERING (SAFE)
# ================================================================

def add_interaction_features(X_raw):
    """
    Add interaction features to capture nonlinear relationships.
    Input: X_raw (n_samples, 8)
    Output: X_augmented (n_samples, 13)
    """
    pressure = X_raw[:, 5:6]
    binder = X_raw[:, 4:5]
    api = X_raw[:, 0:1]
    speed = X_raw[:, 6:7]
    mcc = X_raw[:, 1:2]
    
    # Safe division with clipping to avoid infinity
    pressure_speed = pressure / (speed + 0.1)
    pressure_speed = np.clip(pressure_speed, 0, 1000)
    
    api_mcc = api / (mcc + 0.1)
    api_mcc = np.clip(api_mcc, 0, 1000)
    
    binder_speed = binder / (speed + 0.1)
    binder_speed = np.clip(binder_speed, 0, 100)
    
    # Interaction features
    pressure_binder = pressure * binder
    pressure_api = pressure * api
    
    return np.concatenate([
        X_raw, 
        pressure_binder, 
        pressure_api, 
        pressure_speed, 
        api_mcc, 
        binder_speed
    ], axis=1)

# ================================================================
# 3. MULTI-TASK TRUE PINN MODEL
# ================================================================

class MultiTaskTruePINN(nn.Module):
    """
    Multi-Task Physics-Informed Neural Network with:
    - 5 outputs: Density, Tensile, ER, k(X), A(X)
    - Physical activation functions for primary outputs
    - All physics constraints embedded in loss
    """
    
    def __init__(self, input_dim=13, output_dim=5):
        super(MultiTaskTruePINN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
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

    def get_heckel_params(self, X):
        output = self.forward(X)
        return output[:, 3], output[:, 4]

    def compute_loss(self, X_scaled, X_raw, y_true,
                     epoch=0, max_epochs=4000,
                     w_data_init=2.0, w_physics_init=0.2,
                     w_data_final=1.0, w_physics_final=1.0,
                     w_mcc=0.5, w_density=0.5,
                     efrf_target=0.35,
                     mcc_max=8.0,
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

        density_penalty = torch.mean(torch.relu(density_pred - 0.99) ** 2)

        if compute_grad:
            if not X_scaled.requires_grad:
                X_scaled.requires_grad_(True)

            y_pred_grad = self.forward(X_scaled)
            density_grad = y_pred_grad[:, 0]

            grad_density = torch.autograd.grad(
                outputs=density_grad,
                inputs=X_scaled,
                grad_outputs=torch.ones_like(density_grad),
                create_graph=True,
                retain_graph=True
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

        k_regularization = torch.mean(torch.relu(k_pred - 0.1) ** 2) + torch.mean(torch.relu(0.005 - k_pred) ** 2)
        A_regularization = torch.mean(torch.relu(A_pred - 2.0) ** 2) + torch.mean(torch.relu(0.5 - A_pred) ** 2)

        total_loss = (
            w_data * data_loss +
            w_physics * (heckel_loss + efrf_loss + monotonic_loss + boundary_loss) +
            w_mcc * mcc_loss +
            w_density * density_penalty +
            0.1 * k_regularization +
            0.1 * A_regularization
        )

        loss_dict = {
            'data_loss': data_loss.item(),
            'heckel_loss': heckel_loss.item(),
            'efrf_loss': efrf_loss.item(),
            'mcc_loss': mcc_loss.item(),
            'density_penalty': density_penalty.item(),
            'monotonic_loss': monotonic_loss.item() if compute_grad else 0.0,
            'boundary_loss': boundary_loss.item(),
            'k_reg': k_regularization.item(),
            'A_reg': A_regularization.item(),
            'w_data': w_data,
            'w_physics': w_physics,
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict

    def predict(self, X_scaled):
        self.eval()
        with torch.no_grad():
            if not isinstance(X_scaled, torch.Tensor):
                X_scaled = torch.FloatTensor(X_scaled)
            output = self.forward(X_scaled)
            return output[:, :3].numpy()


# ================================================================
# 4. DATA GENERATION
# ================================================================

def generate_pinn_data(n_samples=600, random_state=42):
    np.random.seed(random_state)
    X = np.zeros((n_samples, 8))
    y = np.zeros((n_samples, 3))

    n_random = n_samples // 2

    for i in range(n_samples):
        if i < n_random:
            api = np.random.uniform(85, 95)
            binder = np.random.uniform(0.5, 4.0)
            mgst = np.random.uniform(0.01, 1.2)
            pvpp = np.random.uniform(0.5, 6.0)
            mcc = np.random.uniform(0, 8.0)
            pressure = np.random.uniform(80, 280)
            speed = np.random.uniform(1, 50)
            granule = np.random.uniform(30, 250)
        else:
            api = np.random.normal(90.5, 1.5)
            api = np.clip(api, 85, 95)
            binder = np.random.normal(2.8, 0.4)
            binder = np.clip(binder, 0.5, 4.0)
            mgst = np.random.normal(0.15, 0.06)
            mgst = np.clip(mgst, 0.01, 1.2)
            pvpp = np.random.normal(3.0, 0.5)
            pvpp = np.clip(pvpp, 0.5, 6.0)
            mcc = np.random.normal(5.0, 1.0)
            mcc = np.clip(mcc, 0, 8.0)
            pressure = np.random.normal(230, 15)
            pressure = np.clip(pressure, 80, 280)
            speed = np.random.normal(10, 3)
            speed = np.clip(speed, 1, 50)
            granule = np.random.normal(125, 20)
            granule = np.clip(granule, 30, 250)

        total_others = api + binder + mgst + pvpp + mcc
        if total_others > 100:
            scale = 100 / total_others
            api *= scale
            binder *= scale
            mgst *= scale
            pvpp *= scale
            mcc *= scale
        else:
            remainder = 100 - (api + binder + mgst + pvpp)
            if mcc + remainder <= 8.0:
                mcc += remainder
            else:
                excess = (mcc + remainder) - 8.0
                mcc = 8.0
                api -= excess

        api = np.clip(api, 85, 95)
        binder = np.clip(binder, 0.5, 4.0)
        mgst = np.clip(mgst, 0.01, 1.2)
        pvpp = np.clip(pvpp, 0.5, 6.0)
        mcc = np.clip(mcc, 0, 8.0)

        X[i] = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]

        k_true = 0.035 * (1 - 0.4 * (api - 85)/10) * (1 - 0.2 * (speed - 10)/30)
        k_true = max(k_true, 0.008)
        A_true = 1.2 + 0.1 * (binder - 1.5) - 0.2 * (mgst - 0.5)
        D = 1 - np.exp(-(k_true * pressure + A_true))
        D = np.clip(D, 0.4, 0.99)

        strength = 3.5 - 0.15 * (api - 85) + 0.3 * binder + 0.008 * (pressure - 100) - 1.5 * mgst - 0.02 * (speed - 10)
        strength = np.clip(strength, 0.5, 6.0)

        er = 1.8 + 0.3 * (api - 85)/10 + 0.08 * (speed - 10)/30 - 0.1 * (pressure - 100)/150
        er = np.clip(er, 0.5, 4.0)

        y[i] = [D, strength, er]

    feature_names = ['API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
                     'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm']
    df = pd.DataFrame(X, columns=feature_names)
    df['Density'] = y[:, 0]
    df['Tensile_Strength_MPa'] = y[:, 1]
    df['Elastic_Recovery_%'] = y[:, 2]
    return df, feature_names


# ================================================================
# 5. PDF GENERATION (NO UNICODE ERRORS)
# ================================================================

def sanitize_text(text):
    """Replace problematic Unicode characters with ASCII equivalents"""
    replacements = {
        'σ': 'sigma',
        'µ': 'um',
        '≥': '>=',
        '≤': '<=',
        '✅': '[PASS]',
        '❌': '[FAIL]',
        '🎉': '[SUCCESS]',
        '⚠️': '[WARNING]',
        '🧠': 'AI',
        '🧬': 'PINN',
        '→': '->',
        '↑': '↑',
        '↓': '↓'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def create_pdf_report(api, mcc, pvpp, mgst, binder, pressure, speed, granule,
                      density, tensile, er, efrf, status, timestamp):
    """Generate a professional PDF report with no Unicode errors."""
    
    pdf = FPDF()
    pdf.add_page()
    
    # HEADER
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, sanitize_text("Formulation Report"), ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 6, sanitize_text("Hybrid AI Framework for Tablet Manufacturing Optimization"), ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Date: {timestamp}", ln=True, align="C")
    pdf.ln(8)
    
    # 1. FORMULATION SUMMARY
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, sanitize_text("1. Formulation Summary"), ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    
    components = [
        ("API", f"{api:.1f}%", "Paracetamol"),
        ("MCC", f"{mcc:.1f}%", "Filler/Binder"),
        ("PVPP", f"{pvpp:.1f}%", "Superdisintegrant"),
        ("Mg-St", f"{mgst:.2f}%", "Lubricant"),
        ("Binder", f"{binder:.1f}%", "Binding Agent"),
        ("TOTAL", f"{api+binder+pvpp+mgst+mcc:.1f}%", "100% Complete")
    ]
    
    pdf.set_font("Arial", "B", 10)
    pdf.cell(50, 6, sanitize_text("Component"), 1, 0, "C")
    pdf.cell(30, 6, sanitize_text("Value"), 1, 0, "C")
    pdf.cell(80, 6, sanitize_text("Function"), 1, 1, "C")
    
    pdf.set_font("Arial", "", 10)
    for comp, val, func in components:
        pdf.cell(50, 6, sanitize_text(comp), 1, 0, "L")
        pdf.cell(30, 6, sanitize_text(val), 1, 0, "C")
        pdf.cell(80, 6, sanitize_text(func), 1, 1, "L")
    
    pdf.ln(5)
    
    # 2. PROCESS PARAMETERS
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, sanitize_text("2. Process Parameters"), ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    
    params = [
        ("Compaction Pressure", f"{pressure:.1f} MPa"),
        ("Punch Speed", f"{speed:.1f} rpm"),
        ("Granule Size", f"{granule:.1f} um"),
    ]
    
    pdf.set_font("Arial", "B", 10)
    pdf.cell(60, 6, sanitize_text("Parameter"), 1, 0, "C")
    pdf.cell(60, 6, sanitize_text("Value"), 1, 1, "C")
    
    pdf.set_font("Arial", "", 10)
    for p, v in params:
        pdf.cell(60, 6, sanitize_text(p), 1, 0, "L")
        pdf.cell(60, 6, sanitize_text(v), 1, 1, "C")
    
    pdf.ln(5)
    
    # 3. PREDICTION RESULTS
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, sanitize_text("3. Prediction Results"), ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    
    results = [
        ("Density", f"{density:.3f}"),
        ("Tensile Strength", f"{tensile:.3f} MPa", ">= 2.0 MPa"),
        ("Elastic Recovery", f"{er:.3f} %"),
        ("EFRF", f"{efrf:.4f}", "< 0.35")
    ]
    
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
    
    pdf.ln(5)
    
    # 4. STATUS
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
    pdf.ln(5)
    
    # 5. RECOMMENDATIONS
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, sanitize_text("5. Recommendations"), ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    
    if status == "PASS":
        recommendations = [
            sanitize_text("1. Proceed with experimental validation."),
            sanitize_text("2. Confirm tensile strength with physical testing."),
            sanitize_text("3. Evaluate disintegration time and dissolution."),
            sanitize_text("4. Assess stability under ICH conditions."),
            sanitize_text("5. Scale-up for process optimization.")
        ]
    else:
        recommendations = [
            sanitize_text("1. Reduce API or adjust binder concentration."),
            sanitize_text("2. Optimize Mg-St level."),
            sanitize_text("3. Increase compaction pressure."),
            sanitize_text("4. Reduce punch speed."),
            sanitize_text("5. Re-run with adjusted parameters.")
        ]
    
    for rec in recommendations:
        pdf.cell(0, 6, rec, ln=True)
    
    pdf.ln(5)
    
    # 6. CONTACT
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, sanitize_text("6. Contact Information"), ln=True, fill=True)
    pdf.set_font("Arial", "", 11)
    
    pdf.cell(0, 8, "Chem. Eng. Babuker A. Abdalla", ln=True)
    pdf.cell(0, 7, "Email: babuker@protonmail.com", ln=True)
    pdf.cell(0, 7, "Phone: +249-123-638-638", ln=True)
    pdf.cell(0, 7, "Sudan", ln=True)
    
    pdf.ln(3)
    
    # FOOTER
    pdf.set_y(270)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 6, "Generated by: Hybrid AI Framework", ln=True, align="C")
    
    # RETURN PDF
    pdf_bytes = pdf.output(dest="S")
    
    if isinstance(pdf_bytes, bytearray):
        return bytes(pdf_bytes)
    elif isinstance(pdf_bytes, bytes):
        return pdf_bytes
    else:
        return str(pdf_bytes).encode('latin1')


# ================================================================
# 6. NSGA-II IMPLEMENTATION
# ================================================================

class NSGAII:
    def __init__(self, model, scaler, bounds, pop_size=100, n_generations=60):
        self.model = model
        self.scaler = scaler
        self.bounds = bounds
        self.pop_size = pop_size
        self.n_generations = n_generations
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
                api, mcc, pvpp, mgst, binder, pressure, speed, granule = population[i, 0], population[i, 1], population[i, 2], population[i, 3], population[i, 4], population[i, 5], population[i, 6], population[i, 7]
                
                # Ensure sum = 100% and MCC <= 8%
                used = api + binder + mgst + pvpp + mcc
                if used > 100:
                    scale = 100 / used
                    api *= scale
                    binder *= scale
                    mgst *= scale
                    pvpp *= scale
                    mcc *= scale
                elif used < 100:
                    # Add remainder to MCC but cap at 8%
                    remainder = 100 - used
                    if mcc + remainder <= 8.0:
                        mcc += remainder
                    else:
                        excess = (mcc + remainder) - 8.0
                        mcc = 8.0
                        api -= excess

                api = np.clip(api, 85, 95)
                binder = np.clip(binder, 0.5, 4.0)
                mgst = np.clip(mgst, 0.01, 1.2)
                pvpp = np.clip(pvpp, 0.5, 6.0)
                mcc = np.clip(mcc, 0, 8.0)

                inputs = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
                inputs_with_features = add_interaction_features(np.array([inputs]))[0]
                inputs_scaled = self.scaler.transform([inputs_with_features])
                X_tensor = torch.FloatTensor(inputs_scaled)
                
                with torch.no_grad():
                    pred = self.model.predict(X_tensor)[0]
                
                tensile = float(pred[1])
                er = float(pred[2])
                
                if tensile < 0.01:
                    tensile = 0.01
                    efrf = 10.0
                else:
                    efrf = er / tensile
                    efrf = max(0.0001, min(efrf, 5.0))
                
                constraints[i] = (tensile >= 1.99 and efrf < 0.35)
                objectives[i, 0] = -api
                objectives[i, 1] = efrf
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
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        if not fronts[-1]:
            fronts.pop()
        return fronts, rank

    def _crowding_distance(self, objectives, front):
        n = len(front)
        if n <= 2:
            return np.ones(n) * np.inf
        distance = np.zeros(n)
        obj_min = objectives.min(axis=0)
        obj_max = objectives.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1
        for m in range(2):
            sorted_idx = sorted(range(n), key=lambda i: objectives[front[i], m])
            distance[sorted_idx[0]] = np.inf
            distance[sorted_idx[-1]] = np.inf
            for i in range(1, n - 1):
                if distance[sorted_idx[i]] != np.inf:
                    diff = objectives[front[sorted_idx[i+1]], m] - objectives[front[sorted_idx[i-1]], m]
                    distance[sorted_idx[i]] += diff / obj_range[m]
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
                if crowding[i1] > crowding[i2]:
                    selected.append(i1)
                else:
                    selected.append(i2)
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
                    child1, child2 = self._simulated_binary_crossover(self.population[selected[i]], self.population[selected[i+1]])
                    child1 = self._polynomial_mutation(child1)
                    child2 = self._polynomial_mutation(child2)
                    offspring.append(child1)
                    offspring.append(child2)
                else:
                    child = self._polynomial_mutation(self.population[selected[i]])
                    offspring.append(child)
            offspring = np.array(offspring[:self.pop_size])
            objectives_off, constraints_off, tensile_off, _ = self._evaluate(offspring)
            combined_pop = np.vstack([self.population, offspring])
            combined_obj = np.vstack([self.objectives, objectives_off])
            combined_const = np.concatenate([self.constraints, constraints_off])
            combined_fronts, combined_ranks = self._fast_non_dominated_sort(combined_obj, combined_const)
            combined_crowding = np.zeros(len(combined_pop))
            for front in combined_fronts:
                dist = self._crowding_distance(combined_obj, front)
                combined_crowding[front] = dist
            new_pop = []
            new_obj = []
            new_const = []
            for front in combined_fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    for idx in front:
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])
                        new_const.append(combined_const[idx])
                else:
                    front_sorted = sorted(front, key=lambda i: combined_crowding[i], reverse=True)
                    remaining = self.pop_size - len(new_pop)
                    for idx in front_sorted[:remaining]:
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
# 7. TRAIN MODEL
# ================================================================

@st.cache_resource
def load_pinn_model():
    df, feature_names = generate_pinn_data(n_samples=600)
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
    X_scaled_train_t.requires_grad_(True)
    X_raw_train_t = torch.FloatTensor(X_raw_train)
    y_train_t = torch.FloatTensor(y_train)

    X_scaled_val_t = torch.FloatTensor(X_scaled_val)
    X_scaled_val_t.requires_grad_(True)
    X_raw_val_t = torch.FloatTensor(X_raw_val)
    y_val_t = torch.FloatTensor(y_val)

    model = MultiTaskTruePINN(input_dim=X_augmented.shape[1])
    
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, patience=100, factor=0.5)

    best_val_loss = float('inf')
    patience = 150
    counter = 0
    best_state = None

    progress_bar = st.progress(0)
    adam_epochs = 2000
    max_epochs = 2500

    for epoch in range(adam_epochs):
        model.train()
        optimizer_adam.zero_grad()
        total_loss, loss_dict = model.compute_loss(
            X_scaled_train_t, X_raw_train_t, y_train_t,
            epoch=epoch, max_epochs=max_epochs,
            w_data_init=2.0, w_physics_init=0.2,
            w_data_final=1.0, w_physics_final=1.0,
            w_mcc=0.5, w_density=0.5,
            efrf_target=0.35, mcc_max=8.0, compute_grad=True
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_adam.step()

        model.eval()
        with torch.set_grad_enabled(False):
            val_loss, _ = model.compute_loss(
                X_scaled_val_t, X_raw_val_t, y_val_t,
                epoch=epoch, max_epochs=max_epochs,
                w_data_init=2.0, w_physics_init=0.2,
                w_data_final=1.0, w_physics_final=1.0,
                w_mcc=0.5, w_density=0.5,
                efrf_target=0.35, mcc_max=8.0, compute_grad=False
            )
        val_loss_value = val_loss.item()
        scheduler.step(val_loss_value)

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            counter = 0
            best_state = model.state_dict().copy()
        else:
            counter += 1

        if counter > patience:
            break

        if (epoch + 1) % 200 == 0:
            progress_bar.progress((epoch + 1) / max_epochs)

    if best_state is not None:
        model.load_state_dict(best_state)
    
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=500, line_search_fn='strong_wolfe')
    
    def closure():
        optimizer_lbfgs.zero_grad()
        total_loss, _ = model.compute_loss(
            X_scaled_train_t, X_raw_train_t, y_train_t,
            epoch=adam_epochs, max_epochs=max_epochs,
            w_data_init=1.0, w_physics_init=1.0,
            w_data_final=1.0, w_physics_final=1.0,
            w_mcc=0.5, w_density=0.5,
            efrf_target=0.35, mcc_max=8.0, compute_grad=True
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        return total_loss

    for i in range(10):
        optimizer_lbfgs.step(closure)
        progress_bar.progress((adam_epochs + (i + 1) * 50) / max_epochs)
        
        model.eval()
        with torch.set_grad_enabled(False):
            val_loss, _ = model.compute_loss(
                X_scaled_val_t, X_raw_val_t, y_val_t,
                epoch=adam_epochs + (i + 1) * 50,
                max_epochs=max_epochs,
                w_data_init=1.0, w_physics_init=1.0,
                w_data_final=1.0, w_physics_final=1.0,
                w_mcc=0.5, w_density=0.5,
                efrf_target=0.35, mcc_max=8.0, compute_grad=False
            )
            val_loss_value = val_loss.item()
            if val_loss_value < best_val_loss:
                best_val_loss = val_loss_value
                best_state = model.state_dict().copy()

    progress_bar.progress(1.0)

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model.feature_names = feature_names
    model.scaler = scaler
    model.y_scaler = y_scaler

    torch.save(model.state_dict(), 'true_pinn_checkpoint.pt')

    return model, scaler, y_scaler, feature_names, df, {'train': [], 'val': []}


# ================================================================
# 8. PREDICTION
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


# ================================================================
# 9. PLOT FUNCTIONS
# ================================================================

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
                name='Pareto Front'
            ))

            feasible_indices = [i for i, f in enumerate(feasible) if f]
            if feasible_indices:
                feasible_api = [pareto_api[i] for i in feasible_indices]
                feasible_efrf = [pareto_efrf[i] for i in feasible_indices]
                fig.add_trace(go.Scatter(
                    x=feasible_api, y=feasible_efrf,
                    mode='markers', marker=dict(size=12, color='green', symbol='star'),
                    name=f'Feasible ({len(feasible_indices)})'
                ))

            if api and efrf and 85 <= api <= 95 and 0.0001 <= efrf <= 2:
                fig.add_trace(go.Scatter(
                    x=[api], y=[efrf],
                    mode='markers+text',
                    marker=dict(size=16, color='blue', symbol='diamond'),
                    text=[f"Your Formulation"],
                    name='Your Formulation'
                ))

            fig.add_hline(y=0.35, line_dash='dash', line_color='red', annotation_text='EFRF=0.35')
            fig.update_layout(
                title='Pareto Front (NSGA-II)',
                xaxis=dict(title='API Loading (%)', range=[85, 95]),
                yaxis=dict(title='EFRF', range=[0, 1.0]),
                height=500,
                hovermode='closest'
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
        fig.add_vline(x=np.mean(sensitivities), line_dash='dash', line_color='red', annotation_text=f'Avg: {np.mean(sensitivities):.4f}')
        fig.update_layout(
            title='Sensitivity Analysis (EFRF)',
            xaxis_title='Sensitivity (ΔEFRF)',
            yaxis_title='Parameters',
            height=450
        )
        return fig
    except Exception:
        return None


# ================================================================
# 10. MODEL COMPARISON (FIXED XGBOOST IMPORT)
# ================================================================

def train_and_compare(X_train, X_test, y_train, y_test):
    models = {}
    
    # MLP
    models['MLP'] = MLPRegressor(hidden_layer_sizes=(64, 64, 64), max_iter=1000, random_state=42)
    
    # Random Forest
    models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # XGBoost (if available)
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        results.append({
            'Model': name,
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Physics': 'Not enforced'
        })
    return pd.DataFrame(results)


# ================================================================
# 11. STREAMLIT UI
# ================================================================

st.set_page_config(page_title="PINN Framework", page_icon="🧬", layout="wide")

# --- HERO SECTION ---
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); 
            padding: 2rem; border-radius: 1rem; margin-bottom: 1.5rem; text-align: center;">
    <h1 style="color: #ffffff; font-size: 2.5rem; margin: 0;">
        🧬 Hybrid AI Framework
    </h1>
    <p style="color: #a8b2d1; font-size: 1.2rem; margin: 0.5rem 0 0 0;">
        Physics-Informed Neural Network · Multi-Objective Optimization
    </p>
    <p style="color: #64ffda; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
        Nile Valley University · Sudan
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 📚 Physics Constraints")
    st.markdown("""
    - ✅ **Heckel:** ln(1/(1-D)) = kP + A
    - ✅ **EFRF:** ER / σt < 0.35
    - ✅ **Monotonicity:** ∂D/∂P > 0
    - ✅ **Boundary:** 0.4 < D < 0.98
    - ✅ **MCC:** ≤ 8%
    - ✅ **Density:** ≤ 1.0
    
    **Multi-Task PINN:**
    - 5 outputs (D, σt, ER, k, A)
    - k and A learned directly
    - Adam → LBFGS hybrid
    - Feature engineering
    - Output normalization
    """)
    st.warning("⚠️ **Production-Ready**")

# --- LOAD MODEL ---
with st.spinner("🔄 Training Multi-Task PINN..."):
    model, scaler, y_scaler, feature_names, df, loss_history = load_pinn_model()
st.success("✅ Multi-Task True PINN trained successfully")

# --- EXPERIMENTS ---
st.markdown("### 🧪 Quick Experiments")
st.caption("Click any button to auto-set parameters")

exp_cols = st.columns(4)
experiments = {
    "Baseline": {'api': 90.5, 'binder': 2.7, 'pvpp': 3.0, 'mgst': 0.20, 'mcc': 5.0, 'pressure': 230, 'speed': 12, 'granule': 125},
    "Exp 1": {'api': 90.5, 'binder': 2.9, 'pvpp': 3.0, 'mgst': 0.15, 'mcc': 5.0, 'pressure': 235, 'speed': 10, 'granule': 125},
    "Exp 2": {'api': 90.5, 'binder': 2.8, 'pvpp': 3.0, 'mgst': 0.12, 'mcc': 5.0, 'pressure': 240, 'speed': 9, 'granule': 125},
    "Exp 3": {'api': 90.5, 'binder': 3.0, 'pvpp': 3.0, 'mgst': 0.10, 'mcc': 5.0, 'pressure': 245, 'speed': 8, 'granule': 125}
}
for i, (name, params) in enumerate(experiments.items()):
    with exp_cols[i]:
        if st.button(f"📌 {name}", key=f"exp_{i}", use_container_width=True):
            for key in params:
                st.session_state[key] = params[key]
            st.rerun()

st.markdown("---")

# --- MAIN LAYOUT ---
col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    st.markdown("### 📊 Formulation Parameters")
    
    with st.container(border=True):
        api = st.slider("🧪 API Loading (%)", 85.0, 95.0, st.session_state.api, 0.1, key="api")
        binder = st.slider("🔗 Binder (%)", 0.5, 4.0, st.session_state.binder, 0.1, key="binder")
        pvpp = st.slider("💊 PVPP (%)", 0.5, 6.0, st.session_state.pvpp, 0.1, key="pvpp")
        mgst = st.slider("🧴 Mg-St (%)", 0.01, 1.2, st.session_state.mgst, 0.01, key="mgst")
        mcc = st.slider("📦 MCC (%)", 0.0, 8.0, st.session_state.mcc, 0.1, key="mcc")
        
        total = api + binder + pvpp + mgst + mcc
        if abs(total - 100) < 0.1:
            st.success(f"✅ Total = {total:.2f}%")
        elif total > 100:
            st.error(f"❌ Total = {total:.2f}% (exceeds 100%)")
        else:
            st.warning(f"⚠️ Total = {total:.2f}% (adds to 100%)")
    
    st.markdown("### ⚙️ Process Parameters")
    with st.container(border=True):
        pressure = st.slider("⚙️ Pressure (MPa)", 80.0, 280.0, st.session_state.pressure, 1.0, key="pressure")
        speed = st.slider("🔄 Speed (rpm)", 1.0, 50.0, st.session_state.speed, 0.5, key="speed")
        granule = st.slider("🔬 Granule Size (µm)", 30.0, 250.0, st.session_state.granule, 1.0, key="granule")
    
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
            kpi_cols[0].metric("Density", f"{density:.3f}", delta="0.99 ideal")
            kpi_cols[1].metric("Tensile", f"{tensile:.3f} MPa", delta=">= 2.0 PASS" if tensile >= 2.0 else "< 2.0 FAIL")
            kpi_cols[2].metric("EFRF", f"{efrf:.4f}", delta="< 0.35 PASS" if efrf < 0.35 else ">= 0.35 FAIL")
            
            st.markdown("---")
            
            # --- Status ---
            if tensile >= 2.0 and efrf < 0.35:
                st.success("✅ Formulation satisfies all constraints!")
            elif tensile >= 2.0 and efrf < 0.40:
                st.warning("⚠️ Feasible but EFRF > 0.35")
            else:
                st.error("❌ Formulation fails constraints")
            
            # --- Feasibility ---
            st.markdown("### ✅ Feasibility")
            pass_cols = st.columns(2)
            pass_cols[0].metric("σt ≥ 2.0 MPa", "✅ PASS" if tensile >= 2.0 else "❌ FAIL")
            pass_cols[1].metric("EFRF < 0.40", "✅ PASS" if efrf < 0.40 else "❌ FAIL")
            
            # --- Physics Verification ---
            with st.expander("🔬 Physics Verification"):
                try:
                    inputs_with_features = add_interaction_features(np.array([inputs]))[0]
                    inputs_scaled = scaler.transform([inputs_with_features])
                    X_tensor = torch.FloatTensor(inputs_scaled)
                    with torch.no_grad():
                        full_output = model.forward(X_tensor).numpy()[0]
                    k_learned = full_output[3]
                    A_learned = full_output[4]
                    st.metric("k (Plasticity)", f"{k_learned:.4f}")
                    st.metric("A (Rearrangement)", f"{A_learned:.4f}")
                except:
                    pass
            
            # --- NSGA-II ---
            st.markdown("### ⚙️ NSGA-II")
            bounds = np.array([
                [85, 95], [0, 8], [0.5, 6.0], [0.01, 1.2], [0.5, 4.0],
                [80, 280], [1, 50], [30, 250]
            ])
            with st.spinner("🔄 Running NSGA-II..."):
                nsga = NSGAII(model, scaler, bounds, pop_size=80, n_generations=40)
                pop, objectives, constraints, fronts = nsga.run()
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
            
            # --- Pareto Front ---
            st.markdown("### 📉 Pareto Front")
            fig_p = plot_pareto_plotly(objectives, constraints, fronts, nsga, api, efrf)
            if fig_p:
                st.plotly_chart(fig_p, use_container_width=True)
            
            # --- Sensitivity ---
            st.markdown("### 🔍 Sensitivity")
            fig_s = plot_sensitivity_plotly(inputs, model, scaler, y_scaler)
            if fig_s:
                st.plotly_chart(fig_s, use_container_width=True)
            
            # --- Model Comparison ---
            st.markdown("### 📊 Model Comparison")
            X_train, X_test, y_train, y_test = train_test_split(
                df[feature_names].values,
                df['Tensile_Strength_MPa'].values,
                test_size=0.2,
                random_state=42
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
            pinn_row = pd.DataFrame([{
                'Model': 'PINN (Proposed)',
                'R²': pinn_r2,
                'RMSE': pinn_rmse,
                'MAE': pinn_mae,
                'Physics': '✅ Enforced'
            }])
            comp_df = pd.concat([pinn_row, comp_df], ignore_index=True)
            st.dataframe(comp_df.style.highlight_max(subset=['R²'], color='lightgreen'), use_container_width=True, hide_index=True)
            
            # --- PDF Report ---
            st.markdown("### 📄 Report")
            if st.button("📥 Download PDF Report", use_container_width=True):
                try:
                    status = "PASS" if (tensile >= 2.0 and efrf < 0.35) else "FAIL"
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    pdf_data = create_pdf_report(
                        api, mcc, pvpp, mgst, binder, pressure, speed, granule,
                        density, tensile, er, efrf, status, timestamp
                    )
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_data,
                        file_name=f"formulation_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF generation error: {e}")

st.markdown("---")
st.caption("🔬 **Multi-Task True PINN — Production-Ready**")
st.caption("📧 Contact: babuker@protonmail.com")
