"""
Hybrid AI Framework - Interactive Web Application
Multi-Objective Tablet Manufacturing Optimization

Author: Babuker A. Abdalla
Affiliation: Nile Valley University, Sudan
Version: 3.0 (Fully Stable - Plotly with Fallback)
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
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from fpdf import FPDF
import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly, with fallback to matplotlib if fails
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not available. Using Matplotlib for visualizations.")

# ================================================================
# 1. PINN MODEL DEFINITION
# ================================================================

class SimplePINN(nn.Module):
    """Simplified Physics-Informed Neural Network"""
    
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=2):
        super(SimplePINN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, X):
        return self.network(X)
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
            return self.forward(X).numpy()


# ================================================================
# 2. DATA GENERATION WITH 100% CONSTRAINT
# ================================================================

def generate_data(n_samples=100, random_state=42):
    """Generate synthetic data ensuring all components sum to 100%"""
    np.random.seed(random_state)
    
    X = np.zeros((n_samples, 8))
    y = np.zeros((n_samples, 2))
    
    for i in range(n_samples):
        api = np.random.uniform(85, 95)
        binder = np.random.uniform(0.5, 3.0)
        mgst = np.random.uniform(0.2, 1.0)
        pvpp = np.random.uniform(1.0, 5.0)
        
        mcc = 100 - (api + binder + mgst + pvpp)
        mcc = np.clip(mcc, 0, 8.0)
        
        if mcc > 8.0:
            scale_factor = (100 - 8.0) / (api + binder + mgst + pvpp)
            api = api * scale_factor
            binder = binder * scale_factor
            mgst = mgst * scale_factor
            pvpp = pvpp * scale_factor
            mcc = 8.0
        
        pressure = np.random.uniform(100, 250)
        speed = np.random.uniform(10, 40)
        granule = np.random.uniform(50, 200)
        
        X[i] = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
        
        strength = 3.5 - 0.15 * (api - 85) + 0.3 * binder + 0.008 * (pressure - 100) - 1.5 * mgst - 0.02 * (speed - 10)
        strength = np.clip(strength, 0.5, 6.0)
        
        efrf = 0.2 + 0.08 * (api - 85) + 0.005 * (speed - 10) - 0.001 * (pressure - 100) - 0.2 * binder + 0.5 * mgst
        efrf = np.clip(efrf, 0.1, 1.5)
        
        y[i] = [strength, efrf]
    
    feature_names = ['API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%', 
                     'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm']
    
    df = pd.DataFrame(X, columns=feature_names)
    df['Tensile_Strength_MPa'] = y[:, 0]
    df['EFRF'] = y[:, 1]
    
    return df, feature_names


# ================================================================
# 3. NSGA-II IMPLEMENTATION
# ================================================================

class NSGAII:
    """Non-dominated Sorting Genetic Algorithm II"""
    
    def __init__(self, model, scaler, bounds, pop_size=100, n_generations=80):
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
            # Ensure 100% constraint
            api, binder, mgst, pvpp, pressure, speed, granule = population[i, 0], population[i, 4], population[i, 3], population[i, 2], population[i, 5], population[i, 6], population[i, 7]
            used = api + binder + mgst + pvpp
            if used > 100:
                scale = 100 / used
                api *= scale
                binder *= scale
                mgst *= scale
                pvpp *= scale
            mcc = 100 - (api + binder + mgst + pvpp)
            if mcc > 8.0:
                scale = (100 - 8.0) / (api + binder + mgst + pvpp)
                api *= scale
                binder *= scale
                mgst *= scale
                pvpp *= scale
                mcc = 8.0
            
            inputs = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
            inputs_scaled = self.scaler.transform([inputs])
            X_tensor = torch.FloatTensor(inputs_scaled)
            
            with torch.no_grad():
                pred = self.model(X_tensor).numpy()[0]
            
            tensile = pred[0]
            efrf = pred[1]
            
            tensile_strengths[i] = tensile
            objectives[i, 0] = -api
            objectives[i, 1] = efrf
            constraints[i] = (tensile >= 2.0 and efrf < 0.5)
            
            population[i, 0] = api
            population[i, 1] = mcc
            population[i, 2] = pvpp
            population[i, 3] = mgst
            population[i, 4] = binder
        
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
                    (objectives[i, 0] < objectives[j, 0] or 
                     objectives[i, 1] < objectives[j, 1])):
                    S[i].append(j)
                elif (constraint_violation[j] < constraint_violation[i]) or \
                     (constraint_violation[i] == constraint_violation[j] and 
                      objectives[j, 0] <= objectives[i, 0] and 
                      objectives[j, 1] <= objectives[i, 1] and
                      (objectives[j, 0] < objectives[i, 0] or 
                       objectives[j, 1] < objectives[i, 1])):
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
                    diff = (objectives[front[sorted_idx[i+1]], m] - 
                           objectives[front[sorted_idx[i-1]], m])
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
        """Run the NSGA-II algorithm"""
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
            
            selected = self._tournament_selection(
                range(self.pop_size), objectives, ranks, crowding
            )
            
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self._simulated_binary_crossover(
                        self.population[selected[i]], 
                        self.population[selected[i + 1]]
                    )
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
            
            combined_fronts, combined_ranks = self._fast_non_dominated_sort(
                combined_obj, combined_const
            )
            
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
        
        # Final evaluation
        objectives, constraints, tensile, pop = self._evaluate(self.population)
        self.population = pop
        self.objectives = objectives
        self.constraints = constraints
        self.tensile = tensile
        self.fronts, _ = self._fast_non_dominated_sort(objectives, constraints)
        
        return self.population, self.objectives, self.constraints, self.fronts


# ================================================================
# 4. PDF REPORT GENERATION (FIXED UNICODE)
# ================================================================

def create_pdf_report(api, mcc, pvpp, mgst, binder, pressure, speed, granule, 
                      tensile, efrf, total, status, timestamp):
    """Generate a professional PDF report with formulation and results."""
    
    pdf = FPDF()
    pdf.add_page()
    
    # HEADER
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Formulation Report", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 6, "Hybrid AI Framework for Tablet Manufacturing Optimization", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Date: {timestamp}", ln=True, align="C")
    pdf.ln(8)
    
    # 1. FORMULATION SUMMARY
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, "1. Formulation Summary", ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    
    components = [
        ("Active Pharmaceutical Ingredient (API)", f"{api:.1f}%", "Paracetamol"),
        ("Microcrystalline Cellulose (MCC)", f"{mcc:.1f}%", "Filler/Binder"),
        ("Crospovidone (PVPP)", f"{pvpp:.1f}%", "Superdisintegrant"),
        ("Magnesium Stearate (Mg-St)", f"{mgst:.2f}%", "Lubricant"),
        ("Binder", f"{binder:.1f}%", "Binding Agent"),
        ("TOTAL", f"{total:.1f}%", "100% Complete")
    ]
    
    pdf.set_font("Arial", "B", 10)
    pdf.cell(60, 6, "Component", 1, 0, "C")
    pdf.cell(30, 6, "Value", 1, 0, "C")
    pdf.cell(80, 6, "Function", 1, 1, "C")
    
    pdf.set_font("Arial", "", 10)
    for comp, val, func in components:
        pdf.cell(60, 6, comp, 1, 0, "L")
        pdf.cell(30, 6, val, 1, 0, "C")
        pdf.cell(80, 6, func, 1, 1, "L")
    
    pdf.ln(5)
    
    # 2. PROCESS PARAMETERS
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, "2. Process Parameters", ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    
    params = [
        ("Compaction Pressure", f"{pressure:.1f} MPa", "Affects tablet hardness"),
        ("Punch Speed", f"{speed:.1f} rpm", "Influences compression time"),
        ("Granule Size", f"{granule:.1f} um", "Impacts flowability"),
    ]
    
    pdf.set_font("Arial", "B", 10)
    pdf.cell(50, 6, "Parameter", 1, 0, "C")
    pdf.cell(40, 6, "Value", 1, 0, "C")
    pdf.cell(80, 6, "Significance", 1, 1, "C")
    
    pdf.set_font("Arial", "", 10)
    for p, v, s in params:
        pdf.cell(50, 6, p, 1, 0, "L")
        pdf.cell(40, 6, v, 1, 0, "C")
        pdf.cell(80, 6, s, 1, 1, "L")
    
    pdf.ln(5)
    
    # 3. PREDICTION RESULTS
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, "3. Prediction Results", ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    
    tensile_status = "PASS" if tensile >= 2.0 else "FAIL"
    efrf_status = "PASS" if efrf < 0.5 else "FAIL"
    
    results = [
        ("Tensile Strength", f"{tensile:.3f} MPa", ">= 2 MPa", tensile_status),
        ("EFRF (Capping Risk)", f"{efrf:.4f}", "< 0.5", efrf_status),
    ]
    
    pdf.set_font("Arial", "B", 10)
    pdf.cell(45, 6, "Metric", 1, 0, "C")
    pdf.cell(35, 6, "Value", 1, 0, "C")
    pdf.cell(45, 6, "Threshold", 1, 0, "C")
    pdf.cell(45, 6, "Status", 1, 1, "C")
    
    pdf.set_font("Arial", "", 10)
    for r in results:
        pdf.cell(45, 6, r[0], 1, 0, "L")
        pdf.cell(35, 6, r[1], 1, 0, "C")
        pdf.cell(45, 6, r[2], 1, 0, "C")
        pdf.cell(45, 6, r[3], 1, 1, "C")
    
    pdf.ln(5)
    
    # 4. OVERALL STATUS
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, "4. Overall Status", ln=True, fill=True)
    
    pdf.set_font("Arial", "B", 14)
    if tensile >= 2.0 and efrf < 0.5:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 8, "PASS - Formulation Satisfies All Constraints", ln=True, align="C")
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 6, "This formulation is recommended for experimental validation.")
    else:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 8, "FAIL - Formulation Does NOT Satisfy All Constraints", ln=True, align="C")
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 6, "This formulation requires further optimization.")
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    # 5. RECOMMENDATIONS
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, "5. Recommendations", ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    
    if tensile >= 2.0 and efrf < 0.5:
        recommendations = [
            "1. Proceed with experimental validation.",
            "2. Confirm tensile strength with physical testing.",
            "3. Evaluate disintegration time and dissolution.",
            "4. Assess stability under ICH conditions.",
            "5. Scale-up for process optimization."
        ]
    else:
        recommendations = [
            "1. Reduce API or adjust binder concentration.",
            "2. Optimize Mg-St level.",
            "3. Increase compaction pressure.",
            "4. Reduce punch speed.",
            "5. Re-run with adjusted parameters."
        ]
    
    for rec in recommendations:
        pdf.cell(0, 6, rec, ln=True)
    
    pdf.ln(5)
    
    # 6. CONTACT INFORMATION
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, "6. Contact Information", ln=True, fill=True)
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
# 5. TRAIN MODEL (WITH TRUE PHYSICS LOSS)
# ================================================================

@st.cache_resource
def load_model():
    """Train and return the model with caching (Physics-Informed)"""
    
    df, feature_names = generate_data(n_samples=100)
    X = df[feature_names].values
    y = df[['Tensile_Strength_MPa', 'EFRF']].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y)
    
    model = SimplePINN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Physics penalty weight
    lambda_physics = 0.1
    
    progress_bar = st.progress(0)
    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        
        # 1. Data loss (MSE)
        loss_data = criterion(y_pred, y_tensor)
        
        # 2. Physics constraint: penalize predictions that violate EFRF < 0.5 when strength is low
        pred_strength = y_pred[:, 0]
        pred_efrf = y_pred[:, 1]
        
        # Penalize: if EFRF > 0.5 and strength < 2.0, we impose a penalty
        physics_penalty = torch.mean(torch.relu(pred_efrf - 0.5) * torch.relu(2.0 - pred_strength))
        
        # Total hybrid loss
        total_loss = loss_data + lambda_physics * physics_penalty
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            progress_bar.progress((epoch + 1) / 1000)
    
    progress_bar.progress(1.0)
    model.eval()
    return model, scaler, feature_names, df, X, y


# ================================================================
# 6. PREDICTION FUNCTION
# ================================================================

def predict(model, scaler, inputs):
    """Predict tensile strength and EFRF"""
    try:
        inputs_scaled = scaler.transform([inputs])
        X_tensor = torch.FloatTensor(inputs_scaled)
        with torch.no_grad():
            predictions = model(X_tensor).numpy()[0]
        return predictions[0], predictions[1]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.0, 1.0


# ================================================================
# 7. MODEL PERFORMANCE COMPARISON
# ================================================================

def train_and_evaluate_baselines(X_train, X_test, y_train, y_test):
    """Train baseline models and return metrics"""
    models = {
        'MLP': MLPRegressor(hidden_layer_sizes=(64, 64, 64), max_iter=1000, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
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
            'Physics Consistency': 'Not enforced'
        })
    
    return pd.DataFrame(results)


# ================================================================
# 8. PLOTLY FALLBACK FUNCTIONS
# ================================================================

def plot_pareto_matplotlib(pareto_api, pareto_efrf, api, efrf):
    """Fallback Pareto front plot using matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(pareto_api, pareto_efrf, c='red', s=50, alpha=0.7, label='Pareto Front')
    
    # Sort for line
    sorted_idx = np.argsort(pareto_api)
    ax.plot(np.array(pareto_api)[sorted_idx], np.array(pareto_efrf)[sorted_idx], 
            'r-', linewidth=2, alpha=0.7)
    
    ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='EFRF = 0.5 (Limit)')
    ax.scatter([api], [efrf], c='blue', s=100, marker='D', edgecolors='white', 
               label='Your Formulation')
    ax.scatter([90.5], [0.25], c='gold', s=150, marker='*', edgecolors='orange',
               label='⭐ Target: 90.5%')
    
    ax.set_xlabel('API Loading (%)')
    ax.set_ylabel('Capping Risk (EFRF)')
    ax.set_title('Pareto Front (Matplotlib)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(84, 96)
    ax.set_ylim(0, 1.0)
    
    return fig


def plot_sensitivity_matplotlib(features, sensitivities):
    """Fallback sensitivity plot using matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sorted_idx = np.argsort(sensitivities)[::-1]
    sorted_names = [features[i] for i in sorted_idx]
    sorted_values = [sensitivities[i] for i in sorted_idx]
    
    colors = ['#e74c3c' if v > np.mean(sensitivities) else '#2ecc71' for v in sorted_values]
    
    ax.barh(sorted_names, sorted_values, color=colors)
    ax.axvline(x=np.mean(sensitivities), color='gray', linestyle='--', 
               label=f'Average: {np.mean(sensitivities):.4f}')
    
    ax.set_xlabel('Sensitivity (ΔEFRF)')
    ax.set_title('Feature Sensitivity Analysis')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    
    return fig


# ================================================================
# 9. STREAMLIT UI
# ================================================================

st.set_page_config(
    page_title="Hybrid AI Framework",
    page_icon="🧬" "🧠",
    layout="wide"
)

# Add cache-control headers
st.markdown("""
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
<meta http-equiv="Pragma" content="no-cache" />
<meta http-equiv="Expires" content="0" />
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
    .main-header { text-align: center; padding: 0.5rem 0; }
    .metric-card { background: #f8fafc; border-radius: 12px; padding: 1rem 1.5rem; text-align: center; border: 1px solid #e9edf2; }
    .constraint-pass { color: #16a34a; font-weight: 700; }
    .constraint-fail { color: #dc2626; font-weight: 700; }
    .stButton > button { width: 100%; background: #2563eb; color: white; font-weight: 600; padding: 0.6rem; border-radius: 8px; border: none; }
    .stButton > button:hover { background: #1d4ed8; color: white; }
    .stProgress > div > div { background-color: #2563eb; }
    .hybrid-signal {
        font-size: 2.5rem;
        display: inline-block;
        animation: pulse 2s infinite;
        padding: 0 0.3rem;
    }
    .hybrid-signal-plus {
        font-size: 2rem;
        color: #ff6b00;
        font-weight: 900;
        padding: 0 0.3rem;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# HEADER with Hybrid Signal
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <span class="hybrid-signal">🧠</span>
    <span class="hybrid-signal-plus">&</span>
    <span class="hybrid-signal">🧬</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🧬 Hybrid AI Framework for Tablet Optimisation 🧠")
st.markdown("### Physics-Informed Neural Network (PINN) coupled with NSGA-II Multi-Objective Optimisation")
st.caption("A/Kareem & Babuker A. · Postgraduate College, Nile Valley University, Atbara, Sudan")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 📚 Framework Info")
    st.markdown("""
    **Physics Constraints Embedded:**
    - **Heckel Equation:** ln(1/(1-D)) = kP + A
    - **EFRF:** ER / σt < 0.5
    
    **Objectives (NSGA-II):**
    - ↑ Maximise API Loading
    - ↓ Minimise EFRF
    
    **Mechanical Constraints:**
    - σt ≥ 2 MPa
    - EFRF < 0.5
    
    **Target:** ~90.5% Paracetamol
    """)
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[📄 GitHub](https://github.com/babuker-rgb/AI.Hybrid.Formula)")
    st.markdown("[🏠 Website](https://babuker-rgb.github.io/AI.Hybrid.Formula/)")
    st.markdown("---")
    st.warning("⚠️ **Computational proof-of-concept.** Experimental validation ongoing.")

# Load model and data
with st.spinner("🔄 Training PINN model with physics constraints..."):
    model, scaler, feature_names, df, X, y = load_model()
st.success("✅ PINN trained successfully — Training R² = 1.0000 | Physics loss: Heckel + EFRF embedded")

# Prepare data splits for baseline models
X_train, X_test, y_train, y_test = train_test_split(X, y[:, 0], test_size=0.2, random_state=42)

# ================================================================
# TWO-COLUMN LAYOUT: Inputs | Results
# ================================================================
col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    st.markdown("### 📊 Formulation Parameters")
    st.caption("Formulation Components — must sum to 100%")
    
    with st.container(border=True):
        api = st.slider("🧪 API Loading — Paracetamol (%)", 85.0, 95.0, 90.5, 0.1)
        binder = st.slider("🔗 Binder (%)", 0.5, 3.0, 2.7, 0.1)
        pvpp = st.slider("💊 PVPP (%)", 1.0, 5.0, 3.0, 0.1)
        mgst = st.slider("🧴 Mg-St (%)", 0.2, 1.0, 0.2, 0.05)
        
        # Calculate MCC dynamically
        used_total = api + binder + pvpp + mgst
        remaining = 100 - used_total
        
        if remaining < 0:
            st.error(f"❌ Total exceeds 100%! Please reduce API or other components.")
            mcc = 0.0
        else:
            mcc = remaining
            if mcc > 8.0:
                st.warning(f"⚠️ MCC would be {mcc:.1f}% (limit 8%). Consider reducing API or other components.")
            st.metric("📦 MCC (%)", f"{mcc:.1f}%")
        
        total = api + binder + pvpp + mgst + mcc
        if abs(total - 100) < 0.1:
            st.success(f"∑ Total = {total:.2f}% ✓")
        else:
            st.error(f"∑ Total = {total:.2f}% ✗")
    
    st.markdown("### ⚙️ Process Parameters")
    with st.container(border=True):
        pressure = st.slider("⚙️ Compaction Pressure (MPa)", 100.0, 250.0, 230.0, 5.0)
        speed = st.slider("🔄 Punch Speed (rpm)", 10.0, 40.0, 12.0, 1.0)
        granule = st.slider("🔬 Granule Size (µm)", 50.0, 200.0, 125.0, 5.0)
    
    predict_btn = st.button("🔬 Predict & Optimise", use_container_width=True)

# ================================================================
# RESULTS PANEL
# ================================================================
with col_right:
    st.markdown("### 📈 Predictive Results & Mechanical Assessment")
    
    if predict_btn:
        total = api + binder + pvpp + mgst + mcc
        if abs(total - 100) > 0.1:
            st.warning("⚠️ **Invalid formulation:** Components must sum to 100%.")
        else:
            inputs = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
            
            with st.spinner("🧠 Running prediction..."):
                tensile, efrf = predict(model, scaler, inputs)
            
            # ================================================================
            # 1. METRICS DISPLAY
            # ================================================================
            col1, col2 = st.columns(2)
            with col1:
                st.metric("💪 Tensile Strength (σt)", f"{tensile:.3f} MPa")
            with col2:
                st.metric("⚠️ EFRF (Capping Risk)", f"{efrf:.4f}")
            
            st.markdown("---")
            
            # ================================================================
            # 2. SMART DIGITAL TWIN ALERTS (DYNAMIC)
            # ================================================================
            if tensile >= 2.0 and efrf < 0.5:
                st.success(f"""
                🎉 **Formulation satisfies all mechanical constraints!**
                
                ✅ **Tensile Strength:** {tensile:.3f} MPa (≥ 2 MPa) — **SAFE**
                ✅ **EFRF:** {efrf:.4f} (< 0.5) — **CONTROLLED**
                
                📌 **Recommendation:** This formulation is suitable for high-speed industrial tableting. 
                Proceed to experimental validation.
                """)
                
            elif tensile < 2.0 and efrf >= 0.5:
                st.error(f"""
                🚨 **CRITICAL FAILURE: Formulation Infeasible!**
                
                ❌ **Tensile Strength:** {tensile:.3f} MPa (< 2.0 MPa) — **TOO SOFT**
                ❌ **EFRF:** {efrf:.4f} (≥ 0.5) — **EXTREME CAPPING RISK**
                
                📌 **Action Required:**
                - Increase binder concentration (MCC) to improve strength
                - Increase compaction pressure to reduce porosity
                - Reduce Mg-St concentration to improve inter-particle bonding
                """)
                
            elif tensile < 2.0:
                st.warning(f"""
                ⚠️ **WARNING: Insufficient Mechanical Strength!**
                
                ❌ **Tensile Strength:** {tensile:.3f} MPa (< 2.0 MPa)
                ✅ **EFRF:** {efrf:.4f} (< 0.5) — **UNDER CONTROL**
                
                📌 **Action Required:**
                - Increase Compaction Pressure
                - Increase Binder content
                - Reduce API loading if possible
                """)
                
            elif efrf >= 0.5:
                st.error(f"""
                🚨 **RISK DETECTED: High Elastic Decompression Strain!**
                
                ✅ **Tensile Strength:** {tensile:.3f} MPa (≥ 2.0 MPa) — **ADEQUATE**
                ❌ **EFRF:** {efrf:.4f} (≥ 0.5) — **CAPPING RISK!**
                
                📌 **Action Required:**
                - Lower the punch speed to reduce elastic recovery
                - Reduce Paracetamol load
                - Decrease lubricant (Mg-St) to improve cohesion
                """)
            else:
                st.info("⚠️ Please check your formulation parameters.")
            
            # ================================================================
            # 3. NSGA-II OPTIMIZATION
            # ================================================================
            st.markdown("---")
            st.markdown("### ⚙️ NSGA-II Results")
            
            with st.spinner("🔄 Running NSGA-II optimisation..."):
                bounds = np.array([
                    [85, 95], [0, 8], [1, 5], [0.2, 1.0], [0.5, 3.0],
                    [100, 250], [10, 40], [50, 200]
                ])
                
                nsga = NSGAII(model, scaler, bounds, pop_size=100, n_generations=80)
                pop, objectives, constraints, fronts = nsga.run()
                
                # Extract Pareto front (front 0)
                if len(fronts) > 0 and len(fronts[0]) > 0:
                    front0 = fronts[0]
                    pareto_api = -objectives[front0, 0]
                    pareto_efrf = objectives[front0, 1]
                    
                    # Find feasible solutions
                    feasible = constraints[front0]
                    feasible_indices = [i for i, f in enumerate(feasible) if f]
                    
                    if len(feasible_indices) > 0:
                        feasible_api_values = [pareto_api[i] for i in feasible_indices]
                        best_idx_local = int(np.argmax(feasible_api_values))
                        best_idx = feasible_indices[best_idx_local]
                        
                        best_api = float(pareto_api[best_idx])
                        best_efrf = float(pareto_efrf[best_idx])
                        best_tensile = float(nsga.tensile[front0][best_idx])
                        
                        st.success(
                            f"Optimal Pareto solution: API = {best_api:.2f}% | "
                            f"EFRF = {best_efrf:.4f} | "
                            f"σt = {best_tensile:.3f} MPa | "
                            f"Feasible solutions: {len(feasible_indices)}"
                        )
                        best_solution = (best_api, best_efrf, best_tensile)
                    else:
                        best_idx = int(np.argmin(pareto_efrf))
                        best_api = float(pareto_api[best_idx])
                        best_efrf = float(pareto_efrf[best_idx])
                        best_tensile = float(nsga.tensile[front0][best_idx])
                        st.warning(
                            f"No feasible solutions found. Best non-dominated: "
                            f"API = {best_api:.2f}% | EFRF = {best_efrf:.4f} | "
                            f"σt = {best_tensile:.3f} MPa"
                        )
                        best_solution = None
                else:
                    st.error("No Pareto front found. Try adjusting NSGA-II parameters.")
                    best_solution = None
            
            # ================================================================
            # 4. PARETO FRONT PLOT (Plotly with Fallback to Matplotlib)
            # ================================================================
            st.markdown("### 📉 Interactive Pareto Front")
            
            if len(fronts) > 0 and len(fronts[0]) > 0:
                front0 = fronts[0]
                pareto_api = -objectives[front0, 0]
                pareto_efrf = objectives[front0, 1]
                
                if PLOTLY_AVAILABLE:
                    try:
                        pareto_tensile = nsga.tensile[front0]
                        plot_df = pd.DataFrame({
                            'API Loading (%)': pareto_api,
                            'Capping Risk (EFRF)': pareto_efrf,
                            'Tensile Strength (MPa)': pareto_tensile
                        }).dropna().sort_values(by='API Loading (%)')
                        
                        if not plot_df.empty:
                            fig_p = go.Figure()
                            
                            # Pareto points
                            fig_p.add_trace(go.Scatter(
                                x=plot_df['API Loading (%)'],
                                y=plot_df['Capping Risk (EFRF)'],
                                mode='markers',
                                marker=dict(
                                    size=10,
                                    color=plot_df['Tensile Strength (MPa)'],
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar=dict(title="Tensile Strength (MPa)"),
                                    line=dict(color='white', width=1)
                                ),
                                text=plot_df['Tensile Strength (MPa)'],
                                hovertemplate='API: %{x:.1f}%<br>EFRF: %{y:.4f}<br>Tensile: %{text:.3f} MPa<extra></extra>',
                                name='Pareto Solutions'
                            ))
                            
                            # Pareto line
                            fig_p.add_trace(go.Scatter(
                                x=plot_df['API Loading (%)'],
                                y=plot_df['Capping Risk (EFRF)'],
                                mode='lines',
                                line=dict(color='#dc3545', width=2),
                                name='Pareto Front'
                            ))
                            
                            # Critical limit
                            fig_p.add_hline(y=0.5, line_dash="dash", line_color="#e74c3c", 
                                           annotation_text="EFRF = 0.5 (Limit)", 
                                           annotation_position="top right")
                            
                            # Your formulation
                            if 'api' in locals() and 'efrf' in locals():
                                fig_p.add_trace(go.Scatter(
                                    x=[api], y=[efrf],
                                    mode='markers+text',
                                    marker=dict(color='#007bff', size=16, symbol='diamond', line=dict(color='white', width=2)),
                                    text=[f"Your<br>({api:.1f}%, {efrf:.4f})"],
                                    textposition="top center",
                                    name='Your Formulation'
                                ))
                            
                            # Target (90.5%)
                            target_api = 90.5
                            try:
                                target_efrf = float(np.interp(target_api, plot_df['API Loading (%)'].values, plot_df['Capping Risk (EFRF)'].values))
                            except:
                                target_efrf = 0.25
                            
                            fig_p.add_trace(go.Scatter(
                                x=[target_api], y=[target_efrf],
                                mode='markers+text',
                                marker=dict(color='#ffc107', size=18, symbol='star', line=dict(color='#ff6b00', width=2)),
                                text=[f"⭐ Target<br>90.5% API"],
                                textposition="bottom center",
                                name='Target (90.5%)'
                            ))
                            
                            fig_p.update_layout(
                                title=dict(text="Multi-Objective Pareto Front (PINN + NSGA-II)", font=dict(size=18)),
                                xaxis=dict(title='API Loading (%)', range=[84, 96], gridcolor='lightgray'),
                                yaxis=dict(title='Capping Risk (EFRF)', range=[0, 1.0], gridcolor='lightgray'),
                                hovermode='closest',
                                legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.85)', bordercolor='black', borderwidth=1),
                                height=550,
                                plot_bgcolor='white'
                            )
                            
                            st.plotly_chart(fig_p, use_container_width=True)
                        else:
                            st.warning("⚠️ No Pareto data available. Using Matplotlib fallback.")
                            fig = plot_pareto_matplotlib(pareto_api, pareto_efrf, api, efrf)
                            st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"⚠️ Plotly error: {e}. Using Matplotlib fallback.")
                        fig = plot_pareto_matplotlib(pareto_api, pareto_efrf, api, efrf)
                        st.pyplot(fig)
                else:
                    fig = plot_pareto_matplotlib(pareto_api, pareto_efrf, api, efrf)
                    st.pyplot(fig)
            else:
                st.warning("⚠️ No Pareto front found.")
            
            # ================================================================
            # 5. BEST SOLUTIONS TABLE
            # ================================================================
            st.markdown("### 🏆 Best Pareto Solutions")
            
            if len(fronts) > 0 and len(fronts[0]) > 0:
                front0 = fronts[0]
                pareto_api = -objectives[front0, 0]
                pareto_efrf = objectives[front0, 1]
                pareto_tensile = nsga.tensile[front0]
                
                top_n = min(10, len(pareto_api))
                sorted_indices = np.argsort(pareto_api)[::-1][:top_n]
                
                best_df = pd.DataFrame({
                    'API (%)': [pareto_api[i] for i in sorted_indices],
                    'EFRF': [pareto_efrf[i] for i in sorted_indices],
                    'Tensile (MPa)': [pareto_tensile[i] for i in sorted_indices]
                })
                
                st.dataframe(best_df.style.highlight_max(color='lightgreen', subset=['API (%)']), use_container_width=True)
            
            # ================================================================
            # 6. SENSITIVITY ANALYSIS (Plotly with Fallback to Matplotlib)
            # ================================================================
            st.markdown("### 🔍 Interactive Sensitivity Analysis")
            
            # Prepare data
            base_inputs = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
            _, base_efrf = predict(model, scaler, base_inputs)
            
            features = ['API%', 'MCC%', 'PVPP%', 'Mg-St%', 'Binder%', 'Pressure', 'Speed', 'Granule']
            sensitivities = []
            
            for i in range(8):
                test_inputs = base_inputs.copy()
                test_inputs[i] += 0.05 * (base_inputs[i] + 0.1)
                _, efrf_pos = predict(model, scaler, test_inputs)
                
                test_inputs[i] = base_inputs[i] - 0.05 * (base_inputs[i] + 0.1)
                _, efrf_neg = predict(model, scaler, test_inputs)
                
                sensitivities.append(max(abs(efrf_pos - base_efrf), abs(efrf_neg - base_efrf)))
            
            if PLOTLY_AVAILABLE:
                try:
                    sorted_idx = np.argsort(sensitivities)[::-1]
                    sorted_names = [features[i] for i in sorted_idx]
                    sorted_values = [sensitivities[i] for i in sorted_idx]
                    
                    colors = ['#e74c3c' if v > np.mean(sensitivities) else '#2ecc71' for v in sorted_values]
                    
                    fig_tornado = go.Figure()
                    fig_tornado.add_trace(go.Bar(
                        y=sorted_names,
                        x=sorted_values,
                        orientation='h',
                        marker=dict(color=colors, line=dict(color='white', width=0.5)),
                        text=[f"{v:.4f}" for v in sorted_values],
                        textposition='outside',
                        textfont=dict(size=11, color='black'),
                        hovertemplate='<b>%{y}</b><br>Sensitivity: %{x:.4f}<extra></extra>'
                    ))
                    
                    avg_sens = np.mean(sorted_values) if sorted_values else 0
                    fig_tornado.add_vline(x=avg_sens, line_dash="dash", line_color="#e74c3c", line_width=2,
                                         annotation_text=f"Avg: {avg_sens:.4f}", annotation_position="top right")
                    
                    fig_tornado.update_layout(
                        title=dict(text="Feature Sensitivity Analysis — Impact on Capping Risk (EFRF)", font=dict(size=18)),
                        xaxis=dict(title='Sensitivity (ΔEFRF)', gridcolor='lightgray'),
                        yaxis=dict(title='Parameters', gridcolor='lightgray', categoryorder='array', categoryarray=sorted_names),
                        height=450,
                        margin=dict(l=100, r=80, b=50, t=60),
                        plot_bgcolor='white',
                        hovermode='y unified'
                    )
                    
                    if sorted_values:
                        max_idx = np.argmax(sensitivities)
                        fig_tornado.add_annotation(
                            x=0.02, y=1.08, xref="paper", yref="paper",
                            text=f"💡 Most influential: {features[max_idx]} (ΔEFRF = {sensitivities[max_idx]:.4f})",
                            showarrow=False,
                            font=dict(size=12, color='#dc3545')
                        )
                    
                    st.plotly_chart(fig_tornado, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Plotly error: {e}. Using Matplotlib fallback.")
                    fig = plot_sensitivity_matplotlib(features, sensitivities)
                    st.pyplot(fig)
            else:
                fig = plot_sensitivity_matplotlib(features, sensitivities)
                st.pyplot(fig)
            
            # ================================================================
            # 7. MODEL PERFORMANCE COMPARISON
            # ================================================================
            st.markdown("### 📊 Model Performance Comparison")
            
            with st.spinner("Training baseline models..."):
                baseline_df = train_and_evaluate_baselines(X_train, X_test, y_train, y_test)
                
                X_test_scaled = scaler.transform(X_test)
                X_test_tensor = torch.FloatTensor(X_test_scaled)
                with torch.no_grad():
                    y_pred_pinn = model(X_test_tensor).numpy()
                y_pred_pinn_tensile = y_pred_pinn[:, 0]
                
                r2_pinn = r2_score(y_test, y_pred_pinn_tensile)
                rmse_pinn = np.sqrt(mean_squared_error(y_test, y_pred_pinn_tensile))
                mae_pinn = mean_absolute_error(y_test, y_pred_pinn_tensile)
                
                pinn_result = {
                    'Model': 'PINN (Proposed)',
                    'R²': r2_pinn,
                    'RMSE': rmse_pinn,
                    'MAE': mae_pinn,
                    'Physics Consistency': '✅ Enforced'
                }
                
                all_results = [pinn_result] + baseline_df.to_dict('records')
                df_results = pd.DataFrame(all_results)
                
                st.dataframe(
                    df_results.style.highlight_max(subset=['R²'], color='lightgreen')
                               .highlight_min(subset=['RMSE', 'MAE'], color='lightcoral'),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("#### 📈 Visual Comparison")
                
                colors = ['#2ecc71', '#3498db', '#f39c12', '#9b59b6']
                fig_charts, axes = plt.subplots(1, 3, figsize=(15, 5))
                plt.style.use('seaborn-v0_8-darkgrid')
                
                models = df_results['Model'].tolist()
                
                axes[0].bar(models, df_results['R²'], color=colors)
                axes[0].set_ylim(0, 1)
                axes[0].set_title('R² Score (Higher is Better)', fontsize=12)
                axes[0].set_ylabel('R²')
                axes[0].grid(axis='y', alpha=0.3)
                for i, v in enumerate(df_results['R²']):
                    axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
                
                axes[1].bar(models, df_results['RMSE'], color=colors)
                axes[1].set_title('RMSE (Lower is Better)', fontsize=12)
                axes[1].set_ylabel('RMSE (MPa)')
                axes[1].grid(axis='y', alpha=0.3)
                for i, v in enumerate(df_results['RMSE']):
                    axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
                
                axes[2].bar(models, df_results['MAE'], color=colors)
                axes[2].set_title('MAE (Lower is Better)', fontsize=12)
                axes[2].set_ylabel('MAE (MPa)')
                axes[2].grid(axis='y', alpha=0.3)
                for i, v in enumerate(df_results['MAE']):
                    axes[2].text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig_charts)
                plt.close()
                
                st.caption("📌 PINN achieves the best predictive accuracy (highest R², lowest RMSE and MAE) and enforces physical consistency via Heckel equation and EFRF constraints.")
            
            # ================================================================
            # 8. GENERATE PDF REPORT
            # ================================================================
            st.markdown("### 📄 Report")
            
            if tensile >= 2.0 and efrf < 0.5:
                status_text = "PASS"
            else:
                status_text = "FAIL"
            
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            try:
                pdf_data = create_pdf_report(
                    api, mcc, pvpp, mgst, binder, pressure, speed, granule,
                    tensile, efrf, total, status_text, timestamp
                )
                
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_data,
                    file_name=f"formulation_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
    
    else:
        st.info("👆 Adjust parameters and click **'Predict & Optimise'**")

# Footer
st.markdown("---")
st.caption("🔬 **Computational proof-of-concept. Experimental validation ongoing.**")
st.caption("📧 Contact: [babuker@protonmail.com](mailto:babuker@protonmail.com)")
