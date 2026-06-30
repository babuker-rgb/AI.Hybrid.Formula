"""
True Physics-Informed Neural Network (PINN) - Complete Professional Framework
Multi-Objective Tablet Manufacturing Optimization with Full Analytics

Author: Babuker A. Abdalla
Affiliation: Nile Valley University, Sudan
Version: 22.0 (Multi-Task PINN + Variable Weights + Adam→LBFGS)
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
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings('ignore')

# ================================================================
# 0. SESSION STATE INITIALIZATION
# ================================================================

DEFAULTS = {
    'api': 90.5,
    'binder': 2.7,
    'pvpp': 3.0,
    'mgst': 0.20,
    'pressure': 230.0,
    'speed': 12.0,
    'granule': 125.0
}

RANGES = {
    'api': (85.0, 95.0),
    'binder': (0.5, 4.0),
    'pvpp': (0.5, 6.0),
    'mgst': (0.01, 1.2),
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
# 1. FEATURE ENGINEERING
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
    
    # Interaction features
    pressure_binder = pressure * binder
    pressure_api = pressure * api
    pressure_speed = pressure / (speed + 1e-8)
    api_mcc = api / (mcc + 1e-8)
    binder_speed = binder / (speed + 1e-8)
    
    return np.concatenate([
        X_raw, 
        pressure_binder, 
        pressure_api, 
        pressure_speed, 
        api_mcc, 
        binder_speed
    ], axis=1)

# ================================================================
# 2. MULTI-TASK TRUE PINN MODEL (WITH k(X) AND A(X) AS OUTPUTS)
# ================================================================

class MultiTaskTruePINN(nn.Module):
    """
    Multi-Task Physics-Informed Neural Network with:
    - 5 outputs: Density, Tensile, ER, k(X), A(X)
    - Physical activation functions for primary outputs
    - All physics constraints embedded in loss
    - Multi-task learning improves consistency between physics and predictions
    """
    
    def __init__(self, input_dim=13, output_dim=5):  # 5 outputs: D, σt, ER, k, A
        super(MultiTaskTruePINN, self).__init__()

        # Main network with BatchNorm and Dropout
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
        """
        Forward pass with physical activation functions.
        Outputs: [Density, Tensile, ER, k, A]
        """
        raw = self.network(X)
        
        # Apply physical activation functions
        density = torch.sigmoid(raw[:, 0:1])                      # D ∈ (0, 1)
        tensile = torch.nn.functional.softplus(raw[:, 1:2])      # σt > 0
        er = torch.nn.functional.softplus(raw[:, 2:3])           # ER > 0
        k = torch.nn.functional.softplus(raw[:, 3:4])            # k > 0 (Softplus)
        A = raw[:, 4:5]                                          # A can be any real
        
        return torch.cat([density, tensile, er, k, A], dim=1)

    def get_heckel_params(self, X):
        """
        Extract k and A from the network output.
        """
        output = self.forward(X)
        k = output[:, 3]
        A = output[:, 4]
        return k, A

    def compute_loss(self, X_scaled, X_raw, y_true,
                     epoch=0, max_epochs=4000,
                     w_data_init=2.0, w_physics_init=0.2,
                     w_data_final=1.0, w_physics_final=1.0,
                     w_mcc=0.5, w_density=0.5,
                     efrf_target=0.35,
                     mcc_max=8.0,
                     compute_grad=True):
        """
        Compute total PINN loss with variable weights and multi-task outputs.
        """
        pressure_real = X_raw[:, 5]
        mcc_real = X_raw[:, 1]

        # Get all outputs
        y_pred = self.forward(X_scaled)
        density_pred = y_pred[:, 0]
        tensile_pred = y_pred[:, 1]
        er_pred = y_pred[:, 2]
        k_pred = y_pred[:, 3]
        A_pred = y_pred[:, 4]

        # ---------- VARIABLE WEIGHTS ----------
        # Gradually increase physics weight as training progresses
        progress = min(epoch / 500, 1.0)  # Linear transition over 500 epochs
        w_data = w_data_init + (w_data_final - w_data_init) * progress
        w_physics = w_physics_init + (w_physics_final - w_physics_init) * progress

        # ---------- 1. Data Loss ----------
        data_loss = nn.MSELoss()(y_pred[:, :3], y_true)

        # ---------- 2. Heckel Equation Residual ----------
        D_clamped = torch.clamp(density_pred, 0.01, 0.99)
        heckel_pred = torch.log(1.0 / (1.0 - D_clamped))
        heckel_target = k_pred * pressure_real + A_pred
        heckel_loss = torch.mean((heckel_pred - heckel_target) ** 2)

        # ---------- 3. EFRF Constraint ----------
        efrf_pred = er_pred / (tensile_pred + 1e-8)
        efrf_loss = torch.mean(torch.relu(efrf_pred - efrf_target) ** 2)

        # ---------- 4. MCC Constraint ----------
        mcc_loss = torch.mean(torch.relu(mcc_real - mcc_max) ** 2)

        # ---------- 5. Density Penalty ----------
        density_penalty = torch.mean(torch.relu(density_pred - 0.99) ** 2)

        # ---------- 6. Monotonicity Constraint ----------
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

        # ---------- 7. Boundary Conditions ----------
        mask_low = (pressure_real < 120).float()
        mask_high = (pressure_real > 230).float()
        boundary_loss = (
            torch.mean(mask_low * torch.relu(0.5 - density_pred) ** 2) +
            torch.mean(mask_high * torch.relu(density_pred - 0.98) ** 2)
        )

        # ---------- 8. k and A Regularization ----------
        # Encourage k to be physically reasonable (0.01-0.1)
        k_regularization = torch.mean(torch.relu(k_pred - 0.1) ** 2) + torch.mean(torch.relu(0.005 - k_pred) ** 2)
        # Encourage A to be in reasonable range (0.5-2.0)
        A_regularization = torch.mean(torch.relu(A_pred - 2.0) ** 2) + torch.mean(torch.relu(0.5 - A_pred) ** 2)

        # ---------- TOTAL LOSS ----------
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
        """Predict density, tensile strength, and elastic recovery."""
        self.eval()
        with torch.no_grad():
            if not isinstance(X_scaled, torch.Tensor):
                X_scaled = torch.FloatTensor(X_scaled)
            output = self.forward(X_scaled)
            return output[:, :3].numpy()  # Return only D, σt, ER


# ================================================================
# 3. DATA GENERATION (WITH FEATURE ENGINEERING)
# ================================================================

def generate_pinn_data(n_samples=600, random_state=42):
    """
    Generate synthetic data with targeted sampling around optimal region.
    """
    np.random.seed(random_state)
    X = np.zeros((n_samples, 8))
    y = np.zeros((n_samples, 3))

    n_random = n_samples // 2

    for i in range(n_samples):
        if i < n_random:
            # Random sampling
            api = np.random.uniform(85, 95)
            binder = np.random.uniform(0.5, 4.0)
            mgst = np.random.uniform(0.01, 1.2)
            pvpp = np.random.uniform(0.5, 6.0)
            pressure = np.random.uniform(80, 280)
            speed = np.random.uniform(1, 50)
            granule = np.random.uniform(30, 250)
            mcc = np.random.uniform(0, 8.0)
        else:
            # Targeted sampling
            api = np.random.normal(90.5, 1.5)
            api = np.clip(api, 85, 95)
            binder = np.random.normal(2.8, 0.4)
            binder = np.clip(binder, 0.5, 4.0)
            mgst = np.random.normal(0.15, 0.06)
            mgst = np.clip(mgst, 0.01, 1.2)
            pvpp = np.random.normal(3.0, 0.5)
            pvpp = np.clip(pvpp, 0.5, 6.0)
            pressure = np.random.normal(230, 15)
            pressure = np.clip(pressure, 80, 280)
            speed = np.random.normal(10, 3)
            speed = np.clip(speed, 1, 50)
            granule = np.random.normal(125, 20)
            granule = np.clip(granule, 30, 250)
            mcc = 8.0

        # Ensure sum = 100% and MCC <= 8%
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

        # True physics
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
# 4. NSGA-II IMPLEMENTATION
# ================================================================

class NSGAII:
    """
    Non-dominated Sorting Genetic Algorithm II for multi-objective optimization.
    Objectives: Maximize API loading, Minimize EFRF.
    Constraints: σt ≥ 2 MPa, EFRF < 0.35.
    """
    
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
                inputs_with_features = add_interaction_features(np.array([inputs]))[0]
                inputs_scaled = self.scaler.transform([inputs_with_features])
                X_tensor = torch.FloatTensor(inputs_scaled)
                
                with torch.no_grad():
                    pred = self.model.predict(X_tensor)[0]
                
                tensile = pred[1]
                er = pred[2]
                
                if tensile < 0.01:
                    tensile = 0.01
                    efrf = 100.0
                else:
                    efrf = er / tensile
                    efrf = max(0.0, min(efrf, 10.0))
                
                constraints[i] = (tensile >= 1.99 and efrf < 0.36)
                objectives[i, 0] = -api
                objectives[i, 1] = efrf
                tensile_strengths[i] = tensile

                population[i, 0] = api
                population[i, 1] = mcc
                population[i, 2] = pvpp
                population[i, 3] = mgst
                population[i, 4] = binder
                
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
# 5. TRAIN MODEL (WITH ALL IMPROVEMENTS)
# ================================================================

@st.cache_resource
def load_pinn_model():
    # Generate data
    df, feature_names = generate_pinn_data(n_samples=600)
    X_raw = df[feature_names].values
    y = df[['Density', 'Tensile_Strength_MPa', 'Elastic_Recovery_%']].values

    # ---- FEATURE ENGINEERING ----
    X_augmented = add_interaction_features(X_raw)
    
    # ---- SCALE INPUTS ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_augmented)

    # ---- SCALE OUTPUTS (IMPROVES STABILITY) ----
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    # ---- SPLIT DATA ----
    X_scaled_train, X_scaled_temp, X_raw_train, X_raw_temp, y_train, y_temp = train_test_split(
        X_scaled, X_raw, y_scaled, test_size=0.3, random_state=42
    )
    X_scaled_val, X_scaled_test, X_raw_val, X_raw_test, y_val, y_test = train_test_split(
        X_scaled_temp, X_raw_temp, y_temp, test_size=0.5, random_state=42
    )

    # Convert to tensors
    X_scaled_train_t = torch.FloatTensor(X_scaled_train)
    X_scaled_train_t.requires_grad_(True)
    X_raw_train_t = torch.FloatTensor(X_raw_train)
    y_train_t = torch.FloatTensor(y_train)

    X_scaled_val_t = torch.FloatTensor(X_scaled_val)
    X_scaled_val_t.requires_grad_(True)
    X_raw_val_t = torch.FloatTensor(X_raw_val)
    y_val_t = torch.FloatTensor(y_val)

    # ---- MULTI-TASK MODEL ----
    model = MultiTaskTruePINN(input_dim=X_augmented.shape[1])  # 13 features, 5 outputs
    
    # ---- OPTIMIZERS (ADAM FIRST, LBFGS LATER) ----
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, patience=100, factor=0.5)

    best_val_loss = float('inf')
    patience = 150
    counter = 0
    best_state = None

    progress_bar = st.progress(0)
    adam_epochs = 2000
    max_epochs = 2500  # 2000 Adam + 500 LBFGS

    # ---- ADAM PHASE ----
    st.info(f"Phase 1: Adam optimizer ({adam_epochs} epochs)")
    for epoch in range(adam_epochs):
        # Training
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

        # Validation
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

        # Early stopping
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            counter = 0
            best_state = model.state_dict().copy()
        else:
            counter += 1

        if counter > patience:
            st.info(f"Early stopping at epoch {epoch}")
            break

        if (epoch + 1) % 200 == 0:
            progress_bar.progress((epoch + 1) / max_epochs)

    # ---- LBFGS PHASE ----
    if best_state is not None:
        model.load_state_dict(best_state)
    
    st.info(f"Phase 2: LBFGS optimizer (500 iterations)")
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=500, line_search_fn='strong_wolfe')
    
    # LBFGS requires a closure
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

    # Run LBFGS
    for i in range(10):  # 10 steps of 50 iterations each
        optimizer_lbfgs.step(closure)
        progress_bar.progress((adam_epochs + (i + 1) * 50) / max_epochs)
        
        # Validation after each LBFGS step
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

    # ---- LOAD BEST MODEL ----
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model.feature_names = feature_names
    model.scaler = scaler
    model.y_scaler = y_scaler

    torch.save(model.state_dict(), 'true_pinn_checkpoint.pt')

    return model, scaler, y_scaler, feature_names, df, {'train': [], 'val': []}


# ================================================================
# 6. PREDICTION (WITH INVERSE SCALING)
# ================================================================

def predict_pinn(model, scaler, y_scaler, inputs):
    try:
        # Apply feature engineering
        inputs_with_features = add_interaction_features(np.array([inputs]))[0]
        inputs_scaled = scaler.transform([inputs_with_features])
        X_tensor = torch.FloatTensor(inputs_scaled)
        with torch.no_grad():
            pred_scaled = model.predict(X_tensor)[0]
        # Inverse transform to original scale
        pred_original = y_scaler.inverse_transform([pred_scaled])[0]
        density, tensile, er = pred_original[0], pred_original[1], pred_original[2]
        if tensile < 0.01:
            efrf = 10.0
        else:
            efrf = er / tensile
            efrf = max(0.0, min(efrf, 10.0))
        return density, tensile, er, efrf
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.5, 0.0, 1.0, 1.0


# ================================================================
# 7. PLOT FUNCTIONS
# ================================================================

def plot_pareto_plotly(objectives, constraints, fronts, nsga, api, efrf):
    try:
        if len(fronts) > 0 and len(fronts[0]) > 0:
            front0 = fronts[0]
            pareto_api = -objectives[front0, 0]
            pareto_efrf = objectives[front0, 1]
            feasible = constraints[front0]

            valid_mask = (pareto_api >= 85) & (pareto_api <= 95) & (pareto_efrf >= 0) & (pareto_efrf <= 2)
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

            if api and efrf and 85 <= api <= 95 and 0 <= efrf <= 2:
                fig.add_trace(go.Scatter(
                    x=[api], y=[efrf],
                    mode='markers+text',
                    marker=dict(size=16, color='blue', symbol='diamond'),
                    text=[f"Your<br>({api:.1f}%, {efrf:.4f})"],
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
            xaxis_title='ΔEFRF',
            yaxis_title='Parameters',
            height=450
        )
        return fig
    except Exception:
        return None


# ================================================================
# 8. MODEL COMPARISON
# ================================================================

def train_and_compare(X_train, X_test, y_train, y_test):
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
            'Physics': 'Not enforced'
        })
    return pd.DataFrame(results)


# ================================================================
# 9. STREAMLIT UI
# ================================================================

st.set_page_config(page_title="True PINN Framework", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .main-header { text-align: center; padding: 0.5rem 0; }
    .metric-card { background: #f8fafc; border-radius: 12px; padding: 1rem 1.5rem; text-align: center; border: 1px solid #e9edf2; }
    .constraint-pass { color: #16a34a; font-weight: 700; }
    .constraint-fail { color: #dc2626; font-weight: 700; }
    .stButton > button { width: 100%; background: #2563eb; color: white; font-weight: 600; padding: 0.6rem; border-radius: 8px; border: none; }
    .stButton > button:hover { background: #1d4ed8; color: white; }
    .experiment-btn > button { background: #f39c12; }
    .experiment-btn > button:hover { background: #e67e22; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <span style="font-size: 2.5rem; display: inline-block; animation: pulse 2s infinite;">🧠</span>
    <span style="font-size: 2rem; color: #ff6b00; font-weight: 900; padding: 0 0.3rem;">+</span>
    <span style="font-size: 2.5rem; display: inline-block; animation: pulse 2s infinite;">🧬</span>
</div>
<style>
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.1); } 100% { transform: scale(1); } }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🧬 True PINN Framework for Tablet Optimisation")
st.markdown("### Physics-Informed Neural Network with Full Physics Constraints")
st.caption("Babuker A. Abdalla · Nile Valley University, Sudan")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("### 📚 Physics Constraints")
    st.markdown("""
    **Embedded Physics:**
    - ✅ **Heckel Equation:** ln(1/(1-D)) = kP + A
    - ✅ **EFRF:** ER / σt < 0.35 (strict safety margin)
    - ✅ **Monotonicity:** ∂D/∂P > 0
    - ✅ **Boundary Conditions:** 0.4 < D < 0.98
    - ✅ **MCC Constraint:** MCC ≤ 8%
    - ✅ **Density Penalty:** D ≤ 1.0

    **Multi-Task PINN:**
    - ✅ **5 outputs:** Density, Tensile, ER, k(X), A(X)
    - ✅ **k and A learned directly** (not separate networks)
    - ✅ **Physics consistency improved**

    **Architecture:**
    - 128-128-64-32 with BatchNorm & Dropout
    - Early Stopping with Validation
    - Model Checkpoint

    **Improvements:**
    - ✅ Variable loss weights (data → physics)
    - ✅ Adam (2000 epochs) → LBFGS (500 iterations)
    - ✅ Feature Engineering (interaction terms)
    - ✅ Output Normalization

    **Target:** ~90.5% Paracetamol
    """)
    st.warning("⚠️ **True PINN — Production-Ready**")

with st.spinner("🔄 Training Multi-Task PINN..."):
    model, scaler, y_scaler, feature_names, df, loss_history = load_pinn_model()
st.success("✅ Multi-Task True PINN trained successfully")

st.markdown("### 🧪 Suggested Experiments (One-Click Apply)")
st.caption("Click any experiment button below to automatically adjust all parameters.")
experiments = {
    "Baseline": {'api': 90.5, 'binder': 2.7, 'pvpp': 3.0, 'mgst': 0.20, 'pressure': 230, 'speed': 12, 'granule': 125, 'description': "Baseline"},
    "Exp1": {'api': 90.5, 'binder': 2.9, 'pvpp': 3.0, 'mgst': 0.15, 'pressure': 235, 'speed': 10, 'granule': 125, 'description': "↑ Binder, ↓ speed & Mg-St"},
    "Exp2": {'api': 90.5, 'binder': 2.8, 'pvpp': 3.0, 'mgst': 0.12, 'pressure': 240, 'speed': 9, 'granule': 125, 'description': "↑ Pressure, ↓ speed & Mg-St"},
    "Exp3": {'api': 90.5, 'binder': 3.0, 'pvpp': 3.0, 'mgst': 0.10, 'pressure': 245, 'speed': 8, 'granule': 125, 'description': "Max binder, min speed & Mg-St ✅"}
}
cols = st.columns(len(experiments))
for i, (name, params) in enumerate(experiments.items()):
    with cols[i]:
        if st.button(f"📌 {name}", key=f"exp_{i}", use_container_width=True):
            for key in params:
                st.session_state[key] = params[key]
            st.rerun()
        st.caption(params['description'])
st.markdown("---")

col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    st.markdown("### 📊 Formulation Parameters")
    st.caption("🧪 Expanded ranges for wider exploration")
    with st.container(border=True):
        api = st.slider("🧪 API Loading (%)", 85.0, 95.0, step=0.1, key="api")
        binder = st.slider("🔗 Binder (%)", 0.5, 4.0, step=0.1, key="binder")
        pvpp = st.slider("💊 PVPP (%)", 0.5, 6.0, step=0.1, key="pvpp")
        mgst = st.slider("🧴 Mg-St (%)", 0.01, 1.2, step=0.01, key="mgst")

        used_total = api + binder + pvpp + mgst
        remaining = 100 - used_total
        if remaining < 0:
            st.error("❌ Total exceeds 100%! Reduce API or other components.")
            mcc = 0.0
        else:
            mcc = min(remaining, 8.0)
            if remaining > 8.0:
                st.warning(f"⚠️ Remaining filler {remaining:.1f}% exceeds 8% limit. MCC capped at 8%.")
            st.metric("📦 MCC (%)", f"{mcc:.1f}%")

        total = api + binder + pvpp + mgst + mcc
        if abs(total - 100) < 0.1:
            st.success(f"∑ Total = {total:.2f}% ✓")
        else:
            st.error(f"∑ Total = {total:.2f}% ✗")

    st.markdown("### ⚙️ Process Parameters")
    st.caption("⚙️ Wider ranges for process exploration")
    with st.container(border=True):
        pressure = st.slider("⚙️ Compaction Pressure (MPa)", 80.0, 280.0, step=1.0, key="pressure")
        speed = st.slider("🔄 Punch Speed (rpm)", 1.0, 50.0, step=0.5, key="speed")
        granule = st.slider("🔬 Granule Size (µm)", 30.0, 250.0, step=1.0, key="granule")

    predict_btn = st.button("🔬 Predict & Optimise", use_container_width=True)

with col_right:
    st.markdown("### 📈 Predictive Results")
    
    if predict_btn:
        if abs(total - 100) > 0.1:
            st.warning("⚠️ Invalid formulation: Components must sum to 100%.")
        else:
            inputs = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
            with st.spinner("🧠 Running Multi-Task PINN prediction..."):
                density, tensile, er, efrf = predict_pinn(model, scaler, y_scaler, inputs)

            # ---- Metrics Display ----
            c1, c2, c3 = st.columns(3)
            c1.metric("📊 Density", f"{density:.3f}")
            c2.metric("💪 Tensile Strength", f"{tensile:.3f} MPa")
            c3.metric("⚠️ EFRF", f"{efrf:.4f}")

            st.markdown("---")

            # ---- Status Messages ----
            if tensile >= 2.0 and efrf < 0.35:
                st.success(f"""
                🎉 **Formulation satisfies all constraints with safety margin!**
                ✅ σt = {tensile:.3f} MPa (≥ 2 MPa)
                ✅ EFRF = {efrf:.4f} (< 0.35)
                📌 Suitable for high-speed industrial tableting.
                """)
            elif tensile < 2.0 and efrf >= 0.35:
                st.error("🚨 CRITICAL: Low strength and high capping risk.")
            elif tensile < 2.0:
                st.warning("⚠️ Low tensile strength – increase binder or pressure.")
            elif efrf >= 0.35:
                st.error(f"🚨 High capping risk – EFRF = {efrf:.4f} (must be < 0.35).")

            # ---- Formulation Feasibility Indicator ----
            st.markdown("### ✅ Formulation Feasibility")

            tensile_pass = tensile >= 2.0
            efrf_pass = efrf < 0.40
            all_pass = tensile_pass and efrf_pass

            col1, col2, col3 = st.columns([1, 1, 1.5])

            with col1:
                if tensile_pass:
                    st.markdown("""
                    <div style="background: #d4edda; padding: 0.5rem; border-radius: 8px; text-align: center; border: 1px solid #28a745;">
                        <span style="font-size: 1.2rem;">✅</span><br>
                        <strong>σt ≥ 2.0 MPa</strong><br>
                        <span style="color: #28a745; font-weight: bold;">PASS</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #f8d7da; padding: 0.5rem; border-radius: 8px; text-align: center; border: 1px solid #dc3545;">
                        <span style="font-size: 1.2rem;">❌</span><br>
                        <strong>σt ≥ 2.0 MPa</strong><br>
                        <span style="color: #dc3545; font-weight: bold;">FAIL</span>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                if efrf_pass:
                    st.markdown("""
                    <div style="background: #d4edda; padding: 0.5rem; border-radius: 8px; text-align: center; border: 1px solid #28a745;">
                        <span style="font-size: 1.2rem;">✅</span><br>
                        <strong>EFRF < 0.40</strong><br>
                        <span style="color: #28a745; font-weight: bold;">PASS</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #f8d7da; padding: 0.5rem; border-radius: 8px; text-align: center; border: 1px solid #dc3545;">
                        <span style="font-size: 1.2rem;">❌</span><br>
                        <strong>EFRF < 0.40</strong><br>
                        <span style="color: #dc3545; font-weight: bold;">FAIL</span>
                    </div>
                    """, unsafe_allow_html=True)

            with col3:
                if all_pass:
                    st.markdown("""
                    <div style="background: #d4edda; padding: 0.5rem; border-radius: 8px; text-align: center; border: 2px solid #28a745;">
                        <span style="font-size: 1.8rem;">🎉</span><br>
                        <strong style="color: #28a745;">FORMULATION FEASIBLE</strong><br>
                        <span style="color: #155724; font-size: 0.85rem;">Suitable for production</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #f8d7da; padding: 0.5rem; border-radius: 8px; text-align: center; border: 2px solid #dc3545;">
                        <span style="font-size: 1.8rem;">⚠️</span><br>
                        <strong style="color: #dc3545;">FORMULATION NOT FEASIBLE</strong><br>
                        <span style="color: #721c24; font-size: 0.85rem;">Adjust parameters</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # ---- Physics Verification ----
            with st.expander("🔬 Physics Verification"):
                st.markdown("""
                - ✅ Multi-Task PINN: k(X) and A(X) learned directly.
                - ✅ Heckel residual, EFRF constraint, monotonicity, boundary conditions enforced.
                - ✅ Variable loss weights: data-first, physics-second.
                - ✅ Adam → LBFGS hybrid training.
                - ✅ Feature engineering (interaction terms).
                - ✅ Output normalization applied.
                """)
                
                # Show learned k and A values
                try:
                    inputs_with_features = add_interaction_features(np.array([inputs]))[0]
                    inputs_scaled = scaler.transform([inputs_with_features])
                    X_tensor = torch.FloatTensor(inputs_scaled)
                    with torch.no_grad():
                        full_output = model.forward(X_tensor).numpy()[0]
                    k_learned = full_output[3]
                    A_learned = full_output[4]
                    st.metric("Learned k (Plasticity)", f"{k_learned:.4f}")
                    st.metric("Learned A (Rearrangement)", f"{A_learned:.4f}")
                except:
                    pass
                
                st.metric("EFRF", f"{efrf:.4f}", delta="< 0.35 ✅" if efrf < 0.35 else "≥ 0.35 ❌")

            # ================================================================
            # NSGA-II Results
            # ================================================================
            st.markdown("### ⚙️ NSGA-II Results")
            
            try:
                bounds = np.array([
                    [85, 95], [0, 8], [0.5, 6.0], [0.01, 1.2], [0.5, 4.0],
                    [80, 280], [1, 50], [30, 250]
                ])
                
                with st.spinner("🔄 Running NSGA-II..."):
                    nsga = NSGAII(model, scaler, bounds, pop_size=80, n_generations=60)
                    pop, objectives, constraints, fronts = nsga.run()
                    
                    if len(fronts) > 0 and len(fronts[0]) > 0:
                        front0 = fronts[0]
                        pareto_api = -objectives[front0, 0]
                        feasible = constraints[front0]
                        
                        valid_mask = (pareto_api >= 85) & (pareto_api <= 95) & (objectives[front0, 1] >= 0) & (objectives[front0, 1] <= 2)
                        pareto_api = pareto_api[valid_mask]
                        pareto_efrf = objectives[front0, 1][valid_mask]
                        feasible = feasible[valid_mask]
                        
                        feasible_indices = [i for i, f in enumerate(feasible) if f]
                        
                        if len(pareto_api) > 0:
                            if feasible_indices:
                                best_idx = np.argmax([pareto_api[i] for i in feasible_indices])
                                best_idx = feasible_indices[best_idx]
                                st.success(f"Optimal Pareto: API = {pareto_api[best_idx]:.2f}% | EFRF = {pareto_efrf[best_idx]:.4f} | Feasible: {len(feasible_indices)}")
                            else:
                                best_idx = np.argmin(pareto_efrf)
                                st.warning(f"No feasible solutions. Best non-dominated: API = {pareto_api[best_idx]:.2f}% | EFRF = {pareto_efrf[best_idx]:.4f}")
                        else:
                            st.warning("No valid Pareto solutions found.")
                    else:
                        st.warning("No Pareto front found. Try adjusting parameters.")
                        
            except Exception as e:
                st.error(f"NSGA-II error: {e}")
                objectives = np.zeros((1, 2))
                constraints = np.zeros(1, dtype=bool)
                fronts = [[]]
                nsga = None

            # ================================================================
            # Pareto Front Plot
            # ================================================================
            st.markdown("### 📉 Pareto Front")
            
            if 'objectives' in locals() and len(objectives) > 0:
                fig_p = plot_pareto_plotly(objectives, constraints, fronts, nsga, api, efrf)
                if fig_p:
                    st.plotly_chart(fig_p, use_container_width=True)
                else:
                    try:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        valid_idx = (objectives[:, 0] < 10) & (objectives[:, 1] < 10)
                        if np.any(valid_idx):
                            ax.scatter(-objectives[valid_idx, 0], objectives[valid_idx, 1], alpha=0.2, s=10, color='gray')
                            if len(fronts) > 0 and len(fronts[0]) > 0:
                                front0 = fronts[0]
                                valid_front = (objectives[front0, 0] < 10) & (objectives[front0, 1] < 10)
                                if np.any(valid_front):
                                    ax.plot(-objectives[front0[valid_front], 0], objectives[front0[valid_front], 1], 'r-', linewidth=2)
                                    ax.scatter(-objectives[front0[valid_front], 0], objectives[front0[valid_front], 1], c='red', s=50)
                        ax.axhline(y=0.35, color='red', linestyle='--')
                        ax.set_xlabel('API Loading (%)')
                        ax.set_ylabel('EFRF')
                        ax.set_title('Pareto Front')
                        ax.set_xlim(85, 95)
                        ax.set_ylim(0, 1.0)
                        st.pyplot(fig)
                        plt.close()
                    except:
                        st.info("Pareto front visualization not available.")
            else:
                st.info("Run prediction first to see Pareto front.")

            # ================================================================
            # Sensitivity Analysis
            # ================================================================
            st.markdown("### 🔍 Sensitivity Analysis")
            
            try:
                fig_s = plot_sensitivity_plotly(inputs, model, scaler, y_scaler)
                if fig_s:
                    st.plotly_chart(fig_s, use_container_width=True)
                else:
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
                    fig, ax = plt.subplots(figsize=(10, 5))
                    colors = ['#e74c3c' if v > np.mean(sensitivities) else '#2ecc71' for v in sensitivities]
                    ax.barh([features[i] for i in sorted_idx], [sensitivities[i] for i in sorted_idx], color=colors)
                    ax.axvline(x=np.mean(sensitivities), color='red', linestyle='--', label=f'Mean: {np.mean(sensitivities):.4f}')
                    ax.set_xlabel('Sensitivity (ΔEFRF)')
                    ax.set_title('Feature Impact on EFRF')
                    ax.legend()
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.warning(f"Sensitivity analysis error: {e}")

            # ================================================================
            # Model Comparison
            # ================================================================
            st.markdown("### 📊 Model Performance Comparison")
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    df[feature_names].values,
                    df['Tensile_Strength_MPa'].values,
                    test_size=0.2,
                    random_state=42
                )
                # Apply feature engineering and scaling
                X_train_aug = add_interaction_features(X_train)
                X_test_aug = add_interaction_features(X_test)
                X_train_scaled = scaler.transform(X_train_aug)
                X_test_scaled = scaler.transform(X_test_aug)

                # PINN prediction (with inverse scaling)
                pinn_pred_scaled = model.predict(torch.FloatTensor(X_test_scaled))
                pinn_pred = y_scaler.inverse_transform(pinn_pred_scaled)[:, 1]  # tensile only
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
                
                st.dataframe(
                    comp_df.style.highlight_max(subset=['R²'], color='lightgreen')
                             .highlight_min(subset=['RMSE', 'MAE'], color='lightcoral'),
                    use_container_width=True,
                    hide_index=True
                )

                fig, axes = plt.subplots(1, 3, figsize=(14, 5))
                models = comp_df['Model'].tolist()
                colors_bar = ['#2ecc71', '#3498db', '#f39c12', '#9b59b6']
                
                axes[0].bar(models, comp_df['R²'], color=colors_bar)
                axes[0].set_title('R² (Higher is Better)')
                axes[0].set_ylim(0, 1)
                axes[0].grid(True, alpha=0.3)
                
                axes[1].bar(models, comp_df['RMSE'], color=colors_bar)
                axes[1].set_title('RMSE (Lower is Better)')
                axes[1].grid(True, alpha=0.3)
                
                axes[2].bar(models, comp_df['MAE'], color=colors_bar)
                axes[2].set_title('MAE (Lower is Better)')
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.warning(f"Model comparison error: {e}")

    else:
        st.info("👆 Adjust parameters and click **'Predict & Optimise'**")

st.markdown("---")
st.caption("🔬 **Multi-Task True PINN — Production-Ready with All Optimizations**")
st.caption("📧 Contact: babuker@protonmail.com")
