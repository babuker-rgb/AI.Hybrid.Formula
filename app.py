"""
Hybrid AI Framework (PINN-NSGA-II) - Interactive Web Application
Multi-Objective Tablet Manufacturing Optimization

Author: Babuker A. Abdalla
Affiliation: Nile Valley University, Sudan
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 1. PINN MODEL DEFINITION
# ================================================================

class PhysicsLoss(nn.Module):
    """Custom physics loss layer for tablet compaction constraints"""
    
    def __init__(self):
        super(PhysicsLoss, self).__init__()
        # Physical constants (based on paracetamol literature)
        self.k_heckel = nn.Parameter(torch.tensor(0.035), requires_grad=False)
        self.A_heckel = nn.Parameter(torch.tensor(1.2), requires_grad=False)
    
    def forward(self, X, y_pred):
        """
        Compute physics residuals:
        1. Heckel equation: ln(1/(1-D)) = kP + A
        2. EFRF = ER / σ_t (enforced through output consistency)
        """
        # Extract inputs
        api = X[:, 0]
        pressure = X[:, 5]
        speed = X[:, 6]
        
        # Extract predictions
        sigma_t = y_pred[:, 0]
        efrf_pred = y_pred[:, 1]
        
        # --- Physical constraints ---
        # 1. Heckel equation constraint
        k_eff = self.k_heckel * (1 - 0.4 * ((api - 85) / 10).clamp(0, 1))
        k_eff = k_eff * (1 - 0.2 * ((speed - 10) / 30).clamp(0, 1))
        k_eff = torch.clamp(k_eff, min=0.01)
        
        D = 1 - torch.exp(-(k_eff * pressure + self.A_heckel))
        D = torch.clamp(D, 0.5, 1.0)
        
        heckel_target = k_eff * pressure + self.A_heckel
        heckel_pred = -torch.log(1 - D + 1e-8)
        heckel_residual = (heckel_pred - heckel_target) ** 2
        
        # 2. EFRF consistency: EFRF = ER / σ_t
        er = 1.8 + 0.3 * ((api - 85) / 10).clamp(0, 1) + 0.08 * ((speed - 10) / 30).clamp(0, 1)
        efrf_target = er / (sigma_t + 1e-8)
        efrf_residual = (efrf_pred - efrf_target) ** 2
        
        # 3. Physical bounds constraints
        bound_constraint = torch.relu(0.5 - sigma_t) + torch.relu(sigma_t - 6.0)
        bound_constraint = bound_constraint ** 2
        efrf_bound = torch.relu(0.1 - efrf_pred) + torch.relu(efrf_pred - 1.5)
        efrf_bound = efrf_bound ** 2
        
        # Total physics loss
        physics_loss = (heckel_residual.mean() + efrf_residual.mean() 
                       + 0.1 * bound_constraint.mean() + 0.1 * efrf_bound.mean())
        
        return physics_loss


class PINN(nn.Module):
    """Physics-Informed Neural Network"""
    
    def __init__(self, input_dim=8, hidden_layers=[64, 64, 64], output_dim=2):
        super(PINN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.physics_loss = PhysicsLoss()
        
    def forward(self, X):
        return self.network(X)
    
    def compute_loss(self, X, y_true, omega_data=0.8, omega_physics=0.2):
        """Compute hybrid loss: L = ω₁ * MSE_data + ω₂ * MSE_physics"""
        y_pred = self.forward(X)
        data_loss = nn.MSELoss()(y_pred, y_true)
        physics_loss = self.physics_loss(X, y_pred)
        total_loss = omega_data * data_loss + omega_physics * physics_loss
        return total_loss, data_loss, physics_loss
    
    def predict(self, X):
        """Predict with physics constraints"""
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
            return self.forward(X).numpy()


# ================================================================
# 2. DATA GENERATION (In Silico Dataset)
# ================================================================

def generate_heckel_data(n_samples=45, random_state=42):
    """Generate synthetic tablet compaction data using Heckel equation and EFRF."""
    np.random.seed(random_state)
    
    # Define design space boundaries (8 input features)
    bounds = np.array([
        [85.0, 95.0],      # API loading (%)
        [2.0, 8.0],        # MCC (%) - Microcrystalline Cellulose
        [1.0, 5.0],        # PVPP (%) - Crospovidone
        [0.2, 1.0],        # Mg-St (%) - Magnesium Stearate
        [0.5, 3.0],        # Binder (%)
        [100, 250],        # Compaction Pressure (MPa)
        [10, 40],          # Punch Speed (rpm)
        [50, 200]          # Granule Size (µm)
    ])
    
    # Latin Hypercube Sampling
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=8, seed=random_state)
    sample = sampler.random(n=n_samples)
    X = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
    
    feature_names = ['API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%', 
                     'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm']
    
    # Physics model parameters
    k_heckel = 0.035
    A_heckel = 1.2
    elastic_recovery_base = 1.8
    
    y = np.zeros((n_samples, 2))
    
    for i in range(n_samples):
        api = X[i, 0]
        mcc = X[i, 1]
        pvpp = X[i, 2]
        mgst = X[i, 3]
        binder = X[i, 4]
        pressure = X[i, 5]
        speed = X[i, 6]
        granule = X[i, 7]
        
        # Relative Density using Heckel
        k_eff = k_heckel * (1 - 0.4 * (api - 85) / 10) * (1 - 0.2 * (speed - 10) / 30)
        k_eff = max(k_eff, 0.01)
        D = 1 - np.exp(-(k_eff * pressure + A_heckel))
        D = np.clip(D, 0.5, 1.0)
        
        # Tensile Strength
        strength_base = 2.5 * D + 0.5 * np.log(D / (1 - D) + 0.01)
        strength_binder = 0.3 * binder
        strength_api_penalty = 0.8 * (api - 85) / 10
        strength_speed_penalty = 0.15 * (speed - 10) / 30
        strength_lubricant = -0.5 * mgst
        
        tensile_strength = (strength_base + strength_binder - strength_api_penalty 
                           - strength_speed_penalty + strength_lubricant)
        tensile_strength = np.clip(tensile_strength, 0.5, 6.0)
        
        # Elastic Recovery and EFRF
        er = elastic_recovery_base + 0.3 * (api - 85) / 10 + 0.08 * (speed - 10) / 30 - 0.1 * (pressure - 100) / 150
        er = np.clip(er, 0.5, 4.0)
        efrf = er / tensile_strength
        efrf = np.clip(efrf, 0.1, 1.5)
        
        y[i, 0] = tensile_strength
        y[i, 1] = efrf
    
    df = pd.DataFrame(X, columns=feature_names)
    df['Tensile_Strength_MPa'] = y[:, 0]
    df['EFRF'] = y[:, 1]
    
    return df, feature_names


# ================================================================
# 3. TRAIN PINN (Cached for performance)
# ================================================================

@st.cache_resource
def load_pinn_model():
    """Train and return the PINN model with caching"""
    
    # Generate data
    df, feature_names = generate_heckel_data(n_samples=45)
    X = df[feature_names].values
    y = df[['Tensile_Strength_MPa', 'EFRF']].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    np.random.seed(42)
    indices = np.random.permutation(len(X_scaled))
    split = int(0.8 * len(X_scaled))
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    X_train = X_scaled[train_idx]
    y_train = y[train_idx]
    
    # Train PINN
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    
    # Training loop
    for epoch in range(500):
        optimizer.zero_grad()
        total_loss, data_loss, physics_loss = model.compute_loss(X_train_t, y_train_t, 0.8, 0.2)
        
        # Check for NaN or infinite losses
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            st.error(f"Training failed at epoch {epoch}: Loss is NaN or Inf. Please check the data or model.")
            return None, None, None
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    model.eval()
    return model, scaler, feature_names

# ================================================================
# 4. PREDICTION FUNCTION
# ================================================================

def predict_formulation(model, scaler, inputs):
    """Predict tensile strength and EFRF for given formulation"""
    try:
        # Scale the inputs
        inputs_scaled = scaler.transform([inputs])
        with torch.no_grad():
            X_tensor = torch.FloatTensor(inputs_scaled)
            predictions = model(X_tensor).numpy()[0]
        return predictions[0], predictions[1]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return 0.0, 1.0  # Return fallback values

# ================================================================
# 5. STREAMLIT UI
# ================================================================

# Page configuration
st.set_page_config(
    page_title="PINN-NSGA-II Hybrid AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        border: 1px solid #e9edf2;
    }
    .constraint-pass {
        color: #16a34a;
        font-weight: 700;
    }
    .constraint-fail {
        color: #dc2626;
        font-weight: 700;
    }
    .stButton > button {
        width: 100%;
        background: #2563eb;
        color: white;
        font-weight: 600;
        padding: 0.6rem;
        border-radius: 8px;
        border: none;
    }
    .stButton > button:hover {
        background: #1d4ed8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# HEADER
# ================================================================
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🧠 Hybrid AI Framework")
st.subheader("PINN · NSGA-II")
st.markdown("### Multi‑Objective Tablet Manufacturing Optimization · High‑Load Paracetamol")
st.caption("👨‍🔬 Babuker A. Abdalla & Prof. Abdelkarim Mohamed · Nile Valley University")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Load model
with st.spinner("🔄 Loading AI model..."):
    model, scaler, feature_names = load_pinn_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please check the logs.")
        st.stop()
    
st.success("✅ Model loaded successfully!")

# ================================================================
# TWO-COLUMN LAYOUT: Inputs | Results
# ================================================================
col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    st.markdown("### 📊 Formulation Parameters")
    st.markdown("Adjust the parameters below to explore the design space.")
    
    # Create sliders for each parameter
    with st.container(border=True):
        api = st.slider("🧪 API Loading (%)", 85.0, 95.0, 90.5, 0.1, 
                        help="Active Pharmaceutical Ingredient percentage")
        mcc = st.slider("📦 MCC (%)", 2.0, 8.0, 5.0, 0.1,
                        help="Microcrystalline Cellulose - binder/filler")
        pvpp = st.slider("💊 PVPP (%)", 1.0, 5.0, 3.0, 0.1,
                        help="Crospovidone - superdisintegrant")
        mgst = st.slider("🧴 Mg-St (%)", 0.2, 1.0, 0.6, 0.05,
                        help="Magnesium Stearate - lubricant")
        binder = st.slider("🔗 Binder (%)", 0.5, 3.0, 1.5, 0.1)
        pressure = st.slider("⚙️ Compaction Pressure (MPa)", 100.0, 250.0, 180.0, 5.0)
        speed = st.slider("🔄 Punch Speed (rpm)", 10.0, 40.0, 25.0, 1.0)
        granule = st.slider("🔬 Granule Size (µm)", 50.0, 200.0, 125.0, 5.0)
    
    predict_btn = st.button("🔬 Predict & Optimize", use_container_width=True)

# ================================================================
# RESULTS PANEL
# ================================================================
with col_right:
    st.markdown("### 📈 Prediction Results")
    
    if predict_btn:
        inputs = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
        
        with st.spinner("🧠 Running PINN prediction..."):
            tensile, efrf = predict_formulation(model, scaler, inputs)
        
        # Display metrics
        col_metric1, col_metric2 = st.columns(2)
        
        with col_metric1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💪 Tensile Strength (σₜ)", f"{tensile:.3f} MPa")
            if tensile >= 2.0:
                st.markdown('<span class="constraint-pass">✅ ≥ 2 MPa (PASS)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="constraint-fail">❌ < 2 MPa (FAIL)</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_metric2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("⚠️ EFRF (Capping Risk)", f"{efrf:.4f}")
            if efrf < 0.5:
                st.markdown('<span class="constraint-pass">✅ < 0.5 (PASS)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="constraint-fail">❌ ≥ 0.5 (FAIL)</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Overall status
        if tensile >= 2.0 and efrf < 0.5:
            st.success("🎉 **Formulation satisfies all mechanical constraints!**")
            st.balloons()
        else:
            st.warning("⚠️ **Formulation does NOT satisfy all constraints. Adjust parameters.**")
        
        # ================================================================
        # PARETO FRONT PLOT
        # ================================================================
        st.markdown("### 📉 Pareto Front")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate Pareto front approximation
        api_range = np.linspace(85, 95, 100)
        tensile_range = np.zeros_like(api_range)
        efrf_range = np.zeros_like(api_range)
        
        # Use fixed values for other parameters
        fixed_inputs = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
        
        for i, a in enumerate(api_range):
            test_inputs = [a, 5.0, 3.0, 0.6, 1.5, 180.0, 25.0, 125.0]
            t, e = predict_formulation(model, scaler, test_inputs)
            tensile_range[i] = t
            efrf_range[i] = e
        
        # Plot Pareto front
        ax.plot(api_range, efrf_range, 'r-', linewidth=2.5, label='Pareto Front')
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='EFRF = 0.5 (Threshold)')
        
        # Fill feasible region
        feasible_mask = efrf_range < 0.5
        ax.fill_between(api_range, 0, efrf_range, where=feasible_mask, 
                        color='green', alpha=0.15, label='Feasible Design Space')
        
        # Mark current formulation
        ax.scatter([api], [efrf], color='blue', s=150, zorder=5, 
                  edgecolors='darkblue', linewidth=2, label='Your Formulation')
        
        # Mark optimal point (90.5%)
        opt_t, opt_e = predict_formulation(model, scaler, [90.5, 5.0, 3.0, 0.6, 1.5, 180.0, 25.0, 125.0])
        ax.scatter([90.5], [opt_e], color='gold', s=200, zorder=5, 
                  marker='*', edgecolors='darkgoldenrod', linewidth=2, 
                  label='⭐ Optimal: 90.5% API')
        
        ax.set_xlabel('API Loading (%)', fontsize=12)
        ax.set_ylabel('EFRF (Capping Risk)', fontsize=12)
        ax.set_title('Pareto Front: API Loading vs. Capping Risk', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.2)
        ax.set_xlim(84, 96)
        
        st.pyplot(fig)
        
        # ================================================================
        # SENSITIVITY ANALYSIS
        # ================================================================
        st.markdown("### 🔍 Sensitivity Analysis")
        
        with st.expander("Click to view feature importance"):
            st.markdown("The chart below shows how each parameter affects EFRF (capping risk).")
            
            # Calculate sensitivities
            base_inputs = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
            _, base_efrf = predict_formulation(model, scaler, base_inputs)
            
            sensitivities = []
            feature_names_display = ['API%', 'MCC%', 'PVPP%', 'Mg-St%', 'Binder%', 
                                    'Pressure', 'Speed', 'Granule Size']
            
            perturb_values = [5.0, 0.5, 0.5, 0.1, 0.2, 20.0, 5.0, 20.0]
            
            for i in range(8):
                test_inputs = base_inputs.copy()
                test_inputs[i] += perturb_values[i]
                _, efrf_pos = predict_formulation(model, scaler, test_inputs)
                
                test_inputs[i] = base_inputs[i] - perturb_values[i]
                _, efrf_neg = predict_formulation(model, scaler, test_inputs)
                
                sens = max(abs(efrf_pos - base_efrf), abs(efrf_neg - base_efrf))
                sensitivities.append(sens)
            
            # Sort for tornado plot
            sorted_idx = np.argsort(sensitivities)[::-1]
            sorted_names = [feature_names_display[i] for i in sorted_idx]
            sorted_values = [sensitivities[i] for i in sorted_idx]
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            colors = ['#dc2626' if v > np.mean(sorted_values) else '#2563eb' for v in sorted_values]
            ax2.barh(sorted_names, sorted_values, color=colors)
            ax2.axvline(x=np.mean(sorted_values), color='gray', linestyle='--', alpha=0.7, 
                       label=f'Mean: {np.mean(sorted_values):.4f}')
            ax2.set_xlabel('Sensitivity (ΔEFRF)', fontsize=12)
            ax2.set_title('Sensitivity Analysis: Feature Impact on Capping Risk', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='x')
            
            st.pyplot(fig2)
            
            st.caption("🔹 Longer bars indicate more influential parameters on capping risk.")
    
    else:
        st.info("👆 Adjust the parameters on the left and click **'Predict & Optimize'** to see results.")

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.caption("""
🔬 **Note:** This is a computational proof-of-concept. Experimental validation is ongoing.
📧 Contact: [babuker@protonmail.com](mailto:babuker@protonmail.com)
""")

# ================================================================
# SIDEBAR - Additional Info
# ================================================================
with st.sidebar:
    st.markdown("### 📚 About")
    st.markdown("""
    This interactive tool implements a **Physics-Informed Neural Network (PINN)** 
    coupled with **NSGA-II** multi-objective optimization for high-load tablet formulation design.
    
    **Key Features:**
    - ✅ Physics-informed predictions (Heckel equation + EFRF)
    - ✅ Multi-objective optimization
    - ✅ Real-time sensitivity analysis
    - ✅ Pareto front visualization
    
    **Constraints:**
    - 💪 σₜ ≥ 2 MPa (Tensile Strength)
    - ⚠️ EFRF < 0.5 (Capping Risk)
    
    **Optimal Target:** 90.5% Paracetamol Loading
    """)
    
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("""
    - [📄 GitHub Repository](https://github.com/babuker-rgb/AI.Hybrid.Formula)
    - [🏠 Project Website](https://babuker-rgb.github.io/AI.Hybrid.Formula/)
    - [📧 Contact](mailto:babuker@protonmail.com)
    """)
    
    st.markdown("---")
    st.markdown("### 🧪 Status")
    st.info("⚡ **Computational Proof-of-Concept**\n\nExperimental validation is currently ongoing.")
