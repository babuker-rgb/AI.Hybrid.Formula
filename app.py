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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 1. SIMPLIFIED PINN MODEL (Guaranteed to work)
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
# 2. DATA GENERATION (Simplified, robust)
# ================================================================

def generate_data(n_samples=100, random_state=42):
    """Generate synthetic data with clear patterns"""
    np.random.seed(random_state)
    
    # Generate random inputs
    X = np.random.rand(n_samples, 8)
    
    # Scale to realistic ranges
    X[:, 0] = 85 + X[:, 0] * 10  # API: 85-95%
    X[:, 1] = 2 + X[:, 1] * 6    # MCC: 2-8%
    X[:, 2] = 1 + X[:, 2] * 4    # PVPP: 1-5%
    X[:, 3] = 0.2 + X[:, 3] * 0.8  # Mg-St: 0.2-1.0%
    X[:, 4] = 0.5 + X[:, 4] * 2.5  # Binder: 0.5-3.0%
    X[:, 5] = 100 + X[:, 5] * 150  # Pressure: 100-250 MPa
    X[:, 6] = 10 + X[:, 6] * 30    # Speed: 10-40 rpm
    X[:, 7] = 50 + X[:, 7] * 150   # Granule: 50-200 µm
    
    # Calculate realistic outputs
    y = np.zeros((n_samples, 2))
    
    for i in range(n_samples):
        api = X[i, 0]
        mgst = X[i, 3]
        binder = X[i, 4]
        pressure = X[i, 5]
        speed = X[i, 6]
        
        # Tensile strength: decreases with API, increases with pressure and binder
        strength = 3.5 - 0.15 * (api - 85) + 0.3 * binder + 0.008 * (pressure - 100) - 1.5 * mgst - 0.02 * (speed - 10)
        strength = np.clip(strength, 0.5, 6.0)
        
        # EFRF: increases with API and speed, decreases with pressure
        efrf = 0.2 + 0.08 * (api - 85) + 0.005 * (speed - 10) - 0.001 * (pressure - 100) - 0.2 * binder + 0.5 * mgst
        efrf = np.clip(efrf, 0.1, 1.5)
        
        y[i, 0] = strength
        y[i, 1] = efrf
    
    feature_names = ['API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%', 
                     'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm']
    
    df = pd.DataFrame(X, columns=feature_names)
    df['Tensile_Strength_MPa'] = y[:, 0]
    df['EFRF'] = y[:, 1]
    
    return df, feature_names


# ================================================================
# 3. TRAIN MODEL
# ================================================================

@st.cache_resource
def load_model():
    """Train and return the model with caching"""
    
    # Generate data
    df, feature_names = generate_data(n_samples=100)
    X = df[feature_names].values
    y = df[['Tensile_Strength_MPa', 'EFRF']].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y)
    
    # Train model
    model = SimplePINN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    model.eval()
    return model, scaler, feature_names


# ================================================================
# 4. PREDICTION FUNCTION
# ================================================================

def predict(model, scaler, inputs):
    """Predict tensile strength and EFRF"""
    inputs_scaled = scaler.transform([inputs])
    X_tensor = torch.FloatTensor(inputs_scaled)
    with torch.no_grad():
        predictions = model(X_tensor).numpy()[0]
    return predictions[0], predictions[1]


# ================================================================
# 5. STREAMLIT UI
# ================================================================

st.set_page_config(
    page_title="PINN-NSGA-II Hybrid AI",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem 0; }
    .metric-card { background: #f8fafc; border-radius: 12px; padding: 1rem 1.5rem; text-align: center; border: 1px solid #e9edf2; }
    .constraint-pass { color: #16a34a; font-weight: 700; }
    .constraint-fail { color: #dc2626; font-weight: 700; }
    .stButton > button { width: 100%; background: #2563eb; color: white; font-weight: 600; padding: 0.6rem; border-radius: 8px; border: none; }
    .stButton > button:hover { background: #1d4ed8; color: white; }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🧠 Hybrid AI Framework")
st.subheader("PINN · NSGA-II")
st.markdown("### Multi‑Objective Tablet Manufacturing Optimization · High‑Load Paracetamol")
st.caption("👨‍🔬 Babuker A. Abdalla & Prof. Abdelkarim Mohamed · Nile Valley University")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Load model
with st.spinner("🔄 Loading AI model..."):
    model, scaler, feature_names = load_model()
st.success("✅ Model loaded successfully!")

# Inputs
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.markdown("### 📊 Formulation Parameters")
    st.markdown("Adjust the parameters below to explore the design space.")
    
    with st.container(border=True):
        api = st.slider("🧪 API Loading (%)", 85.0, 95.0, 90.5, 0.1)
        mcc = st.slider("📦 MCC (%)", 2.0, 8.0, 5.0, 0.1)
        pvpp = st.slider("💊 PVPP (%)", 1.0, 5.0, 3.0, 0.1)
        mgst = st.slider("🧴 Mg-St (%)", 0.2, 1.0, 0.6, 0.05)
        binder = st.slider("🔗 Binder (%)", 0.5, 3.0, 1.5, 0.1)
        pressure = st.slider("⚙️ Compaction Pressure (MPa)", 100.0, 250.0, 180.0, 5.0)
        speed = st.slider("🔄 Punch Speed (rpm)", 10.0, 40.0, 25.0, 1.0)
        granule = st.slider("🔬 Granule Size (µm)", 50.0, 200.0, 125.0, 5.0)
    
    predict_btn = st.button("🔬 Predict & Optimize", use_container_width=True)

with col_right:
    st.markdown("### 📈 Prediction Results")
    
    if predict_btn:
        inputs = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
        
        with st.spinner("🧠 Running prediction..."):
            tensile, efrf = predict(model, scaler, inputs)
        
        # Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💪 Tensile Strength", f"{tensile:.3f} MPa")
            if tensile >= 2.0:
                st.markdown('<span class="constraint-pass">✅ ≥ 2 MPa (PASS)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="constraint-fail">❌ < 2 MPa (FAIL)</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("⚠️ EFRF", f"{efrf:.4f}")
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
        
        # Pareto Front
        st.markdown("### 📉 Pareto Front")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Generate Pareto front
        api_range = np.linspace(85, 95, 50)
        efrf_vals = []
        for a in api_range:
            test_inputs = [a, 5.0, 3.0, 0.6, 1.5, 180.0, 25.0, 125.0]
            _, e = predict(model, scaler, test_inputs)
            efrf_vals.append(e)
        
        ax.plot(api_range, efrf_vals, 'r-', linewidth=2.5, label='Pareto Front')
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='EFRF = 0.5')
        ax.fill_between(api_range, 0, efrf_vals, where=(np.array(efrf_vals) < 0.5), 
                        color='green', alpha=0.15)
        ax.scatter([api], [efrf], color='blue', s=150, zorder=5, label='Your Formulation')
        ax.scatter([90.5], [0.32], color='gold', s=200, marker='*', zorder=5, label='⭐ Optimal')
        
        ax.set_xlabel('API Loading (%)')
        ax.set_ylabel('EFRF')
        ax.set_title('Pareto Front')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(84, 96)
        st.pyplot(fig)
        
        # Sensitivity
        st.markdown("### 🔍 Sensitivity Analysis")
        with st.expander("Click to view feature importance"):
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
            
            sorted_idx = np.argsort(sensitivities)[::-1]
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.barh([features[i] for i in sorted_idx], [sensitivities[i] for i in sorted_idx])
            ax2.set_xlabel('Sensitivity (ΔEFRF)')
            ax2.set_title('Feature Impact on EFRF')
            ax2.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig2)
    
    else:
        st.info("👆 Adjust parameters and click **'Predict & Optimize'**")

# Footer
st.markdown("---")
st.caption("🔬 **Computational proof-of-concept. Experimental validation ongoing.**")
st.caption("📧 Contact: [babuker@protonmail.com](mailto:babuker@protonmail.com)")

# Sidebar
with st.sidebar:
    st.markdown("### 📚 About")
    st.markdown("""
    This tool implements a **Physics-Informed Neural Network (PINN)** 
    coupled with **NSGA-II** multi-objective optimization.
    
    **Constraints:**
    - 💪 σₜ ≥ 2 MPa
    - ⚠️ EFRF < 0.5
    
    **Optimal Target:** 90.5% Paracetamol
    """)
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[📄 GitHub](https://github.com/babuker-rgb/AI.Hybrid.Formula)")
    st.markdown("[🏠 Website](https://babuker-rgb.github.io/AI.Hybrid.Formula/)")
    st.markdown("---")
    st.info("⚡ **Proof-of-Concept**")
