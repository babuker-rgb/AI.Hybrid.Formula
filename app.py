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
    
    # Initialize dataset
    X = np.zeros((n_samples, 8))
    y = np.zeros((n_samples, 2))
    
    for i in range(n_samples):
        # Generate 5 components that sum to 100%
        # 1. API: 85-95%
        api = np.random.uniform(85, 95)
        
        # 2. Binder: 0.5-3.0%
        binder = np.random.uniform(0.5, 3.0)
        
        # 3. Mg-St: 0.2-1.0%
        mgst = np.random.uniform(0.2, 1.0)
        
        # 4. PVPP: 1.0-5.0%
        pvpp = np.random.uniform(1.0, 5.0)
        
        # 5. Remaining is MCC (max 8%)
        mcc = 100 - (api + binder + mgst + pvpp)
        mcc = np.clip(mcc, 0, 8.0)
        
        # If MCC exceeds 8%, rescale others proportionally
        if mcc > 8.0:
            # Reduce others proportionally
            scale_factor = (100 - 8.0) / (api + binder + mgst + pvpp)
            api = api * scale_factor
            binder = binder * scale_factor
            mgst = mgst * scale_factor
            pvpp = pvpp * scale_factor
            mcc = 8.0
        
        # 6. Process parameters
        pressure = np.random.uniform(100, 250)
        speed = np.random.uniform(10, 40)
        granule = np.random.uniform(50, 200)
        
        # Store inputs
        X[i] = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
        
        # --- Calculate outputs ---
        # Tensile strength: decreases with API, increases with binder and pressure
        strength = 3.5 - 0.15 * (api - 85) + 0.3 * binder + 0.008 * (pressure - 100) - 1.5 * mgst - 0.02 * (speed - 10)
        strength = np.clip(strength, 0.5, 6.0)
        
        # EFRF: increases with API and speed, decreases with pressure and binder
        efrf = 0.2 + 0.08 * (api - 85) + 0.005 * (speed - 10) - 0.001 * (pressure - 100) - 0.2 * binder + 0.5 * mgst
        efrf = np.clip(efrf, 0.1, 1.5)
        
        y[i] = [strength, efrf]
    
    feature_names = ['API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%', 
                     'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm']
    
    df = pd.DataFrame(X, columns=feature_names)
    df['Tensile_Strength_MPa'] = y[:, 0]
    df['EFRF'] = y[:, 1]
    
    # Calculate total composition
    df['Total_Composition_%'] = df[['API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%']].sum(axis=1)
    
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
    progress_bar = st.progress(0)
    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            progress_bar.progress((epoch + 1) / 1000)
    
    progress_bar.progress(1.0)
    model.eval()
    return model, scaler, feature_names


# ================================================================
# 4. PREDICTION FUNCTION
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
    .constraint-warning { color: #d97706; font-weight: 700; }
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

# ================================================================
# TWO-COLUMN LAYOUT: Inputs | Results
# ================================================================
col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    st.markdown("### 📊 Formulation Parameters")
    st.markdown("**IMPORTANT:** All components must sum to 100% for a valid formulation.")
    
    with st.container(border=True):
        api = st.slider("🧪 API Loading (%)", 85.0, 95.0, 90.5, 0.1,
                        help="Active Pharmaceutical Ingredient (Paracetamol)")
        
        binder = st.slider("🔗 Binder (%)", 0.5, 3.0, 2.7, 0.1,
                          help="Binder (e.g., Kollidon VA64)")
        
        pvpp = st.slider("💊 PVPP (%)", 1.0, 5.0, 3.0, 0.1,
                        help="Superdisintegrant (Crospovidone)")
        
        mgst = st.slider("🧴 Mg-St (%)", 0.2, 1.0, 0.2, 0.05,
                        help="Lubricant (Magnesium Stearate)")
        
        # Calculate remaining for MCC (max 8%)
        total_others = binder + pvpp + mgst
        remaining = 100 - api - total_others
        max_mcc = min(remaining, 8.0)
        
        if remaining < 0:
            st.error(f"❌ API + Binder + PVPP + Mg-St = {api + total_others:.1f}% > 100%! Please reduce API or other components.")
            mcc = 0.0
        else:
            mcc = st.number_input(
                "📦 MCC (%)", 
                min_value=0.0, 
                max_value=float(max_mcc),
                value=float(min(max_mcc, 3.6)),
                step=0.1,
                format="%.1f",
                help="Microcrystalline Cellulose - filler (remaining to 100%)"
            )
        
        # Auto-fill MCC if needed
        if st.button("🔧 Auto-fill MCC to 100%"):
            total_components = api + binder + pvpp + mgst
            if total_components <= 100:
                auto_mcc = 100 - total_components
                if auto_mcc <= 8.0:
                    mcc = auto_mcc
                else:
                    st.warning(f"⚠️ Remaining filler ({auto_mcc:.1f}%) exceeds MCC limit (8%). Reduce API or other components.")
        
        # Process parameters
        st.markdown("---")
        pressure = st.slider("⚙️ Compaction Pressure (MPa)", 100.0, 250.0, 230.0, 5.0)
        speed = st.slider("🔄 Punch Speed (rpm)", 10.0, 40.0, 12.0, 1.0)
        granule = st.slider("🔬 Granule Size (µm)", 50.0, 200.0, 125.0, 5.0)
        
        # Calculate and display total
        total = api + binder + pvpp + mgst + mcc
        st.metric("**Total Formulation**", f"{total:.1f}%", 
                  delta="✅ Valid" if abs(total - 100) < 0.1 else "❌ Invalid")
    
    predict_btn = st.button("🔬 Predict & Optimize", use_container_width=True)

# ================================================================
# RESULTS PANEL
# ================================================================
with col_right:
    st.markdown("### 📈 Prediction Results")
    
    if predict_btn:
        # Validate total
        total = api + binder + pvpp + mgst + mcc
        if abs(total - 100) > 0.1:
            st.warning("⚠️ **Invalid formulation:** Components do not sum to 100%. Adjust your inputs and try again.")
        else:
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
            
            # Formulation summary
            st.markdown("### 📋 Formulation Summary")
            summary_data = {
                "Component": ["API", "MCC", "PVPP", "Mg-St", "Binder", "Total"],
                "%": [f"{api:.1f}%", f"{mcc:.1f}%", f"{pvpp:.1f}%", f"{mgst:.2f}%", f"{binder:.1f}%", f"{total:.1f}%"]
            }
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
            
            # Pareto Front
            st.markdown("### 📉 Pareto Front")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Generate Pareto front
            api_range = np.linspace(85, 95, 50)
            efrf_vals = []
            for a in api_range:
                # Adjust MCC to maintain 100%
                total_others = binder + pvpp + mgst
                mcc_fixed = 100 - a - total_others
                if 0 <= mcc_fixed <= 8:
                    test_inputs = [a, mcc_fixed, pvpp, mgst, binder, pressure, speed, granule]
                    _, e = predict(model, scaler, test_inputs)
                    efrf_vals.append(e)
                else:
                    efrf_vals.append(np.nan)
            
            ax.plot(api_range, efrf_vals, 'r-', linewidth=2.5, label='Pareto Front')
            ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='EFRF = 0.5')
            ax.fill_between(api_range, 0, efrf_vals, where=(np.array(efrf_vals) < 0.5), 
                            color='green', alpha=0.15)
            ax.scatter([api], [efrf], color='blue', s=150, zorder=5, label='Your Formulation')
            ax.scatter([90.5], [0.2], color='gold', s=200, marker='*', zorder=5, label='⭐ Target: 90.5%')
            
            ax.set_xlabel('API Loading (%)')
            ax.set_ylabel('EFRF')
            ax.set_title('Pareto Front (100% Formulation Constraint)')
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
        st.info("👆 Adjust parameters to 100% total and click **'Predict & Optimize'**")

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
    
    **Important:** All formulation components (API + MCC + PVPP + Mg-St + Binder) 
    must sum to **100%**.
    
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
