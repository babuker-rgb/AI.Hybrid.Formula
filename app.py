```python
"""
Hybrid AI Framework - Interactive Web Application
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
from fpdf import FPDF
import datetime
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 1. AI MODEL DEFINITION
# ================================================================

class SimplePINN(nn.Module):
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
# 2. DATA GENERATION
# ================================================================

def generate_data(n_samples=100, random_state=42):
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
# 3. PDF REPORT GENERATION
# ================================================================

def create_pdf_report(api, mcc, pvpp, mgst, binder, pressure, speed, granule, 
                      tensile, efrf, total, status, timestamp):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Formulation Optimization Report", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 6, "Hybrid AI Framework for Tablet Manufacturing", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Date: {timestamp}", ln=True, align="C")
    pdf.ln(8)
    
    # 1. Formulation Summary
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
    
    # 2. Process Parameters
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, "2. Process Parameters", ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    
    params = [
        ("Compaction Pressure", f"{pressure:.1f} MPa", "Affects tablet hardness and density"),
        ("Punch Speed", f"{speed:.1f} rpm", "Influences compression time and elastic recovery"),
        ("Granule Size", f"{granule:.1f} µm", "Impacts flowability and content uniformity"),
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
    
    # 3. Prediction Results
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
    
    # 4. Overall Status
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, "4. Overall Status", ln=True, fill=True)
    
    pdf.set_font("Arial", "B", 14)
    if tensile >= 2.0 and efrf < 0.5:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 8, "PASS - Formulation Satisfies All Constraints", ln=True, align="C")
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 6, "This formulation is recommended for experimental validation and scale-up studies.")
    else:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 8, "FAIL - Formulation Does NOT Satisfy All Constraints", ln=True, align="C")
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 6, "This formulation requires further optimization.")
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    # 5. Recommendations
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, "5. Recommendations", ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    
    if tensile >= 2.0 and efrf < 0.5:
        recommendations = [
            "1. Proceed with experimental validation.",
            "2. Confirm tensile strength and capping resistance.",
            "3. Evaluate disintegration time and dissolution profile.",
            "4. Assess stability under accelerated ICH conditions.",
            "5. Scale-up to pilot batch."
        ]
    else:
        recommendations = [
            "1. Reduce API loading or adjust binder concentration.",
            "2. Optimize lubricant level.",
            "3. Increase compaction pressure.",
            "4. Reduce punch speed.",
            "5. Re-run prediction with adjusted parameters."
        ]
    
    for rec in recommendations:
        pdf.cell(0, 6, rec, ln=True)
    
    pdf.ln(5)
    
    # 6. Contact Information
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, "6. Contact Information", ln=True, fill=True)
    pdf.set_font("Arial", "", 11)
    
    pdf.cell(0, 8, "Chem. Eng. Babuker A. Abdalla", ln=True)
    pdf.cell(0, 7, "Email: babuker@protonmail.com", ln=True)
    pdf.cell(0, 7, "Phone: +249-123-638-638", ln=True)
    pdf.cell(0, 7, "Sudan", ln=True)
    
    pdf.ln(3)
    
    # Footer
    pdf.set_y(270)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 6, "Generated by: Hybrid AI Framework", ln=True, align="C")
    
    pdf_bytes = pdf.output(dest="S")
    
    if isinstance(pdf_bytes, bytearray):
        return bytes(pdf_bytes)
    elif isinstance(pdf_bytes, bytes):
        return pdf_bytes
    else:
        return str(pdf_bytes).encode('latin1')


# ================================================================
# 4. TRAIN MODEL
# ================================================================

@st.cache_resource
def load_model():
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
# 5. PREDICTION FUNCTION
# ================================================================

def predict(model, scaler, inputs):
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
# 6. STREAMLIT UI
# ================================================================

st.set_page_config(
    page_title="Hybrid AI Framework",
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
    st.markdown("**Important:** All components must sum to 100%")
    
    with st.container(border=True):
        # API slider
        api = st.slider("🧪 API Loading (%)", 85.0, 95.0, 90.5, 0.1)
        
        binder = st.slider("🔗 Binder (%)", 0.5, 3.0, 2.7, 0.1)
        pvpp = st.slider("💊 PVPP (%)", 1.0, 5.0, 3.0, 0.1)
        mgst = st.slider("🧴 Mg-St (%)", 0.2, 1.0, 0.2, 0.05)
        
        # Auto-calculate MCC to maintain 100%
        total_others = binder + pvpp + mgst
        remaining = 100 - api - total_others
        max_mcc = min(remaining, 8.0)
        
        if remaining < 0:
            st.error(f"❌ API + Binder + PVPP + Mg-St = {api + total_others:.1f}% > 100%! Please reduce values.")
            mcc = 0.0
        else:
            # MCC is auto-calculated but user can still adjust up to max_mcc
            default_mcc = float(min(max_mcc, 3.6))
            mcc = st.number_input(
                "📦 MCC (%)", 
                min_value=0.0, 
                max_value=float(max_mcc),
                value=default_mcc,
                step=0.1,
                format="%.1f",
                help="Microcrystalline Cellulose - auto-calculated to maintain 100%"
            )
        
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
        total = api + binder + pvpp + mgst + mcc
        if abs(total - 100) > 0.1:
            st.warning("⚠️ **Invalid formulation:** Components must sum to 100%")
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
                st.success("🎉 **Formulation satisfies all constraints!**")
                st.balloons()
            else:
                st.warning("⚠️ **Formulation does NOT satisfy all constraints.**")
            
            # Formulation summary
            st.markdown("### 📋 Formulation Summary")
            summary_data = {
                "Component": ["API", "MCC", "PVPP", "Mg-St", "Binder", "Total"],
                "%": [f"{api:.1f}%", f"{mcc:.1f}%", f"{pvpp:.1f}%", f"{mgst:.2f}%", f"{binder:.1f}%", f"{total:.1f}%"]
            }
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
            
            # ================================================================
            # PARETO FRONT PLOT
            # ================================================================
            st.markdown("### 📉 Pareto Front")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            api_range = np.linspace(85, 95, 50)
            efrf_vals = []
            for a in api_range:
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
            ax.scatter([90.5], [0.2], color='gold', s=200, marker='*', zorder=5, label='Target: 90.5%')
            
            ax.set_xlabel('API Loading (%)')
            ax.set_ylabel('EFRF')
            ax.set_title('Pareto Front')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.0)
            ax.set_xlim(84, 96)
            st.pyplot(fig)
            
            # ================================================================
            # SENSITIVITY ANALYSIS
            # ================================================================
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
                ax2.set_xlabel('Sensitivity')
                ax2.set_title('Feature Impact on EFRF')
                ax2.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig2)
            
            # ================================================================
            # GENERATE PDF REPORT
            # ================================================================
            st.markdown("---")
            st.markdown("### 📄 Generate Report")
            
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
                
                st.caption("PDF includes formulation, results, and contact information.")
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
    
    else:
        st.info("👆 Adjust parameters and click **'Predict & Optimize'**")

# Footer
st.markdown("---")
st.caption("🔬 **Computational proof-of-concept**")
st.caption("📧 Contact: [babuker@protonmail.com](mailto:babuker@protonmail.com)")

# Sidebar
with st.sidebar:
    st.markdown("### 📚 About")
    st.markdown("""
    Formulation optimization tool using AI technology.
    
    **Important:** All components must sum to **100%**.
    
    **Targets:**
    - 💪 Tensile Strength ≥ 2 MPa
    - ⚠️ EFRF < 0.5
    
    **Target API:** 90.5% Paracetamol
    """)
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[📄 GitHub](https://github.com/babuker-rgb/AI.Hybrid.Formula)")
    st.markdown("[🏠 Website](https://babuker-rgb.github.io/AI.Hybrid.Formula/)")
    st.markdown("---")
    st.info("⚡ **Proof-of-Concept**")
```
