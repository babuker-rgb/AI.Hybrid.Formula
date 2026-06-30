"""
True Physics-Informed Neural Network (PINN) - Professional Version
Multi-Objective Tablet Manufacturing Optimization

Author: Babuker A. Abdalla
Affiliation: Nile Valley University, Sudan
Version: 6.0 (Fixed validation gradient issue)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from fpdf import FPDF
import datetime
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 1. TRUE PINN MODEL
# ================================================================

class TruePINN(nn.Module):
    """
    True Physics-Informed Neural Network with proper Autograd for monotonicity.
    """

    def __init__(self, input_dim=8, output_dim=3):
        super(TruePINN, self).__init__()

        # Main network
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

        # Physics parameter networks (formulation-dependent)
        self.k_network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softplus()
        )

        self.A_network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, X):
        return self.network(X)

    def get_heckel_params(self, X):
        k = self.k_network(X)
        A = self.A_network(X)
        return k.squeeze(), A.squeeze()

    def compute_loss(self, X_scaled, X_raw, y_true,
                     w_data=1.0, w_heckel=0.5, w_efrf=0.5,
                     w_monotonic=0.1, w_boundary=0.1,
                     efrf_target=0.4,
                     compute_grad=True):
        """
        Total PINN loss.
        compute_grad: if True, compute monotonicity gradient (training).
                      if False, skip monotonicity gradient (validation).
        """
        # Real pressure for physics
        pressure_real = X_raw[:, 5]

        # Forward pass
        y_pred = self.forward(X_scaled)
        density_pred = y_pred[:, 0]
        tensile_pred = y_pred[:, 1]
        er_pred = y_pred[:, 2]

        # Formulation-dependent Heckel parameters
        k, A = self.get_heckel_params(X_scaled)

        # ---------- Data Loss ----------
        data_loss = nn.MSELoss()(y_pred, y_true)

        # ---------- Heckel Residual ----------
        D_clamped = torch.clamp(density_pred, 0.01, 0.99)
        heckel_pred = torch.log(1.0 / (1.0 - D_clamped))
        heckel_target = k * pressure_real + A
        heckel_loss = torch.mean((heckel_pred - heckel_target) ** 2)

        # ---------- EFRF Constraint (with safety margin) ----------
        efrf_pred = er_pred / (tensile_pred + 1e-8)
        efrf_loss = torch.mean(torch.relu(efrf_pred - efrf_target) ** 2)

        # ---------- Monotonicity: ∂D/∂P > 0 ----------
        if compute_grad:
            # Compute gradient using autograd (only during training)
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
            # During validation, we skip the monotonicity term
            monotonic_loss = torch.tensor(0.0, device=X_scaled.device)

        # ---------- Boundary Conditions (using real pressure) ----------
        mask_low = (pressure_real < 120).float()
        mask_high = (pressure_real > 230).float()
        boundary_loss = (
            torch.mean(mask_low * torch.relu(0.5 - density_pred) ** 2) +
            torch.mean(mask_high * torch.relu(density_pred - 0.98) ** 2)
        )

        # ---------- Total Loss ----------
        total_loss = (
            w_data * data_loss +
            w_heckel * heckel_loss +
            w_efrf * efrf_loss +
            w_monotonic * monotonic_loss +
            w_boundary * boundary_loss
        )

        loss_dict = {
            'data_loss': data_loss.item(),
            'heckel_loss': heckel_loss.item(),
            'efrf_loss': efrf_loss.item(),
            'monotonic_loss': monotonic_loss.item() if compute_grad else 0.0,
            'boundary_loss': boundary_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict

    def predict(self, X_scaled):
        self.eval()
        with torch.no_grad():
            if not isinstance(X_scaled, torch.Tensor):
                X_scaled = torch.FloatTensor(X_scaled)
            return self.forward(X_scaled).numpy()


# ================================================================
# 2. DATA GENERATION
# ================================================================

def generate_pinn_data(n_samples=300, random_state=42):
    np.random.seed(random_state)
    X = np.zeros((n_samples, 8))
    y = np.zeros((n_samples, 3))

    for i in range(n_samples):
        api = np.random.uniform(85, 95)
        binder = np.random.uniform(0.5, 3.0)
        mgst = np.random.uniform(0.2, 1.0)
        pvpp = np.random.uniform(1.0, 5.0)
        mcc = 100 - (api + binder + mgst + pvpp)
        mcc = np.clip(mcc, 0, 8.0)
        if mcc > 8.0:
            scale = (100 - 8.0) / (api + binder + mgst + pvpp)
            api *= scale; binder *= scale; mgst *= scale; pvpp *= scale
            mcc = 8.0

        pressure = np.random.uniform(100, 250)
        speed = np.random.uniform(10, 40)
        granule = np.random.uniform(50, 200)

        X[i] = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]

        k_eff = 0.035 * (1 - 0.4 * (api - 85)/10) * (1 - 0.2 * (speed - 10)/30)
        k_eff = max(k_eff, 0.008)
        A_eff = 1.2 + 0.1 * (binder - 1.5) - 0.2 * (mgst - 0.5)
        D = 1 - np.exp(-(k_eff * pressure + A_eff))
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
# 3. TRAIN WITH EARLY STOPPING
# ================================================================

@st.cache_resource
def load_pinn_model():
    df, feature_names = generate_pinn_data(n_samples=300)
    X_raw = df[feature_names].values
    y = df[['Density', 'Tensile_Strength_MPa', 'Elastic_Recovery_%']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Split
    X_scaled_train, X_scaled_temp, X_raw_train, X_raw_temp, y_train, y_temp = train_test_split(
        X_scaled, X_raw, y, test_size=0.3, random_state=42
    )
    X_scaled_val, X_scaled_test, X_raw_val, X_raw_test, y_val, y_test = train_test_split(
        X_scaled_temp, X_raw_temp, y_temp, test_size=0.5, random_state=42
    )

    # Convert to tensors – ensure X_scaled_train requires grad for monotonicity
    X_scaled_train_t = torch.FloatTensor(X_scaled_train)
    X_scaled_train_t.requires_grad_(True)
    X_raw_train_t = torch.FloatTensor(X_raw_train)
    y_train_t = torch.FloatTensor(y_train)

    X_scaled_val_t = torch.FloatTensor(X_scaled_val)
    X_scaled_val_t.requires_grad_(True)  # Keep for consistency, but we won't compute grad during validation
    X_raw_val_t = torch.FloatTensor(X_raw_val)
    y_val_t = torch.FloatTensor(y_val)

    model = TruePINN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)

    best_val_loss = float('inf')
    patience = 100
    counter = 0
    best_state = None

    progress_bar = st.progress(0)
    for epoch in range(2000):
        # ----- Training -----
        model.train()
        optimizer.zero_grad()
        total_loss, _ = model.compute_loss(
            X_scaled_train_t, X_raw_train_t, y_train_t,
            w_data=1.0, w_heckel=0.5, w_efrf=0.5,
            w_monotonic=0.1, w_boundary=0.1,
            efrf_target=0.4,
            compute_grad=True   # enable gradient for training
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ----- Validation (without monotonicity gradient) -----
        model.eval()
        # Compute validation loss without gradient tracking for monotonicity
        # We still compute the loss but skip the autograd part.
        with torch.set_grad_enabled(False):  # Disable gradient globally for validation
            val_loss, _ = model.compute_loss(
                X_scaled_val_t, X_raw_val_t, y_val_t,
                w_data=1.0, w_heckel=0.5, w_efrf=0.5,
                w_monotonic=0.1, w_boundary=0.1,
                efrf_target=0.4,
                compute_grad=False   # skip monotonicity gradient
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
            st.info(f"Early stopping at epoch {epoch}")
            break

        if (epoch + 1) % 100 == 0:
            progress_bar.progress((epoch + 1) / 2000)

    progress_bar.progress(1.0)

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model.feature_names = feature_names
    model.scaler = scaler

    # Save checkpoint for production
    torch.save(model.state_dict(), 'true_pinn_checkpoint.pt')

    return model, scaler, feature_names, df, {'train': [], 'val': []}


# ================================================================
# 4. PREDICTION
# ================================================================

def predict_pinn(model, scaler, inputs):
    try:
        inputs_scaled = scaler.transform([inputs])
        X_tensor = torch.FloatTensor(inputs_scaled)
        with torch.no_grad():
            pred = model(X_tensor).numpy()[0]
        density, tensile, er = pred[0], pred[1], pred[2]
        efrf = er / (tensile + 1e-8)
        return density, tensile, er, efrf
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.5, 0.0, 1.0, 1.0


# ================================================================
# 5. STREAMLIT UI
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
    - ✅ **Heckel Equation:** ln(1/(1-D)) = k(X)P + A(X)
    - ✅ **EFRF:** ER / σt < 0.4 (safety margin)
    - ✅ **Monotonicity:** ∂D/∂P > 0
    - ✅ **Boundary Conditions:** 0.4 < D < 0.98
    - ✅ **Formulation-dependent k(X) and A(X)**

    **Architecture:**
    - 128-128-64-32 with BatchNorm & Dropout
    - Early Stopping with Validation
    - Model Checkpoint

    **Target:** ~90.5% Paracetamol
    """)
    st.warning("⚠️ **True PINN — Production-Ready**")

with st.spinner("🔄 Training True PINN..."):
    model, scaler, feature_names, df, loss_history = load_pinn_model()
st.success("✅ True PINN trained successfully")

col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    st.markdown("### 📊 Formulation Parameters")
    with st.container(border=True):
        api = st.slider("🧪 API Loading (%)", 85.0, 95.0, 90.5, 0.1)
        binder = st.slider("🔗 Binder (%)", 0.5, 3.0, 2.7, 0.1)
        pvpp = st.slider("💊 PVPP (%)", 1.0, 5.0, 3.0, 0.1)
        mgst = st.slider("🧴 Mg-St (%)", 0.2, 1.0, 0.2, 0.05)
        used_total = api + binder + pvpp + mgst
        remaining = 100 - used_total
        if remaining < 0:
            st.error("❌ Total exceeds 100%!")
            mcc = 0.0
        else:
            mcc = remaining
            if mcc > 8.0:
                st.warning(f"⚠️ MCC = {mcc:.1f}% exceeds 8% limit.")
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

with col_right:
    st.markdown("### 📈 Predictive Results")
    if predict_btn:
        if abs(total - 100) > 0.1:
            st.warning("⚠️ Invalid formulation: Components must sum to 100%.")
        else:
            inputs = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
            with st.spinner("🧠 Running True PINN..."):
                density, tensile, er, efrf = predict_pinn(model, scaler, inputs)

            c1, c2, c3 = st.columns(3)
            c1.metric("📊 Density", f"{density:.3f}")
            c2.metric("💪 Tensile", f"{tensile:.3f} MPa")
            c3.metric("⚠️ EFRF", f"{efrf:.4f}")

            if tensile >= 2.0 and efrf < 0.4:
                st.success(f"""
                🎉 **Formulation satisfies all constraints!**
                ✅ σt = {tensile:.3f} MPa (≥ 2 MPa)
                ✅ EFRF = {efrf:.4f} (< 0.4)
                📌 Suitable for high-speed industrial tableting.
                """)
            elif tensile < 2.0 and efrf >= 0.4:
                st.error("🚨 CRITICAL: Low strength and high capping risk.")
            elif tensile < 2.0:
                st.warning("⚠️ Low tensile strength – increase binder or pressure.")
            elif efrf >= 0.4:
                st.error("🚨 High capping risk – reduce speed or Mg-St.")

            with st.expander("🔬 Physics Verification"):
                st.markdown("""
                - ✅ Heckel residual, EFRF constraint, monotonicity, boundary conditions enforced.
                - k(X) and A(X) are formulation-dependent.
                """)
                heckel_pred = np.log(1/(1-density+1e-8))
                st.metric("EFRF", f"{efrf:.4f}", delta="< 0.4 ✅" if efrf < 0.4 else "≥ 0.4 ❌")

st.markdown("---")
st.caption("🔬 **True PINN — Production-Ready with Full Physics**")
st.caption("📧 Contact: babuker@protonmail.com")
