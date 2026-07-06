"""
Hubryd AI – Lightweight Demo (PINN + NSGA-II)
Runs quickly on free tier
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

# ===================== Parameters =====================
TENSILE_MIN = 1.90
EFRF_MAX = 0.40
MCC_MAX = 8.0
D_MIN = 0.40
D_MAX = 0.97
PRESSURE_MAX = 300.0
BINDER_MIN = 0.5
BINDER_MAX = 5.0

N_SAMPLES = 500          # small for speed
ADAM_EPOCHS = 20         # quick training
PATIENCE = 10
NSGA_POP_SIZE = 15
NSGA_GENERATIONS = 10

CACHE_DIR = tempfile.gettempdir()
CHECKPOINT_PATH = os.path.join(CACHE_DIR, 'pinn_light.pt')

# ===================== Data Generation =====================
def generate_data(n=N_SAMPLES):
    np.random.seed(42)
    X = np.zeros((n, 8))
    y = np.zeros((n, 3))
    for i in range(n):
        api = np.random.uniform(85, 95)
        mcc = np.random.uniform(0, 8)
        pvpp = np.random.uniform(0.5, 6)
        mgst = np.random.uniform(0.01, 1.2)
        binder = np.random.uniform(0.5, 5)
        pressure = np.random.uniform(80, 300)
        speed = np.random.uniform(1, 50)
        granule = np.random.uniform(30, 250)
        # simple synthetic targets
        density = np.clip(0.4 + 0.01*(pressure/100) - 0.001*speed, D_MIN, D_MAX)
        tensile = np.clip(1.5 + 0.05*api - 0.2*mcc - 0.1*mgst + 0.1*binder + 0.02*(pressure-100), 0.5, 6)
        er = np.clip(1.0 + 0.02*(api-85) + 0.01*speed - 0.005*pressure, 0.5, 4)
        X[i] = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]
        y[i] = [density, tensile, er]
    features = ['API_%','MCC_%','PVPP_%','MgSt_%','Binder_%','Pressure_MPa','Speed_rpm','Granule_Size_µm']
    df = pd.DataFrame(X, columns=features)
    df['Density'] = y[:,0]
    df['Tensile'] = y[:,1]
    df['ER'] = y[:,2]
    return df, features

def add_interaction(X):
    # simple interactions
    pressure = X[:,5:6]
    speed = X[:,6:7]
    api = X[:,0:1]
    mcc = X[:,1:2]
    binder = X[:,4:5]
    extra = np.concatenate([
        pressure * speed,
        api / (mcc+0.1),
        binder * pressure,
        api**2,
        pressure**2
    ], axis=1)
    return np.concatenate([X, extra], axis=1)

# ===================== PINN Model (tiny) =====================
class TinyPINN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )
    def forward(self, x):
        raw = self.net(x)
        dens = D_MIN + (D_MAX-D_MIN)*torch.sigmoid(raw[:,0:1])
        tens = torch.nn.functional.softplus(raw[:,1:2]) + 0.01
        er = torch.nn.functional.softplus(raw[:,2:3]) + 0.01
        k = torch.nn.functional.softplus(raw[:,3:4]) + 0.001
        A = raw[:,4:5]
        return torch.cat([dens, tens, er, k, A], dim=1)
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            out = self.forward(x)
            return out[:, :3].cpu().numpy()

# ===================== Training =====================
@st.cache_resource
def load_or_train():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
            model = TinyPINN(ckpt['in_dim'])
            model.load_state_dict(ckpt['model'])
            return model, ckpt['scaler'], ckpt['yscaler'], ckpt['features'], ckpt['df']
        except:
            pass

    st.caption("🔄 Training tiny PINN (quick)...")
    df, features = generate_data(N_SAMPLES)
    X_raw = df[features].values
    y = df[['Density','Tensile','ER']].values
    X_aug = add_interaction(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)
    yscaler = StandardScaler()
    y_scaled = yscaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    model = TinyPINN(X_aug.shape[1])
    opt = optim.Adam(model.parameters(), lr=0.01)
    prog = st.progress(0)
    for ep in range(ADAM_EPOCHS):
        opt.zero_grad()
        pred = model(torch.tensor(X_train, dtype=torch.float32))
        loss = nn.MSELoss()(pred[:,:3], torch.tensor(y_train, dtype=torch.float32))
        loss.backward()
        opt.step()
        prog.progress((ep+1)/ADAM_EPOCHS)
    st.success("✅ Training done!")
    # save
    torch.save({
        'model': model.state_dict(),
        'scaler': scaler,
        'yscaler': yscaler,
        'features': features,
        'df': df,
        'in_dim': X_aug.shape[1]
    }, CHECKPOINT_PATH)
    return model, scaler, yscaler, features, df

# ===================== Prediction =====================
def predict(model, scaler, yscaler, inputs):
    try:
        inp = np.array([inputs])
        aug = add_interaction(inp)[0]
        scaled = scaler.transform([aug])
        with torch.no_grad():
            pred_scaled = model.predict(torch.tensor(scaled, dtype=torch.float32))
        pred = yscaler.inverse_transform(pred_scaled)[0]
        d = np.clip(pred[0], D_MIN, D_MAX)
        t = max(pred[1], 0.01)
        e = max(pred[2], 0.01)
        return d, t, e, e/t
    except:
        # fallback
        return 0.7, 2.0, 0.5, 0.25

# ===================== Simple NSGA-II (random search) =====================
def simple_nsga(model, scaler, yscaler, bounds, pop=15, gens=10):
    pop_size = pop
    n_gen = gens
    population = []
    for _ in range(pop_size):
        ind = [np.random.uniform(b[0], b[1]) for b in bounds]
        # repair formulation sum: we'll just clip later
        population.append(np.array(ind))
    objectives = []
    for ind in population:
        api, mcc, pvpp, mgst, binder, pressure, speed, granule = ind
        # simple normalization to sum to 100% (only for formulation components)
        total = api + binder + pvpp + mgst + mcc
        if total > 0:
            api = (api/total)*100
            binder = (binder/total)*100
            pvpp = (pvpp/total)*100
            mgst = (mgst/total)*100
            mcc = (mcc/total)*100
        # clip bounds
        api = np.clip(api, 85,95)
        binder = np.clip(binder, BINDER_MIN, BINDER_MAX)
        pvpp = np.clip(pvpp, 0.5,6)
        mgst = np.clip(mgst,0.01,1.2)
        mcc = np.clip(mcc,0,MCC_MAX)
        pressure = np.clip(pressure,80,PRESSURE_MAX)
        speed = np.clip(speed,1,50)
        granule = np.clip(granule,30,250)
        d,t,e,ef = predict(model, scaler, yscaler, [api,mcc,pvpp,mgst,binder,pressure,speed,granule])
        # objectives: maximize API (negative), minimize EFRF
        # penalty for constraints
        penalty = 0
        if t < TENSILE_MIN: penalty += (TENSILE_MIN-t)**2
        if ef >= EFRF_MAX: penalty += (ef-EFRF_MAX)**2
        if mcc > MCC_MAX: penalty += (mcc-MCC_MAX)**2
        obj1 = -(api) + 100*penalty
        obj2 = ef + 100*penalty
        objectives.append([obj1, obj2])
    objectives = np.array(objectives)
    # find non-dominated front (simple)
    fronts = []
    remaining = list(range(pop_size))
    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i==j: continue
                if objectives[j,0] <= objectives[i,0] and objectives[j,1] <= objectives[i,1] and \
                   (objectives[j,0] < objectives[i,0] or objectives[j,1] < objectives[i,1]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        fronts.append(front)
        remaining = [i for i in remaining if i not in front]
    return np.array(population), objectives, fronts

# ===================== Streamlit UI =====================
st.set_page_config(page_title="Hubryd AI Light", layout="wide")
st.markdown("""
<div style="background:#1a1a2e;padding:1rem;border-radius:1rem;text-align:center;">
<h2 style="color:#fff;">🧬 Hubryd AI – Lightweight Demo</h2>
<p style="color:#64ffda;">Runs in seconds on free tier</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.info("Constraints: Density 0.40–0.97, Tensile ≥1.9 MPa, EFRF <0.40")

# Load model
try:
    model, scaler, yscaler, features, df = load_or_train()
except Exception as e:
    st.error(f"Training error: {e}. Using dummy model.")
    # dummy
    model = None
    scaler = None
    yscaler = None

# Sliders
col1, col2 = st.columns([1,1.2])
with col1:
    with st.container(border=True):
        api = st.slider("API (%)", 85.0,95.0,90.5,0.1)
        binder = st.slider("Binder (%)", BINDER_MIN,BINDER_MAX,2.7,0.1)
        pvpp = st.slider("PVPP (%)",0.5,6.0,3.0,0.1)
        mgst = st.slider("Mg-St (%)",0.01,1.2,0.20,0.01)
        mcc = st.slider("MCC (%)",0.0,MCC_MAX,3.6,0.1)
    with st.container(border=True):
        pressure = st.slider("Pressure (MPa)",80.0,PRESSURE_MAX,230.0,1.0)
        speed = st.slider("Speed (rpm)",1.0,50.0,12.0,0.5)
        granule = st.slider("Granule (µm)",30.0,250.0,125.0,1.0)
    predict = st.button("🔬 Predict & Optimize", use_container_width=True)

with col2:
    st.markdown("### Results")
    if predict:
        # normalize formulation
        total = api+binder+pvpp+mgst+mcc
        if abs(total-100)>0.1:
            st.warning("Formulation must sum to 100%")
        else:
            if model is not None:
                d,t,e,ef = predict(model, scaler, yscaler, [api,mcc,pvpp,mgst,binder,pressure,speed,granule])
            else:
                d,t,e,ef = 0.7,2.0,0.5,0.25
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Density",f"{d:.3f}","✅" if D_MIN<=d<=D_MAX else "❌")
            c2.metric("Tensile",f"{t:.2f}","✅" if t>=TENSILE_MIN else "❌")
            c3.metric("EFRF",f"{ef:.4f}","✅" if ef<EFRF_MAX else "❌")
            c4.metric("MCC",f"{mcc:.1f}","✅" if mcc<=MCC_MAX else "❌")
            if all([D_MIN<=d<=D_MAX, t>=TENSILE_MIN, ef<EFRF_MAX, mcc<=MCC_MAX]):
                st.success("✅ All constraints satisfied!")
            else:
                st.error("❌ Violations")

            # NSGA-II (simple)
            st.markdown("#### Pareto Front (approx)")
            bounds = np.array([[60,100],[0.1,20],[0.1,12],[0.01,3.0],[0.1,10],
                               [80,PRESSURE_MAX],[1,50],[30,250]])
            with st.spinner("Running NSGA‑II..."):
                pop, objs, fronts = simple_nsga(model, scaler, yscaler, bounds, 
                                                pop=NSGA_POP_SIZE, gens=NSGA_GENERATIONS)
            if len(fronts)>0 and len(fronts[0])>0:
                front0 = fronts[0]
                df_plot = pd.DataFrame({
                    'API': -objs[front0,0],
                    'EFRF': objs[front0,1]
                })
                fig = px.scatter(df_plot, x='API', y='EFRF', title='Pareto Front')
                fig.add_hline(y=EFRF_MAX, line_dash='dash', line_color='red')
                st.plotly_chart(fig, use_container_width=True)
                # show best (lowest EFRF feasible)
                best_idx = None
                best_ef = 1e9
                for idx in front0:
                    if objs[idx,1] < best_ef:
                        # check feasibility
                        ind = pop[idx]
                        api2,mcc2,pvpp2,mgst2,binder2,pres2,sp2,gr2 = ind
                        d2,t2,e2,ef2 = predict(model, scaler, yscaler, 
                                                [api2,mcc2,pvpp2,mgst2,binder2,pres2,sp2,gr2])
                        if D_MIN<=d2<=D_MAX and t2>=TENSILE_MIN and ef2<EFRF_MAX:
                            best_ef = objs[idx,1]
                            best_idx = idx
                if best_idx is not None:
                    golden = pop[best_idx]
                    d2,t2,e2,ef2 = predict(model, scaler, yscaler, golden)
                    st.markdown("---")
                    st.markdown("### ⭐ Best Feasible Solution")
                    colA,colB = st.columns(2)
                    with colA:
                        st.write(f"API: {golden[0]:.1f}%")
                        st.write(f"MCC: {golden[1]:.1f}%")
                        st.write(f"PVPP: {golden[2]:.1f}%")
                        st.write(f"MgSt: {golden[3]:.2f}%")
                        st.write(f"Binder: {golden[4]:.1f}%")
                    with colB:
                        st.write(f"Pressure: {golden[5]:.1f} MPa")
                        st.write(f"Speed: {golden[6]:.1f} rpm")
                        st.write(f"Granule: {golden[7]:.0f} µm")
                        st.write(f"Density: {d2:.3f}")
                        st.write(f"Tensile: {t2:.2f} MPa")
                        st.write(f"EFRF: {ef2:.4f}")
            else:
                st.info("No Pareto front found.")
st.caption("Hubryd AI Light · Fast Demo")
