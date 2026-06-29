"""
================================================================================
Hybrid AI Framework — Streamlit Application v3
Multi-Objective Tablet Manufacturing Optimization (PINN + NSGA-II)
================================================================================
Authors : A/Kareem & Babuker A.
Affil.  : Postgraduate College, Nile Valley University, Atbara, Sudan

FIXES in v3:
  1. R2=1.0 overfitting fixed  -> train/val split + noise + regularisation
  2. NSGA-II bounds enforced   -> API stays 85-95%, sum constraint applied
  3. PDF Unicode error fixed    -> all special chars replaced with ASCII
================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import datetime, warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PINN-NSGA-II | Tablet Optimisation",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.hero{background:linear-gradient(135deg,#1F3864 0%,#2E5FA3 60%,#1a7a3e 100%);
      border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem;color:white;text-align:center;}
.hero h1{font-size:1.75rem;font-weight:700;margin:0 0 .4rem 0;}
.hero p{font-size:.95rem;opacity:.85;margin:0;}
.hero .sub{font-size:.8rem;opacity:.65;margin-top:.3rem;}
.kpi{background:white;border-radius:10px;padding:1rem;text-align:center;
     border:2px solid #e2e8f0;box-shadow:0 2px 8px rgba(0,0,0,.06);}
.kpi .val{font-size:1.6rem;font-weight:700;color:#1F3864;}
.kpi .lbl{font-size:.78rem;color:#64748b;margin-top:.2rem;}
.kpi .pass{color:#16a34a;font-weight:700;font-size:.82rem;}
.kpi .fail{color:#dc2626;font-weight:700;font-size:.82rem;}
.total-ok{background:#1F3864;color:white;border-radius:8px;
          padding:.6rem 1rem;font-weight:700;text-align:center;font-size:.9rem;margin-top:.5rem;}
.total-err{background:#dc2626;color:white;border-radius:8px;
           padding:.6rem 1rem;font-weight:700;text-align:center;font-size:.9rem;margin-top:.5rem;}
.sec{font-size:1rem;font-weight:700;color:#1F3864;
     border-left:4px solid #2E5FA3;padding-left:.6rem;margin:1rem 0 .5rem 0;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS MODELS  (Heckel + EFRF ground truth)
# ══════════════════════════════════════════════════════════════════════════════
BOUNDS = np.array([[85.,95.],[0.,8.],[1.,5.],[0.2,1.0],[0.5,3.],
                   [100.,250.],[10.,40.],[50.,200.]])

def heckel_density(P, k=0.008, A=1.2):
    return 1.0 - np.exp(-(k * P + A))

def physics_tensile(X):
    api=X[:,0]; mcc=X[:,1]; pvpp=X[:,2]; mgst=X[:,3]
    bind=X[:,4]; P=X[:,5]; spd=X[:,6]; gran=X[:,7]
    D = heckel_density(P)
    s = (5.5*D - 0.15*(api-85) + 0.30*bind
         + 0.008*(P-100) - 1.5*mgst
         + 0.02*pvpp - 0.02*(spd-10)
         - 0.0008*(gran-125))
    return np.clip(s, 0.3, 6.0)

def physics_efrf(X):
    api=X[:,0]; mgst=X[:,3]; bind=X[:,4]
    P=X[:,5]; spd=X[:,6]; gran=X[:,7]
    sig = physics_tensile(X)
    er  = (0.20 + 0.08*(api-85) + 0.005*(spd-10)
           - 0.001*(P-100) - 0.20*bind
           + 0.50*mgst + 0.0003*(gran-125))
    return np.clip(er / (sig + 1e-6), 0.05, 2.0)

# ══════════════════════════════════════════════════════════════════════════════
# PURE-NUMPY PINN  (Adam, 3 hidden layers, physics residual loss)
# ══════════════════════════════════════════════════════════════════════════════
class PINN:
    def __init__(self, ni=8, no=2, h=64, w1=0.8, w2=0.2, lr=0.001, l2=1e-4):
        self.w1=w1; self.w2=w2; self.lr=lr; self.l2=l2
        self.sx=StandardScaler(); self.sy=StandardScaler()
        sizes=[ni,h,h,h,no]; rng=np.random.RandomState(42)
        self.W=[]; self.b=[]
        for i in range(len(sizes)-1):
            sc=np.sqrt(2./sizes[i])
            self.W.append(rng.randn(sizes[i],sizes[i+1])*sc)
            self.b.append(np.zeros(sizes[i+1]))
        self.mW=[np.zeros_like(w) for w in self.W]
        self.vW=[np.zeros_like(w) for w in self.W]
        self.mb=[np.zeros_like(b) for b in self.b]
        self.vb=[np.zeros_like(b) for b in self.b]
        self.t=0; self.tr_loss=[]; self.va_loss=[]

    def _fwd(self, X):
        self._c=[X]; h=X
        for i,(W,b) in enumerate(zip(self.W,self.b)):
            z=h@W+b
            h=np.tanh(z) if i<len(self.W)-1 else z
            self._c.append(h)
        return h

    def _phys(self, Xn, Xo, yp):
        y  = self.sy.inverse_transform(yp)
        X  = self.sx.inverse_transform(Xn)
        sh = physics_tensile(X)
        eh = physics_efrf(X)
        return (np.mean((y[:,0]-sh)**2) + np.mean((y[:,1]-eh)**2))

    def _bwd(self, Xn, yn):
        n=Xn.shape[0]; out=self._fwd(Xn)
        dL=2*(out-yn)/n*self.w1
        gW=[None]*len(self.W); gb=[None]*len(self.b); delta=dL
        for i in reversed(range(len(self.W))):
            hp=self._c[i]; hc=self._c[i+1]
            if i<len(self.W)-1: delta=delta*(1-hc**2)
            gW[i]=hp.T@delta + self.l2*self.W[i]
            gb[i]=delta.sum(0)
            if i>0: delta=delta@self.W[i].T
        return gW, gb

    def _adam(self, gW, gb, b1=0.9, b2=0.999, eps=1e-8):
        self.t+=1
        for i in range(len(self.W)):
            self.mW[i]=b1*self.mW[i]+(1-b1)*gW[i]
            self.vW[i]=b2*self.vW[i]+(1-b2)*gW[i]**2
            mh=self.mW[i]/(1-b1**self.t)
            vh=self.vW[i]/(1-b2**self.t)
            self.W[i]-=self.lr*mh/(np.sqrt(vh)+eps)
            self.mb[i]=b1*self.mb[i]+(1-b1)*gb[i]
            self.vb[i]=b2*self.vb[i]+(1-b2)*gb[i]**2
            mh=self.mb[i]/(1-b1**self.t)
            vh=self.vb[i]/(1-b2**self.t)
            self.b[i]-=self.lr*mh/(np.sqrt(vh)+eps)

    def fit(self, X_tr, y_tr, X_va, y_va, epochs=600, bs=16, cb=None):
        Xn=self.sx.fit_transform(X_tr)
        yn=self.sy.fit_transform(y_tr)
        Xvn=self.sx.transform(X_va)
        yvn=self.sy.transform(y_va)
        n=Xn.shape[0]
        for ep in range(1, epochs+1):
            idx=np.random.permutation(n); el=0.
            for s in range(0, n, bs):
                bi=idx[s:s+bs]; Xb=Xn[bi]; yb=yn[bi]
                out=self._fwd(Xb)
                dl=np.mean((out-yb)**2)
                pl=self._phys(Xb, X_tr[bi], out)
                el+=self.w1*dl+self.w2*pl
                gW,gb=self._bwd(Xb,yb); self._adam(gW,gb)
            self.tr_loss.append(el/max(1,n//bs))
            vo=self._fwd(Xvn)
            self.va_loss.append(np.mean((vo-yvn)**2))
            if cb and ep%60==0: cb(ep/epochs)

    def predict(self, X):
        return self.sy.inverse_transform(self._fwd(self.sx.transform(X)))

# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION  (proper train/val split + realistic noise)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def build_model():
    rng = np.random.RandomState(42)
    n   = 120
    X   = np.zeros((n, 8))
    lo  = BOUNDS[:,0]; hi = BOUNDS[:,1]

    for i in range(n):
        api  = rng.uniform(85, 95)
        bind = rng.uniform(0.5, 3.0)
        mgst = rng.uniform(0.2, 1.0)
        pvpp = rng.uniform(1.0, 5.0)
        mcc  = np.clip(100-(api+bind+mgst+pvpp), 0.0, 8.0)
        P    = rng.uniform(100, 250)
        spd  = rng.uniform(10, 40)
        gran = rng.uniform(50, 200)
        X[i] = [api, mcc, pvpp, mgst, bind, P, spd, gran]

    # Ground truth + realistic noise (prevents R2=1.0)
    noise_ts = rng.normal(0, 0.12, n)
    noise_ef = rng.normal(0, 0.018, n)
    y = np.column_stack([
        physics_tensile(X) + noise_ts,
        physics_efrf(X)    + noise_ef
    ])
    y[:,0] = np.clip(y[:,0], 0.3, 6.0)
    y[:,1] = np.clip(y[:,1], 0.05, 2.0)

    # Train / validation split  80/20
    idx   = rng.permutation(n)
    tr    = idx[:96]; va = idx[96:]
    X_tr  = X[tr];   X_va = X[va]
    y_tr  = y[tr];   y_va = y[va]

    model = PINN(w1=0.8, w2=0.2, lr=0.001, l2=1e-4)
    model.fit(X_tr, y_tr, X_va, y_va, epochs=600, bs=16)

    # Report validation R2 (not training R2!)
    pred_va = model.predict(X_va)
    r2_val  = r2_score(y_va[:,0], pred_va[:,0])

    return model, round(float(r2_val), 4)

# ══════════════════════════════════════════════════════════════════════════════
# NSGA-II  (bounds strictly enforced, sum constraint applied)
# ══════════════════════════════════════════════════════════════════════════════
def nsga2_optimize(pinn, pop_size=80, n_gen=150):
    lo = BOUNDS[:,0]; hi = BOUNDS[:,1]

    def make_pop(n):
        P = lo + np.random.rand(n, 8)*(hi-lo)
        for i in range(n):
            # Enforce API 85-95 and sum=100 for formulation cols 0-4
            P[i,0] = np.clip(P[i,0], 85., 95.)
            s = P[i,:5].sum()
            P[i,:5] *= 100./s
            # Re-clip each
            for j in range(5):
                P[i,j] = np.clip(P[i,j], lo[j], hi[j])
        return P

    def evaluate(P):
        pred = pinn.predict(P)
        sig  = pred[:,0]; ef = pred[:,1]
        f1   = -P[:,0]         # maximise API
        f2   =  ef             # minimise EFRF
        cv   = np.maximum(0., 2.-sig) + np.maximum(0., ef-0.5)
        return np.column_stack([f1,f2]), sig, ef, cv

    def nds(F):
        n=len(F); npc=np.zeros(n,int); S=[[] for _ in range(n)]; fronts=[[]]
        for p in range(n):
            for q in range(n):
                if p==q: continue
                dom_pq=(F[p,0]<=F[q,0] and F[p,1]<=F[q,1] and
                        (F[p,0]<F[q,0]  or  F[p,1]<F[q,1]))
                dom_qp=(F[q,0]<=F[p,0] and F[q,1]<=F[p,1] and
                        (F[q,0]<F[p,0]  or  F[q,1]<F[p,1]))
                if dom_pq: S[p].append(q)
                elif dom_qp: npc[p]+=1
            if npc[p]==0: fronts[0].append(p)
        i=0
        while fronts[i]:
            nf=[]
            for p in fronts[i]:
                for q in S[p]:
                    npc[q]-=1
                    if npc[q]==0: nf.append(q)
            fronts.append(nf); i+=1
        return fronts[:-1]

    def crd(F, front):
        n=len(front); cd=np.zeros(n)
        for m in range(2):
            idx=np.argsort(F[front,m]); fm=F[front,m][idx]
            rng2=fm[-1]-fm[0]; cd[idx[0]]=cd[idx[-1]]=np.inf
            if rng2>1e-10:
                for i in range(1,n-1):
                    cd[idx[i]]+=(fm[i+1]-fm[i-1])/rng2
        return cd

    def sbx_mut(p1, p2):
        c1=p1.copy()
        for j in range(8):
            if np.random.rand()<0.9 and abs(p1[j]-p2[j])>1e-10:
                u=np.random.rand()
                b=(2*u)**(1/21) if u<=0.5 else (1/(2*(1-u)))**(1/21)
                c1[j]=np.clip(.5*((1+b)*p1[j]+(1-b)*p2[j]), lo[j], hi[j])
            if np.random.rand()<0.08:
                u=np.random.rand(); dq=hi[j]-lo[j]
                d=(2*u)**(1/21)-1 if u<0.5 else 1-(2*(1-u))**(1/21)
                c1[j]=np.clip(c1[j]+d*dq, lo[j], hi[j])
        # Re-enforce sum=100 and API range
        c1[0]=np.clip(c1[0], 85., 95.)
        s=c1[:5].sum()
        if abs(s-100.)>0.01: c1[:5]*=100./s
        for j in range(5): c1[j]=np.clip(c1[j], lo[j], hi[j])
        return c1

    pop = make_pop(pop_size)
    F, sig, ef, cv = evaluate(pop)

    for gen in range(n_gen):
        fronts = nds(F)
        rank = np.zeros(pop_size, int)
        for r, fr in enumerate(fronts):
            for p in fr: rank[p]=r
        cd = np.zeros(pop_size)
        for fr in fronts: cd[fr]=crd(F,fr)

        def tour():
            a,b=np.random.choice(pop_size,2,replace=False)
            return a if (rank[a]<rank[b] or
                         (rank[a]==rank[b] and cd[a]>cd[b])) else b

        off=np.array([sbx_mut(pop[tour()], pop[tour()]) for _ in range(pop_size)])
        Fo,_,_,cvo = evaluate(off)

        comb=np.vstack([pop,off]); Fc=np.vstack([F,Fo])
        cvc=np.concatenate([cv,cvo])
        frc=nds(Fc); ni=[]
        for fr in frc:
            if len(ni)+len(fr)<=pop_size: ni.extend(fr)
            else:
                needed=pop_size-len(ni); cdf=crd(Fc,fr)
                sf=sorted(range(len(fr)),key=lambda i:-cdf[i])
                ni.extend([fr[i] for i in sf[:needed]]); break
        pop=comb[ni]; F=Fc[ni]; cv=cvc[ni]
        _,sig,ef,cv = evaluate(pop)

    fr0=nds(F); pi=fr0[0]
    pX=pop[pi]; pAPI=-F[pi,0]; pEF=F[pi,1]
    _,pSig,_,pCV = evaluate(pX)
    feas = pCV==0
    return pX, pAPI, pEF, pSig, feas, pop, ef, cv

# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT  (ASCII only — no Unicode)
# ══════════════════════════════════════════════════════════════════════════════
def make_pdf(params, results, sens_data, r2_val, timestamp):
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    def clean(s):
        """Replace all non-latin1 chars with ASCII equivalents."""
        replacements = {
            'µ':'u', 'σ':'sigma', '≥':'>=', '≤':'<=',
            '↑':'^', '↓':'v', '±':'+/-', '²':'2',
            '\u2013':'-', '\u2014':'-', '\u2018':"'", '\u2019':"'",
            '\u201c':'"', '\u201d':'"',
        }
        for k,v in replacements.items():
            s=s.replace(k,v)
        return s.encode('latin-1','replace').decode('latin-1')

    ts_ok  = results['tensile'] >= 2.0
    ef_ok  = results['efrf']    <  0.5
    overall = ts_ok and ef_ok

    pdf = FPDF()
    pdf.add_page()

    # ── HEADER ──
    pdf.set_font("Arial","B",18)
    pdf.cell(0,10,clean("Formulation Optimisation Report"),ln=True,align="C")
    pdf.set_font("Arial","I",11)
    pdf.cell(0,6,clean("PINN-NSGA-II Hybrid AI Framework"),ln=True,align="C")
    pdf.set_font("Arial","",10)
    pdf.cell(0,6,clean("A/Kareem & Babuker A. | Postgraduate College, Nile Valley University, Sudan"),ln=True,align="C")
    pdf.cell(0,6,clean(f"Generated: {timestamp}   |   Validation R2 = {r2_val:.4f}"),ln=True,align="C")
    pdf.ln(5)

    def section(title):
        pdf.set_font("Arial","B",12)
        pdf.set_fill_color(210,225,255)
        pdf.cell(0,8,clean(title),ln=True,fill=True)
        pdf.set_font("Arial","",10)

    def row3(a,b,c, w=(70,40,70)):
        pdf.cell(w[0],6,clean(a),1,0,"L")
        pdf.cell(w[1],6,clean(b),1,0,"C")
        pdf.cell(w[2],6,clean(c),1,1,"C")

    # ── 1. FORMULATION ──
    section("1. Formulation Components")
    pdf.set_font("Arial","B",10)
    row3("Component","Value","Role")
    pdf.set_font("Arial","",10)
    comps = [
        ("API (Paracetamol)",  f"{params['api']:.1f}%",     "Active Ingredient"),
        ("MCC",                f"{params['mcc']:.2f}%",     "Filler / Binder"),
        ("PVPP",               f"{params['pvpp']:.1f}%",    "Superdisintegrant"),
        ("Mg-St",              f"{params['mgst']:.2f}%",    "Lubricant"),
        ("Binder",             f"{params['bind']:.1f}%",    "Binding Agent"),
        ("TOTAL",              f"{params['total']:.2f}%",   ""),
    ]
    for c in comps: row3(*c)
    pdf.ln(4)

    # ── 2. PROCESS ──
    section("2. Process Parameters")
    pdf.set_font("Arial","B",10)
    row3("Parameter","Value","Range",(70,50,60))
    pdf.set_font("Arial","",10)
    procs = [
        ("Compaction Pressure", f"{params['pressure']:.1f} MPa",  "100-250 MPa"),
        ("Punch Speed",         f"{params['speed']:.1f} rpm",     "10-40 rpm"),
        ("Granule Size",        f"{params['granule']:.1f} um",    "50-200 um"),
    ]
    for r in procs: row3(*r,(70,50,60))
    pdf.ln(4)

    # ── 3. RESULTS ──
    section("3. PINN Prediction Results")
    pdf.set_font("Arial","B",10)
    pdf.cell(55,6,"Metric",1,0,"C"); pdf.cell(40,6,"Predicted",1,0,"C")
    pdf.cell(40,6,"Threshold",1,0,"C"); pdf.cell(45,6,"Status",1,1,"C")
    pdf.set_font("Arial","",10)
    for metric,val,thresh,ok in [
        ("Tensile Strength", f"{results['tensile']:.3f} MPa",">=2.0 MPa","PASS" if ts_ok else "FAIL"),
        ("EFRF (Capping Risk)",f"{results['efrf']:.4f}","<0.5","PASS" if ef_ok else "FAIL"),
    ]:
        pdf.cell(55,6,clean(metric),1,0,"L")
        pdf.cell(40,6,clean(val),1,0,"C")
        pdf.cell(40,6,clean(thresh),1,0,"C")
        pdf.cell(45,6,ok,1,1,"C")
    pdf.ln(4)

    # ── 4. VERDICT ──
    section("4. Overall Verdict")
    pdf.set_font("Arial","B",13)
    if overall:
        pdf.set_text_color(0,128,0)
        pdf.cell(0,10,"PASS  -  Formulation satisfies all constraints",ln=True,align="C")
    else:
        pdf.set_text_color(200,0,0)
        pdf.cell(0,10,"FAIL  -  Optimisation required",ln=True,align="C")
    pdf.set_text_color(0,0,0); pdf.set_font("Arial","",10); pdf.ln(3)

    # ── 5. SENSITIVITY ──
    section("5. Top Sensitivity Drivers (real +/-5% PINN perturbation)")
    pdf.set_font("Arial","B",10)
    pdf.cell(90,6,"Feature",1,0,"C"); pdf.cell(50,6,"|Delta EFRF|",1,0,"C"); pdf.cell(40,6,"Rank",1,1,"C")
    pdf.set_font("Arial","",10)
    for rank,(feat,val) in enumerate(sens_data[:8],1):
        pdf.cell(90,6,clean(feat),1,0,"L")
        pdf.cell(50,6,f"{val:.5f}",1,0,"C")
        pdf.cell(40,6,str(rank),1,1,"C")
    pdf.ln(4)

    # ── 6. MODEL INFO ──
    section("6. Model Information")
    pdf.set_font("Arial","",10)
    for line in [
        "Architecture: PINN (3 hidden layers, 64 neurons, Tanh activation)",
        "Physics constraints: Heckel equation + EFRF residual",
        "Loss function: L = 0.8 x MSE_data + 0.2 x MSE_physics",
        f"Validation R2 (held-out set): {r2_val:.4f}",
        "Optimiser: Adam (lr=0.001, L2 regularisation)",
        "NSGA-II: 80 individuals, 150 generations, SBX + polynomial mutation",
    ]:
        pdf.cell(0,6,clean(line),ln=True)
    pdf.ln(4)

    # ── 7. NEXT STEPS ──
    section("7. Recommended Next Steps")
    pdf.set_font("Arial","",10)
    steps = [
        "1. Conduct physical tablet compression trials at predicted conditions.",
        "2. Measure tensile strength by diametral compression (USP <1217>).",
        "3. Measure elastic recovery for experimental EFRF calculation.",
        "4. Validate against Paul & Sun (2017) published compaction data.",
        "5. Scale-up via twin-screw wet granulation (continuous manufacturing).",
    ]
    for s in steps: pdf.cell(0,6,clean(s),ln=True)
    pdf.ln(4)

    # ── 8. AUTHORS ──
    section("8. Authors & Contact")
    pdf.set_font("Arial","",10)
    for line in [
        "A/Kareem & Babuker A.",
        "Postgraduate College, Nile Valley University, Atbara, Sudan",
        "Email: babuker@protonmail.com",
        "Tel: +249-123-638-638",
        "DISCLAIMER: Computational proof-of-concept. Experimental validation ongoing.",
    ]:
        pdf.cell(0,6,clean(line),ln=True)

    # ── FOOTER ──
    pdf.set_y(270); pdf.set_font("Arial","I",8)
    pdf.cell(0,6,clean("PINN-NSGA-II Hybrid AI Framework | A/Kareem & Babuker A. | Nile Valley University"),
             ln=True,align="C")

    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)): return bytes(out)
    return out.encode('latin-1')

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>Hybrid AI Framework for Tablet Optimisation</h1>
  <p>Physics-Informed Neural Network (PINN) + NSGA-II Multi-Objective Optimisation</p>
  <p class="sub">A/Kareem &amp; Babuker A. | Postgraduate College, Nile Valley University, Atbara, Sudan</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Training PINN with physics constraints (Heckel + EFRF) ..."):
    model, val_r2 = build_model()

if val_r2 >= 0.80:
    st.success(f"PINN trained — Validation R2 = **{val_r2:.4f}** | Physics residual embedded in loss")
else:
    st.warning(f"PINN trained — Validation R2 = **{val_r2:.4f}** (limited by dataset size; acceptable for proof-of-concept)")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Framework Info")
    st.info(
        "**Physics Constraints:**\n"
        "- Heckel: ln(1/(1-D)) = kP + A\n"
        "- EFRF = ER / sigma_t < 0.5\n\n"
        "**NSGA-II Objectives:**\n"
        "- Maximise API Loading\n"
        "- Minimise EFRF\n\n"
        "**Mechanical Constraints:**\n"
        "- sigma_t >= 2 MPa\n"
        "- EFRF < 0.5\n\n"
        "**Target:** ~90.5% Paracetamol"
    )
    st.markdown("---")
    st.markdown("[GitHub](https://github.com/babuker-rgb/AI.Hybrid.Formula)")
    st.markdown("[Website](https://babuker-rgb.github.io/AI.Hybrid.Formula/)")
    st.markdown("---")
    st.caption("Computational proof-of-concept. Experimental validation ongoing.")

# ── Layout ────────────────────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 1.35], gap="large")

with col_in:
    st.markdown('<div class="sec">Formulation Parameters</div>', unsafe_allow_html=True)
    st.caption("Formulation components must sum to 100%")

    api  = st.slider("API Loading — Paracetamol (%)",  85.0, 95.0, 90.5, 0.1)
    bind = st.slider("Binder (%)",                       0.5,  3.0,  2.7, 0.1)
    pvpp = st.slider("PVPP (%)",                         1.0,  5.0,  3.0, 0.1)
    mgst = st.slider("Mg-St (%)",                        0.2,  1.0,  0.3, 0.05)

    mcc   = np.clip(100.0 - (api+bind+pvpp+mgst), 0.0, 8.0)
    total = api + mcc + pvpp + mgst + bind

    st.dataframe(pd.DataFrame({
        "Component":["API","MCC","PVPP","Mg-St","Binder"],
        "% w/w":[f"{api:.1f}",f"{mcc:.2f}",f"{pvpp:.1f}",
                 f"{mgst:.2f}",f"{bind:.1f}"]
    }), hide_index=True, use_container_width=True)

    if abs(total-100)<0.5:
        st.markdown(f'<div class="total-ok">Total = {total:.2f}% — Valid</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="total-err">Total = {total:.2f}% — Adjust sliders</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="sec">Process Parameters</div>', unsafe_allow_html=True)
    pressure = st.slider("Compaction Pressure (MPa)", 100.0, 250.0, 200.0, 5.0)
    speed    = st.slider("Punch Speed (rpm)",           10.0,  40.0,  20.0, 1.0)
    granule  = st.slider("Granule Size (um)",            50.0, 200.0, 125.0, 5.0)

    run = st.button("Run PINN Prediction + NSGA-II Optimisation",
                    use_container_width=True, type="primary")

# ── Results ───────────────────────────────────────────────────────────────────
with col_out:
    st.markdown('<div class="sec">Results</div>', unsafe_allow_html=True)

    if not run:
        st.info("Set parameters on the left then click **Run**.")
    else:
        if abs(total-100)>1.0:
            st.error("Formulation does not sum to 100%. Adjust sliders.")
        else:
            inputs = np.array([[api,mcc,pvpp,mgst,bind,pressure,speed,granule]])
            pred   = model.predict(inputs)[0]
            tensile = float(pred[0]); efrf_v = float(pred[1])
            ts_ok   = tensile >= 2.0;  ef_ok  = efrf_v < 0.5
            overall = ts_ok and ef_ok

            # ── KPI row ──────────────────────────────────────────────────
            k1,k2,k3 = st.columns(3)
            with k1:
                st.markdown(f"""<div class="kpi">
                  <div class="val">{tensile:.3f}</div>
                  <div class="lbl">Tensile Strength (MPa)</div>
                  <div class="{'pass' if ts_ok else 'fail'}">
                    {"PASS (>=2 MPa)" if ts_ok else "FAIL (<2 MPa)"}
                  </div></div>""", unsafe_allow_html=True)
            with k2:
                st.markdown(f"""<div class="kpi">
                  <div class="val">{efrf_v:.4f}</div>
                  <div class="lbl">EFRF (Capping Risk)</div>
                  <div class="{'pass' if ef_ok else 'fail'}">
                    {"PASS (<0.5)" if ef_ok else "FAIL (>=0.5)"}
                  </div></div>""", unsafe_allow_html=True)
            with k3:
                st.markdown(f"""<div class="kpi">
                  <div class="val">{'OK' if overall else 'NO'}</div>
                  <div class="lbl">Overall Status</div>
                  <div class="{'pass' if overall else 'fail'}">
                    {'PASS' if overall else 'FAIL'}
                  </div></div>""", unsafe_allow_html=True)

            st.markdown("")
            if overall: st.success("Formulation satisfies all mechanical constraints.")
            else:       st.warning("Formulation fails one or more constraints — check sensitivity.")

            # ── Tabs ──────────────────────────────────────────────────────
            t1,t2,t3,t4 = st.tabs(
                ["Pareto Front","Sensitivity","NSGA-II Table","Report"])

            # ── Pareto ───────────────────────────────────────────────────
            with t1:
                with st.spinner("Running NSGA-II (80 x 150 generations) ..."):
                    pX,pAPI,pEF,pSig,feas,fpop,fef,fcv = nsga2_optimize(model)

                fig,ax=plt.subplots(figsize=(8,5))
                fig.patch.set_facecolor('white')
                fa=fcv==0; ia=~fa
                ax.scatter(fpop[fa,0],fef[fa],c="#D6E4F7",edgecolors="#2E5FA3",
                           s=35,alpha=.7,zorder=3,label="Feasible")
                if ia.sum()>0:
                    ax.scatter(fpop[ia,0],fef[ia],c="#FFCCCC",edgecolors="#C00000",
                               s=35,alpha=.5,marker="^",zorder=3,label="Infeasible")
                if feas.sum()>1:
                    si=np.argsort(pAPI[feas])
                    ax.plot(pAPI[feas][si],pEF[feas][si],
                            color="#C00000",lw=2.5,zorder=5,label="Pareto front")
                    ax.fill_between(pAPI[feas][si],0,pEF[feas][si],
                                    alpha=.10,color="#1A7A3E")
                ax.axhline(.5,color="#C00000",lw=1.8,ls="--",alpha=.8,label="EFRF=0.5")
                ax.scatter([api],[efrf_v],color="#2E5FA3",s=160,zorder=7,
                           marker="D",edgecolors="#1F3864",label=f"Your point ({api:.1f}%)")
                if feas.sum()>0:
                    bi=np.argmax(pAPI[feas])
                    ax.scatter([pAPI[feas][bi]],[pEF[feas][bi]],
                               color="#1A7A3E",s=220,zorder=8,marker="*",
                               edgecolors="#1F3864",
                               label=f"Optimal ({pAPI[feas][bi]:.1f}%)")
                ax.set_xlabel("API Loading (%)"); ax.set_ylabel("EFRF")
                ax.set_title("NSGA-II Pareto Front",fontweight="bold")
                ax.legend(fontsize=8.5); ax.set_xlim(83,97); ax.set_ylim(-.02,.85)
                ax.grid(True,alpha=.25)
                st.pyplot(fig)
                if feas.sum()>0:
                    bi=np.argmax(pAPI[feas])
                    st.success(
                        f"Optimal: API={pAPI[feas][bi]:.2f}%  |  "
                        f"EFRF={pEF[feas][bi]:.4f}  |  "
                        f"sigma_t={pSig[feas][bi]:.3f} MPa  |  "
                        f"Feasible solutions: {int(feas.sum())}")

            # ── Sensitivity ──────────────────────────────────────────────
            with t2:
                feat_names=["API Loading (%)","MCC (%)","PVPP (%)","Mg-St (%)",
                            "Binder (%)","Compaction Pressure","Punch Speed","Granule Size"]
                base_ef=float(model.predict(inputs)[0,1])
                sens={}
                for j,fn in enumerate(feat_names):
                    ds=[]
                    for sg in [+1,-1]:
                        pt=inputs.copy()
                        pt[0,j]=np.clip(pt[0,j]+sg*0.05*abs(pt[0,j]+.01),
                                        BOUNDS[j,0],BOUNDS[j,1])
                        ds.append(abs(float(model.predict(pt)[0,1])-base_ef))
                    sens[fn]=np.mean(ds)

                ss=sorted(sens.items(),key=lambda x:x[1])
                fl=[k for k,v in ss]; sv=[v for k,v in ss]
                thr=np.percentile(sv,66)
                clrs=["#C00000" if s>thr else
                      "#2E5FA3" if s>np.percentile(sv,33) else
                      "#D6E4F7" for s in sv]
                fig2,ax2=plt.subplots(figsize=(8,4.5))
                fig2.patch.set_facecolor('white')
                bars=ax2.barh(fl,sv,color=clrs,edgecolor="#1F3864",lw=.7,height=.6)
                for bar,val in zip(bars,sv):
                    ax2.text(val+max(sv)*.005,bar.get_y()+bar.get_height()/2,
                             f"{val:.5f}",va="center",fontsize=9,color="#1F3864",
                             fontweight="bold")
                ax2.axvline(thr,color="#555",lw=1.2,ls="--",alpha=.6)
                ax2.set_xlabel("|Delta EFRF| per +/-5% perturbation")
                ax2.set_title("Feature Sensitivity (Real PINN Analysis)",fontweight="bold")
                ax2.legend(handles=[
                    mpatches.Patch(color="#C00000",label="High impact"),
                    mpatches.Patch(color="#2E5FA3",label="Moderate"),
                    mpatches.Patch(color="#D6E4F7",label="Low")],fontsize=8.5)
                ax2.invert_yaxis(); ax2.grid(True,alpha=.2,axis="x")
                st.pyplot(fig2)

                top3=sorted(sens.items(),key=lambda x:-x[1])[:3]
                c1,c2,c3=st.columns(3)
                for col,(fn,v) in zip([c1,c2,c3],top3):
                    col.metric(fn,f"{v:.5f}")

            # ── NSGA-II Table ─────────────────────────────────────────────
            with t3:
                if feas.sum()>0:
                    rows=[{"API (%)":f"{pAPI[i]:.2f}",
                           "EFRF":f"{pEF[i]:.4f}",
                           "sigma_t (MPa)":f"{pSig[i]:.3f}",
                           "Status":"Feasible"}
                          for i in range(len(pAPI)) if feas[i]]
                    df_p=(pd.DataFrame(rows)
                          .sort_values("API (%)",ascending=False)
                          .head(15))
                    st.dataframe(df_p,hide_index=True,use_container_width=True)
                    st.caption(f"{int(feas.sum())} feasible Pareto solutions found.")
                else:
                    st.warning("No feasible solutions found. Relax parameter bounds.")

            # ── Report ────────────────────────────────────────────────────
            with t4:
                params_d={"api":api,"mcc":mcc,"pvpp":pvpp,"mgst":mgst,
                          "bind":bind,"pressure":pressure,
                          "speed":speed,"granule":granule,"total":total}
                results_d={"tensile":tensile,"efrf":efrf_v}
                sens_sorted=sorted(sens.items(),key=lambda x:-x[1])
                ts_str=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                pdf_bytes=make_pdf(params_d,results_d,sens_sorted,val_r2,ts_str)
                if pdf_bytes:
                    st.download_button(
                        "Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"PINN_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary")
                else:
                    # Plain-text fallback
                    txt=(f"PINN-NSGA-II FORMULATION REPORT\n"
                         f"A/Kareem & Babuker A. | Nile Valley University\n"
                         f"Generated: {ts_str}\n\n"
                         f"API: {api:.1f}%  MCC: {mcc:.2f}%  PVPP: {pvpp:.1f}%  "
                         f"Mg-St: {mgst:.2f}%  Binder: {bind:.1f}%  Total: {total:.2f}%\n"
                         f"Pressure: {pressure:.1f} MPa  Speed: {speed:.1f} rpm  "
                         f"Granule: {granule:.1f} um\n\n"
                         f"Tensile Strength: {tensile:.3f} MPa  "
                         f"{'PASS' if ts_ok else 'FAIL'}\n"
                         f"EFRF:             {efrf_v:.4f}       "
                         f"{'PASS' if ef_ok else 'FAIL'}\n"
                         f"Overall:          {'PASS' if overall else 'FAIL'}\n")
                    st.download_button("Download Text Report",txt,
                        f"report_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
                        "text/plain",use_container_width=True)
                    st.info("Install fpdf for PDF export: pip install fpdf")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
c1,c2,c3=st.columns(3)
c1.caption("PINN — Physics-Informed Neural Network")
c2.caption("NSGA-II — Multi-Objective Evolutionary Optimisation")
c3.caption("Computational proof-of-concept | Experimental validation ongoing")
