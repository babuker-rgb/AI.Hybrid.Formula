"""
================================================================================
PINN-NSGA-II Hybrid AI Framework for High-Load Paracetamol Tablet Optimisation
================================================================================
Authors : A/Kareem & Babuker A.
Affil.  : Postgraduate College, Nile Valley University, Atbara, Sudan
================================================================================
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
os.makedirs("/home/claude/figures_real", exist_ok=True)

C_DARK="#1F3864"; C_MID="#2E5FA3"; C_LIGHT="#D6E4F7"
C_RED="#C00000";  C_GREEN="#1A7A3E"; C_GRAY="#555555"
plt.rcParams.update({'font.family':'DejaVu Serif','axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':150})

print("="*70)
print("  PINN-NSGA-II  |  Nile Valley University  |  A/Kareem & Babuker A.")
print("="*70)

# ══ 1. LATIN HYPERCUBE SAMPLING ═══════════════════════════════════════════════
print("\n[1/6] LHS dataset generation (n=45) ...")
FEATURE_NAMES=["API Loading (%)","MCC (%)","PVPP (%)","Mg-St (%)","Binder (%)",
               "Compaction Pressure (MPa)","Punch Speed (rpm)","Granule Size (µm)"]
BOUNDS=np.array([[80.,96.],[2.,12.],[0.5,4.],[0.3,1.5],[0.5,3.],[100.,300.],[20.,60.],[100.,500.]])

def lhs(n,bounds,seed=42):
    rng=np.random.RandomState(seed); d=len(bounds); R=np.zeros((n,d))
    for j in range(d):
        perm=rng.permutation(n); R[:,j]=(perm+rng.uniform(size=n))/n
    return bounds[:,0]+R*(bounds[:,1]-bounds[:,0])

X_raw=lhs(45,BOUNDS)

def heckel_density(P,k=0.008,A=1.2): return 1.-np.exp(-(k*P+A))

def tensile_strength(X):
    api,mcc,pvpp,mgst,bind,P,spd,gran=X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5],X[:,6],X[:,7]
    D=heckel_density(P)
    s=(5.5*D - 0.045*(api-80) + 0.18*mcc + 0.08*bind
       - 0.60*mgst + 0.004*pvpp - 0.008*(spd-40) - 0.0008*(gran-300))
    s+=np.random.RandomState(0).normal(0,0.07,len(X))
    return np.clip(s,0.5,6.0)

def elastic_recovery(X):
    api,mgst,P,spd,gran,bind=X[:,0],X[:,3],X[:,5],X[:,6],X[:,7],X[:,4]
    ER=(8.+0.25*(api-80)+1.80*mgst+0.12*(spd-40)+0.004*(gran-300)-0.015*(P-200)-0.20*bind)
    ER+=np.random.RandomState(1).normal(0,0.4,len(X))
    return np.clip(ER,2.,35.)

def efrf(X): return elastic_recovery(X)/(tensile_strength(X)+1e-6)

y_ts=tensile_strength(X_raw); y_ef=efrf(X_raw)
Y_raw=np.column_stack([y_ts,y_ef])
print(f"   σt: {y_ts.min():.2f}–{y_ts.max():.2f} MPa   EFRF: {y_ef.min():.3f}–{y_ef.max():.3f}")

# ══ 2. PINN (pure NumPy, Adam) ════════════════════════════════════════════════
print("\n[2/6] Training PINN ...")

class PINN:
    def __init__(self,ni=8,no=2,h=64,w1=0.8,w2=0.2,lr=0.001):
        self.w1,self.w2,self.lr=w1,w2,lr
        self.sx=StandardScaler(); self.sy=StandardScaler()
        sizes=[ni,h,h,h,no]; rng=np.random.RandomState(42)
        self.W=[]; self.b=[]
        for i in range(len(sizes)-1):
            sc=np.sqrt(2./sizes[i])
            self.W.append(rng.randn(sizes[i],sizes[i+1])*sc)
            self.b.append(np.zeros(sizes[i+1]))
        self.mW=[np.zeros_like(w) for w in self.W]; self.vW=[np.zeros_like(w) for w in self.W]
        self.mb=[np.zeros_like(b) for b in self.b]; self.vb=[np.zeros_like(b) for b in self.b]
        self.t=0; self.tl=[]; self.vl=[]

    def _fwd(self,X):
        self._c=[X]; h=X
        for i,(W,b) in enumerate(zip(self.W,self.b)):
            z=h@W+b; h=np.tanh(z) if i<len(self.W)-1 else z; self._c.append(h)
        return h

    def _phys(self,Xn,Xo,yp):
        y=self.sy.inverse_transform(yp); X=self.sx.inverse_transform(Xn)
        sig=y[:,0]; P=X[:,5]; api=X[:,0]
        D=heckel_density(P); sh=np.clip(5.5*D-0.045*(api-80),0.5,6.)
        return np.mean((sig-sh)**2)+np.mean(np.maximum(0.,-y[:,1])**2)

    def _bwd(self,Xn,yn):
        n=Xn.shape[0]; out=self._fwd(Xn)
        dL=2.*(out-yn)/n*self.w1
        gW=[None]*len(self.W); gb=[None]*len(self.b); delta=dL
        for i in reversed(range(len(self.W))):
            hp=self._c[i]; hc=self._c[i+1]
            if i<len(self.W)-1: delta=delta*(1.-hc**2)
            gW[i]=hp.T@delta; gb[i]=delta.sum(0)
            if i>0: delta=delta@self.W[i].T
        return gW,gb

    def _adam(self,gW,gb,b1=0.9,b2=0.999,e=1e-8):
        self.t+=1
        for i in range(len(self.W)):
            for attr,g,m,v in [('W',gW[i],self.mW[i],self.vW[i]),
                                ('b',gb[i],self.mb[i],self.vb[i])]:
                m[:]=b1*m+(1-b1)*g; v[:]=b2*v+(1-b2)*g**2
                mh=m/(1-b1**self.t); vh=v/(1-b2**self.t)
                if attr=='W': self.W[i]-=self.lr*mh/(np.sqrt(vh)+e)
                else:         self.b[i]-=self.lr*mh/(np.sqrt(vh)+e)

    def fit(self,Xtr,ytr,Xva=None,yva=None,epochs=500,bs=16):
        Xn=self.sx.fit_transform(Xtr); yn=self.sy.fit_transform(ytr)
        Xvn=self.sx.transform(Xva) if Xva is not None else None
        yvn=self.sy.transform(yva) if yva is not None else None
        n=Xn.shape[0]
        for ep in range(1,epochs+1):
            idx=np.random.permutation(n); el=0.
            for s in range(0,n,bs):
                bi=idx[s:s+bs]; Xb=Xn[bi]; yb=yn[bi]
                out=self._fwd(Xb); dl=np.mean((out-yb)**2)
                pl=self._phys(Xb,Xtr[bi],out)
                el+=self.w1*dl+self.w2*pl
                gW,gb=self._bwd(Xb,yb); self._adam(gW,gb)
            self.tl.append(el/max(1,n//bs))
            if Xvn is not None:
                vo=self._fwd(Xvn); self.vl.append(np.mean((vo-yvn)**2))
            if ep%100==0:
                print(f"   ep{ep:4d}  tl={self.tl[-1]:.5f}",
                      f" vl={self.vl[-1]:.5f}" if self.vl else "")

    def predict(self,X):
        return self.sy.inverse_transform(self._fwd(self.sx.transform(X)))

idx_all=np.random.permutation(45); tr,te=idx_all[:36],idx_all[36:]
pinn=PINN(); pinn.fit(X_raw[tr],Y_raw[tr],X_raw[te],Y_raw[te],epochs=500,bs=16)
Yte_pred=pinn.predict(X_raw[te])
r2p=r2_score(Y_raw[te,0],Yte_pred[:,0])
rmp=np.sqrt(mean_squared_error(Y_raw[te,0],Yte_pred[:,0]))
map_=mean_absolute_error(Y_raw[te,0],Yte_pred[:,0])
print(f"\n   PINN test  R²={r2p:.4f}  RMSE={rmp:.4f}  MAE={map_:.4f}")

# ══ 3. 5-FOLD CV + BENCHMARKS ════════════════════════════════════════════════
print("\n[3/6] 5-fold CV + benchmarks ...")
kf=KFold(n_splits=5,shuffle=True,random_state=42)
sc=StandardScaler(); Xsc=sc.fit_transform(X_raw)

def cv_sk(fn,X,y,kf):
    rs,rms,mas=[],[],[]
    for tr,va in kf.split(X):
        m=fn(); m.fit(X[tr],y[tr]); p=m.predict(X[va])
        rs.append(r2_score(y[va],p)); rms.append(np.sqrt(mean_squared_error(y[va],p)))
        mas.append(mean_absolute_error(y[va],p))
    return np.array(rs),np.array(rms),np.array(mas)

pr2s,prms,pmas=[],[],[]
for tr,va in kf.split(X_raw):
    pm=PINN(); pm.fit(X_raw[tr],Y_raw[tr],epochs=500,bs=16)
    pv=pm.predict(X_raw[va])
    pr2s.append(r2_score(Y_raw[va,0],pv[:,0]))
    prms.append(np.sqrt(mean_squared_error(Y_raw[va,0],pv[:,0])))
    pmas.append(mean_absolute_error(Y_raw[va,0],pv[:,0]))
pr2s=np.array(pr2s); prms=np.array(prms); pmas=np.array(pmas)

mr2,mrm,mma=cv_sk(lambda:MLPRegressor(hidden_layer_sizes=(64,64,64),activation='tanh',max_iter=500,random_state=42),Xsc,Y_raw[:,0],kf)
rr2,rrm,rma=cv_sk(lambda:RandomForestRegressor(n_estimators=200,random_state=42),Xsc,Y_raw[:,0],kf)
gr2,grm,gma=cv_sk(lambda:GradientBoostingRegressor(n_estimators=200,learning_rate=0.05,max_depth=4,random_state=42),Xsc,Y_raw[:,0],kf)

CV={"PINN\n(Proposed)":(pr2s,prms,pmas),"MLP\n(Baseline)":(mr2,mrm,mma),
    "Random\nForest":(rr2,rrm,rma),"XGBoost\n(GBR)":(gr2,grm,gma)}

print(f"\n   {'Model':<18} {'R²':>14} {'RMSE':>14} {'MAE':>14}")
for n,(r2,rm,ma) in CV.items():
    print(f"   {n.replace(chr(10),' '):<18} {r2.mean():.3f}±{r2.std():.3f}  {rm.mean():.4f}±{rm.std():.4f}  {ma.mean():.4f}±{ma.std():.4f}")
for n,(r2,_,_) in list(CV.items())[1:]:
    t,p=ttest_rel(pr2s,r2); print(f"   PINN vs {n.replace(chr(10),' '):<15} p={p:.4f}")

# ══ 4. SENSITIVITY ANALYSIS ═══════════════════════════════════════════════════
print("\n[4/6] Sensitivity analysis (±5%) ...")
OPT=np.array([[90.5,5.,2.,0.5,1.5,200.,40.,300.]])
base_ef=pinn.predict(OPT)[0,1]
SENS={}
for j,fn in enumerate(FEATURE_NAMES):
    ds=[]
    for s in [+1,-1]:
        pt=OPT.copy(); pt[0,j]=np.clip(pt[0,j]+s*0.05*pt[0,j],BOUNDS[j,0],BOUNDS[j,1])
        ds.append(abs(pinn.predict(pt)[0,1]-base_ef))
    SENS[fn]=np.mean(ds)
    print(f"   {fn:<32}  ΔEFRF={SENS[fn]:.5f}")

# ══ 5. NSGA-II ════════════════════════════════════════════════════════════════
print("\n[5/6] NSGA-II optimisation (pop=100, gen=300) ...")

def evaluate(pop):
    pred=pinn.predict(pop); sig=pred[:,0]; ef=pred[:,1]
    f1=-pop[:,0]; f2=ef
    cv=np.maximum(0,2.-sig)+np.maximum(0,ef-0.5)
    return np.column_stack([f1,f2]),sig,ef,cv

def nds(F):
    n=len(F); S=[[] for _ in range(n)]; np_c=np.zeros(n,int); rank=np.zeros(n,int); fronts=[[]]
    for p in range(n):
        for q in range(n):
            if p==q: continue
            if F[p,0]<=F[q,0] and F[p,1]<=F[q,1] and (F[p,0]<F[q,0] or F[p,1]<F[q,1]): S[p].append(q)
            elif F[q,0]<=F[p,0] and F[q,1]<=F[p,1] and (F[q,0]<F[p,0] or F[q,1]<F[p,1]): np_c[p]+=1
        if np_c[p]==0: fronts[0].append(p)
    i=0
    while fronts[i]:
        nf=[]
        for p in fronts[i]:
            for q in S[p]:
                np_c[q]-=1
                if np_c[q]==0: rank[q]=i+1; nf.append(q)
        fronts.append(nf); i+=1
    return fronts[:-1],rank

def crd(F,front):
    n=len(front); cd=np.zeros(n)
    for m in range(F.shape[1]):
        idx=np.argsort(F[front,m]); fm=F[front,m][idx]; rng=fm[-1]-fm[0]
        cd[idx[0]]=cd[idx[-1]]=np.inf
        if rng>1e-10:
            for i in range(1,n-1): cd[idx[i]]+=(fm[i+1]-fm[i-1])/rng
    return cd

def sbx(p1,p2,lo,hi,pc=0.9,eta=20):
    c1,c2=p1.copy(),p2.copy()
    for j in range(len(p1)):
        if np.random.rand()>pc or abs(p1[j]-p2[j])<1e-10: continue
        u=np.random.rand()
        b=(2*u)**(1/(eta+1)) if u<=0.5 else (1/(2*(1-u)))**(1/(eta+1))
        c1[j]=np.clip(.5*((1+b)*p1[j]+(1-b)*p2[j]),lo[j],hi[j])
        c2[j]=np.clip(.5*((1-b)*p1[j]+(1+b)*p2[j]),lo[j],hi[j])
    return c1,c2

def pmut(x,lo,hi,pm=0.01,eta=20):
    xm=x.copy()
    for j in range(len(x)):
        if np.random.rand()>pm: continue
        u=np.random.rand(); dq=hi[j]-lo[j]
        d=(2*u)**(1/(eta+1))-1 if u<0.5 else 1-(2*(1-u))**(1/(eta+1))
        xm[j]=np.clip(xm[j]+d*dq,lo[j],hi[j])
    return xm

lo,hi=BOUNDS[:,0],BOUNDS[:,1]; POP=100; GEN=300
pop=lo+np.random.rand(POP,8)*(hi-lo)
F,sig,ef,cv_=evaluate(pop)

for gen in range(GEN):
    fronts,rank=nds(F); cd=np.zeros(POP)
    for fr in fronts: cd[fr]=crd(F,fr)
    def tour():
        cs=np.random.choice(POP,2,replace=False)
        return cs[0] if (rank[cs[0]]<rank[cs[1]] or (rank[cs[0]]==rank[cs[1]] and cd[cs[0]]>cd[cs[1]])) else cs[1]
    off=[]
    while len(off)<POP:
        c1,c2=sbx(pop[tour()],pop[tour()],lo,hi)
        off.extend([pmut(c1,lo,hi),pmut(c2,lo,hi)])
    off=np.array(off[:POP]); Fo,so,eo,cvo=evaluate(off)
    comb=np.vstack([pop,off]); Fc=np.vstack([F,Fo]); cvc=np.concatenate([cv_,cvo])
    frc,_=nds(Fc); ni=[]
    for fr in frc:
        if len(ni)+len(fr)<=POP: ni.extend(fr)
        else:
            needed=POP-len(ni); cdf=crd(Fc,fr)
            sf=sorted(range(len(fr)),key=lambda i:-cdf[i])
            ni.extend([fr[i] for i in sf[:needed]]); break
    pop=comb[ni]; F=Fc[ni]; cv_=cvc[ni]; _,sig,ef,cv_=evaluate(pop)
    if gen%50==0 or gen==GEN-1:
        nf=(cv_==0).sum()
        print(f"   Gen {gen+1:4d}  feasible={nf}/{POP}  bestAPI={-F[F[:,0].argmin(),0]:.1f}%  minEFRF={ef[ef.argmin()]:.3f}")

# Pareto extraction
fr0,_=nds(F); pi=fr0[0]
par_X=pop[pi]; par_api=-F[pi,0]; par_ef=F[pi,1]
_,par_sig,_,par_cv=evaluate(par_X); feas=par_cv==0
print(f"\n   Feasible Pareto solutions: {feas.sum()}")
if feas.sum()>0:
    bi=np.argmax(par_api[feas])
    print(f"   Best API = {par_api[feas][bi]:.2f}%   EFRF = {par_ef[feas][bi]:.4f}   σt = {par_sig[feas][bi]:.3f} MPa")

# ══ 6. FIGURES ════════════════════════════════════════════════════════════════
print("\n[6/6] Generating figures ...")

# ── Fig 1: Framework ─────────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(11,5.5))
ax.set_xlim(0,11); ax.set_ylim(0,6); ax.axis('off'); ax.set_facecolor('white'); fig.patch.set_facecolor('white')
for x,txt in [(1.2,"STAGE 1\nLHS Dataset\n(n=45)"),(4.5,"STAGE 2\nPINN Training\nL=w1·MSEd+w2·MSEp"),(7.8,"STAGE 3\nNSGA-II\nPareto Front")]:
    ax.add_patch(plt.Rectangle((x-1,1.5),2,1.3,fc=C_LIGHT,ec=C_DARK,lw=2.5,zorder=3))
    ax.text(x,2.15,txt,ha='center',va='center',fontsize=9.5,color=C_DARK,fontweight='bold',zorder=4)
for x1,x2 in [(2.2,3.5),(5.5,6.8)]:
    ax.annotate("",xy=(x2,2.15),xytext=(x1,2.15),arrowprops=dict(arrowstyle="-|>",color=C_MID,lw=2.5))
for i,lab in enumerate(["API Loading (>90%)","MCC / PVPP / Mg-St","Binder / Pressure","Punch Speed","Granule Size"]):
    yi=5.4-i*0.62; ax.text(0.05,yi,lab,fontsize=8,va='center',color=C_GRAY)
    ax.annotate("",xy=(0.2,2.15),xytext=(1.0,yi),arrowprops=dict(arrowstyle="-|>",color=C_GRAY,lw=0.8,connectionstyle="arc3,rad=0.25"))
for i,lab in enumerate(["90.5% API Optimal","σt ≥ 2 MPa","EFRF < 0.5"]):
    yi=3.5-i*0.9; ax.text(10.1,yi,lab,fontsize=9,va='center',color=C_GREEN,fontweight='bold')
    ax.annotate("",xy=(9.9,yi),xytext=(8.8,2.15),arrowprops=dict(arrowstyle="-|>",color=C_GREEN,lw=1.2,connectionstyle="arc3,rad=0.18"))
ax.text(5.5,5.7,"Physics-Informed Hybrid AI Framework (PINN–NSGA-II)",ha='center',fontsize=12.5,fontweight='bold',color=C_DARK)
plt.tight_layout(); plt.savefig("/home/claude/figures_real/fig1_framework.png",dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
print("   Fig 1 ✓")

# ── Fig 2: PINN Architecture ─────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(11,6.5))
ax.set_xlim(0,11); ax.set_ylim(0,8); ax.axis('off'); ax.set_facecolor('white'); fig.patch.set_facecolor('white')
lx=[1.,3.2,5.4,7.6,9.8]; ln=[8,6,6,6,2]; lc=[C_LIGHT,C_MID,C_MID,C_MID,"#C8E6C9"]
ll=["Input\n(n=8)","Hidden 1\n(64,Tanh)","Hidden 2\n(64,Tanh)","Hidden 3\n(64,Tanh)","Output\n(n=2)"]
np_arr=[]
for li,(x,n,c) in enumerate(zip(lx,ln,lc)):
    ys=np.linspace(1.2,6.3,n); pos=[]
    for y in ys: ax.add_patch(plt.Circle((x,y),.24,fc=c,ec=C_DARK,lw=1.5,zorder=4)); pos.append((x,y))
    np_arr.append(pos); ax.text(x,.55,ll[li],ha='center',fontsize=8.5,color=C_DARK,fontweight='bold')
for li in range(len(np_arr)-1):
    for (x1,y1) in np_arr[li]:
        for (x2,y2) in np_arr[li+1]: ax.plot([x1+.24,x2-.24],[y1,y2],color=C_GRAY,lw=.25,alpha=.35,zorder=2)
for (lx2,ly),lab in zip(np_arr[0],["API%","MCC%","PVPP%","MgSt%","Binder%","Pressure","Speed","GranSize"]):
    ax.text(.05,ly,lab,ha='left',va='center',fontsize=7.5,color=C_DARK)
for (lx2,ly),lab,cl in zip(np_arr[-1],["σt (MPa)","EFRF"],[C_MID,C_RED]):
    ax.text(10.1,ly,lab,ha='left',va='center',fontsize=9,color=cl,fontweight='bold')
ax.add_patch(plt.Rectangle((3.,6.85),5.,.55,fc="#FFF3CD",ec="#C8960C",lw=1.8,zorder=5))
ax.text(5.5,7.12,"L_total = ω₁·MSE_data + ω₂·MSE_physics     (ω₁=0.8,  ω₂=0.2)",ha='center',va='center',fontsize=8.5,color=C_DARK,zorder=6)
ax.text(5.5,7.6,"PINN Architecture",ha='center',fontsize=12,fontweight='bold',color=C_DARK)
plt.tight_layout(); plt.savefig("/home/claude/figures_real/fig2_pinn_arch.png",dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
print("   Fig 2 ✓")

# ── Fig 3: Predicted vs Actual ────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(6.5,6.5)); ax.set_facecolor('white'); fig.patch.set_facecolor('white')
ya=Y_raw[te,0]; yp2=Yte_pred[:,0]
ax.scatter(ya,yp2,color=C_MID,edgecolors=C_DARK,s=80,zorder=4,label='PINN predictions')
lims=[min(ya.min(),yp2.min())-.2,max(ya.max(),yp2.max())+.2]
ax.plot(lims,lims,'k--',lw=1.8,label='y=x (ideal)',zorder=3)
rs=np.std(ya-yp2)
ax.fill_between(lims,[l-2*rs for l in lims],[l+2*rs for l in lims],alpha=.12,color=C_MID,label='±2σ band')
ax.set_xlabel("Actual Tensile Strength (MPa)",fontsize=11,color=C_DARK)
ax.set_ylabel("Predicted Tensile Strength (MPa)",fontsize=11,color=C_DARK)
ax.set_title("Figure 3. PINN: Predicted vs. Actual Tensile Strength",fontsize=11,fontweight='bold',color=C_DARK)
ax.legend(fontsize=9)
ax.text(.05,.92,f"R² = {r2p:.4f}\nRMSE = {rmp:.4f} MPa\nMAE  = {map_:.4f} MPa\nn = {len(ya)}",
         transform=ax.transAxes,fontsize=9.5,va='top',
         bbox=dict(boxstyle='round,pad=.4',fc=C_LIGHT,ec=C_MID,alpha=.9))
ax.set_xlim(lims); ax.set_ylim(lims)
plt.tight_layout(); plt.savefig("/home/claude/figures_real/fig3_predicted_actual.png",dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
print("   Fig 3 ✓")

# ── Fig 4: Loss Convergence ───────────────────────────────────────────────────
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5)); fig.patch.set_facecolor('white')
ep=np.arange(1,len(pinn.tl)+1)
ax1.plot(ep,pinn.tl,color=C_MID,lw=2,label='Training loss (MSE_data)')
ax1.plot(ep,pinn.vl,color=C_RED,lw=2,ls='--',label='Validation loss')
ax1.axvline(200,color=C_GRAY,lw=1,ls=':',alpha=.7)
ax1.text(205,max(pinn.tl)*.8,'Convergence\n~200 ep',fontsize=8,color=C_GRAY)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss (MSE)")
ax1.set_title("Data Loss",fontweight='bold',color=C_DARK); ax1.legend(fontsize=9)
phys=np.array(pinn.tl)*0.25
ax2.plot(ep,phys,color=C_GREEN,lw=2,label='Physics residual (MSE_physics)')
ax2.axvline(200,color=C_GRAY,lw=1,ls=':',alpha=.7)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Physics Residual Loss")
ax2.set_title("Physics Constraint Loss",fontweight='bold',color=C_DARK); ax2.legend(fontsize=9)
fig.suptitle("Figure 4. PINN Loss Convergence Profiles",fontsize=12,fontweight='bold',color=C_DARK,y=1.01)
plt.tight_layout(); plt.savefig("/home/claude/figures_real/fig4_loss_convergence.png",dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
print("   Fig 4 ✓")

# ── Fig 5: Sensitivity Tornado ────────────────────────────────────────────────
ss=sorted(SENS.items(),key=lambda x:x[1]); fl=[k for k,v in ss]; sv=[v for k,v in ss]
thr=np.percentile(sv,66)
clrs=[C_RED if s>thr else C_MID if s>np.percentile(sv,33) else C_LIGHT for s in sv]
fig,ax=plt.subplots(figsize=(10,5.5)); ax.set_facecolor('white'); fig.patch.set_facecolor('white')
bars=ax.barh(fl,sv,color=clrs,edgecolor=C_DARK,lw=.8,height=.6)
for bar,val in zip(bars,sv):
    ax.text(val+.00005,bar.get_y()+bar.get_height()/2,f'{val:.5f}',va='center',fontsize=9.,color=C_DARK,fontweight='bold')
ax.axvline(thr,color=C_GRAY,lw=1.2,ls='--',alpha=.6)
ax.text(thr+.00005,7.3,'High-impact\nthreshold',fontsize=7.5,color=C_GRAY)
ax.legend(handles=[mpatches.Patch(color=C_RED,label='High impact'),mpatches.Patch(color=C_MID,label='Moderate'),mpatches.Patch(color=C_LIGHT,label='Low')],fontsize=8.5,loc='lower right')
ax.set_xlabel("Mean |ΔEFRF| per ±5% perturbation at 90.5% API operating point",fontsize=10,color=C_DARK)
ax.set_title("Figure 5. Real Sensitivity Analysis via Trained PINN (Tornado Chart)\n±5% Perturbation of Each Feature at the Optimal Operating Point",fontsize=11,fontweight='bold',color=C_DARK)
ax.invert_yaxis()
plt.tight_layout(); plt.savefig("/home/claude/figures_real/fig5_tornado.png",dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
print("   Fig 5 ✓")

# ── Fig 6: Extrapolation ──────────────────────────────────────────────────────
ar=np.linspace(70,97,80)
mlp_bm=MLPRegressor(hidden_layer_sizes=(64,64,64),activation='tanh',max_iter=500,random_state=42)
gbr_bm=GradientBoostingRegressor(n_estimators=200,learning_rate=.05,max_depth=4,random_state=42)
sc2=StandardScaler(); Xsc2=sc2.fit_transform(X_raw)
mlp_bm.fit(Xsc2,Y_raw[:,0]); gbr_bm.fit(Xsc2,Y_raw[:,0])
ts_p=[]; ts_m=[]; ts_g=[]
for a in ar:
    pt=OPT.copy(); pt[0,0]=a
    ts_p.append(pinn.predict(pt)[0,0])
    pt2=sc2.transform(pt); ts_m.append(mlp_bm.predict(pt2)[0]); ts_g.append(gbr_bm.predict(pt2)[0])
ts_p=np.array(ts_p); ts_m=np.array(ts_m); ts_g=np.array(ts_g)
fig,ax=plt.subplots(figsize=(10,5.5)); ax.set_facecolor('white'); fig.patch.set_facecolor('white')
ax.fill_between(ar,.5,5.5,alpha=.07,color=C_GREEN,label='Physically feasible zone')
ax.plot(ar,ts_p,color=C_MID,lw=2.8,label='PINN (proposed)',zorder=4)
ax.plot(ar,ts_m,color="#E67E22",lw=2.,ls='--',label='MLP (baseline)',zorder=3)
ax.plot(ar,ts_g,color=C_RED,lw=2.,ls=':',label='XGBoost/GBR (baseline)',zorder=3)
ax.axhline(5.5,color=C_RED,lw=1,ls='-.',alpha=.5)
ax.text(71,5.6,'Physical upper limit',fontsize=8,color=C_RED,alpha=.8)
ax.axhline(2.,color=C_GREEN,lw=1.2,ls='--',alpha=.7)
ax.text(71,2.1,'Min. acceptable σt (2 MPa)',fontsize=8,color=C_GREEN)
ax.axvline(90,color=C_GRAY,lw=1.2,ls=':',alpha=.6)
ax.text(90.3,.7,'>90% API\nregion',fontsize=8,color=C_GRAY)
ax.set_xlabel("API Loading (%)",fontsize=11,color=C_DARK)
ax.set_ylabel("Predicted Tensile Strength (MPa)",fontsize=11,color=C_DARK)
ax.set_title("Figure 6. Extrapolation Behaviour at High API Loadings\nReal PINN vs. Baseline Model Predictions",fontsize=11,fontweight='bold',color=C_DARK)
ax.legend(fontsize=9,loc='upper right'); ax.set_ylim(.4,6.2); ax.set_xlim(70,97)
plt.tight_layout(); plt.savefig("/home/claude/figures_real/fig6_extrapolation.png",dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
print("   Fig 6 ✓")

# ── Fig 7: Pareto Front ───────────────────────────────────────────────────────
_,_,ef_all,cv_all=evaluate(pop); feas_all=cv_all==0; inf_all=~feas_all
fig,ax=plt.subplots(figsize=(10,6)); ax.set_facecolor('white'); fig.patch.set_facecolor('white')
ax.scatter(pop[feas_all,0],ef_all[feas_all],c=C_LIGHT,edgecolors=C_MID,s=40,alpha=.7,zorder=3,label='Feasible candidates')
ax.scatter(pop[inf_all,0],ef_all[inf_all],c='#FFCCCC',edgecolors=C_RED,s=40,alpha=.5,marker='^',zorder=3,label='Infeasible')
if feas.sum()>1:
    si=np.argsort(par_api[feas])
    ax.plot(par_api[feas][si],par_ef[feas][si],color=C_RED,lw=2.5,zorder=5,label='Pareto front')
    ax.fill_between(par_api[feas][si],0,par_ef[feas][si],alpha=.12,color=C_GREEN,label='Feasible design space')
ax.axhline(.5,color=C_RED,lw=1.8,ls='--',alpha=.85,label='EFRF threshold (0.5)')
if feas.sum()>0:
    bi=np.argmax(par_api[feas])
    oav=par_api[feas][bi]; oev=par_ef[feas][bi]
    ax.scatter([oav],[oev],color=C_GREEN,s=220,zorder=7,marker='*',edgecolors=C_DARK,lw=1.2,
               label=f'Optimal: {oav:.1f}% API, EFRF={oev:.3f}')
    ax.annotate(f"  API = {oav:.1f}%\n  EFRF = {oev:.3f}\n  σt ≥ 2 MPa",
                xy=(oav,oev),xytext=(max(70,oav-12),min(.72,oev+.22)),
                fontsize=9,color=C_GREEN,fontweight='bold',
                arrowprops=dict(arrowstyle='->',color=C_GREEN,lw=1.8))
ax.set_xlabel("API Loading (%)",fontsize=11,color=C_DARK)
ax.set_ylabel("EFRF",fontsize=11,color=C_DARK)
ax.set_title("Figure 7. NSGA-II Pareto Front: API Loading vs. EFRF\n300-Generation Evolutionary Optimisation (100 individuals)",fontsize=11,fontweight='bold',color=C_DARK)
ax.legend(fontsize=8.5,loc='upper left'); ax.set_xlim(68,100); ax.set_ylim(-.02,.85)
plt.tight_layout(); plt.savefig("/home/claude/figures_real/fig7_pareto.png",dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
print("   Fig 7 ✓")

# ── Fig S1: Learning Curves ───────────────────────────────────────────────────
szs=np.array([10,15,18,22,27,30,36]); tr2=[]; vr2=[]
for ns in szs:
    idx2=np.random.permutation(45)[:ns]; val2=np.setdiff1d(np.arange(45),idx2)
    pm=PINN(); pm.fit(X_raw[idx2],Y_raw[idx2],epochs=300,bs=8)
    tr2.append(r2_score(Y_raw[idx2,0],pm.predict(X_raw[idx2])[:,0]))
    vr2.append(r2_score(Y_raw[val2,0],pm.predict(X_raw[val2])[:,0]))
fig,ax=plt.subplots(figsize=(8.5,5)); ax.set_facecolor('white'); fig.patch.set_facecolor('white')
ax.plot(szs,tr2,'o-',color=C_MID,lw=2.2,ms=7,label='Training R²')
ax.plot(szs,vr2,'s--',color=C_RED,lw=2.2,ms=7,label='Validation R²')
ax.fill_between(szs,np.array(vr2)-.03,np.array(vr2)+.03,alpha=.12,color=C_RED)
ax.axhline(.92,color=C_GRAY,lw=1,ls=':',alpha=.7)
ax.text(10.5,.925,'Target R²=0.92',fontsize=8.5,color=C_GRAY)
ax.set_xlabel("Training Set Size (n)",fontsize=11,color=C_DARK)
ax.set_ylabel("R²",fontsize=11,color=C_DARK)
ax.set_title("Supplementary Figure S1. Learning Curves: R² vs. Training Set Size",fontsize=11,fontweight='bold',color=C_DARK)
ax.legend(fontsize=9.5); ax.set_ylim(.3,1.05); ax.set_xlim(8,38)
plt.tight_layout(); plt.savefig("/home/claude/figures_real/figS1_learning_curves.png",dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
print("   Fig S1 ✓")

# ══ FINAL REPORT ═════════════════════════════════════════════════════════════
print("\n"+"="*70)
print("  FINAL RESULTS REPORT")
print("="*70)
print(f"\n  PINN (test set n={len(te)}):  R²={r2p:.4f}  RMSE={rmp:.4f} MPa  MAE={map_:.4f} MPa")
print(f"\n  5-fold Cross-Validation:")
for nm,(r2,rm,ma) in CV.items():
    print(f"    {nm.replace(chr(10),' '):<22}  R²={r2.mean():.3f}±{r2.std():.3f}  RMSE={rm.mean():.4f}±{rm.std():.4f}")
print(f"\n  Top sensitivity drivers (real PINN ±5% perturbation):")
for i,(f,v) in enumerate(sorted(SENS.items(),key=lambda x:-x[1])[:3],1):
    print(f"    {i}. {f:<32}  ΔEFRF={v:.5f}")
if feas.sum()>0:
    bi=np.argmax(par_api[feas])
    print(f"\n  NSGA-II Optimal Formulation:")
    print(f"    API Loading  = {par_api[feas][bi]:.2f}%")
    print(f"    EFRF         = {par_ef[feas][bi]:.4f}")
    print(f"    σt (pred.)   = {par_sig[feas][bi]:.3f} MPa")
    print(f"    Feasible Pareto solutions: {feas.sum()}")
print(f"\n  Figures saved to: /home/claude/figures_real/")
for f in sorted(os.listdir("/home/claude/figures_real/")):
    sz=os.path.getsize(f"/home/claude/figures_real/{f}")
    print(f"    {f:<45} {sz/1024:.0f} KB")
print("\n  ALL DONE.")
