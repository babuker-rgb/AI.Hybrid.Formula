import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pinn_model import FEATURE_NAMES, TENSILE_MIN, EFRF_MAX, MCC_MAX, BINDER_MIN, BINDER_MAX

BOUNDS = np.array([
    [85.0, 95.0],
    [0.0, MCC_MAX],
    [0.5, 6.0],
    [0.01, 1.2],
    [BINDER_MIN, BINDER_MAX],
    [80.0, 300.0],
    [1.0, 50.0],
    [30.0, 250.0],
], dtype=float)

class TabletOptimizationProblem(Problem):
    def __init__(self, trainer):
        super().__init__(
            n_var=8,
            n_obj=2,
            n_constr=2,
            xl=BOUNDS[:, 0],
            xu=BOUNDS[:, 1]
        )
        self.trainer = trainer

    def repair(self, X):
        Xr = X.copy()
        Xr[:, 0] = np.clip(Xr[:, 0], 85, 95)
        Xr[:, 1] = np.clip(Xr[:, 1], 0, MCC_MAX)
        Xr[:, 2] = np.clip(Xr[:, 2], 0.5, 6.0)
        Xr[:, 3] = np.clip(Xr[:, 3], 0.01, 1.2)
        Xr[:, 4] = np.clip(Xr[:, 4], BINDER_MIN, BINDER_MAX)
        Xr[:, 5] = np.clip(Xr[:, 5], 80, 300)
        Xr[:, 6] = np.clip(Xr[:, 6], 1, 50)
        Xr[:, 7] = np.clip(Xr[:, 7], 30, 250)

        used = Xr[:, 0] + Xr[:, 1] + Xr[:, 2] + Xr[:, 3] + Xr[:, 4]
        over = used > 100
        if np.any(over):
            scale = 100.0 / used[over]
            Xr[over, 0] *= scale
            Xr[over, 1] *= scale
            Xr[over, 2] *= scale
            Xr[over, 3] *= scale
            Xr[over, 4] *= scale

        under = used < 100
        if np.any(under):
            remainder = 100 - used[under]
            add_mcc = np.minimum(remainder, MCC_MAX - Xr[under, 1])
            Xr[under, 1] += add_mcc
            rem2 = remainder - add_mcc
            Xr[under, 0] = np.clip(Xr[under, 0] + rem2, 85, 95)

        return Xr

    def _evaluate(self, X, out, *args, **kwargs):
        Xr = self.repair(X)
        pred = self.trainer.predict(Xr)

        tensile = pred[:, 1]
        er = pred[:, 2]
        efrf = er / (tensile + 1e-8)

        f1 = -Xr[:, 0]
        f2 = efrf

        g1 = TENSILE_MIN - tensile
        g2 = efrf - EFRF_MAX

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])

def run_nsga2(trainer, pop_size=100, n_gen=80, seed=42):
    problem = TabletOptimizationProblem(trainer)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", n_gen)
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=True,
        save_history=False
    )

    X_opt = problem.repair(res.X)
    F_opt = res.F

    pred = trainer.predict(X_opt)
    efrf = pred[:, 2] / (pred[:, 1] + 1e-8)

    opt_df = pd.DataFrame(X_opt, columns=FEATURE_NAMES)
    opt_df["Objective_API"] = -F_opt[:, 0]
    opt_df["Objective_EFRF"] = F_opt[:, 1]
    opt_df["Pred_Density"] = pred[:, 0]
    opt_df["Pred_Tensile"] = pred[:, 1]
    opt_df["Pred_ER"] = pred[:, 2]
    opt_df["Pred_EFRF"] = efrf
    opt_df["Feasible"] = (
        (opt_df["Pred_Tensile"] >= TENSILE_MIN) &
        (opt_df["Pred_EFRF"] <= EFRF_MAX)
    )

    feasible_df = opt_df[opt_df["Feasible"]].copy()
    if len(feasible_df) > 0:
        best_idx = feasible_df["Pred_EFRF"].idxmin()
        best_formulation = feasible_df.loc[best_idx]
    else:
        best_formulation = opt_df.iloc[opt_df["Pred_EFRF"].idxmin()]

    return opt_df, best_formulation, res
