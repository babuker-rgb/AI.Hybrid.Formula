# ================================================================
# NSGA-II OPTIMIZATION
# ================================================================
st.markdown("### ⚙️ NSGA-II Results")

with st.spinner("🔄 Running NSGA-II optimisation..."):
    bounds = np.array([
        [85, 95], [0, 8], [1, 5], [0.2, 1.0], [0.5, 3.0],
        [100, 250], [10, 40], [50, 200]
    ])
    
    nsga = NSGAII(model, scaler, bounds, pop_size=100, n_generations=80)
    pop, objectives, constraints, fronts = nsga.run()
    
    # Extract Pareto front
    front0 = fronts[0]
    pareto_api = [-objectives[front0, 0]]
    pareto_efrf = objectives[front0, 1]
    
    # Find best feasible solution
    feasible = constraints[front0]
    feasible_api = [a for i, a in enumerate(pareto_api) if feasible[i]]
    feasible_efrf = [e for i, e in enumerate(pareto_efrf) if feasible[i]]
    
    # Ensure we have feasible solutions
    if len(feasible_api) > 0:
        # Find index of maximum API among feasible
        best_idx = int(np.argmax(feasible_api))  # Convert to int
        best_api = float(feasible_api[best_idx])
        best_efrf = float(feasible_efrf[best_idx])
        # Get tensile strength for that index
        tensile_values = nsga.tensile[front0][feasible]
        best_tensile = float(tensile_values[best_idx])
        st.success(f"Optimal Pareto solution: API = {best_api:.2f}% | EFRF = {best_efrf:.4f} | σt = {best_tensile:.3f} MPa | Feasible solutions: {len(feasible_api)}")
    else:
        # No feasible solutions: show best non-dominated (lowest EFRF)
        best_idx = int(np.argmin(pareto_efrf))  # or argmax API? we choose lowest EFRF
        best_api = float(pareto_api[best_idx])
        best_efrf = float(pareto_efrf[best_idx])
        best_tensile = float(nsga.tensile[front0][best_idx])
        st.warning(f"No feasible solutions found. Best non-dominated: API = {best_api:.2f}% | EFRF = {best_efrf:.4f} | σt = {best_tensile:.3f} MPa")
