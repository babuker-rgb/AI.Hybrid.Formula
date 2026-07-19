"""
Hubryd AI – v29.27-R11 (Fixed Toggle Keys)
Hybrid AI for Multi-Objective Optimization of Tablet Formulation
- Fixed: toggles now use consistent session-state keys.
- All UI labels in English.
Nile Valley University · Sudan
"""

# ... (all imports and constants remain exactly as in v29.27-R10, up to the Streamlit UI section)

# ================================================================
# Streamlit UI
# ================================================================
st.set_page_config(page_title="Hybrid AI for Multi-Objective Optimization", layout="wide")

st.markdown("""
<div style="background: #0b1a33; padding:1rem; border-radius:0.5rem; text-align:center; margin-bottom:1rem;">
    <h2 style="color:#fff; margin:0;">🧬 Hybrid AI · Multi‑Objective Tablet Optimization</h2>
    <p style="color:#64ffda; margin:0;">PINN + NSGA‑II | Nile Valley University, Sudan</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📚 Physics Constraints")
    st.markdown(f"""
    ✅ **API:** {API_MIN:.0f}–{API_MAX:.0f}%  
    ✅ **Density:** {D_MIN:.2f}–{D_MAX:.2f}  
    ✅ **Tensile:** ≥ {TENSILE_MIN:.2f} MPa  
    ✅ **EFRF:** &lt; 0.40 (feasible)  
    ✅ **MCC:** ≤ {MCC_MAX:.1f}%  
    ✅ **Pressure:** ≤ {PRESSURE_MAX:.0f} MPa  
    ✅ **NSGA‑II:** Pop=80, Gen=50
    """)
    st.caption("🔬 v29.27-R11 — Fixed Toggle Keys")

# ... (load model, left column sliders unchanged)

# ================================================================
# RIGHT COLUMN – Results
# ================================================================
with col_right:
    st.markdown("### 📈 Results")
    # ... (all prediction and NSGA-II code remains the same, up to the point where results are displayed)

    if st.session_state.run_optimized:
        # ... (Pareto Front and Golden Solution unchanged)

        st.markdown("---")

        # ---- 3. Toggle: Cost-wise solution (FIXED) ----
        st.toggle(
            "💰 Cost-wise Solution (Max API, Min Pressure)",
            value=st.session_state.get("show_cost_solution", False),
            key="show_cost_solution"
        )
        if st.session_state.show_cost_solution and cost_solution is not None:
            st.markdown("#### 💰 Cost-Optimised Formulation (Max API, Min Pressure)")
            d, t, e, ef = predict_pinn(model, scaler, y_scaler, cost_solution)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Formulation:**")
                st.write(f"API: {cost_solution[0]:.1f}%")
                st.write(f"MCC: {cost_solution[1]:.1f}%")
                st.write(f"PVPP: {cost_solution[2]:.1f}%")
                st.write(f"Mg-St: {cost_solution[3]:.2f}%")
                st.write(f"Binder: {cost_solution[4]:.1f}%")
            with col2:
                st.write("**Process & CQAs:**")
                st.write(f"Pressure: {cost_solution[5]:.1f} MPa")
                st.write(f"Speed: {cost_solution[6]:.1f} rpm")
                st.write(f"Granule: {cost_solution[7]:.0f} µm")
                st.write(f"Density: {d:.3f}")
                st.write(f"Tensile: {t:.3f} MPa")
                st.write(f"EFRF: {ef:.4f}")
            st.session_state.cost_pred = (d, t, e, ef)

        # ---- 4. Toggle: Quality-wise solution (FIXED) ----
        st.toggle(
            "🏆 Quality-wise Solution (Max Tensile Strength)",
            value=st.session_state.get("show_quality_solution", False),
            key="show_quality_solution"
        )
        if st.session_state.show_quality_solution and quality_solution is not None:
            st.markdown("#### 🏆 Quality-Optimised Formulation (Max Tensile Strength)")
            d, t, e, ef = predict_pinn(model, scaler, y_scaler, quality_solution)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Formulation:**")
                st.write(f"API: {quality_solution[0]:.1f}%")
                st.write(f"MCC: {quality_solution[1]:.1f}%")
                st.write(f"PVPP: {quality_solution[2]:.1f}%")
                st.write(f"Mg-St: {quality_solution[3]:.2f}%")
                st.write(f"Binder: {quality_solution[4]:.1f}%")
            with col2:
                st.write("**Process & CQAs:**")
                st.write(f"Pressure: {quality_solution[5]:.1f} MPa")
                st.write(f"Speed: {quality_solution[6]:.1f} rpm")
                st.write(f"Granule: {quality_solution[7]:.0f} µm")
                st.write(f"Density: {d:.3f}")
                st.write(f"Tensile: {t:.3f} MPa")
                st.write(f"EFRF: {ef:.4f}")
            st.session_state.quality_pred = (d, t, e, ef)

        # ---- 5. Toggle: Model Comparison (FIXED) ----
        st.toggle(
            "📊 Model Comparison",
            value=st.session_state.get("show_comparison", True),
            key="show_comparison"
        )
        if st.session_state.show_comparison:
            st.markdown("### 📊 Model Comparison (Tensile R²)")
            bench_df, chart_data = run_model_comparison(model, scaler, y_scaler, features, df, device)
            st.session_state.benchmark_df = bench_df

            fig_bar = px.bar(pd.DataFrame(chart_data), x='Model', y='R² Score', color='Model',
                             title='Real R² Comparison (Tensile Strength)',
                             text=pd.DataFrame(chart_data)['R² Score'].round(3))
            fig_bar.update_layout(height=380, template='plotly_white')
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(bench_df, use_container_width=True)

        # ---- 6. Toggle: Sensitivity Analysis (FIXED) ----
        st.toggle(
            "🔬 Sensitivity Analysis",
            value=st.session_state.get("show_sensitivity", False),
            key="show_sensitivity"
        )
        if st.session_state.show_sensitivity:
            f = st.session_state.formulation
            if f['api_n'] is not None:
                st.markdown("### 🔬 Sensitivity Analysis – Parameter Impact on EFRF")
                fig_bars = plot_sensitivity_bars(f, model, scaler, y_scaler, efrf_max=0.40)
                if fig_bars:
                    st.plotly_chart(fig_bars, use_container_width=True)

        # ---- 7. Toggle: Particle Size Effect (FIXED) ----
        st.toggle(
            "📊 Particle Size Effect",
            value=st.session_state.get("show_particle_plot", False),
            key="show_particle_plot"
        )
        if st.session_state.show_particle_plot:
            f = st.session_state.formulation
            if f['api_n'] is not None:
                st.markdown("### 📊 Particle Size Effect with Pressure Variation")
                fig = plot_particle_pressure_density(f, model, scaler, y_scaler)
                st.plotly_chart(fig, use_container_width=True)

        # ---- 8. Report button (unchanged) ----
        generate_report_btn = st.button("📄 Generate Report (PDF)", key="knob_report")
        if generate_report_btn and st.session_state.benchmark_df is not None:
            # ... (PDF generation code unchanged)
            pass

    else:
        st.info("Adjust sliders and click '🔬 Predict & Optimise' to see results.")

st.caption("📧 Contact: babuker@protonmail.com | 🏛️ Nile Valley University, Sudan")
