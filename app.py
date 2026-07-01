import streamlit as st
import pandas as pd
from pathlib import Path

from pinn_model import generate_pinn_data, PINNTrainer, FEATURE_NAMES, OUTPUT_NAMES
from nsga2_optimizer import run_nsga2
from plots import plot_pareto, plot_loss_curves, plot_predicted_vs_actual
from report_generator import generate_pdf_report

st.set_page_config(page_title="PINN + NSGA-II Tablet Optimization", layout="wide")

st.title("Hybrid AI Framework: PINN + NSGA-II for Tablet Optimization")

with st.sidebar:
    st.header("Run Settings")
    n_samples = st.number_input("Samples", 1000, 20000, 6000, 500)
    epochs = st.number_input("Epochs", 50, 2000, 500, 50)
    patience = st.number_input("Patience", 10, 300, 60, 10)
    pop_size = st.number_input("NSGA-II Population", 20, 300, 100, 10)
    n_gen = st.number_input("NSGA-II Generations", 10, 300, 80, 10)
    run_button = st.button("Run Optimization")

if run_button:
    with st.spinner("Generating data and training PINN..."):
        df = generate_pinn_data(n_samples=n_samples)
        trainer = PINNTrainer()
        trainer.fit(df, epochs=epochs, patience=patience)

        metrics_df = trainer.evaluate(df)
        X_raw = df[FEATURE_NAMES].values.astype(float)
        y_true = df[OUTPUT_NAMES].values.astype(float)
        y_pred = trainer.predict(X_raw)

        with st.spinner("Running NSGA-II optimization..."):
            opt_df, best_formulation, res = run_nsga2(
                trainer,
                pop_size=pop_size,
                n_gen=n_gen
            )

        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)

        metrics_df.to_csv(out_dir / "metrics.csv", index=False)
        opt_df.to_csv(out_dir / "nsga2_pareto_solutions.csv", index=False)

        plot_pareto(opt_df, out_dir / "pareto_front.png")
        plot_loss_curves(trainer.loss_history, out_dir / "loss_curves.png")
        plot_predicted_vs_actual(y_true, y_pred, OUTPUT_NAMES, out_dir / "prediction_plot.png")

        generate_pdf_report(
            out_path=str(out_dir / "report.pdf"),
            best_formulation=best_formulation,
            metrics_df=metrics_df,
            opt_df=opt_df
        )

    st.success("Optimization completed successfully.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Model Performance")
        st.dataframe(metrics_df, use_container_width=True)

    with c2:
        st.subheader("Best Formulation")
        st.dataframe(pd.DataFrame([best_formulation]), use_container_width=True)

    st.subheader("Pareto Solutions")
    st.dataframe(opt_df.head(20), use_container_width=True)

    st.subheader("Pareto Front")
    st.image(str(out_dir / "pareto_front.png"))

    st.subheader("Loss Curves")
    st.image(str(out_dir / "loss_curves.png"))

    st.subheader("Prediction Plot")
    st.image(str(out_dir / "prediction_plot.png"))

    st.download_button(
        "Download PDF Report",
        data=open(out_dir / "report.pdf", "rb").read(),
        file_name="report.pdf",
        mime="application/pdf"
    )
