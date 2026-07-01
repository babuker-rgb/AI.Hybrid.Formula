from fpdf import FPDF

def sanitize_text(text):
    replacements = {
        "σ": "sigma",
        "µ": "um",
        "≥": ">=",
        "≤": "<=",
        "✅": "[PASS]",
        "❌": "[FAIL]",
        "⚠️": "[WARNING]"
    }
    for old, new in replacements.items():
        text = str(text).replace(old, new)
    return text

def generate_pdf_report(out_path, best_formulation, metrics_df, opt_df,
                        title="Hybrid AI Framework (PINN-NSGA-II) Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, sanitize_text(title), ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "1. Best Formulation", ln=True)
    pdf.set_font("Arial", "", 10)
    for col in best_formulation.index:
        pdf.cell(0, 6, sanitize_text(f"{col}: {best_formulation[col]}"), ln=True)

    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "2. Model Performance", ln=True)

    pdf.set_font("Arial", "B", 9)
    cols = ["Output", "R2", "RMSE", "MAE", "PearsonR"]
    widths = [40, 25, 25, 25, 25]
    for c, w in zip(cols, widths):
        pdf.cell(w, 7, c, 1, 0, "C")
    pdf.ln()

    pdf.set_font("Arial", "", 9)
    for _, row in metrics_df.iterrows():
        pdf.cell(40, 7, sanitize_text(row["Output"]), 1)
        pdf.cell(25, 7, f'{row["R2"]:.4f}', 1, 0, "C")
        pdf.cell(25, 7, f'{row["RMSE"]:.4f}', 1, 0, "C")
        pdf.cell(25, 7, f'{row["MAE"]:.4f}', 1, 0, "C")
        pdf.cell(25, 7, f'{row["PearsonR"]:.4f}', 1, 0, "C")
        pdf.ln()

    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "3. Pareto Best Solutions", ln=True)

    subset = opt_df[["API_%", "MCC_%", "PVPP_%", "MgSt_%", "Binder_%", "Pred_Tensile", "Pred_EFRF", "Feasible"]].head(10)
    pdf.set_font("Arial", "B", 8)
    widths = [16, 16, 16, 16, 16, 20, 20, 15]
    for c, w in zip(subset.columns, widths):
        pdf.cell(w, 7, sanitize_text(c), 1, 0, "C")
    pdf.ln()

    pdf.set_font("Arial", "", 8)
    for _, row in subset.iterrows():
        for c, w in zip(subset.columns, widths):
            pdf.cell(w, 7, sanitize_text(row[c]), 1, 0, "C")
        pdf.ln()

    pdf.output(out_path)
