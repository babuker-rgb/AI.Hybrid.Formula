import matplotlib.pyplot as plt

def plot_pareto(opt_df, path="pareto_front.png"):
    plt.figure(figsize=(7, 5))
    feasible = opt_df[opt_df["Feasible"] == True]
    infeasible = opt_df[opt_df["Feasible"] == False]

    if len(infeasible) > 0:
        plt.scatter(
            infeasible["Objective_API"],
            infeasible["Objective_EFRF"],
            c="gray", s=16, alpha=0.5, label="Infeasible"
        )
    if len(feasible) > 0:
        plt.scatter(
            feasible["Objective_API"],
            feasible["Objective_EFRF"],
            c="green", s=20, alpha=0.8, label="Feasible"
        )

    plt.xlabel("API Loading (%)")
    plt.ylabel("EFRF")
    plt.title("NSGA-II Pareto Solutions")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_loss_curves(loss_history, path="loss_curves.png"):
    plt.figure(figsize=(7, 5))
    plt.plot(loss_history["train"], label="Train Loss")
    plt.plot(loss_history["val"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN Training Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_predicted_vs_actual(y_true, y_pred, output_names, path="prediction_plot.png"):
    n = len(output_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.scatter(y_true[:, i], y_pred[:, i], s=12, alpha=0.7)
        mn = min(y_true[:, i].min(), y_pred[:, i].min())
        mx = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([mn, mx], [mn, mx], "r--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(output_names[i])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
