from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_subject_bars(all_folds_df: pd.DataFrame, out_path: Path, title: str):
    """
    Bar chart: x=subject, bars grouped by model, y=test_acc
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # pivot to subjects x models
    pivot = all_folds_df.pivot_table(index="subject", columns="model",
                                     values="test_acc")
    ax = pivot.plot(kind="bar")
    ax.set_xlabel("Subject (LOSO test subject)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_model_means(all_folds_df: pd.DataFrame, out_path: Path, title: str):
    """
    Bar chart of mean test accuracy per model (with std error bars).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    g = all_folds_df.groupby("model")["test_acc"]
    means = g.mean()
    stds = g.std(ddof=0)

    ax = means.plot(kind="bar", yerr=stds)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Test Accuracy")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
