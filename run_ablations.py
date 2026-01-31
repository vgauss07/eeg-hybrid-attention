from pathlib import Path
import pandas as pd

from datasets.bciciv2a import load_bciciv2a_all
from datasets.things_eeg import load_things_eeg_npz

from models.eegnet import EEGNetAblation
from train.loso import run_loso_for_model
from viz.plots import plot_subject_bars, plot_model_means


def build_model_fn(n_channels, n_times, n_classes, attention: str | None):
    def _fn():
        return EEGNetAblation(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            attention=attention,   # None / "se" / "cbam" / "mha"
            F1=8, D=2, F2=16,
            kernel_length=64,
            dropout=0.25,
            mha_heads=4,
            mha_dropout=0.1
        )
    return _fn


def run_dataset_bciciv2a(results_root: Path):
    dataset_name = "bciciv2a"
    out_dir = results_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, subjects = load_bciciv2a_all(
        data_dir="data/bciciv2a",
        session_suffix="T",
        tmin=0.0,
        tmax=4.0,
        l_freq=4.0,
        h_freq=38.0,
    )
    n_channels = X.shape[2]
    n_times = X.shape[3]
    n_classes = len(set(y.tolist()))

    ablations = [
        ("cnn_only", None),
        ("cnn_se", "se"),
        ("cnn_cbam", "cbam"),
        ("cnn_mha", "mha"),
    ]

    all_folds = []
    summaries = []

    for model_name, attn in ablations:
        model_fn = build_model_fn(n_channels, n_times, n_classes, attn)
        df_folds, summary = run_loso_for_model(
            X=X, y=y, subjects=subjects,
            model_name=model_name,
            model_fn=model_fn,
            out_dir=out_dir,
            epochs=120,
            batch_size=32,
            lr=1e-3,
            val_ratio=0.15,
            seed=42,
            save_checkpoints=False
        )
        all_folds.append(df_folds)
        summaries.append(summary)

    all_folds_df = pd.concat(all_folds, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    (out_dir / "tables").mkdir(exist_ok=True)
    all_folds_df.to_csv(out_dir / "tables" / "all_models_folds.csv", index=False)
    summary_df.to_csv(out_dir / "tables" / "all_models_summary.csv", index=False)

    (out_dir / "figures").mkdir(exist_ok=True)
    plot_subject_bars(all_folds_df, out_dir / "figures" / "subject_bars.png",
                      title="BCICIV-2a LOSO: Test Accuracy per Subject")
    plot_model_means(all_folds_df, out_dir / "figures" / "model_means.png",
                     title="BCICIV-2a LOSO: Mean Test Accuracy by Ablation")

    print(f"\n✅ Done: {dataset_name} saved to {out_dir}")


def run_dataset_things_eeg(results_root: Path):
    """
    End-to-end if you provide:
      data/things_eeg/things_eeg_preprocessed.npz
    with keys: X (N,1,C,T), y (N,), subjects (N,)
    """
    dataset_name = "things_eeg"
    out_dir = results_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, subjects = load_things_eeg_npz("data/things_eeg/things_eeg_preprocessed.npz")
    n_channels = X.shape[2]
    n_times = X.shape[3]
    n_classes = len(set(y.tolist()))

    ablations = [
        ("cnn_only", None),
        ("cnn_se", "se"),
        ("cnn_cbam", "cbam"),
        ("cnn_mha", "mha"),
    ]

    all_folds = []
    summaries = []

    for model_name, attn in ablations:
        model_fn = build_model_fn(n_channels, n_times, n_classes, attn)
        df_folds, summary = run_loso_for_model(
            X=X, y=y, subjects=subjects,
            model_name=model_name,
            model_fn=model_fn,
            out_dir=out_dir,
            epochs=120,
            batch_size=32,
            lr=1e-3,
            val_ratio=0.15,
            seed=42,
            save_checkpoints=False
        )
        all_folds.append(df_folds)
        summaries.append(summary)

    all_folds_df = pd.concat(all_folds, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    (out_dir / "tables").mkdir(exist_ok=True)
    all_folds_df.to_csv(out_dir / "tables" / "all_models_folds.csv", index=False)
    summary_df.to_csv(out_dir / "tables" / "all_models_summary.csv", index=False)

    (out_dir / "figures").mkdir(exist_ok=True)
    plot_subject_bars(all_folds_df, out_dir / "figures" / "subject_bars.png",
                      title="THINGS-EEG LOSO: Test Accuracy per Subject")
    plot_model_means(all_folds_df, out_dir / "figures" / "model_means.png",
                     title="THINGS-EEG LOSO: Mean Test Accuracy by Ablation")

    print(f"\n✅ Done: {dataset_name} saved to {out_dir}")


def main():
    results_root = Path("results")
    results_root.mkdir(exist_ok=True)

    # Run per-dataset
    run_dataset_bciciv2a(results_root)

    # THINGS-EEG is optional until you generate the preprocessed NPZ
    # run_dataset_things_eeg(results_root)


if __name__ == "__main__":
    main()
