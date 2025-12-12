from eeg_hybrid.datasets.bciciv2a import load_bciciv2a_all
from train.loso_trainer import run_loso
from eeg_hybrid.models.eegnet import EEGNet


def main():

    data_dir = "data/raw/bciciv2a"
    results_dir = "eeg_hybrid/results"

    X, y, subjects = load_bciciv2a_all(data_dir)

    n_channels = X.shape[2]
    n_times = X.shape[3]
    n_classes = len(set(y))

    def model_fn():
        return EEGNet(n_channels, n_times, n_classes)

    run_loso(
        X=X,
        y=y,
        subjects=subjects,
        model_fn=model_fn,
        results_dir=results_dir,
        epochs=120,
        batch_size=64,
        lr=1e-3
    )


if __name__ == "__main__":
    main()
