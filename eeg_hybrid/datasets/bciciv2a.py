import os
import numpy as np
import mne


def load_bciciv2a_subject(data_dir, subject_id,
                          tmin=0, tmax=4, l_freq=4, h_freq=38):

    fname = f"A{subject_id:02d}T.gdf"
    path = os.path.join(data_dir, fname)
    raw = mne.io.read_raw_gdf(path, preload=True)

    raw.drop_channels([ch for ch in raw.ch_names if "EOG" in ch.upper()])
    raw.filter(l_freq, h_freq)

    event_id = dict(left=769, right=770, feet=771, tongue=772)
    events, _ = mne.events_from_annotations(raw, event_id=event_id)

    epochs = mne.Epochs(raw, events, event_id,
                        tmin=tmin, tmax=tmax,
                        baseline=None, preload=True)

    X = epochs.get_data()
    labels = epochs.events[:, 2]

    mapping = {769: 0, 770: 1, 771: 2, 772: 3}
    y = np.array([mapping[int(i)] for i in labels])

    return X, y


def load_bciciv2a_all(data_dir):

    Xs, ys, subjects = [], [], []

    for sid in range(1, 10):
        X, y = load_bciciv2a_subject(data_dir, sid)
        Xs.append(X)
        ys.append(y)
        subjects.append(np.full(len(y), sid))

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    subjects = np.concatenate(subjects)

    X = X[:, None, :, :]  # (N,1,C,T)

    return X, y, subjects
