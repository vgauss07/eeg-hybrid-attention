import logging
import os

from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    'configs/bciciv2a_eegnet.yaml',
    'configs/bciciv2a_deepconvnet.yaml',
    'configs/things_eegnet.yaml',
    'configs/things_deepconvnet.yaml',
    'data/raw/data',
    'data/processed/data',
    'eeg_hybrid/__init__.py',
    'eeg_hybrid/datasets/bciciv2a.py',
    'eeg_hybrid/datasets/things.py',
    'eeg_hybrid/models/eegnet.py',
    'eeg_hybrid/models/deepconvnet.py',
    'eeg_hybrid/models/modules/se.py',
    'eeg_hybrid/models/modules/cbam.py',
    'eeg_hybrid/models/modules/mha.py',
    'eeg_hybrid/models/builders.py',
    'eeg_hybrid/train.py',
    'eeg_hybrid/eval.py',
    'eeg_hybrid/utils/metrics.py',
    'eeg_hybrid/utils/seed.py',
    'eeg_hybrid/utils/logging.py',
    'eeg_hybrid/utils/visualization.py',
    'eeg_hybrid/experiments/',
    'eeg_hybrid/results/logs',
    'eeg_hybrid/results/checkpoints',
    'eeg_hybrid/results/figures',
    'README.md'
]

for file_path in list_of_files:
    filepath = Path(file_path)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating Directory: {filedir} for the file {filename}')

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f'Creating empty file: {filepath}')
    else:
        logging.info(f'{filename} already exists')
