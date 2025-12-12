from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requuirements = f.read().splitlines()

setup(
    name='EEG-Hybrid-Attention-Project',
    version='0.1',
    author='Jeffrey Voke Ojuederhie',
    packages=find_packages(),
    install_requires=requuirements
)
