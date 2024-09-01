from setuptools import setup, find_packages

# Base requirements for the custom transformers model
base_requirements = [
    "hivemind @ git+https://github.com/learning-at-home/hivemind.git@213bff98a62accb91f254e2afdccbf1d69ebdea9",
    "torch",
    "transformers",
]

# Additional requirements for training and orchestration
train_requirements = [
    "datasets",
    "lightning",
    "numpy",
]

setup(
    name="thorns",
    packages=find_packages(),
    install_requires=base_requirements,
    extras_require={
        "train": train_requirements,
        "all": base_requirements + train_requirements,
    },
)
