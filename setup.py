from setuptools import find_packages, setup

# Base requirements for the custom transformers model
base_requirements = [
    "hivemind @ git+https://github.com/learning-at-home/hivemind.git@213bff98a62accb91f254e2afdccbf1d69ebdea9",
    "torch",
    "tokenmonster",
    "transformers",
]

# Additional requirements for training and orchestration
host_requirements = [
    "asciichartpy",
    "blessed",
    "datasets",
    "flask",
    "lightning",
    "pytorch_optimizer @ git+https://github.com/kozistr/pytorch_optimizer.git",
    "werkzeug",
]

setup(
    name="praxis",
    packages=find_packages(),
    install_requires=base_requirements,
    extras_require={
        "all": base_requirements + host_requirements,
    },
)
