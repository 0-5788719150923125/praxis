from setuptools import find_packages, setup

# Base requirements for the custom transformers model
base_requirements = [
    "hivemind @ git+https://github.com/learning-at-home/hivemind.git@213bff98a62accb91f254e2afdccbf1d69ebdea9",
    "torch",
    "transformers",
]

# Additional requirements for training and orchestration
server_requirements = [
    "asciichartpy",
    "blessed",
    "datasets",
    "flask",
    "lightning",
    "numpy",
    "pytorch_optimizer",
    "werkzeug",
]

setup(
    name="praxis",
    packages=find_packages(),
    install_requires=base_requirements,
    extras_require={
        "serve": server_requirements,
        "all": base_requirements + server_requirements,
    },
)
