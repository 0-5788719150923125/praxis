from setuptools import find_packages, setup

# Base requirements for the custom transformers model
base_requirements = [
    "hivemind @ git+https://github.com/learning-at-home/hivemind.git@213bff98a62accb91f254e2afdccbf1d69ebdea9",
    "torch",
    "transformers",
]

# Additional requirements for training and orchestration
host_requirements = [
    "asciichartpy",
    "blessed",
    "datasets",
    "flask",
    "lightning",
    "lm_eval",
    "pytorch_optimizer @ git+https://github.com/kozistr/pytorch_optimizer.git",
    "tokenmonster",
    "werkzeug",
]

# Requirements primarily for IDE-related tooling
dev_requirements = ["attrs", "isort", "lsprotocol", "pygls"]

setup(
    name="praxis",
    version="0.1.0",
    decription="as above, so below",
    author="Ryan Brooks",
    author_email="ryan@src.eco",
    url="https://src.eco",
    packages=find_packages(),
    license="MIT",
    install_requires=base_requirements,
    extras_require={
        "all": base_requirements + host_requirements + dev_requirements,
        "dev": dev_requirements,
    },
)
