from setuptools import setup, find_packages

setup(name='thorns',
      packages=find_packages(), 
      install_requires=[
            'hivemind @ git+https://github.com/learning-at-home/hivemind.git@213bff98a62accb91f254e2afdccbf1d69ebdea9',
            'numpy',
            'torch',
            'transformers'
      ])