from setuptools import setup, find_packages

setup(name='thorns',
      packages=find_packages(), 
      install_requires=[
            'numpy',
            'torch',
            'transformers'
      ])