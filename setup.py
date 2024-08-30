from setuptools import setup, find_packages

setup(name='vine',
      packages=find_packages(), 
      install_requires=[
            'numpy',
            'torch',
            'transformers'
      ])