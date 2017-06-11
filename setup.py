from setuptools import setup
from setuptools import find_packages


setup(name='cleverhans',
      version='1.0.0',
      url='https://github.com/tensorflow/cleverhans',
      license='MIT',
      install_requires=[
          'nose',
          'pycodestyle',
          'scipy',
          'tensorflow',
          'matplotlib'],
      packages=find_packages())
