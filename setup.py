from setuptools import setup
from setuptools import find_packages


setup(name='cleverhans',
      version='1.0.0',
      url='https://github.com/openai/cleverhans',
      license='MIT',
      install_requires=[
          'keras==1.2',
          'nose',
          'pycodestyle',
          'theano',
          'scipy',
          'tensorflow',
          'matplotlib'],
      packages=find_packages())
