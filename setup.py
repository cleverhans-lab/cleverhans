from setuptools import setup
from setuptools import find_packages


setup(name='cleverhans',
      version='1.0.0',
      url='https://github.com/openai/cleverhans',
      license='MIT',
      install_requires=['keras', 'nose', 'pycodestyle', 'theano'],
      packages=find_packages())
