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
          'matplotlib'],
      # Explicit dependence on TensorFlow is not supported.
      # See https://github.com/tensorflow/tensorflow/issues/7166
      extras_require={
        "tf": ["tensorflow>=1.0.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0.0"],
      },
      packages=find_packages())
