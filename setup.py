from setuptools import find_packages
from setuptools import setup

setup(name='cleverhans',
      version='2.1.0',
      url='https://github.com/tensorflow/cleverhans',
      license='MIT',
      install_requires=[
          'nose',
          'pycodestyle',
          'scipy',
          'matplotlib',
          "mnist ~= 0.2",
          "numpy",
      ],
      # Explicit dependence on TensorFlow is not supported.
      # See https://github.com/tensorflow/tensorflow/issues/7166
      extras_require={
          "tf": ["tensorflow>=1.0.0"],
          "tf_gpu": ["tensorflow-gpu>=1.0.0"],
          "test": [
              "keras == 2.1.5",  # Keras 2.1.6 is incompatible with TF 1.4
          ],
      },
      packages=find_packages())
