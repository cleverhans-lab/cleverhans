from setuptools import find_packages
from setuptools import setup

setup(
    name="cleverhans",
    version="3.0.1",
    url="https://github.com/tensorflow/cleverhans",
    license="MIT",
    install_requires=[
        "nose",
        "pycodestyle",
        "scipy",
        "matplotlib",
        "mnist ~= 0.2",
        "numpy",
        "tensorflow-probability",
        "joblib",
    ],
    # Explicit dependence on TensorFlow is not supported.
    # See https://github.com/tensorflow/tensorflow/issues/7166
    extras_require={
        "tf": ["tensorflow>=1.0.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0.0"],
        "pytorch": ["torch>=1.1.0", "torchvision==0.3.0"],
    },
    packages=find_packages(),
)
