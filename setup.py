from setuptools import find_packages
from setuptools import setup

setup(
    name="cleverhans",
    version="4.0.0",
    url="https://github.com/cleverhans-lab/cleverhans",
    license="MIT",
    install_requires=[
        "nose",
        "pycodestyle",
        "scipy",
        "matplotlib",
        "mnist",
        "numpy",
        "tensorflow-probability",
        "joblib",
        "easydict",
        "absl-py",
        "six",
    ],
    extras_require={
        "jax": ["jax>=0.2.9", "jaxlib"],
        "tf": ["tensorflow>=2.4.0", "tensorflow-probability", "tensorflow-datasets"],
        "pytorch": ["torch>=1.7.0", "torchvision>=0.8.0"],
    },
    packages=find_packages(),
)
