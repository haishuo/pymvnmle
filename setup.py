from setuptools import setup, find_packages

setup(
    name="pymvnmle",
    version="0.1.0",
    description="ML Estimation for Multivariate Normal Data with Missing Values",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0"
    ],
    python_requires=">=3.8",
)