from setuptools import setup, find_packages

setup(
    name="meridian",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.1.1",
        "tensorflow-cpu>=2.13.0",
        "tensorflow-probability>=0.21.0",
        "scipy>=1.11.0",
        "xarray>=2023.1.0",
        "h5py>=3.9.0",
        "arviz>=0.15.0"
    ],
    python_requires=">=3.10",
)
