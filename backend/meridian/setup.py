from setuptools import setup, find_packages

setup(
    name="meridian",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0.2",
        "pandas>=2.3.0",
        "tensorflow>=2.18.1",
        "tensorflow-probability>=0.25.0",
        "scipy>=1.15.3",
        "xarray>=2025.6.0",
        "h5py>=3.14.0",
        "arviz>=0.21.0"
    ],
    python_requires=">=3.10",
)
