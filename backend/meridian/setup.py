from setuptools import setup, find_packages

setup(
    name="meridian",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23.5",
        "pandas>=1.5.3",
        "tensorflow-cpu>=2.10.1",
        "tensorflow-probability>=0.18.0",
        "scipy>=1.9.3",
        "xarray>=2023.1.0",
        "h5py>=3.8.0",
        "arviz>=0.14.0"
    ],
    python_requires=">=3.10",
)
