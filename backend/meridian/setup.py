from setuptools import setup, find_packages

setup(
    name="meridian",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.1.1",
        "tensorflow-cpu>=2.13.0",
        "tensorflow-probability[tf]>=0.21.0",
        "tf-keras>=2.13.0",
        "scipy>=1.11.0",
        "xarray>=2023.1.0",
        "h5py>=3.9.0",
        "arviz>=0.15.0",
        "joblib>=1.3.0",
        "altair>=5.0.0",
        "natsort>=8.0.0",
        "jinja2>=3.0.0",
        "absl-py>=1.0.0",
        "immutabledict>=2.2.3",
        "python-dateutil>=2.8.2",
        "openpyxl>=3.1.2",
        "protobuf>=4.21.0"
    ],
    python_requires=">=3.10",
)
