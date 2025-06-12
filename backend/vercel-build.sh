#!/bin/bash
set -e

echo "Python version:"
python --version

echo "Pip version:"
pip --version

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing wheel and setuptools..."
pip install --no-cache-dir wheel setuptools

echo "Installing requirements..."
pip install --no-cache-dir --only-binary=:all: -r requirements.txt

echo "Installing meridian package..."
cd meridian && pip install --no-cache-dir -e .

echo "Build completed successfully"
