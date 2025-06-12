#!/bin/bash
set -e

echo "Python version:"
python --version

echo "Pip version:"
pip --version

echo "Installing requirements..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Installing meridian package..."
cd meridian && pip install -e .

echo "Build completed successfully"
