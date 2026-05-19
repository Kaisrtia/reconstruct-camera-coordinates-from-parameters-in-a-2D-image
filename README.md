# reconstruct-camera-coordinates-from-parameters-in-a-2D-image

## Overview
This project provides a Python scaffold to reconstruct camera coordinates from 2D image parameters. It includes a minimal camera model, projection utilities, and a demo script.

## Structure
- src/camera_reconstruction/: core package (camera model, projection, reconstruction)
- scripts/: runnable helpers
- tests/: basic smoke tests
- data/: input/output data (empty)
- notebooks/: exploration notebooks (empty)

## Setup (Windows PowerShell)
1) python -m venv .venv
2) .venv\Scripts\Activate.ps1
3) pip install -e .

## Run demo
- python -m camera_reconstruction.demo
- python scripts/run_demo.py

## Run tests
- python -m unittest
