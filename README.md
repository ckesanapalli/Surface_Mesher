# 🧱 Surface Mesher

[![Python Package](https://github.com/ckesanapalli//surface-mesher/actions/workflows/python-package.yml/badge.svg)](https://github.com/ckesanapalli/surface-mesher/actions/workflows/python-package.yml.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/ckesanapalli/surface-mesher/badge.svg?branch=master)](https://coveralls.io/github/ckesanapalli/surface-mesher?branch=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-blueviolet.svg)](https://docs.astral.sh/ruff/)
[![Powered by uv](https://img.shields.io/badge/Powered%20by-uv-22272e?logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![Python >= 3.10](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/downloads/)


**Surface Mesher** is a Python library for generating structured 3D surface meshes of primitive shapes, with a strong focus on **quadrilateral-dominant (quad) meshing**. The meshes are particularly suited for **visualization** and **Boundary Element Method (BEM)** simulations.

> ⚠️ This project is currently under active development.

---

## ✨ Features

- Generate clean, structured **cuboid surface meshes**
- Support for mesh generation using:
  - Fixed number of divisions
  - Fixed edge sizes
  - Explicit coordinate arrays
- All mesh faces are quadrilateral with consistent vertex ordering
- Easily visualized using `matplotlib` 3D plotting
- Lightweight and dependency-minimal core

---

## 🎯 Objective

This library aims to provide a minimal, intuitive interface for constructing **quad-based surface meshes** of primitive solids.

Use cases include:

- Geometry visualization
- Boundary Element Methods (BEM)
- Educational tooling
- Preprocessing for surface-based solvers

---

## ⚙️ Requirements

- **Python**: >= 3.10
- **Dependencies**:
  - `numpy>=1.24`
  - Optional (for examples and visualization):
    - `ipykernel`
    - `jupyterlab`
    - `matplotlib`

---

## 🚀 Installation

You can install the latest development version via Git:

```bash
pip install git+https://github.com/ckesanapalli/surface-mesher.git
```

---

## 🧱 Basic Usage

For detailed examples and tutorials, refer to the [Cuboid.ipynb](examples/cuboid.ipynb) file.

---

## 🧪 Running Tests

To run the full test suite:

```bash
pytest tests/
```

The library includes full test coverage including edge cases.

---

## 📁 Project Structure

```bash
surface_mesher/
│
├── cuboid.py                # Cuboid mesh generation logic
├── __init__.py
└── ...
tests/
├── test_cuboid.py           # Unit tests
└── ...
examples/
├── cuboid.ipynb             # Example notebook for cuboid mesh generation
└── ...
```

---

## 📌 Roadmap

- [x] Cuboid surface mesh generation
- [ ] Cylinder, cone, and sphere support
- [ ] STL/PLY export support
- [ ] Mesh visualization utilities
- [ ] Export to BEM-compatible formats

---

## 📄 License

MIT License © 2025 Chaitanya Kesanapalli
