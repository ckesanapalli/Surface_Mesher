# 🧱 Surface Mesher

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
