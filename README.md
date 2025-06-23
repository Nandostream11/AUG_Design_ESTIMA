# AUG_Design_ESTIMA

**Machine Learning-Based Parametric Design Estimation Tool for Underwater Vehicles(Gliders)**
---

### Overview

`AUG_Design_ESTIMA` is a research-driven, ML-powered framework designed to estimate and optimize hydrodynamic design parameters for **Autonomous Underwater Vehicles (AUVs)**, specifically **Autonomous Underwater Gliders (AUGs)**. Unlike traditional CFD pipelines that are time-intensive and computationally expensive, this tool aims to act as a **fast, scalable, and intelligent inverse-design advisor** using historical datasets, generative design strategies, and AI-based estimation models.

This project reimagines how we approach early-stage AUV conceptual design by using **ML regression models**, **parametric shape estimation**, and **inverse problem-solving** techniques to generate and refine vehicle geometries optimized for performance under water.

---

### Goals

-  **Estimate Hydrodynamic Coefficients** of AUGs using Machine Learning (drag, lift, stability parameters, etc.)
-  **Inverse Parameter Estimation**: Predict optimal design parameters (fin shape, hull length, volume, etc.) for desired performance profiles
-  **Integrate Generative Design** workflows to propose new hull shapes satisfying performance constraints
-  Support hybrid workflows that combine **CFD simulation datasets** and **AI model training**
-  Build an open-source **toolbox** for underwater vehicle designers, researchers, and startups

---

### 📁 What’s Inside

| Folder | Description |
|--------|-------------|
| `data/` | Sample CFD simulation results and synthetic datasets for AUGs |
| `models/` | ML models (XGBoost, LSTM, MLP, etc.) to be trained for parameter estimation |
| `notebooks/` | Jupyter notebooks for training, evaluation, and visualization |
| `scripts/` | Core scripts for preprocessing, modeling, and inverse solving(ref. [AUG-Simulator](https://github.com/Bhaswanth-A/AUG-Simulator)) | 
| `generative/` | Integration with generative shape estimation (WIP) |
| `docs/` | Technical documents and literature references |

---

### 📌 Features

- ✅ ML Models trained on hydrodynamic CFD simulation results
- 🔄 Inverse design engine: Suggest design parameters for a desired underwater performance
- 🔄 Support for both 2D and 3D shape estimation
- 🔄 Open dataset loader and preprocessor module
- 🔄 Future support for **Unity/AUVSim/ROS2-Gazebo** integrations for real-time feedback

---

### 🛠 Technologies in Use

- Python, NumPy, Pandas, SciKit-Learn
- PyTorch / TensorFlow (for deep learning models)
- XGBoost / CatBoost (for classical regression)
- OpenFOAM / SimScale (CFD dataset generation)
- Blender / FreeCAD (for parametric design modeling)
- ROS2 / Gazebo (future integration for control simulation)

---

### 📚 References / Inspirations

- ESTIMA (Nonlinear Param Estimation Toolkit)
- Aerodynamic Design ML Tools (UAV/Drone sector)
- AIML for Parametric Shape Estimation
- AI-based Aero-Engine Design Advisors
- Research papers on AI in Hydrodynamics and Naval Architecture

---

###  Getting Started

#### 1. Clone the Repo

```bash
git clone https://github.com/<your-username>/AUG_Design_ESTIMA.git
cd AUG_Design_ESTIMA
```
#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### Project Roadmap
- [x] Implement inverse solver (Design → Param Estimation)
- [ ] Import existing CFD data and build base regression models
- [ ] Add generative geometry design module
- [ ] Build GUI/CLI for quick param estimation
- [ ] ROS2 / Unity simulation integration
- [ ] Research publication & open data release
 
### Contributions
We welcome open-source contributions, dataset sharing, and collaboration with marine engineers, naval architects, and AI researchers.
Feel free to raise issues, request features, or collaborate on research:
> Fork the repo
> Create a branch
> Submit a pull request

📫 Contact
For queries, collaboration, or demo requests:
📧 anandvardhanx@gmail.com
🌐 LinkedIn

🌊 Let’s Redefine Underwater Design Together
"Designing below the surface with intelligence above it."
