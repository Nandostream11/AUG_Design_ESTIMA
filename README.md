# AUG_Design_ESTIMA
**Machine Learning-Based Parametric Design Estimation Tool for Underwater Vehicles(Gliders)**
---

### Overview

`AUG_Design_ESTIMA` is a research-driven, ML-powered framework designed to estimate and optimize hydrodynamic design parameters for **Autonomous Underwater Vehicles (AUVs)**, specifically **Autonomous Underwater Gliders (AUGs)**. Unlike traditional CFD pipelines that are time-intensive and computationally expensive, this tool aims to act as a **fast, scalable, and intelligent inverse-design advisor** using historical datasets, generative design strategies, and AI-based estimation models.

It is a framework for estimating and predicting design parameters of **Autonomous Underwater Gliders (AUGs)** using Machine Learning. The tool leverages simulation data generated from a physics-based solver to train regression models that learn the relationship between glider design parameters and their performance metrics. This enables **inverse design**: predicting suitable design parameter combinations for a desired set of performance characteristics.

Unlike traditional CFD pipelines, this tool provides a **fast and scalable approach** for early-stage conceptual design and rapid design space exploration.

This project reimagines how we approach early-stage AUV conceptual design by using **ML regression models**, **parametric shape estimation**, and **inverse problem-solving** techniques to generate and refine vehicle geometries optimized for performance under water.

---

### Current Functionality

- **Simulation-Driven Dataset Generation:**  
  Iterates over user-selected combinations of glider design parameters (e.g., body length, radius, added masses, configuration variables) using the inbuilt solver to generate performance data.
- **Performance Metric Extraction:**  
  Automatically extracts key performance metrics (e.g., position, velocity, angular rates) from simulation results for each design configuration.
- **Machine Learning Model Training:**  
  Trains a regression model (RandomForestRegressor by default) to learn the mapping from performance metrics to design parameters.
- **Model Inference:**  
  Provides scripts to load the trained model and predict design parameters (such as `BODY_LEN`, `RADIUS`, etc.) for any given set of performance metrics.
- **Flexible Parameter Selection:**  
  Users can choose which parameter groups (body, added mass, configuration) to include in the design space exploration.
- **No Plotting Required:**  
  The workflow is fully automated and does not require manual inspection of plots.

---

### Goals

- **Inverse Parameter Estimation:**  
  Predict optimal design parameters (e.g., hull length, radius, mass properties) for desired performance profiles.
- **Rapid Design Space Exploration:**  
  Enable fast, automated evaluation of large numbers of design configurations.
- **Open-Source Toolbox:**  
  Provide a modular, extensible tool for underwater vehicle designers and researchers.

---

<!-- ### 📁 What’s Inside

| Folder | Description |
|--------|-------------|
| `data/` | Sample CFD simulation results and synthetic datasets for AUGs |
| `models/` | ML models (XGBoost, LSTM, MLP, etc.) to be trained for parameter estimation |
| `notebooks/` | Jupyter notebooks for training, evaluation, and visualization |
| `scripts/` | Core scripts for preprocessing, modeling, and inverse solving(ref. [AUG-Simulator](https://github.com/Bhaswanth-A/AUG-Simulator)) | 
| `generative/` | Integration with generative shape estimation (WIP) |
| `docs/` | Technical documents and literature references |

--- -->
<!-- 
### 📌 Features

- ✅ ML Models trained on hydrodynamic CFD simulation results
- 🔄 Inverse design engine: Suggest design parameters for a desired underwater performance
- 🔄 Support for both 2D and 3D shape estimation
- 🔄 Open dataset loader and preprocessor module
- 🔄 Future support for **Unity/AUVSim/ROS2-Gazebo** integrations for real-time feedback -->

### Requirements

- Python 3.x
- numpy, pandas, scikit-learn, joblib

### 📚 References / Inspirations

- Solver from [Underwater Glider Performance metrics plotter tool](https://github.com/Bhaswanth-A/AUG-Simulator) by Bhaswanth A.
- ESTIMA (Nonlinear Param Estimation Toolkit)
- Aerodynamic Design ML Tools (UAV/Drone sector)
- AIML for Parametric Shape Estimation
- AI-based Aero-Engine Design Advisors
- Research papers on AI in Hydrodynamics and Naval Architecture

---
### Usage

1. **Generate Training Data & Train Model:**  
   Use `model_trainer_lite.py` to run simulations, collect data, and train the ML model.  
   You can select which parameter groups to iterate on by setting flags at the top of the script.

2. **Save Target Column Names:**  
   The script saves the names of the design parameters for use during inference.

3. **Predict Design Parameters:**  
   Use `params_predictor.py` to load the trained model and predict design parameters from new performance metrics.  
   The script prints both the input features and the predicted design parameters with their names.

---

###  Getting Started

#### 1. Clone the Repo

```bash
git clone https://github.com/Nandostream11/AUG_Design_ESTIMA.git
cd AUG_Design_ESTIMA
```
#### 2. Initialize the git submodules
```bash
git submodule update --init --recursive
```
  ⚠ If you get errors fetching submodules, try:
```bash
git submodule update --init --remote
```
#### 3. Install Dependencies
You can create an env and activate it:
```bash
python -m venv .venv

# Activate it (Linux/Mac)
source .venv/bin/activate

# Activate it (Windows PowerShell)
.venv\Scripts\Activate
```
or create a new virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```
💡 Make sure you are using Python 3.10+ if required by the repo.
#### 4. **Train the Model:**
```sh
python3 model_trainer_lite.py
```

#### 5. Predict Parameters:
```sh
python3 params_predictor.py
```
### Notes
1. Since solver_2 is a git submodule, make sure it cloned correctly and contains all code before running Python scripts.
2. All scripts assume you are running from the repo root (AUG_Design_ESTIMA).
3. If Python cannot find modules inside solver_2, you may need to ensure sys.path includes the repo root as we discussed.

The tool currently uses a Random Forest regressor; you can swap in other ML models as needed.
Ensure that the solver and parameter modules are importable from your script locations.
The model predicts only for the range and combinations of parameters it was trained on.

# 🛠 Technologies to be used in the near future

- Python, NumPy, Pandas, SciKit-Learn
- PyTorch / TensorFlow (for deep learning models)
- XGBoost / CatBoost (for classical regression)
- OpenFOAM / SimScale (CFD dataset generation)
- Blender / FreeCAD (for parametric design modeling)
- ROS2 / Gazebo (future integration for control simulation)

---
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
📧 anandvk113@gmail.com
🌐 [LinkedIn](https://www.linkedin.com/in/anandvardhanrbtics/)

🌊 Let’s Redefine Underwater Design Together \
"_Designing below the surface with intelligence above it._"
