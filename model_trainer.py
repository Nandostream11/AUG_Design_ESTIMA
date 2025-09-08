# Python
import sys
import os

# Add repo root
repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Also add solver_2 root specifically so imports inside submodule work
solver_root = os.path.join(repo_root, "solver")
if solver_root not in sys.path:
    sys.path.insert(0, solver_root)

import numpy as np
import itertools
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from solver.Parameters.slocum3D import SLOCUM_PARAMS
from solver.Modeling3d.glider_model_3D import ThreeD_Motion
import joblib

# --- User choices: set to True to include in iteration ---
iterate_body_params = True
iterate_added_mass_params = True
iterate_config_params = True
# ---------------------------------------------------------

# Define design parameter grids
body_params = {
    'BODY_LEN': np.linspace(1.2, 1.8, 4),   # meters
    'RADIUS': np.linspace(0.09, 0.13, 3),   # meters
}
added_mass_params = {
    'MF1': np.linspace(4, 6, 3),
    'MF2': np.linspace(55, 65, 3),
    'MF3': np.linspace(65, 75, 3),
    'J1': np.linspace(3, 5, 3),
    'J2': np.linspace(10, 14, 3),
    'J3': np.linspace(9, 13, 3),
}
config_params = {
    'GLIDE_ANGLE': np.linspace(10, 30, 3),  # degrees
    'SPEED': np.linspace(0.2, 0.5, 3),      # m/s
    'BALLAST_RATE': np.linspace(0.0005, 0.002, 3),
    'rp2': [0.01, 0.02, 0.03],
    'rp3': [0.04, 0.05, 0.06],
    'rb1': [0.0],
    'rb2': [0.0],
    'rb3': [0.0],
    'PHI': [0, 30, 45],
    'THETA': [10, 25, 40],
    'PSI': [0.0],
    'BETA': [0.0, 1.0, 2.0],
}

# Generate combinations (sampled for tractability)
body_combos = list(itertools.product(*body_params.values()))
added_mass_combos = list(itertools.product(*added_mass_params.values()))
config_combos = list(itertools.product(*config_params.values()))

results = []
iteration = 0

for b in body_combos:
    for a in added_mass_combos:
        for c in config_combos:
            print(f"\nIteration {iteration} | Params: {b}, {a}, {c}")
            iteration += 1
            # Set up parameter dicts
            glider_conf = dict(zip(body_params.keys(), b))
            added_mass_conf = dict(zip(added_mass_params.keys(), a))
            config_conf = dict(zip(config_params.keys(), c))

            # Patch SLOCUM_PARAMS for this run
            for k, v in glider_conf.items():
                setattr(SLOCUM_PARAMS.GLIDER_CONFIG, k, v)
            for k, v in added_mass_conf.items():
                setattr(SLOCUM_PARAMS.GLIDER_CONFIG, k, v)
            for k, v in config_conf.items():
                setattr(SLOCUM_PARAMS.VARIABLES, k, v)

            # Prepare dummy args for ThreeD_Motion
            class Args:
                mode = "3D"
                glider = "slocum"
                info = False
                pid = "disable"
                rudder = "disable"
                setrudder = 10.0
                plot = []       # Suppress plotting
                cycle = 1
                angle = config_conf['GLIDE_ANGLE']
                speed = config_conf['SPEED']

            args = Args()
            sim = ThreeD_Motion(args)
            sim.set_desired_trajectory()

            # Extract performance metrics (customize as needed)
            perf = {}
            # Example: mean/max/min of each variable in solver_array
            for idx, name in enumerate(['x', 'y', 'z', 'omega1', 'omega2', 'omega3', 'vel', 'v1', 'v2', 'v3', 'rp1', 'rp2', 'rp3', 'mb', 'phi', 'theta', 'psi']):
                if idx < sim.solver_array.shape[1]:
                    arr = sim.solver_array[:, idx]
                    perf[f'{name}_mean'] = np.mean(arr)
                    perf[f'{name}_max'] = np.max(arr)
                    perf[f'{name}_min'] = np.min(arr)

            # Store design params and performance
            result = {**glider_conf, **added_mass_conf, **config_conf, **perf}
            results.append(result)

# DataFrame for ML
df = pd.DataFrame(results)

# Features: all performance metrics
perf_cols = [col for col in df.columns if any(s in col for s in ['_mean', '_max', '_min'])]
# Targets: design parameters
target_cols = list(body_params.keys()) + list(added_mass_params.keys()) + list(config_params.keys())

X = df[perf_cols]
y = df[target_cols]

# Train ML model
# Define models to test
model = RandomForestRegressor()
model.fit(X, y)

print("Model trained. You can now predict design parameters from performance metrics.")

# Save the model
joblib.dump(model, "detailedglider_design_inference_model.pkl")
print("Model saved to detailedglider_design_inference_model.pkl")