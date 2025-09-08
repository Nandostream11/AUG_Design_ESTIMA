# Python
import sys
import os

# Add repo root
repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Also add solver_2 root specifically so imports inside submodule work
solver_root = os.path.join(repo_root, "solver_2")
if solver_root not in sys.path:
    sys.path.insert(0, solver_root)

import numpy as np
import itertools
import pandas as pd
import joblib
from sklearn.multioutput import MultiOutputRegressor    #a sklearn wrapper to handle multi-output regression for other models than RandomForestRegressor()
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from solver_2.Parameters.slocum3D import SLOCUM_PARAMS
from solver_2.Modeling3d.glider_model_3D import ThreeD_Motion

# --- User choices: set to True to include in iteration ---
iterate_body_params = True
iterate_added_mass_params = False   
iterate_config_params = False
# ---------------------------------------------------------

# Define parameter grids
body_params = {
    'BODY_LEN': np.linspace(1.2, 1.8, 4),
    'RADIUS': np.linspace(0.09, 0.13, 3),
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
    'GLIDE_ANGLE': np.linspace(10, 30, 3),
    'SPEED': np.linspace(0.2, 0.5, 3),
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

# Choose which parameter sets to iterate
param_grids = []
param_names = []

if iterate_body_params:
    param_grids.append(list(body_params.values()))
    param_names.extend(list(body_params.keys()))
else:
    param_grids.append([[v[0]] for v in body_params.values()])
    param_names.extend(list(body_params.keys()))

if iterate_added_mass_params:
    param_grids.append(list(added_mass_params.values()))
    param_names.extend(list(added_mass_params.keys()))
else:
    param_grids.append([[v[0]] for v in added_mass_params.values()])
    param_names.extend(list(added_mass_params.keys()))

if iterate_config_params:
    param_grids.append(list(config_params.values()))
    param_names.extend(list(config_params.keys()))
else:
    param_grids.append([[v[0]] for v in config_params.values()])
    param_names.extend(list(config_params.keys()))

# Flatten param_grids for itertools.product
flat_grids = [g for group in param_grids for g in group]
combinations = list(itertools.product(*flat_grids))

results = []
iteration = 0
total_iterations = len(combinations)
from tqdm import tqdm

for iteration, combo in enumerate(tqdm(combinations, desc="Training Progress")):
    print(f"\n Params: {combo}")
    param_dict = dict(zip(param_names, combo))
        
    # Patch SLOCUM_PARAMS for this run
    for k in body_params.keys():
        setattr(SLOCUM_PARAMS.GLIDER_CONFIG, k, param_dict[k])
    for k in added_mass_params.keys():
        setattr(SLOCUM_PARAMS.GLIDER_CONFIG, k, param_dict[k])
    for k in config_params.keys():
        setattr(SLOCUM_PARAMS.VARIABLES, k, param_dict[k])

    class Args:
        mode = "3D"
        glider = "slocum"
        info = False
        pid = "disable"
        rudder = "disable"
        setrudder = 10.0
        plot = []  # Suppress plotting
        cycle = 1
        angle = param_dict['GLIDE_ANGLE']
        speed = param_dict['SPEED']

    args = Args()
    sim = ThreeD_Motion(args)
    sim.set_desired_trajectory()

    perf = {}
    for idx, name in enumerate(['x', 'y', 'z', 'omega1', 'omega2', 'omega3', 'vel', 'v1', 'v2', 'v3', 'rp1', 'rp2', 'rp3', 'mb', 'phi', 'theta', 'psi']):
        if idx < sim.solver_array.shape[1]:
            arr = sim.solver_array[:, idx]
            perf[f'{name}_mean'] = np.mean(arr)
            perf[f'{name}_max'] = np.max(arr)
            perf[f'{name}_min'] = np.min(arr)

    result = {**param_dict, **perf}
    results.append(result)

df = pd.DataFrame(results)
perf_cols = [col for col in df.columns if any(s in col for s in ['_mean', '_max', '_min'])]
target_cols = param_names

joblib.dump(target_cols, "glider_design_target_cols.pkl")

X = df[perf_cols]
y = df[target_cols]

# Split data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "RandomForestRegressor": RandomForestRegressor(),
    "GradientBoostingRegressor": MultiOutputRegressor(GradientBoostingRegressor()),
    "LinearRegression": MultiOutputRegressor(LinearRegression())
}

best_score = float('inf')
best_model = None
best_model_name = ""

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    perf_mae = []
    for i in range(len(X_test)):
        pred_params = dict(zip(y.columns, y_pred[i]))
        for k in body_params.keys():
            setattr(SLOCUM_PARAMS.GLIDER_CONFIG, k, pred_params[k])
        for k in added_mass_params.keys():
            setattr(SLOCUM_PARAMS.GLIDER_CONFIG, k, pred_params[k])
        for k in config_params.keys():
            setattr(SLOCUM_PARAMS.VARIABLES, k, pred_params[k])

        class Args:
            mode = "3D"
            glider = "slocum"
            info = False
            pid = "disable"
            rudder = "disable"
            setrudder = 10.0
            plot = []
            cycle = 1
            angle = pred_params.get('GLIDE_ANGLE', 20)
            speed = pred_params.get('SPEED', 0.3)
        args = Args()
        sim = ThreeD_Motion(args)
        sim.set_desired_trajectory()

        perf = {}
        for idx, fname in enumerate(X.columns):
            if idx < sim.solver_array.shape[1]:
                arr = sim.solver_array[:, idx % sim.solver_array.shape[1]]
                if fname.endswith('_mean'):
                    perf[fname] = np.mean(arr)
                elif fname.endswith('_max'):
                    perf[fname] = np.max(arr)
                elif fname.endswith('_min'):
                    perf[fname] = np.min(arr)
        common_cols = [k for k in X.columns if k in perf]
        if common_cols:
            perf_mae.append(mean_absolute_error([X_test.iloc[i][k] for k in common_cols], [perf[k] for k in common_cols]))
        else:
            print(f"Warning:{k} No common columns found for iteration {i}")
    avg_mae = np.mean(perf_mae)
    print(f"{name} average MAE on solver-evaluated test set: {avg_mae:.4f}")

    if avg_mae < best_score:
        best_score = avg_mae
        best_model = model
        best_model_name = name

print(f"\nBest model: {best_model_name} with MAE: {best_score:.4f}")
joblib.dump(best_model, "glider_design_inference_model.pkl")
print("Model saved to glider_design_inference_model.pkl")