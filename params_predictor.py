import joblib
import pandas as pd

model = joblib.load("glider_design_inference_model.pkl")

# Get the required feature names
feature_names = model.feature_names_in_
print("Input features and values:")
X_new = pd.DataFrame([{name: 1.0 for name in feature_names}])
for name in feature_names:
    print(f"{name}: {X_new.iloc[0][name]}")

# Predict design parameters
y_pred = model.predict(X_new)

# Print predicted design parameters with names if available
try:
    target_cols = joblib.load("glider_design_target_cols.pkl")
except Exception:
    target_cols = [f"param_{i}" for i in range(len(y_pred[0]))]

print("\nPredicted design parameters:")
for name, value in zip(target_cols, y_pred[0]):
    print(f"{name}: {value}")