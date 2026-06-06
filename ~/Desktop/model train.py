import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=============================================")
print("STARTING DAY 4: MODEL TRAINING & PREDICTIONS")
print("=============================================\n")

# --- STEP 1: LOAD ENGINEERED DAY 3 DATA ---
data_filename = "mahib_week1day3_cleaned_data.csv"
data_path = None

for root, dirs, files in os.walk("."):
    if data_filename in files:
        data_path = os.path.join(root, data_filename)
        break

if data_path:
    df = pd.read_csv(data_path)
    print(f"Loaded engineered dataset from: {data_path}")
else:
    print(f"Error: Missing '{data_filename}'. Please run Day 3 first.")
    exit()

# --- STEP 2: TRAIN-TEST SPLIT (EXPERIMENTAL WORK STEP 2) ---
# Separate features (X) from our target label (y)
X = df.drop(columns=['Survived'])
y = df['Survived']

# 80% training for models to learn, 20% validation to test performance unbiased
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Dataset split completed successfully:")
print(f"   - Training Features Shape: {X_train.shape}")
print(f"   - Evaluation Features Shape: {X_test.shape}")

# --- STEP 3: MODEL TRAINING & HYPERPARAMETER TUNING (STEP 1 & 3) ---
models = {
    "Logistic_Regression_Baseline": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Decision_Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
    "Random_Forest_Advanced": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
}

performance_logs = {}
test_predictions_registry = {"True_Labels": y_test.tolist()}

print("\n--- Training Progress ---")
for name, model in models.items():
    # Fit model to training subsets
    model.fit(X_train, y_train)
    
    # Generate predictions on unseen test split
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print(f"✅ Trained {name} | Unseen Test Accuracy: {acc*100:.2f}%")
    
    # Track metrics and predictions separately
    performance_logs[name] = {
        "Accuracy": float(acc),
        "Hyperparameters": {k: str(v) for k, v in model.get_params().items() if k in ['C', 'max_depth', 'n_estimators', 'min_samples_split']}
    }
    test_predictions_registry[f"{name}_Predictions"] = predictions.tolist()

# --- STEP 4 & 5: SAVE PREDICTIONS & METRICS (NAMING CONVENTION) ---
metrics_output_path = "mahib_week1day4_metrics.json"
predictions_output_path = "mahib_week1day4_predictions.json"

with open(metrics_output_path, 'w') as f:
    json.dump(performance_logs, f, indent=4)

with open(predictions_output_path, 'w') as f:
    json.dump(test_predictions_registry, f, indent=4)

print(f"\n📈 Model evaluation metrics summary saved to: '{metrics_output_path}'")
print(f"🔮 Final model predictions logged to: '{predictions_output_path}'")
print("=============================================")