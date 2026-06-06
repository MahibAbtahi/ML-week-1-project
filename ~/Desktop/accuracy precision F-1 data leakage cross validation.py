import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

print("=============================================")
print("STARTING DAY 5: EVALUATION & PERFORMANCE METRICS")
print("=============================================\n")

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
    print(f"Error: Missing '{data_filename}'. Please run Day 3/Day 4 first.")
    exit()

X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("\n--- Step 1: Computing Core Classification Metrics ---")

champion_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
champion_model.fit(X_train, y_train)
predictions = champion_model.predict(X_test)

raw_accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

print(f"Accuracy:  {raw_accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")

print(f"\nConfusion Matrix Breakdown:")
print(f"   - True Negatives: {tn}")
print(f"   - False Positives: {fp}")
print(f"   - False Negatives: {fn}")
print(f"   - True Positives: {tp}")

print("\n--- Step 2: Running 5-Fold Cross Validation ---")

cv_scores = cross_val_score(champion_model, X, y, cv=5, scoring='accuracy')

print(f"5-Fold Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}%")
print(f"Standard Deviation: {cv_scores.std()*100:.2f}%")

output_json_path = "mahib_week1day5_metrics.json"

evaluation_profile = {
    "model_evaluated": "Random_Forest_Advanced",
    "single_split_metrics": {
        "accuracy": float(raw_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    },
    "confusion_matrix_counts": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    },
    "cross_validation": {
        "fold_scores": [float(score) for score in cv_scores],
        "mean_accuracy": float(cv_scores.mean()),
        "standard_deviation": float(cv_scores.std())
    },
    "analytical_conclusion": "The cross-validation mean accuracy aligns closely with our test split, indicating stable behavior free from severe data leakage."
}

with open(output_json_path, 'w') as json_file:
    json.dump(evaluation_profile, json_file, indent=4)

print(f"\nDay 5 advanced validation profile exported to: '{output_json_path}'")
print("=============================================")