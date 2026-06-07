import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("=============================================")
print("STARTING DAY 6: FINAL INTEGRATION & PIPELINE STRUCTURE")
print("=============================================\n")

raw_data_path = "Titanic-Dataset.csv"
if not os.path.exists(raw_data_path):
    for root, dirs, files in os.walk("."):
        if "Titanic-Dataset.csv" in files:
            raw_data_path = os.path.join(root, "Titanic-Dataset.csv")
            break

if os.path.exists(raw_data_path):
    df = pd.read_csv(raw_data_path)
    print(f"Loaded raw dataset from: {raw_data_path}")
else:
    print("Error: 'Titanic-Dataset.csv' not found. Please place it in your project folder.")
    exit()

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

numeric_features = ['Age', 'Fare', 'FamilySize']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

passthrough_features = ['Pclass', 'IsAlone']

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
    ('pass', 'passthrough', passthrough_features)
])

candidate_models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Decision_Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
}

pipeline_metrics = {}
os.makedirs("saved_models", exist_ok=True)

print("\n--- Training Multiple Pipeline Architectures ---")
for name, model in candidate_models.items():
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    full_pipeline.fit(X_train, y_train)
    predictions = full_pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    print(f" Trained {name} Pipeline | Accuracy: {acc*100:.2f}%")
    
    pipeline_metrics[name] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1)
    }
    
    model_save_path = f"saved_models/mahib_week1day6_{name.lower()}_pipeline.pkl"
    joblib.dump(full_pipeline, model_save_path)

metrics_output_path = "mahib_week1day6_pipeline_metrics.json"
with open(metrics_output_path, 'w') as f:
    json.dump(pipeline_metrics, f, indent=4)

print(f"\n Pipeline evaluation metrics exported to: '{metrics_output_path}'")
print(" All trained pipelines exported for reuse inside directory: '/saved_models/'")
print("=============================================")