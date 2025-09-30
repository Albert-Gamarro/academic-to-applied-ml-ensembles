# --------------------------------
# Imports
# --------------------------------
import os
import yaml
import joblib
import pandas as pd

from data.load_dataset import load_clean_adult_dataset

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------------
# Load config_prod.yaml
# -----------------------------
config_path = "config/config_prod.yml"
with open(config_path) as file:
    config = yaml.safe_load(file)

MODEL_DIR = config["model_dir"]
RESULTS_DIR = config["results_dir"]

# XGBoost hyperparameters
XGB_PARAMS = config["model"]["params"]

# --------------------------------
# Load dataset
# --------------------------------
X, y = load_clean_adult_dataset()
y = y.map({">50K": 1, "<=50K": 0})

# Convert object columns to category
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category")

# --------------------------------
# Preprocessing Data
# --------------------------------
# Identify categorical features. XGBoost require numerical inputs.
categorical_features = X.select_dtypes(include=["category"]).columns

# Define a preprocessing pipeline for categorical features.
# This ensures consistent handling of missing values and encoding during training and prediction.
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",  # Step name: imputer
            SimpleImputer(
                strategy="constant",  # Fill missing values with a constant value
                fill_value="missing",  # Value to fill in missing entries
            ),
        ),
        (
            "onehot",  # Step name: onehot
            OneHotEncoder(
                handle_unknown="ignore",  # Avoid errors if unseen categories appear in prediction
                sparse_output=False,  # Output a dense array (easier to work with for some pipelines)
            ),
        ),
    ]
)

# Create a ColumnTransformer that applies the categorical_transformer to all categorical columns.
# This structure is useful because it allows different preprocessing for different column types.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",  # Name of this transformation step
            categorical_transformer,  # The pipeline to apply
            categorical_features,  # Columns to which this pipeline applies
        )
    ]
)


# --------------------------------
# Train/test split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --------------------------------
# Build full pipeline (preprocessing + model)
# --------------------------------
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(**XGB_PARAMS),
        ),
    ]
)


# --------------------------------
# Fit model and evaluate
# --------------------------------
# trains XGBoost model and applies preprocessing at the same time
model_pipeline.fit(X_train, y_train)
# tests trained pipeline against unseen data
predictions = model_pipeline.predict(X_test)


accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"Prod Model Evaluation:")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")


# --------------------------------
# 8. Save model
# --------------------------------

import joblib

joblib.dump(model_pipeline, os.path.join(MODEL_DIR, "xgb_model_pipeline_prod.joblib"))
