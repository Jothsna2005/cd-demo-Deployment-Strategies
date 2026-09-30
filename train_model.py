import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# =====================================================
# 1. LOAD DATASET
# =====================================================

DATA_FILE = "kidney_disease.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# =====================================================
# 2. BASIC INFORMATION
# =====================================================

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# =====================================================
# 3. CLEAN TARGET
# =====================================================

df["classification"] = (
    df["classification"]
    .astype(str)
    .str.strip()
)

print("\nTarget distribution:")
print(df["classification"].value_counts())


# =====================================================
# 4. SEPARATE FEATURES AND TARGET
# =====================================================

X = df.drop("classification", axis=1)

y = df["classification"]


# Remove identifier
if "id" in X.columns:
    X = X.drop("id", axis=1)


print("\nFeature shape:", X.shape)
print("Target shape :", y.shape)


# =====================================================
# 5. IDENTIFY COLUMN TYPES
# =====================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# =====================================================
# 6. TRAIN-TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining shape:", X_train.shape)
print("Testing shape :", X_test.shape)


# =====================================================
# 7. NUMERICAL PREPROCESSING
# =====================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# =====================================================
# 8. CATEGORICAL PREPROCESSING
# =====================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# =====================================================
# 9. COMBINE PREPROCESSING
# =====================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# =====================================================
# 10. CREATE MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# =====================================================
# 11. COMPLETE PIPELINE
# =====================================================

ml_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# =====================================================
# 12. TRAIN
# =====================================================

print("\nTraining model...")

ml_pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# =====================================================
# 13. EVALUATE
# =====================================================

y_pred = ml_pipeline.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# =====================================================
# 14. SAVE COMPLETE PIPELINE
# =====================================================

MODEL_FILE = "kidney_disease_model.pkl"

joblib.dump(
    ml_pipeline,
    MODEL_FILE
)


print(
    f"\nModel pipeline saved as: {MODEL_FILE}"
)


# =====================================================
# 15. VERIFY FILE
# =====================================================

print(
    "\nFile exists:",
    os.path.exists(MODEL_FILE)
)


if os.path.exists(MODEL_FILE):

    file_size = (
        os.path.getsize(MODEL_FILE) / 1024
    )

    print(
        f"File size: {file_size:.2f} KB"
    )