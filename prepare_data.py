
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.sparse import save_npz

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ==========================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "global_student_digital_behavior_dataset.csv"

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

for folder in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    folder.mkdir(exist_ok=True)


TARGET = "cyberbullying_exposure"
ID_COLUMN = "student_id"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Original dataset shape:", df.shape)


# ==========================================
# 3. BASIC DATA VALIDATION
# ==========================================

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

if df[TARGET].isnull().any():
    raise ValueError("Target contains missing values.")

print("\nTarget distribution:")
print(df[TARGET].value_counts())

print("\nTarget percentages:")
print(
    df[TARGET].value_counts(normalize=True).mul(100).round(2)
)


# ==========================================
# 4. REMOVE UNNECESSARY COLUMNS
# ==========================================

# student_id is only an identifier.
# brain_rot_index and brain_rot_level are excluded
# because they are not required for this target.

columns_to_drop = [
    ID_COLUMN,
    "brain_rot_index",
    "brain_rot_level"
]

columns_to_drop = [
    col for col in columns_to_drop
    if col in df.columns
]

df = df.drop(columns=columns_to_drop)

# Separate input features and target
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(str)

print("\nInput feature count:", X.shape[1])
print("Target column:", TARGET)

if X.shape[1] < 12:
    raise ValueError("Dataset does not meet the 12-feature requirement.")


# ==========================================
# 5. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ==========================================

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))


# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ==========================================
# 7. PREPROCESSING PIPELINES
# ==========================================

# Numerical features:
# - Replace missing values with training median
# - Standardize numerical values

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# Categorical features:
# - Replace missing values with "Unknown"
# - Convert categories using sparse one-hot encoding

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="constant",
            fill_value="Unknown"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )
    )
])


# Combine both pipelines

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ],
    sparse_threshold=1.0
)


# ==========================================
# 8. FIT ON TRAINING DATA ONLY
# ==========================================

print("\nFitting preprocessing pipeline...")

X_train_processed = preprocessor.fit_transform(X_train)

# Only transform test data.
# Never fit preprocessing on test data.

X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

print("Training matrix type:", type(X_train_processed))
print("Testing matrix type:", type(X_test_processed))


# ==========================================
# 9. SAVE PROCESSED DATA
# ==========================================

print("\nSaving processed datasets...")

# Sparse matrices use much less memory than dense arrays.

save_npz(
    DATA_DIR / "X_train_processed.npz",
    X_train_processed
)

save_npz(
    DATA_DIR / "X_test_processed.npz",
    X_test_processed
)

pd.DataFrame({
    TARGET: y_train
}).to_csv(
    DATA_DIR / "y_train.csv",
    index=False
)

pd.DataFrame({
    TARGET: y_test
}).to_csv(
    DATA_DIR / "y_test.csv",
    index=False
)


# ==========================================
# 10. SAVE PREPROCESSING ARTIFACTS
# ==========================================

joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.pkl"
)

joblib.dump(
    X.columns.tolist(),
    MODEL_DIR / "feature_columns.pkl"
)

joblib.dump(
    TARGET,
    MODEL_DIR / "target_column.pkl"
)


# Save class names for the Streamlit application

joblib.dump(
    sorted(y.unique().tolist()),
    MODEL_DIR / "class_names.pkl"
)


# ==========================================
# 11. CREATE STREAMLIT TEST DATA
# ==========================================

# Create a stratified sample of the held-out test set
# so the app has a manageable CSV for demonstration.

X_app, _, y_app, _ = train_test_split(
    X_test,
    y_test,
    train_size=2000,
    random_state=RANDOM_STATE,
    stratify=y_test
)

test_data = X_app.copy()
test_data[TARGET] = y_app

test_data.to_csv(
    BASE_DIR / "test_data.csv",
    index=False
)


# ==========================================
# 12. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 55)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 55)

print("Original records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Input features:", X.shape[1])
print("Processed features:", X_train_processed.shape[1])

print("\nSaved files:")
print("- data/X_train_processed.npz")
print("- data/X_test_processed.npz")
print("- data/y_train.csv")
print("- data/y_test.csv")
print("- models/preprocessor.pkl")
print("- models/feature_columns.pkl")
print("- models/target_column.pkl")
print("- models/class_names.pkl")
print("- test_data.csv")

print("\nTarget distribution:")
print(y.value_counts())

print("\nAll preprocessing steps completed!")
