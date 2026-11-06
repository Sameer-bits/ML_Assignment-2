import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Define project directory
base_path = Path(__file__).resolve().parent

# Load cleaned dataset
df = pd.read_csv(
    base_path / "Dry_Bean_Cleaned.csv"
)

# Separate input features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Combine features and target
train_data = X_train.copy()
train_data["Class"] = y_train

test_data = X_test.copy()
test_data["Class"] = y_test

# Save datasets
train_data.to_csv(
    base_path / "train_data.csv",
    index=False
)

test_data.to_csv(
    base_path / "test_data.csv",
    index=False
)

# Display results
print("TRAINING DATA SHAPE:", train_data.shape)
print("TESTING DATA SHAPE:", test_data.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nTraining and testing datasets saved successfully!")