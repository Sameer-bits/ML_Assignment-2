import pandas as pd
from pathlib import Path

# Define project path
base_path = Path(__file__).resolve().parent

# Load original dataset
df = pd.read_csv(
    base_path  / "Dry_Bean_Dataset.csv"
)

# Display original dataset information
print("Original dataset shape:", df.shape)

# Count duplicate records
print("Duplicate records:", df.duplicated().sum())

# Remove duplicate records
df_cleaned = df.drop_duplicates()

# Check cleaned dataset
print("\nCleaned dataset shape:", df_cleaned.shape)

print(
    "Remaining duplicates:",
    df_cleaned.duplicated().sum()
)

# Check missing values
print(
    "\nMissing values:",
    df_cleaned.isnull().sum().sum()
)

# Save cleaned dataset
output_path = base_path /"Dry_Bean_Cleaned.csv"

df_cleaned.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")

# Display updated class distribution
print("\nUpdated class distribution:")
print(df_cleaned["Class"].value_counts())