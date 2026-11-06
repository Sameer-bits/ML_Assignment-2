import pandas as pd
from pathlib import Path



# Load the dataset
base_path = Path(__file__).resolve().parent
df = pd.read_csv(base_path / "Dry_Bean_Dataset.csv")


print("FIRST FIVE RECORDS")
print(df.head())

print("DATASET SHAPE:", df.shape)


print("\nCOLUMN NAMES:")
print(df.columns.tolist())


print("\nDATA Type:")
df.info()

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDUPLICATE RECORDS:", df.duplicated().sum())

print("\nCLASS DISTRIBUTION:")
print(df["Class"].value_counts())