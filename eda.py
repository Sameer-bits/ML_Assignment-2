import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
base_path = Path(__file__).resolve().parent
df = pd.read_csv(
    base_path / "Dry_Bean_Dataset.csv"
)

# 1. Class distribution
plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="Class",
    order=df["Class"].value_counts().index
)

plt.title("Distribution of Dry Bean Classes")
plt.xlabel("Bean Variety")
plt.ylabel("Number of Instances")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# 2. Statistical summary
print("STATISTICAL SUMMARY")
print(df.describe())


# 3. Correlation heatmap
plt.figure(figsize=(13, 9))

numeric_df = df.select_dtypes(include="number")

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    annot=False
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()