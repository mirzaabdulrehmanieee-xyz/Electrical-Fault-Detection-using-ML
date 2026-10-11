from pathlib import Path
import pandas as pd

file = Path("data") / "detect_dataset.csv"
df = pd.read_csv(file)

# Remove empty columns
df = df.drop(columns=["Unnamed: 7", "Unnamed: 8"])

# Rename the label column
df = df.rename(columns={"Output (S)": "fault"})

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nFault rows:", (df["fault"] == 1).sum())
print("No-fault rows:", (df["fault"] == 0).sum())