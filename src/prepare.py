import numpy as np
import pandas as pd

df = pd.read_csv("data/detect_dataset.csv")

df = df.drop(columns=["Unnamed: 7", "Unnamed: 8"])
df = df.rename(columns={"Output (S)": "fault"})

# Split the data as given in the assignment
rng = np.random.default_rng(42)
idx = rng.permutation(len(df))
n_train, n_val = int(0.6 * len(df)), int(0.2 * len(df))

train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]

# Get the three sets
train_data = df.iloc[train_idx]
val_data = df.iloc[val_idx]
test_data = df.iloc[test_idx]

# Report rows and fault fraction
print("Training rows:", len(train_data))
print("Training fault fraction:", (train_data["fault"] == 1).mean())

print("\nValidation rows:", len(val_data))
print("Validation fault fraction:", (val_data["fault"] == 1).mean())

print("\nTest rows:", len(test_data))
print("Test fault fraction:", (test_data["fault"] == 1).mean())

# Select the six measurement columns
features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

# Calculate mean and standard deviation from training data only
train_mean = train_data[features].mean()
train_std = train_data[features].std(ddof=0)

# Standardise all three sets using training statistics
train_data.loc[:, features] = (
    train_data[features] - train_mean
) / train_std

val_data.loc[:, features] = (
    val_data[features] - train_mean
) / train_std

test_data.loc[:, features] = (
    test_data[features] - train_mean
) / train_std

print("\nTraining feature means after standardisation:")
print(train_data[features].mean().round(4))

print("\nTraining feature standard deviations after standardisation:")
print(train_data[features].std(ddof=0).round(4))

