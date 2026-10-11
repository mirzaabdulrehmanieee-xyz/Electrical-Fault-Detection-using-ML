import joblib
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC


# Load and clean the dataset
df = pd.read_csv("data/detect_dataset.csv")

df = df.drop(columns=["Unnamed: 7", "Unnamed: 8"])
df = df.rename(columns={"Output (S)": "fault"})

# Select the six original features for Set R
features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

# Split the data using the same method as Task 3
rng = np.random.default_rng(42)
idx = rng.permutation(len(df))

n_train = int(0.6 * len(df))
n_val = int(0.2 * len(df))

train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]

train_data = df.iloc[train_idx]
val_data = df.iloc[val_idx]

# Separate features and labels
X_train = train_data[features].to_numpy()
y_train = train_data["fault"].to_numpy()

X_val = val_data[features].to_numpy()
y_val = val_data["fault"].to_numpy()

# Standardize using training data statistics only
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)

X_train = (X_train - train_mean) / train_std
X_val = (X_val - train_mean) / train_std

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# Test the four C values
C_values = [0.1, 1, 10, 100]

results = []
best_model = None
best_C = None
best_accuracy = -1

for C in C_values:

    # Create and train the linear SVM
    model = SVC(kernel="linear", C=C)
    model.fit(X_train, y_train)

    # Predict validation labels
    predictions = model.predict(X_val)

    # Calculate accuracy manually
    accuracy = np.mean(predictions == y_val) * 100

    # Count support vectors
    support_vectors = model.n_support_.sum()

    results.append((C, accuracy, support_vectors))

    print(
        f"C: {C}, "
        f"Validation accuracy: {accuracy:.2f}%, "
        f"Support vectors: {support_vectors}"
    )

    # Keep the model with the highest validation accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_C = C
        best_model = model

print("\nBest Linear SVM")
print("Selected C:", best_C)
print(f"Best validation accuracy: {best_accuracy:.2f}%")
print("Support vectors:", best_model.n_support_.sum())

# Save the best linear SVM and preprocessing information
os.makedirs("results", exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "mean": train_mean,
        "std": train_std,
        "features": features,
        "C": best_C
    },
    "results/best_linear_svm.joblib"
)

print("Best Linear SVM saved to results/best_linear_svm.joblib")
