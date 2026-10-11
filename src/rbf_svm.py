
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC


# Load and clean the dataset
df = pd.read_csv("data/detect_dataset.csv")

df = df.drop(columns=["Unnamed: 7", "Unnamed: 8"])
df = df.rename(columns={"Output (S)": "fault"})

# Set R: six original features only
features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

# Use the same split as the previous tasks
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

# Standardize using training statistics only
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)

X_train = (X_train - train_mean) / train_std
X_val = (X_val - train_mean) / train_std

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# Test the required C and gamma values
C_values = [0.1, 1, 10, 100]
gamma_values = [0.1, 1]

best_model = None
best_C = None
best_gamma = None
best_accuracy = -1

results = []

for gamma in gamma_values:
    for C in C_values:

        # Create and train the RBF SVM
        model = SVC(kernel="rbf", C=C, gamma=gamma)
        model.fit(X_train, y_train)

        # Predict validation labels
        predictions = model.predict(X_val)

        # Calculate validation accuracy manually
        accuracy = np.mean(predictions == y_val) * 100

        # Count support vectors
        support_vectors = model.n_support_.sum()

        results.append((C, gamma, accuracy, support_vectors))

        print(
            f"C: {C}, Gamma: {gamma}, "
            f"Validation accuracy: {accuracy:.2f}%, "
            f"Support vectors: {support_vectors}"
        )

        # Keep the model with the highest validation accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = C
            best_gamma = gamma
            best_model = model

print("\nBest RBF SVM")
print("Selected C:", best_C)
print("Selected gamma:", best_gamma)
print(f"Best validation accuracy: {best_accuracy:.2f}%")
print("Support vectors:", best_model.n_support_.sum())

# Save the best model and its preprocessing information
joblib.dump(
    {
        "model": best_model,
        "mean": train_mean,
        "std": train_std,
        "features": features,
        "C": best_C,
        "gamma": best_gamma
    },
    "results/best_rbf_svm.joblib"
)

print("Best RBF SVM saved to results/best_rbf_svm.joblib")
