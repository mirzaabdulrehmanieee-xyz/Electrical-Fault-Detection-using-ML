
import numpy as np
import pandas as pd
import joblib

from logistic import fit_logistic, predict
from metrics import accuracy, confusion_matrix


# Load and clean the dataset
df = pd.read_csv("data/detect_dataset.csv")
df = df.drop(columns=["Unnamed: 7", "Unnamed: 8"])
df = df.rename(columns={"Output (S)": "fault"})

features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

# Recreate the exact same train, validation, and test split
rng = np.random.default_rng(42)
idx = rng.permutation(len(df))

n_train = int(0.6 * len(df))
n_val = int(0.2 * len(df))

train_idx = idx[:n_train]
test_idx = idx[n_train + n_val:]

train_data = df.iloc[train_idx]
test_data = df.iloc[test_idx]

y_train = train_data["fault"].to_numpy()
y_test = test_data["fault"].to_numpy()

# Prepare Set R: six original features
X_train_R_raw = train_data[features].to_numpy()
X_test_R_raw = test_data[features].to_numpy()

mean_R = X_train_R_raw.mean(axis=0)
std_R = X_train_R_raw.std(axis=0)

X_train_R = (X_train_R_raw - mean_R) / std_R
X_test_R = (X_test_R_raw - mean_R) / std_R

# Add intercept for logistic regression only
X_train_R_log = np.column_stack(
    (np.ones(len(X_train_R)), X_train_R)
)
X_test_R_log = np.column_stack(
    (np.ones(len(X_test_R)), X_test_R)
)

# Train final logistic regression model for Set R
theta_R, costs_R = fit_logistic(
    X_train_R_log, y_train, 1, 2000
)

# Prepare Set Q: original features plus squared features
X_train_Q_raw = np.column_stack(
    (X_train_R_raw, X_train_R_raw ** 2)
)
X_test_Q_raw = np.column_stack(
    (X_test_R_raw, X_test_R_raw ** 2)
)

mean_Q = X_train_Q_raw.mean(axis=0)
std_Q = X_train_Q_raw.std(axis=0)

X_train_Q = (X_train_Q_raw - mean_Q) / std_Q
X_test_Q = (X_test_Q_raw - mean_Q) / std_Q

# Add intercept for logistic regression
X_train_Q_log = np.column_stack(
    (np.ones(len(X_train_Q)), X_train_Q)
)
X_test_Q_log = np.column_stack(
    (np.ones(len(X_test_Q)), X_test_Q)
)

# Train final logistic regression model for Set Q
theta_Q, costs_Q = fit_logistic(
    X_train_Q_log, y_train, 10, 2000
)

# Load the saved best SVM models
linear_data = joblib.load(
    "results/best_linear_svm.joblib"
)
rbf_data = joblib.load(
    "results/best_rbf_svm.joblib"
)

linear_model = linear_data["model"]
rbf_model = rbf_data["model"]

# Use each SVM's saved training statistics
X_test_linear = (
    X_test_R_raw - linear_data["mean"]
) / linear_data["std"]

X_test_rbf = (
    X_test_R_raw - rbf_data["mean"]
) / rbf_data["std"]


# Evaluate a model and display its results
def evaluate_model(name, y_true, y_pred):
    acc = accuracy(y_true, y_pred) * 100
    cm = confusion_matrix(y_true, y_pred)

    tn, fp = cm[0]
    fn, tp = cm[1]

    print(f"\n{name}")
    print(f"Test accuracy: {acc:.2f}%")
    print("Confusion matrix:")
    print(cm)
    print("TN:", tn)
    print("FP:", fp)
    print("FN (missed faults):", fn)
    print("TP:", tp)


# Test each final model once
pred_R = predict(X_test_R_log, theta_R)
evaluate_model("Logistic Regression - Set R", y_test, pred_R)

pred_Q = predict(X_test_Q_log, theta_Q)
evaluate_model("Logistic Regression - Set Q", y_test, pred_Q)

pred_linear = linear_model.predict(X_test_linear)
evaluate_model("Best Linear SVM", y_test, pred_linear)

pred_rbf = rbf_model.predict(X_test_rbf)
evaluate_model("Best RBF SVM", y_test, pred_rbf)
