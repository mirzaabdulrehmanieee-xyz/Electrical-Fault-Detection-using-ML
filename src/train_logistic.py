
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic import hypothesis, cost, fit_logistic, predict


# Create the folder for saving graphs
os.makedirs("results/figures", exist_ok=True)


# Load and clean the dataset
df = pd.read_csv("data/detect_dataset.csv")

df = df.drop(columns=["Unnamed: 7", "Unnamed: 8"])
df = df.rename(columns={"Output (S)": "fault"})


# Original features
features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]


# Split the data using the assignment's exact method
rng = np.random.default_rng(42)
idx = rng.permutation(len(df))

n_train = int(0.6 * len(df))
n_val = int(0.2 * len(df))

train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]

train_data = df.iloc[train_idx]
val_data = df.iloc[val_idx]

y_train = train_data["fault"].to_numpy()
y_val = val_data["fault"].to_numpy()


# Get original feature values for Set R
X_train_R = train_data[features].to_numpy()
X_val_R = val_data[features].to_numpy()


# Set Q: add squared versions of the six features
X_train_Q = np.column_stack(
    (X_train_R, X_train_R ** 2)
)

X_val_Q = np.column_stack(
    (X_val_R, X_val_R ** 2)
)


# Function to standardize features and add the intercept
def prepare_features(X_train, X_val):

    # Calculate statistics using training data only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Avoid division by zero if any feature has zero variation
    std[std == 0] = 1

    # Standardize both sets using training statistics
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Add a column of ones for theta_0
    X_train = np.column_stack(
        (np.ones(len(X_train)), X_train)
    )

    X_val = np.column_stack(
        (np.ones(len(X_val)), X_val)
    )

    return X_train, X_val


# Prepare Set R and Set Q
X_train_R, X_val_R = prepare_features(
    X_train_R, X_val_R
)

X_train_Q, X_val_Q = prepare_features(
    X_train_Q, X_val_Q
)


print("Set R training shape:", X_train_R.shape)
print("Set R validation shape:", X_val_R.shape)

print("Set Q training shape:", X_train_Q.shape)
print("Set Q validation shape:", X_val_Q.shape)


# Test the learning rates
alphas = [0.01, 0.1, 1, 10]
iterations = 2000


def train_and_check(
    X_train, X_val, y_train, y_val, set_name
):

    cost_history = {}
    suitable_alphas = []

    print(f"\n{set_name} Learning Rate Results")

    # Test each learning rate
    for alpha in alphas:

        theta = np.zeros(X_train.shape[1])
        costs = []

        for i in range(iterations):

            h = hypothesis(X_train, theta)

            # Gradient for logistic regression
            gradient = (
                X_train.T @ (y_train - h)
            ) / len(y_train)

            # Update theta
            theta = theta + alpha * gradient

            # Save cost
            costs.append(cost(X_train, y_train, theta))

        cost_history[alpha] = costs

        # Check whether cost decreases at every iteration
        decreasing = all(
            costs[i] <= costs[i - 1]
            for i in range(1, len(costs))
        )

        print(
            f"Alpha: {alpha}, "
            f"Final cost: {costs[-1]:.6f}, "
            f"Decreases every iteration: {decreasing}"
        )

        if decreasing:
            suitable_alphas.append(alpha)

    # Select the largest suitable learning rate
    if len(suitable_alphas) == 0:
        print("No suitable learning rate found.")
        return None

    best_alpha = max(suitable_alphas)

    # Train the final model using the selected learning rate
    theta_final, final_costs = fit_logistic(
        X_train, y_train, best_alpha, iterations
    )

    # Generate predictions
    train_predictions = predict(X_train, theta_final)
    val_predictions = predict(X_val, theta_final)

    # Calculate accuracy manually
    train_accuracy = np.mean(
        train_predictions == y_train
    ) * 100

    val_accuracy = np.mean(
        val_predictions == y_val
    ) * 100

    print(f"\nFinal {set_name} Results")
    print("Selected alpha:", best_alpha)
    print(f"Final training cost: {final_costs[-1]:.6f}")
    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Validation accuracy: {val_accuracy:.2f}%")
    print("Theta values:", theta_final)

    # Plot the cost history for each learning rate
    plt.figure(figsize=(10, 6))

    for alpha in alphas:
        plt.plot(
            cost_history[alpha],
            label=f"Alpha = {alpha}"
        )

    plt.xlabel("Iteration")
    plt.ylabel("Cost J(theta)")
    plt.title(f"Learning Rate Comparison - {set_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the graph before displaying it
    if set_name == "Set R":
        graph_path = "results/figures/logistic_set_R.png"
    else:
        graph_path = "results/figures/logistic_set_Q.png"

    plt.savefig(
        graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    print("Graph saved:", graph_path)

    plt.show()
    plt.close()

    return {
        "alpha": best_alpha,
        "cost": final_costs[-1],
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "theta": theta_final
    }


# Train and evaluate Set R
results_R = train_and_check(
    X_train_R,
    X_val_R,
    y_train,
    y_val,
    "Set R"
)


# Train and evaluate Set Q
results_Q = train_and_check(
    X_train_Q,
    X_val_Q,
    y_train,
    y_val,
    "Set Q"
)


# Save theta values for both logistic regression models
if results_R is not None and results_Q is not None:

    np.savetxt(
        "results/theta_R.txt",
        results_R["theta"]
    )

    np.savetxt(
        "results/theta_Q.txt",
        results_Q["theta"]
    )

    print("\nTheta values saved successfully.")

print("\nLogistic regression training completed.")
