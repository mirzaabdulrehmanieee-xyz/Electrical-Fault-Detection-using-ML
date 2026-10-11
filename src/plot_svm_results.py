
import matplotlib.pyplot as plt

# Validation results from our SVM experiments
C_values = [0.1, 1, 10, 100]

linear_accuracy = [72.58, 72.58, 72.58, 72.58]

rbf_gamma_01 = [95.00, 97.50, 99.21, 99.54]
rbf_gamma_1 = [98.04, 99.17, 99.62, 99.67]

plt.figure(figsize=(10, 6))

plt.plot(
    C_values, linear_accuracy,
    marker="o", label="Linear SVM"
)

plt.plot(
    C_values, rbf_gamma_01,
    marker="s", label="RBF SVM (gamma = 0.1)"
)

plt.plot(
    C_values, rbf_gamma_1,
    marker="^", label="RBF SVM (gamma = 1)"
)

plt.xscale("log")
plt.xlabel("C (log scale)")
plt.ylabel("Validation Accuracy (%)")
plt.title("SVM Validation Accuracy vs C")
plt.xticks(C_values, ["0.1", "1", "10", "100"])
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/figures/svm_validation_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("SVM plot saved successfully!")
