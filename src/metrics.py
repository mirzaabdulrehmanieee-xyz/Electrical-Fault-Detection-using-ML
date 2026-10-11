
import numpy as np


def accuracy(y_true, y_pred):
    # Count correct predictions and divide by total predictions
    correct = np.sum(y_true == y_pred)
    total = len(y_true)

    return correct / total


def confusion_matrix(y_true, y_pred):
    # Actual no-fault, predicted no-fault
    tn = np.sum((y_true == 0) & (y_pred == 0))

    # Actual no-fault, predicted fault
    fp = np.sum((y_true == 0) & (y_pred == 1))

    # Actual fault, predicted no-fault
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Actual fault, predicted fault
    tp = np.sum((y_true == 1) & (y_pred == 1))

    # Rows = actual labels; columns = predicted labels
    return np.array([
        [tn, fp],
        [fn, tp]
    ])


# Small example to check both functions
if __name__ == "__main__":
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])

    print("Accuracy:", accuracy(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
