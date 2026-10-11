
# Table 4 – Test Results

This table compares the final models on the test dataset. TN represents correctly identified no-fault samples, FP represents false alarms, FN represents missed faults, and TP represents correctly detected faults.

| Model | Accuracy | TN | False Alarms (FP) | Missed Faults (FN) | TP |
|---|---:|---:|---:|---:|---:|
| Logistic Regression, Set R | 72.64% | 1,305 | 0 | 657 | 439 |
| Logistic Regression, Set Q | 97.88% | 1,299 | 6 | 45 | 1,051 |
| Linear SVM (C = 0.1) | 73.59% | 1,305 | 0 | 634 | 462 |
| RBF SVM (C = 100, γ = 1) | 99.58% | 1,301 | 4 | 6 | 1,090 |

## Observation

The RBF SVM achieved the highest test accuracy of 99.58% and missed only 6 faults. Set Q Logistic Regression missed 45 faults, while Set R Logistic Regression and the Linear SVM missed substantially more faults. The RBF SVM is the strongest candidate among these models based on the current test results, although further testing is needed before real-world deployment.
