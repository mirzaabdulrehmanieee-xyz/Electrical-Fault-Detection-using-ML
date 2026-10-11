
# Table 3 – SVM Validation Results

This table compares the validation accuracy and number of support vectors for linear and RBF SVM models using different values of C and gamma.

| Kernel | C | γ | Validation Accuracy | Support Vectors |
|---|---:|---:|---:|---:|
| Linear | 0.1 | — | 72.58% | 5,849 |
| Linear | 1 | — | 72.58% | 5,845 |
| Linear | 10 | — | 72.58% | 5,836 |
| Linear | 100 | — | 72.58% | 5,836 |
| RBF | 0.1 | 0.1 | 95.00% | 1,347 |
| RBF | 1 | 0.1 | 97.50% | 723 |
| RBF | 10 | 0.1 | 99.21% | 381 |
| RBF | 100 | 0.1 | 99.54% | 174 |
| RBF | 0.1 | 1 | 98.04% | 848 |
| RBF | 1 | 1 | 99.17% | 482 |
| RBF | 10 | 1 | 99.62% | 342 |
| RBF | 100 | 1 | 99.67% | 266 |

## Observation

The RBF kernel achieved higher validation accuracy than the linear kernel. The best RBF configuration was C = 100 and γ = 1, with 99.67% validation accuracy and 266 support vectors. The linear SVM achieved 72.58% validation accuracy for all tested values of C.
