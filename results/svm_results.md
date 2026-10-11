## 4.3 SVM Validation Results

The following table shows the validation accuracy and number of support vectors for the linear and RBF SVM models using different values of C and gamma.

| Kernel | C | γ | Validation Accuracy | Support Vectors |
|---|---:|---:|---:|---:|
| Linear | 0.1 | — | 72.58% | 5849 |
| Linear | 1 | — | 72.58% | 5845 |
| Linear | 10 | — | 72.58% | 5836 |
| Linear | 100 | — | 72.58% | 5836 |
| RBF | 0.1 | 0.1 | 95.00% | 1347 |
| RBF | 1 | 0.1 | 97.50% | 723 |
| RBF | 10 | 0.1 | 99.21% | 381 |
| RBF | 100 | 0.1 | 99.54% | 174 |
| RBF | 0.1 | 1 | 98.04% | 848 |
| RBF | 1 | 1 | 99.17% | 482 |
| RBF | 10 | 1 | 99.62% | 342 |
| RBF | 100 | 1 | 99.67% | 266 |

The best linear SVM achieved 72.58% validation accuracy with C = 0.1. The best RBF SVM achieved 99.67% validation accuracy with C = 100 and gamma = 1.
