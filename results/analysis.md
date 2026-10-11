# Task 6 – Analysis

## 6.1 Logistic Regression on Set R and Linear SVM

Logistic Regression on Set R achieved 72.64% test accuracy, while the linear SVM achieved 73.59%. The difference is only 0.95 percentage points.

Both models use the original six features and learn a linear decision boundary. Their similar performance suggests that a linear boundary cannot fully separate fault and non-fault samples in this dataset. A more flexible model may capture the relationship between current, voltage, and faults more effectively.

## 6.2 Signs of the Squared Feature Weights in Set Q

In Set Q, all three squared current terms (\(I_a^2\), \(I_b^2\), and \(I_c^2\)) have positive weights. Their coefficients are approximately 18.89, 4.75, and 13.96, respectively. Positive weights mean that larger squared current features increase the model's predicted probability of a fault, assuming other features remain constant. This is physically reasonable because current often increases during an electrical fault.

All three squared voltage terms (\(V_a^2\), \(V_b^2\), and \(V_c^2\)) have negative weights. Their coefficients are approximately -6.03, -6.46, and -6.14. Negative weights mean that larger squared voltage features decrease the model's predicted probability of a fault, assuming other features remain constant. This is also physically reasonable because voltage often drops during a fault.

Therefore, the signs of the learned squared-feature weights agree with the expected physical behaviour of increased current and reduced voltage during a fault.


## 6.3 Support Vectors in Linear and RBF SVM

The selected linear SVM used C = 0.1 and had approximately 5,849 support vectors. The best RBF SVM used C = 100 and gamma = 1, with 266 support vectors.

The linear SVM uses many more support vectors than the RBF SVM. A large number of support vectors means that many training samples influence the position of the decision boundary. This can indicate that the classes are difficult to separate with the selected kernel.

## 6.4 Effect of C on Accuracy and Support Vectors

The parameter C controls the penalty for classification errors in an SVM. A larger C penalizes training errors more strongly and encourages the model to classify training samples correctly.

For the linear SVM, validation accuracy remained around 72.58% as C increased from 0.1 to 100. The number of support vectors decreased slightly, from approximately 5,849 to 5,836.

For the RBF SVM, increasing C generally improved validation accuracy and reduced the number of support vectors. At gamma = 1, accuracy increased from 98.04% at C = 0.1 to 99.67% at C = 100, while the number of support vectors decreased from 848 to 266.

A larger C does not guarantee better performance on unseen data, so validation results should guide model selection.

## 6.5 Is the Test Accuracy Optimistic or Pessimistic?

The test accuracy is likely optimistic for a real transmission-line protection relay. The dataset contains consecutive time samples, and the random split can place very similar neighbouring samples in both the training and test sets.

As a result, the test set may be easier for the model to classify than genuinely new fault events. In real operation, a relay must detect faults under conditions that may not have appeared in its training data.

A chronological split, where later time samples are kept for testing, would provide a more realistic evaluation.

## 6.6 Model Selection for the Protection Relay

I would select the RBF SVM as the initial candidate for deployment because it achieved 99.58% test accuracy, with only 6 missed faults and 4 false alarms on the test set.

In comparison, Set Q Logistic Regression missed 45 faults and produced 6 false alarms, while the linear SVM missed 634 faults.

For a protection relay, missed faults are particularly dangerous because the relay may fail to respond to a real electrical fault. False alarms are also undesirable because they can cause unnecessary trips and interrupt power supply.

The RBF SVM provides the best results among the tested models. However, before actual deployment, it should be tested on independent, chronologically separated fault events and evaluated for response time, reliability, and safety requirements.
## 6.7 Practical Deployment Considerations

Although the RBF SVM achieved the best test accuracy in this experiment, selecting a model for an electrical protection system requires more than comparing accuracy scores.

A missed fault is particularly important because the system may fail to identify a dangerous electrical condition. The RBF SVM missed 6 faults, compared with 45 for Logistic Regression on Set Q. However, false alarms also matter because unnecessary trips can interrupt the power supply.

The dataset was randomly split even though its rows represent consecutive time samples. Similar samples may therefore appear in both the training and test sets. This can make the reported test performance optimistic compared with performance on independent operating conditions.

Before practical deployment, the selected model should be evaluated on independent electrical measurements, different fault scenarios, and unseen operating conditions. Its response time and reliability should also be tested. The current results identify a promising model for further evaluation, rather than proving that it is ready for a real protection relay.
