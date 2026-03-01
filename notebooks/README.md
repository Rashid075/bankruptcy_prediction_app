In this project, four classification models were implemented to predict corporate bankruptcy:

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

The objective was to compare linear, non-linear, tree-based, and margin-based approaches to determine which model best captures bankruptcy risk patterns.

All models achieved very high performance:

Accuracy ≥ 96%

Recall = 100% for bankruptcy class

Cross-validation accuracy ≈ 99.6%

Random Forest and SVM achieved perfect classification on the test set.

The consistently high performance across all models indicates:

Strong separability between bankrupt and non-bankrupt firms

Highly informative risk features

Low noise in dataset

Clear monotonic relationship between risk and bankruptcy

Even the linear model (Logistic Regression) performed extremely well, which suggests that the decision boundary is almost linearly separable.

Tree-based and margin-based models further captured small non-linearities, leading to near-perfect classification.

Random Forest was selected as the final model for its robustness and feature importance capability.
