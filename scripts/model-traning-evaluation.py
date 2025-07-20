from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, mean_absolute_error,
                             r2_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import numpy as np

# Define a dictionary of classifiers
classifiers = {
    'Bernoulli Naive Bayes': BernoulliNB(alpha=0.1, binarize=0.0),
    'Decision Tree': DecisionTreeClassifier(max_depth=9, min_samples_split=5, random_state=101),
    'Logistic Regression': LogisticRegression(solver='liblinear', penalty='l1', C=10, random_state=101),
    'XGB': XGBClassifier(learning_rate=0.2, max_depth=7, n_estimators=500, random_state=101, tree_method='gpu_hist')
}

# === Linear Regression Model ===
print("\nTraining Linear Regression Model...")
linear_reg = LinearRegression()
linear_reg.fit(X_train_tfidf, y_train_encoded)

# Predictions (regression)
y_pred_lr = linear_reg.predict(X_test_tfidf)

# Convert to classes
y_pred_lr_classes = np.round(y_pred_lr).astype(int)
y_pred_lr_classes = np.clip(y_pred_lr_classes, 0, len(le.classes_)-1)

# Evaluation
print("\nLinear Regression Results:")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_lr_classes))
print(classification_report(y_test_encoded, y_pred_lr_classes))

print("\nRegression Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test_encoded, y_pred_lr))
print("Mean Absolute Error:", mean_absolute_error(y_test_encoded, y_pred_lr))
print("R-squared:", r2_score(y_test_encoded, y_pred_lr))

# === XGBoost Classifier (example use) ===
print("\nTraining XGBoost...")
xgb_reg = classifiers['XGB']
xgb_reg.fit(X_train_tfidf, y_train_encoded)
y_pred = xgb_reg.predict(X_test_tfidf)

print("\nXGBoost Results:")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
print(classification_report(y_test_encoded, y_pred))

# === Plotting: Classification Metrics Comparison ===
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
xgb_scores = [
    accuracy_score(y_test_encoded, y_pred),
    precision_score(y_test_encoded, y_pred, average='weighted'),
    recall_score(y_test_encoded, y_pred, average='weighted'),
    f1_score(y_test_encoded, y_pred, average='weighted')
]

lr_scores = [
    accuracy_score(y_test_encoded, y_pred_lr_classes),
    precision_score(y_test_encoded, y_pred_lr_classes, average='weighted'),
    recall_score(y_test_encoded, y_pred_lr_classes, average='weighted'),
    f1_score(y_test_encoded, y_pred_lr_classes, average='weighted')
]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, xgb_scores, width, label='XGBoost')
rects2 = ax.bar(x + width/2, lr_scores, width, label='Linear Regression')

ax.set_ylabel('Scores')
ax.set_title('Model Comparison by Classification Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

fig.tight_layout()
plt.show()

# === Plotting: Regression Metrics Comparison ===
reg_metrics = ['MSE', 'MAE', 'R-squared']
x = np.arange(len(reg_metrics))

xgb_reg_scores = [
    mean_squared_error(y_test_encoded, xgb_reg.predict(X_test_tfidf)),
    mean_absolute_error(y_test_encoded, xgb_reg.predict(X_test_tfidf)),
    r2_score(y_test_encoded, xgb_reg.predict(X_test_tfidf))
]

lr_reg_scores = [
    mean_squared_error(y_test_encoded, y_pred_lr),
    mean_absolute_error(y_test_encoded, y_pred_lr),
    r2_score(y_test_encoded, y_pred_lr)
]

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, xgb_reg_scores, width, label='XGBoost')
rects2 = ax.bar(x + width/2, lr_reg_scores, width, label='Linear Regression')

ax.set_ylabel('Scores')
ax.set_title('Model Comparison by Regression Metrics')
ax.set_xticks(x)
ax.set_xticklabels(reg_metrics)
ax.legend()

fig.tight_layout()
plt.show()