import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
)
from sklearn.naive_bayes import GaussianNB
from time import time


class NaiveBayesScratch:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.variance = {}
        self.priors = {}

    def fit(self, X, Y):
        self.classes = Y.unique()
        self.class_prior = {c: 0 for c in self.classes}
        total_samples = len(Y)
        for c in self.classes:
            self.class_prior[c] = len(Y[Y == c]) / total_samples
        self.mean = {c: X[Y == c].mean() for c in self.classes}
        self.variance = {c: X[Y == c].var() for c in self.classes}

    def calculate_probability(self, x, mean, var):
        eps = 1e-6
        numerator = np.exp(-((x - mean) ** 2) / (2 * var + eps))
        denominator = np.sqrt(2 * np.pi * var + eps)
        return numerator / denominator

    def predict(self, X):
        predictions = [self._predict(X.iloc[i]) for i in range(len(X))]
        return predictions

    def _predict(self, x):
        probabilities = {}
        for c in self.classes:
            probabilities[c] = self.class_prior[c]
            for feature in x.index:
                probabilities[c] *= self.calculate_probability(
                    x[feature], self.mean[c][feature], self.variance[c][feature]
                )
        return max(probabilities, key=probabilities.get)


data = pd.read_csv("./ion_binary_classification.csv")

data = data.drop(columns=["Unnamed: 0"])
data["Class"] = data["Class"].map(
    {"good": 1, "bad": 0}
)  # Convert 'good' to 1 and 'bad' to 0
data["V1"] = data["V1"].astype(int)
data["V2"] = data["V2"].astype(int)
X = data.drop(columns=["Class"])
Y = data["Class"]

print("Class distribution:")
print(Y.value_counts())

zero_variance_features = X.columns[X.std() == 0]
for feature in zero_variance_features:
    X[f"{feature}_flag"] = (X[feature] != 0).astype(int)
train_size = int(0.8 * len(data))
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]
X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)
print(f"Training set shape: {X_train.shape}, {Y_train.shape}")
print(f"Test set shape: {X_test.shape}, {Y_test.shape}")

# print("Basic statistics:")
# print(X.describe())

# plt.figure(figsize=(12, 6))
# sns.boxplot(data=X)
# plt.title("Boxplot of Features")
# plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
# plt.show()

# X.hist(figsize=(12, 10), bins=30, edgecolor="black")
# plt.suptitle("Histograms of Features")
# plt.tight_layout()
# plt.show()


# zero_variance_features = X.columns[X.std() == 0]
# print(f"Features with zero variance: {zero_variance_features}")

# X_filtered = X.drop(columns=zero_variance_features)
# X_filtered = X_filtered.fillna(X_filtered.mean())
# corr_with_target = X_filtered.corrwith(Y.replace({"good": 1, "bad": 0})).abs()
# top_corr_features_with_target = (
#     corr_with_target.sort_values(ascending=False).head(4).index
# )
# data_visual_top = pd.concat([X_filtered[top_corr_features_with_target], Y], axis=1)
# sns.pairplot(data_visual_top, hue="Class")
# plt.show()


# corr_matrix = X.corr()
# top_corr_features = (
#     corr_matrix.abs().unstack().sort_values(ascending=False).drop_duplicates()
# )
# top_features = (
#     top_corr_features[top_corr_features > 0.5].index.get_level_values(0).unique()[:10]
# )
# plt.figure(figsize=(10, 8))
# sns.heatmap(X[top_features].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
# plt.title("Top Feature Correlation Heatmap")
# plt.show()


def confusion_matrix_manual(actual, predicted):
    TP = sum((actual == 1) & (predicted == 1))  # True Positives
    TN = sum((actual == 0) & (predicted == 0))  # True Negatives
    FP = sum((actual == 0) & (predicted == 1))  # False Positives
    FN = sum((actual == 1) & (predicted == 0))  # False Negatives
    return TP, TN, FP, FN


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f1


# Model training
# nb_scratch = NaiveBayesScratch()
# nb_scratch.fit(X_train, Y_train)
# y_pred = nb_scratch.predict(X_test)
# accuracy = np.mean(y_pred == Y_test)
# TP, TN, FP, FN = confusion_matrix_manual(np.array(Y_test), np.array(y_pred))
# precision, recall, f1 = precision_recall_f1(TP, FP, FN)
# confusion_matrix = pd.crosstab(
#     Y_test, y_pred, rownames=["Actual"], colnames=["Predicted"], margins=True
# )


# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"Precision: {precision * 100:.2f}%")
# print(f"Recall: {recall * 100:.2f}%")
# print(f"F1 Score: {f1 * 100:.2f}%")


# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix")
# plt.show()

# fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color="blue", label="ROC curve (area = %0.2f)" % roc_auc)
# plt.plot([0, 1], [0, 1], color="red", linestyle="--")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver Operating Characteristic (ROC) Curve")
# plt.legend(loc="lower right")
# plt.show()


# precision_vals, recall_vals, _ = precision_recall_curve(Y_test, y_pred)

# plt.figure(figsize=(8, 6))
# plt.plot(recall_vals, precision_vals, color="blue")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.show()

# feature_importance = pd.DataFrame(
#     {
#         "Feature": X.columns,
#         "Mean (Class 1)": [nb_scratch.mean[1][feature] for feature in X.columns],
#         "Mean (Class 0)": [nb_scratch.mean[0][feature] for feature in X.columns],
#     }
# )

# feature_importance.set_index("Feature").plot(kind="bar", figsize=(12, 6))
# plt.title("Feature Means for Each Class")
# plt.ylabel("Mean Value")
# plt.xlabel("Features")
# plt.show()


# plt.figure(figsize=(8, 6))
# sns.countplot(x=y_pred)
# plt.title("Distribution of Predicted Classes")
# plt.xlabel("Predicted Class")
# plt.ylabel("Count")
# plt.show()


# plt.figure(figsize=(12, 8))
# for feature in X.columns[:4]:
#     sns.kdeplot(
#         X.iloc[Y_test.index][feature][Y_test == 0],
#         label=f"Class 0 - {feature}",
#         fill=True,
#     )
#     sns.kdeplot(
#         X.iloc[Y_test.index][feature][Y_test == 1],
#         label=f"Class 1 - {feature}",
#         fill=True,
#     )
# plt.title("Feature Distributions by Class")
# plt.legend()
# plt.show()

# Initialize models
nb_scratch = NaiveBayesScratch()
nb_sklearn = GaussianNB()

# Timing Scratch Implementation
start_time = time()
nb_scratch.fit(X_train, Y_train)
y_pred_scratch = nb_scratch.predict(X_test)
scratch_time = time() - start_time

# Timing Sklearn Implementation
start_time = time()
nb_sklearn.fit(X_train, Y_train)
y_pred_sklearn = nb_sklearn.predict(X_test)
sklearn_time = time() - start_time

# Calculate metrics for both
metrics_scratch = classification_report(
    Y_test, np.array(y_pred_scratch), output_dict=True, zero_division=0
)

print("Scratch Implementation:")
print(metrics_scratch)

metrics_sklearn = classification_report(
    Y_test, y_pred_sklearn, output_dict=True, zero_division=0
)

print("Sklearn Implementation:")
print(metrics_sklearn)


comparison_data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "Execution Time (s)"],
    "NaiveBayesScratch": [
        metrics_scratch.get("accuracy", 0),
        metrics_scratch.get("1", {}).get("precision", 0),
        metrics_scratch.get("1", {}).get("recall", 0),
        metrics_scratch.get("1", {}).get("f1-score", 0),
        scratch_time,
    ],
    "GaussianNB": [
        metrics_sklearn.get("accuracy", 0),
        metrics_sklearn.get("1", {}).get("precision", 0),
        metrics_sklearn.get("1", {}).get("recall", 0),
        metrics_sklearn.get("1", {}).get("f1-score", 0),
        sklearn_time,
    ],
}

# Prepare a comparison table
comparison_df = pd.DataFrame(comparison_data)

print(comparison_df)
