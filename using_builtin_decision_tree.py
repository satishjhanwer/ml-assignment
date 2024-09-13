import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess data
df = pd.read_csv("data/Fraud_check.csv")
df["Risk"] = np.where(df["Taxable.Income"] <= 30000, 0, 1)
df = df.drop(columns=["City.Population", "Taxable.Income"])
df["Undergrad"] = df["Undergrad"].map({"YES": 1, "NO": 0})
df["Marital.Status"] = df["Marital.Status"].map(
    {"Single": 0, "Married": 1, "Divorced": 2}
)
df["Urban"] = df["Urban"].map({"YES": 1, "NO": 0})

X = df.drop(columns=["Risk"])
y = df["Risk"]

# Ensure correct 80:20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Check the sizes of the splits
print(f"Training data size: {len(X_train)}, Test data size: {len(X_test)}")

# Train Decision Tree with GINI index and fine-tuned parameters
clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
)
clf.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
