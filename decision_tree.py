import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("data/Fraud_check.csv")

# Create a 'Risk' column based on Taxable.Income --  0 = Risky, 1 = Good
df["Risk"] = np.where(df["Taxable.Income"] <= 30000, 0, 1)

# Drop unnecessary columns (City.Population and Taxable.Income)
df = df.drop(columns=["City.Population", "Taxable.Income"])

# Convert categorical columns into numerical format (binary or multiple values)
df["Undergrad"] = df["Undergrad"].map({"YES": 1, "NO": 0})
df["Marital.Status"] = df["Marital.Status"].map(
    {"Single": 0, "Married": 1, "Divorced": 2}
)
df["Urban"] = df["Urban"].map({"YES": 1, "NO": 0})

# Separate the features (X) and target (y)
X = df.drop(columns=["Risk"])  # Features
y = df["Risk"]  # Target (Risk)


# Split the data into 80% training and 20% testing
train_size = int(0.8 * len(df))
X_train, X_test = X[:train_size], X[train_size:]  # First 80% for training
y_train, y_test = y[:train_size], y[train_size:]  # Remaining 20% for testing


# GINI Index calculation
def gini_index(groups, classes):
    total_samples = sum([len(group) for group in groups])  # Total number of samples

    gini = 0.0  # Initialize GINI index
    for group in groups:
        size = len(group)
        if size == 0:  # If group is empty, skip it
            continue

        # Calculate the score for the group
        score = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / size
            score += proportion * proportion

        # GINI for this group, weighted by its size
        gini += (1.0 - score) * (size / total_samples)

    return gini


# Function to split the dataset based on a feature and its value
def test_split(index, value, dataset):
    left, right = list(), list()  # Create empty lists for left and right branches
    for row in dataset:
        if (
            row[index] < value
        ):  # If the feature's value is less than the split value, go left
            left.append(row)
        else:  # Otherwise, go right
            right.append(row)
    return left, right


# Find the best split based on GINI index
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))  # Get unique classes (0, 1)
    best_index, best_value, best_gini, best_groups = (
        999,
        999,
        999,
        None,
    )  # Initialize best values

    for index in range(len(dataset[0]) - 1):  # For each feature
        for row in dataset:  # For each row in dataset
            groups = test_split(index, row[index], dataset)  # Split the dataset
            gini = gini_index(groups, class_values)  # Calculate GINI index
            if gini < best_gini:  # If GINI is lower, update the best split
                best_index, best_value, best_gini, best_groups = (
                    index,
                    row[index],
                    gini,
                    groups,
                )

    return {"index": best_index, "value": best_value, "groups": best_groups}


# Create a terminal node (decide on the final class if no more splitting is possible)
def to_terminal(group):
    outcomes = [row[-1] for row in group]  # Collect class labels (Risky or Good)
    return max(set(outcomes), key=outcomes.count)  # Return the most common class


# Recursive function to split the node and build the tree
def split(node, max_depth, min_size, depth):
    left, right = node["groups"]  # Get the left and right groups of data
    del node["groups"]  # Remove the groups from the node after splitting

    # Check if no more splitting is needed (either side is empty)
    if not left or not right:
        node["left"] = node["right"] = to_terminal(left + right)
        return

    # If max depth is reached, stop splitting
    if depth >= max_depth:
        node["left"], node["right"] = to_terminal(left), to_terminal(right)
        return

    # Split the left side of the tree
    if len(left) <= min_size:
        node["left"] = to_terminal(left)
    else:
        node["left"] = get_split(left)
        split(node["left"], max_depth, min_size, depth + 1)

    # Split the right side of the tree
    if len(right) <= min_size:
        node["right"] = to_terminal(right)
    else:
        node["right"] = get_split(right)
        split(node["right"], max_depth, min_size, depth + 1)


# Build the decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)  # Get the root node
    split(root, max_depth, min_size, 1)  # Start recursive splitting
    return root


# Make a prediction for a single row of data
def predict(node, row):
    if (
        row[node["index"]] < node["value"]
    ):  # If row's value is less than the node's split value, go left
        if isinstance(
            node["left"], dict
        ):  # If the left branch is another node, recurse
            return predict(node["left"], row)
        else:
            return node["left"]  # Otherwise, return the predicted class
    else:
        if isinstance(
            node["right"], dict
        ):  # If the right branch is another node, recurse
            return predict(node["right"], row)
        else:
            return node["right"]  # Otherwise, return the predicted class


# Convert training and test data to lists
train_data = np.column_stack([X_train.values, y_train.values]).tolist()
test_data = np.column_stack([X_test.values, y_test.values]).tolist()

# Build the decision tree
max_depth = 5  # Maximum depth of the tree
min_size = 10  # Minimum number of samples to further split
tree = build_tree(train_data, max_depth, min_size)

# Make predictions on the test data
predictions = [predict(tree, row) for row in test_data]

# Calculate accuracy
accuracy = sum([pred == row[-1] for pred, row in zip(predictions, test_data)]) / len(
    test_data
)
print(f"Accuracy: {accuracy * 100:.2f}%")
