# ML Assignment - Decision Tree Model for Fraud Detection

## Problem Statement

For the given loan data, use Decision Trees to prepare a model on fraud data, treating those who have `taxable_income <= 30000` as `Risky` and others as `Good`.

## Group Members

- Satish Jhanwer (G24AIT009)
- Aditya Sharma (G24AIT011)
- Jyothsna Sravanthi (G24AIT015)

## Data Description

- **Undergrad**: Whether the person is an undergraduate or not (Binary: YES/NO).
- **Marital.Status**: Marital status of the person (Ternary: Single, Married, Divorced).
- **Work.Experience**: Total years of work experience (Numerical).
- **Urban**: Whether the person belongs to an urban area (Binary: YES/NO).
- **City.Population**: Population of the city (Numerical, dropped).
- **Taxable.Income**: Taxable income of the individual (Numerical, used to create target variable).

## Assumptions

- No missing values in the dataset.
- The decision tree is restricted by `max_depth in [3, 5, 7, 9]` and `min_size = 10` to prevent overfitting.

## Implementation Details

- Created a new target column named `Risk` using values from the `Taxable.Income` column. If `taxable_income <= 30000`, it is classified as `Risky(0)`; otherwise, it is classified as `Good(1)`.
- Dropped the `City.Population` and `Taxable.Income` columns/attributes.
- Used 80% of the data as `training data` and the remaining 20% as `test data`.
- Calculated the GINI index to measure the impurity of the information gain.
- Implemented binary split to feature all the attributes.
- Converted categorical columns (`Undergrad`, `Marital.Status`, `Urban`) into numerical format.
- Implemented recursive splitting to build the decision tree and determine the best splits for each node.

## Model Evaluation

We experimented with several tree depths `[3, 5, 7, 9]` to observe their impact on modal's accuracy and execution time.

**Decision Tree Experiment Results (Reduced Depths):**

| Max Depth | Accuracy | Precision | Recall  | F1 Score | Execution Time |
| --------- | -------- | --------- | ------- | -------- | -------------- |
| 3         | 86.67%   | 86.67%    | 100.00% | 92.86%   | 0.2256 seconds |
| 5         | 81.67%   | 87.27%    | 92.31%  | 89.72%   | 0.2350 seconds |
| 7         | 84.17%   | 87.61%    | 95.19%  | 91.24%   | 0.2531 seconds |
| 9         | 76.67%   | 86.54%    | 86.54%  | 86.54%   | 0.2775 seconds |

## Best Performing Model

- **Max Depth:** `3`
- **Recall**: `100.00%`
- **Accuracy:** `86.67%`
- **F1-Score**: `92.86%`
- **Precision**: `86.67%`
- **Execution Time:** `0.2256 seconds`

## Challenges & Learnings

- The manual implementation of the decision tree provided deeper insights into how GINI index calculations work.
- Ensuring the dataset was properly preprocessed and categorical values were encoded for the decision tree.
- Learned the importance of tuning hyper-parameters like tree depth and node size to avoid overfitting or underfitting the model.

## Conclusion

The decision tree model achieved an accuracy of `86.67%`, which suggests that it performs well on the given dataset. Future improvements could involve experimenting with other algorithms (such as Random Forests or Gradient Boosting) and tuning hyper-parameters more finely to increase accuracy and generalization.
