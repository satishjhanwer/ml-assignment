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
- The decision tree is restricted by `max_depth = 5` and `min_size = 10` to prevent overfitting.

## Implementation Details

- Created a new target column named `Risk` using values from the `Taxable.Income` column. If `taxable_income <= 30000`, it is classified as `Risky(0)`; otherwise, it is classified as `Good(1)`.
- Dropped the `City.Population` and `Taxable.Income` columns/attributes.
- Used 80% of the data as `training data` and the remaining 20% as `test data`.
- Calculated the GINI index to measure the impurity of the information gain.
- Implemented a binary split to feature all the attributes.
- Converted categorical columns (`Undergrad`, `Marital.Status`, `Urban`) into numerical format.
- Implemented recursive splitting to build the decision tree and determine the best splits for each node.

## Model Evaluation

- **Confusion Matrix**:

  |              | Predicted Risky | Predicted Good |
  | ------------ | --------------- | -------------- |
  | Actual Risky | TP              | FN             |
  | Actual Good  | FP              | TN             |

- **Recall**: `92.31%`
- **Accuracy**: `81.67%`
- **F1-Score**: `89.72%`
- **Precision**: `87.27%`

## Result

- **Accuracy:** `81.67%`
- **Execution Time:** `0.1703 seconds`

## Challenges & Learnings

- The manual implementation of the decision tree provided deeper insights into how GINI index calculations work.
- Ensuring the dataset was properly preprocessed and categorical values were encoded for the decision tree.
- Learned the importance of tuning hyper-parameters like tree depth and node size to avoid overfitting or underfitting the model.

## Conclusion

The decision tree model achieved an accuracy of `81.67%`, which suggests that it performs well on the given dataset. Future improvements could involve experimenting with other algorithms (such as Random Forests or Gradient Boosting) and tuning hyper-parameters more finely to increase accuracy and generalization.
