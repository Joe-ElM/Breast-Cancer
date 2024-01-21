# Breast Cancer W.i.s.c.o.n.s.i.n (Original) Dataset

![Breast Cancer](/Images/Front.png)

## Project Overview

Welcome to the Breast Cancer dataset repository! This dataset is a renowned benchmark in the field of machine learning and healthcare. It features a comprehensive set of attributes, including clump thickness, size uniformity, shape uniformity, marginal adhesion, epithelial size, bare nuclei, bland chromatin, normal nucleoli, and mitoses. Each data instance corresponds to a cell nucleus, and the primary objective is to predict whether a tumor is malignant or benign. The 'class' column serves as the ground truth label, with 1 indicating malignancy and 0 indicating benignity. This dataset is a valuable resource for developing and testing machine learning models aimed at aiding in the early detection of breast cancer. Explore the data, delve into insightful analyses, and leverage it to advance your understanding of breast cancer diagnostics. Together, let's contribute to the ongoing efforts in healthcare and data science.

## Dataset

![Download](/Data/)

- **Structured:** Yes
- **Format:** Single .csv file
- **Number of Features:** [9]
- **Target Feature (Vector):** `class`, imbalanced, [35:65]
  ![Distribution](/Images/Distribution-of-the-classes.png)

- **Duplicates:** [0]

## Breast Cancer Classification Problem Space

- Binary classification challenge: Distinguishing between malignant and benign breast tumors.
- Moderate-sized dataset for training: Utilizing a dataset of moderate size to develop an effective machine learning model.
- Imbalanced class distribution: Addressing the inherent imbalance in the distribution of malignant and benign cases.
- Robust model development: Focusing on creating a resilient model capable of accurate predictions.
- Independent performance metrics: Employing metrics for a comprehensive evaluation, ensuring reliability and adaptability.
- Custom scoring with F1 score: Utilizing a customized scoring system that equally emphasizes recall and precision for effective breast cancer classification.

## Best Models

- Notebook 2. RFC Model (Random Forest)

## Process

### 1. Exploratory Data Analysis (EDA)

The EDA.ipynb file encapsulates the process of Exploratory Data Analysis (EDA), where I thoroughly explore and analyze datasets to extract valuable insights and discern patterns

### 2. Models

Developed base classification algorithms: Support Vector Machines (SVC), Logistic Regression, K-Nearest Neighbors, Decision Tree / Random Forest, Artificial Neural Networks, Ensemble.

### 3. Pipeline

## Pipeline and GridSearch Overview

### Random Forest Classifier with PCA

This pipeline incorporates a Random Forest Classifier with Principal Component Analysis (PCA) for effective feature reduction and classification. The components are as follows:

**1. Standardization (`std`):**

- Standardizes numerical features to ensure consistent scaling.

**2. PCA Dimensionality Reduction (`pca`):**

- Utilizes PCA to reduce the dimensionality of the feature space to enhance model efficiency while retaining essential information.

**3. Random Forest Classifier (`RFC`):**

- Implements a Random Forest Classifier with balanced class weights, addressing imbalanced class distribution.

### Hyperparameter GridSearch (`GridSearch_RFC`)

The hyperparameter grid search explores a range of configurations for the Random Forest Classifier within the PCA pipeline. The search space includes variations in the number of estimators, maximum depth, minimum samples per leaf, and minimum samples for splitting nodes.

```python
search_space_grid = {
    'RFC__n_estimators': [24, 25, 26],
    'RFC__max_depth': [1, 2, 3, 4, 5],
    'RFC__min_samples_leaf': [2, 3, 4, 5, 6],
    'RFC__min_samples_split': [2, 3, 4, 5],
}
```

#### Execution of the Pipeline:

To execute this specialized pipeline, ensure the necessary Python packages are installed. Execute the provided code within the relevant environment, adjusting the pipeline or model parameters as needed. This pipeline is crafted to enhance the performance of a classification model through advanced feature engineering.

### 6. Model Interpretation

Derived insights into feature importance through model-specific methodologies, including analyses of feature importance, and a single decision tree visualizations.

## Models' Performance Metrics

|               | K-Nearest Neighbors | Random Forest |
| :------------ | ------------------- | ------------- |
| **F1 Score**  | 96.87%              | 97.67%        |
| **Recall**    | 98.41%              | 100.0%        |
| **Precision** | 95.38%              | 95.45%        |
| **Accuracy**  | 97.36%              | 98.02%        |

## Choosing the Best Model

The best-performing model is `Random Forest Classifier`. It's tuned toward the F1 score

The confusion matrix of best scoring model is:

![Confusion Matrix](/Images/RF-CM.png)

## Model Interpretation

An evaluation of the constructed predictive model has been conducted using various performance metrics, aiming to ascertain its effectiveness in classifying auctioned vehicles into categories of good or bad buys. The model's performance has been quantified using several metrics, defined as follows:

- F1 Score: A metric that is measured to balance precision and recall, providing a single performance indicator, especially valuable when dealing with imbalanced classes.
- Recall: A metric reflecting the ability of the model to accurately identify and label the relevant (positive) cases.
- Precision: A representation of the proficiency of the model to ensure the relevancy of labeled cases.
- Accuracy: A metric showing the proportion of total predictions (both positive and negative) that were determined to be correct.
  Subsequent to the performance evaluation, an analysis of feature importance was performed to identify the most influential features in the predictions made by the model.

## Feature Importance

![Feature Importance](/Images/Features-Importance.png)

```

```
