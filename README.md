# Breast Cancer Wisconsin (Original) Dataset

![Breast Cancer](/Images/Front.png)

## Project Overview

Welcome to the Breast Cancer dataset repository! This dataset is a renowned benchmark in the field of machine learning and healthcare. It features a comprehensive set of attributes, including clump thickness, size uniformity, shape uniformity, marginal adhesion, epithelial size, bare nuclei, bland chromatin, normal nucleoli, and mitoses. Each data instance corresponds to a cell nucleus, and the primary objective is to predict whether a tumor is malignant or benign. The 'class' column serves as the ground truth label, with 1 indicating malignancy and 0 indicating benignity. This dataset is a valuable resource for developing and testing machine learning models aimed at aiding in the early detection of breast cancer. Explore the data, delve into insightful analyses, and leverage it to advance your understanding of breast cancer diagnostics. Together, let's contribute to the ongoing efforts in healthcare and data science.

## Dataset

Data is available in download folder

- **Structured:** Yes
- **Format:** Single .csv file
- **Number of Features:** [9]
- **Target Feature (Vector):** `class`, imbalanced, [35:65]
  ![Distribution](/Images/Distribution-of-the-classes.png)

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

Developed base classification algorithms: K-Nearest Neighbors, Decision Tree and Random Forest.

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

## Visual Exploration of Selected Features

Explore the distribution of selected features through box plots and histograms. The visualizations provide insights into the spread and density of each feature across different classes.

![Feature Visualization](/Images/hist-boxplots-of-features.png)

## Scree Plot for Principal Components

The Scree plot visualizes the explained variance of each principal component obtained through Principal Component Analysis (PCA) in the K-Nearest Neighbors (KNN) pipeline. Each bar represents the proportion of variance explained by a specific principal component, helping to identify the significance of individual components in capturing the dataset's variability.

- **Violet Bars:** Indicate the explained variance for each principal component.
- **Red Line:** Represents the cumulative explained variance across principal components.

The plot aids in determining the optimal number of principal components to retain, striking a balance between dimensionality reduction and preserving the dataset's information. This insightful visualization guides the decision-making process when choosing the appropriate number of principal components for subsequent model training and analysis.

![Scree Plot](/Images/scree-plot.png)

## Feature Importance Analysis with PolynomialFeatures and PCA

Explore the feature importance analysis using PolynomialFeatures and Principal Component Analysis (PCA) within the K-Nearest Neighbors (KNN) pipeline. This pipeline introduces polynomial feature expansion and dimensionality reduction to enhance the model's ability to capture complex relationships within the breast cancer dataset.

### Pipeline Configuration:

- **PolynomialFeatures (`poly`):**

  - Degree: 2 (Quadratic feature expansion)
  - Include Bias: False (To avoid redundant features)

- **StandardScaler (`std`):**

  - Standardizes numerical features for consistent scaling.

- **PCA (`pca`):**

  - Number of Components: 6
  - Random State: 100

- **K-Nearest Neighbors (`knn`):**
  - Optimized parameters obtained from GridSearch_KNN.

### Feature Importance Visualization:

The eigenvectors matrix, representing the influence of features on principal components, is illustrated. This analysis unveils the importance of features in the first two principal components.

#### Eigenvector for Principal Component 1 and 2 (pc1/pc2):

![Eigenvector for pc1](/Images/PCA-eigenvectores.png)

The bar plots provide a visual representation of feature importance within each principal component, aiding in understanding the contribution of individual features to the model's performance. Analyzing these eigenvectors is crucial for interpreting the impact of features on the derived principal components.

## Decision Tree Visualization for Feature Importance

Explore the decision tree from a Random Forest Classifier with balanced class weights. This tree is specifically chosen for visualization to simplify the understanding of feature importance calculations.

The RandomForestClassifier is configured with the following parameters:

- Number of Estimators: 25
- Maximum Depth: 4
- Minimum Samples per Leaf: 2
- Minimum Samples for Splitting Nodes: 2

The decision tree is a fundamental component of the Random Forest model, and visualizing a single tree provides insights into how features are utilized for classification. Each node in the tree represents a decision based on a specific feature, contributing to the overall predictive capability of the model.

The visualization includes:

- **Feature Names:** Corresponding to the features in the dataset.
- **Class Names:** Representing the target classes ('Benign' and 'Malignant').
- **Filled Nodes:** Indicating the class distribution within each node.

This visualization is valuable for understanding the decision-making process of the Random Forest model and identifying influential features in predicting breast cancer classification.

![Tree Plot](/Images/Tree-0.png)
