# Customer Purchase Prediction Using Logistic Regression and Random Forest

## Project Overview

Customer Purchase Prediction is a machine learning project developed using Python to predict whether a customer is likely to purchase a product based on demographic and behavioural information.

The project implements **Logistic Regression** and **Random Forest** for supervised classification. It also explores customer groups using unsupervised learning techniques such as K-Means Clustering, DBSCAN, and Hierarchical Clustering.

The project covers the complete machine learning workflow, including data cleaning, preprocessing, exploratory data analysis, model training, hyperparameter tuning, evaluation, and prediction for a new customer.

## Objectives

- Analyse customer demographic and behavioural data.
- Identify and handle duplicate records and missing values.
- Perform exploratory data analysis (EDA).
- Prepare features using imputation, encoding, and scaling.
- Build Logistic Regression and Random Forest classifiers.
- Explore customer groups using unsupervised learning.
- Tune Random Forest hyperparameters using GridSearchCV.
- Evaluate models using multiple classification metrics.
- Demonstrate purchase prediction for a new customer.

## Dataset Description

The dataset contains **5,000 customer records and 13 columns**.

| Attribute | Description |
|---|---|
| `Customer_ID` | Customer identifier |
| `Age` | Customer's age |
| `Gender` | Customer's gender |
| `Annual_Income` | Annual income |
| `Customer_Segment` | Customer category |
| `Previous_Purchases` | Number of previous purchases |
| `Website_Visits` | Number of website visits |
| `Pages_Viewed` | Number of pages viewed |
| `Time_on_Website_Min` | Time spent on the website |
| `Emails_Opened` | Number of emails opened |
| `Cart_Additions` | Number of cart additions |
| `Days_Since_Last_Purchase` | Days since the previous purchase |
| `Purchase` | Target variable: 0 = No Purchase, 1 = Purchase |

### Dataset Cleaning

- Initial records: 5,000
- Initial columns: 13
- Duplicate rows removed: 30
- Records after duplicate removal: 4,970
- Original missing cells identified: 45
- Identifier excluded from model inputs: `Customer_ID`
- Final supervised inputs: 11 predictors and 1 target

## Technologies Used

- **Programming Language:** Python
- **Environment:** Jupyter Notebook
- **Libraries:**
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - SciPy

## Project Workflow

1. Load the customer dataset.
2. Inspect the dataset structure and data types.
3. Identify and remove duplicate records.
4. Check and handle missing values.
5. Remove the `Customer_ID` identifier from model inputs.
6. Perform exploratory data analysis.
7. Split the dataset into training and testing sets.
8. Apply preprocessing using `SimpleImputer`, `OneHotEncoder`, and `StandardScaler`.
9. Train Logistic Regression and Random Forest classifiers.
10. Explore customer groups using unsupervised learning.
11. Tune Random Forest hyperparameters using GridSearchCV.
12. Evaluate and compare model performance.
13. Generate a prediction for a new customer.

## Data Preprocessing

The following preprocessing techniques are used:

- **Duplicate Removal:** Removes repeated records.
- **Missing-Value Imputation:** Uses median imputation for numerical features and most-frequent imputation for categorical features.
- **Categorical Encoding:** Uses `OneHotEncoder` to convert categorical values into numerical features.
- **Feature Scaling:** Uses `StandardScaler` to standardize numerical features.
- **Identifier Removal:** Excludes `Customer_ID` from model inputs.
- **Train-Test Split:** Uses a stratified 80:20 split with `random_state=42`.

Preprocessing is included in a Scikit-learn `Pipeline` and `ColumnTransformer` so that learned transformations are fitted on training data rather than the test set.

## Exploratory Data Analysis (EDA)

EDA is performed to understand customer characteristics and relationships in the dataset.

The analysis includes:

- Descriptive statistics
- Purchase-class distribution
- Numerical feature distributions
- Categorical feature distributions
- Boxplots comparing features across purchase classes
- Correlation heatmap

The cleaned dataset contains 2,905 No Purchase records and 2,065 Purchase records. The analysis indicates that individual numerical features have generally weak linear correlations with the purchase target.

## Machine Learning Models

### 1. Logistic Regression

Logistic Regression is used as a baseline binary classification model. It estimates the probability of purchase and predicts either Purchase or No Purchase.

### 2. Random Forest Classifier

Random Forest combines multiple decision trees to model nonlinear relationships and interactions between customer features. Feature importance is also analysed to understand which features contribute most to the fitted model.

### 3. Tuned Random Forest

GridSearchCV with five-fold cross-validation is used to explore Random Forest hyperparameters.

The reported best configuration is:

- `n_estimators = 50`
- `max_depth = 10`
- `min_samples_split = 5`

The reported best cross-validation accuracy is 60.89%.

## Unsupervised Learning

The project also explores customer groupings using the following techniques:

### K-Means Clustering
Groups customers based on similarity in standardized numerical features.

### Elbow Method
Examines K-Means inertia for different numbers of clusters to help select a suitable value of K.

### Silhouette Score
Evaluates cluster cohesion and separation. The reported highest tested score is approximately 0.0837 at K = 2.

### DBSCAN
Identifies dense groups and observations labelled as noise or outliers. The reported run identified 32 clusters and 1,353 noise observations using `eps=1.5` and `min_samples=5`.

### Hierarchical Clustering
Uses agglomerative clustering with Ward's linkage and a dendrogram to visualize how customer groups merge.

These clustering methods are exploratory and are separate from the supervised purchase prediction task.

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC

### Reported Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 61.77% | 57.02% | 32.45% | 41.36% | 0.6351 |
| Random Forest | 57.95% | 48.90% | 26.88% | 34.69% | 0.5775 |
| Tuned Random Forest | 59.86% | 53.40% | 26.63% | 35.54% | 0.6046 |

### Results and Discussion

Logistic Regression achieved the highest reported test accuracy and F1-score among the three models. Tuned Random Forest achieved a reported cross-validation accuracy of 60.89%, but its test accuracy was 59.86%.

The relatively low recall values indicate that the models miss a number of actual purchasers. Further experimentation with feature engineering, class weighting, probability calibration, or decision thresholds could be investigated.

## Sample Prediction

A sample customer record is passed through the trained Logistic Regression pipeline.

| Feature | Example Value |
|---|---:|
| Age | 35 |
| Gender | Male |
| Annual Income | 80000 |
| Customer Segment | Regular |
| Previous Purchases | 5 |
| Website Visits | 10 |
| Pages Viewed | 6 |
| Time on Website | 12 minutes |
| Emails Opened | 4 |
| Cart Additions | 2 |
| Days Since Last Purchase | 60 |

**Reported output:**
- Predicted class: `0` — No Purchase
- Estimated purchase probability: `35.04%`

This probability is a model estimate, not a guarantee of future customer behaviour.

## How to Run the Project

### Prerequisites

Install Python and Jupyter Notebook.

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### 3. Add the Dataset

Place `Customer_Purchase_Prediction.csv` in the directory expected by the notebook.

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the project notebook and run the cells in order, from dataset loading through model evaluation and final prediction.

## Project Structure

The repository can be organized as follows:

```text
Customer-Purchase-Prediction/
├── Customer_Purchase_Prediction.csv
├── Customer_Purchase_Prediction.ipynb
├── README.md
├── report/
│   └── Customer_Purchase_Prediction_Report.pdf
└── presentation/
    └── Customer_Purchase_Prediction_Presentation.pptx
```

Update the filenames and folders to match the files actually uploaded to your repository.

## Limitations

- The models' purchase-class recall is relatively low.
- Predictions may be less reliable for customer values outside the training-data range.
- Clustering results depend on feature scaling and parameter selection.
- Feature importance does not establish causation.
- Results should be revalidated before real-world business use.

## Future Enhancements

- Evaluate class weighting and resampling methods.
- Explore alternative classification algorithms.
- Improve purchase-class recall.
- Evaluate probability calibration and decision thresholds.
- Validate performance on newer customer records.
- Develop a user-friendly prediction interface.

## Conclusion

This project demonstrates a complete machine-learning workflow for customer purchase prediction. It combines data cleaning, preprocessing, exploratory analysis, supervised classification, unsupervised clustering, hyperparameter tuning, model evaluation, and prediction for a new customer.

The results show that Logistic Regression performed best on the reported test accuracy and F1-score, while also highlighting the need to improve the identification of actual purchasers.

## Authors

- Gnaneshwar
- Vivek
- Hasini

**Department:** Computer Science and Engineering  
**Academic Year:** 2026–2027

## Acknowledgements

We thank our project guide and faculty for their guidance and support throughout the project.

