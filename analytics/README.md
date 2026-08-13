# Analytics

## Overview
The Analytics module performs exploratory data analysis and predictive modeling using the Titanic customer-style dataset.

## Structure
```text
analytics/
├── 01_eda.py
├── 02_modeling.py
├── titanic.csv
├── model_comparison.csv
├── *.png
├── *.joblib
└── README.md
```

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Joblib

## Exploratory Data Analysis
The EDA workflow includes:
- Dataset shape and data types
- Summary statistics
- Missing-value analysis and handling
- Correlation analysis and heatmap
- Age and fare distributions
- Box plots and IQR-based outlier analysis
- Survival analysis by categorical variables

The correlation analysis uses:
```text
survived, pclass, age, sibsp, parch, fare
```

Missing-value handling follows:
- Less than 5% → drop affected rows
- 5%–30% → impute
- More than 30% → drop the column

## Predictive Modeling
Three classification models are evaluated:
1. Logistic Regression
2. Decision Tree
3. Random Forest

Metrics include:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Confusion matrices and ROC curves are generated.

## Additional Modeling
- Class imbalance analysis
- SMOTE comparison
- Random Forest GridSearchCV
- Random Forest OOB score
- Regression task
- MAE, RMSE, R² and Adjusted R²
- Regression residual plot
- Saved fitted pipelines using Joblib

## Important Outputs
```text
model_comparison.csv
roc_curves.png
regression_residuals.png
logistic_regression_pipeline.joblib
decision_tree_pipeline.joblib
random_forest_pipeline.joblib
best_random_forest_pipeline.joblib
smote_logistic_pipeline.joblib
regression_pipeline.joblib
```

## Running
From the repository root:
```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```

Install required packages:
```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn joblib
```

## Design Decisions
Preprocessing is kept inside scikit-learn pipelines to reduce data leakage and make transformations reproducible. Multiple classifiers are compared using the same evaluation framework, while SMOTE, hyperparameter tuning, OOB evaluation, and regression provide additional validation.
