import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import joblib


# ============================================================
# PART-B — PREDICTIVE MODELING
# ============================================================

# 1. LOAD DATA
# ============================================================

df = pd.read_csv("analytics/titanic.csv")

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print("Shape:", df.shape)


# ============================================================
# 2. SELECT FEATURES AND TARGET
# ============================================================

target = "survived"

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features]
y = df[target]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 4. PREPROCESSING
# ============================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 5. CLASSIFICATION MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        oob_score=True
    )
}


# ============================================================
# 6. TRAIN AND EVALUATE CLASSIFIERS
# ============================================================

results = []
trained_models = {}

roc_data = {}


for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Fit only on training data
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[model_name] = pipeline

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # Store ROC information
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_data[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc
    }


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)

results_df.to_csv(
    "analytics/model_comparison.csv",
    index=False
)


# ============================================================
# 8. CONFUSION MATRICES
# ============================================================

for model_name, pipeline in trained_models.items():

    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n" + "=" * 60)
    print(model_name, "— Confusion Matrix")
    print("=" * 60)

    print(cm)


# ============================================================
# 9. ROC CURVES
# ============================================================

plt.figure(figsize=(8, 6))

for model_name, data in roc_data.items():

    plt.plot(
        data["fpr"],
        data["tpr"],
        label=f"{model_name} (AUC = {data['auc']:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Classification Models")
plt.legend()
plt.tight_layout()

plt.savefig(
    "analytics/roc_curves.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. RANDOM FOREST OOB SCORE
# ============================================================

rf_pipeline = trained_models["Random Forest"]

rf_model = rf_pipeline.named_steps["model"]

print("\n" + "=" * 60)
print("RANDOM FOREST OOB SCORE")
print("=" * 60)

print(
    "OOB Score:",
    round(rf_model.oob_score_, 4)
)


# ============================================================
# 11. RANDOM FOREST GRIDSEARCHCV
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST GRID SEARCH")
print("=" * 60)


rf_pipeline_grid = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                oob_score=True
            )
        )
    ]
)


param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5]
}


grid_search = GridSearchCV(
    estimator=rf_pipeline_grid,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


grid_search.fit(
    X_train,
    y_train
)


best_rf = grid_search.best_estimator_

print("Best Parameters:")
print(grid_search.best_params_)

print(
    "Best CV F1 Score:",
    round(grid_search.best_score_, 4)
)


best_rf_pred = best_rf.predict(X_test)

best_rf_prob = best_rf.predict_proba(X_test)[:, 1]

best_rf_f1 = f1_score(
    y_test,
    best_rf_pred,
    zero_division=0
)

best_rf_auc = roc_auc_score(
    y_test,
    best_rf_prob
)

print(
    "Test F1 Score:",
    round(best_rf_f1, 4)
)

print(
    "Test ROC-AUC:",
    round(best_rf_auc, 4)
)

print(
    "Best RF OOB Score:",
    round(
        best_rf.named_steps["model"].oob_score_,
        4
    )
)


# ============================================================
# 12. CLASS IMBALANCE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print(
    y_train.value_counts()
)

print(
    "\nClass proportions:"
)

print(
    y_train.value_counts(normalize=True)
)


# ============================================================
# 13. SMOTE COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("SMOTE COMPARISON")
print("=" * 60)


smote_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


smote_pipeline.fit(
    X_train,
    y_train
)


smote_pred = smote_pipeline.predict(
    X_test
)

smote_prob = smote_pipeline.predict_proba(
    X_test
)[:, 1]


smote_accuracy = accuracy_score(
    y_test,
    smote_pred
)

smote_precision = precision_score(
    y_test,
    smote_pred,
    zero_division=0
)

smote_recall = recall_score(
    y_test,
    smote_pred,
    zero_division=0
)

smote_f1 = f1_score(
    y_test,
    smote_pred,
    zero_division=0
)

smote_auc = roc_auc_score(
    y_test,
    smote_prob
)


print(
    "SMOTE Accuracy :",
    round(smote_accuracy, 4)
)

print(
    "SMOTE Precision:",
    round(smote_precision, 4)
)

print(
    "SMOTE Recall   :",
    round(smote_recall, 4)
)

print(
    "SMOTE F1       :",
    round(smote_f1, 4)
)

print(
    "SMOTE ROC-AUC  :",
    round(smote_auc, 4)
)


# ============================================================
# 14. REGRESSION TASK
# ============================================================

print("\n" + "=" * 60)
print("REGRESSION TASK")
print("=" * 60)

# Predict fare using the remaining available features

regression_features = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]

regression_target = "fare"

regression_df = df[
    regression_features + [regression_target]
].dropna()


X_reg = regression_df[
    regression_features
]

y_reg = regression_df[
    regression_target
]


X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)


regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            ),
            regression_features
        )
    ]
)


regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            regression_preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)


regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)


y_reg_pred = regression_pipeline.predict(
    X_reg_test
)


mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)


n = len(y_reg_test)
p = X_reg_test.shape[1]

if n > p + 1:
    adjusted_r2 = (
        1
        - (
            (1 - r2)
            * (n - 1)
            / (n - p - 1)
        )
    )
else:
    adjusted_r2 = np.nan


print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))
print(
    "Adjusted R²:",
    round(adjusted_r2, 4)
)


# ============================================================
# 15. REGRESSION RESIDUAL PLOT
# ============================================================

residuals = y_reg_test - y_reg_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Regression Residual Plot")

plt.tight_layout()

plt.savefig(
    "analytics/regression_residuals.png",
    dpi=300
)

plt.show()


# ============================================================
# 16. SAVE COMPLETE FITTED PIPELINES
# ============================================================

joblib.dump(
    trained_models["Logistic Regression"],
    "analytics/logistic_regression_pipeline.joblib"
)

joblib.dump(
    trained_models["Decision Tree"],
    "analytics/decision_tree_pipeline.joblib"
)

joblib.dump(
    trained_models["Random Forest"],
    "analytics/random_forest_pipeline.joblib"
)

joblib.dump(
    best_rf,
    "analytics/best_random_forest_pipeline.joblib"
)

joblib.dump(
    smote_pipeline,
    "analytics/smote_logistic_pipeline.joblib"
)

joblib.dump(
    regression_pipeline,
    "analytics/regression_pipeline.joblib"
)


# ============================================================
# 17. FINAL MODEL RECOMMENDATION
# ============================================================

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]


print("\n" + "=" * 60)
print("BEST CLASSIFICATION MODEL")
print("=" * 60)

print(
    "Model   :",
    best_model["Model"]
)

print(
    "Accuracy:",
    round(best_model["Accuracy"], 4)
)

print(
    "Precision:",
    round(best_model["Precision"], 4)
)

print(
    "Recall  :",
    round(best_model["Recall"], 4)
)

print(
    "F1 Score:",
    round(best_model["F1 Score"], 4)
)

print(
    "ROC-AUC :",
    round(best_model["ROC-AUC"], 4)
)


print("\n" + "=" * 60)
print("FINAL RECOMMENDATION")
print("=" * 60)

print(
    f"{best_model['Model']} is recommended as the "
    "final classification model because it achieved "
    "the highest F1 score among the evaluated baseline models."
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PART-B COMPLETED SUCCESSFULLY")
print("=" * 60)