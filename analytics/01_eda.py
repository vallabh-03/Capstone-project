import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# PART-A — EXPLORATORY DATA ANALYSIS
# ============================================================

# 1. Load the Titanic dataset ONCE
df = sns.load_dataset("titanic")

# Save a local copy for offline use
df.to_csv("analytics/titanic.csv", index=False)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

# 2. Basic dataset information
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Descriptive Statistics ---")
print(df.describe(include="all"))

# ============================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_count = df.isnull().sum()
missing_percentage = (missing_count / len(df)) * 100

missing_summary = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percentage": missing_percentage
})

print(missing_summary)

# ============================================================
# 4. DATA CLEANING
# ============================================================


for column in df.columns:

    missing_pct = df[column].isnull().mean() * 100

    if missing_pct == 0:
        continue

    elif missing_pct < 10:

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])

    elif missing_pct <= 30:

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])

    else:
        df.drop(columns=[column], inplace=True)

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())

# ============================================================
# 5. AGE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("AGE ANALYSIS")
print("=" * 60)

print("\nAge Statistics:")
print(df["age"].describe())

print("\nAge Skewness:")
print(df["age"].skew())

# ============================================================
# 6. FARE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("FARE ANALYSIS")
print("=" * 60)

print("\nFare Statistics:")
print(df["fare"].describe())

print("\nFare Skewness:")
print(df["fare"].skew())

# ============================================================
# 7. BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("BIVARIATE ANALYSIS")
print("=" * 60)

print("\nSurvival Rate by Sex:")
print(df.groupby("sex")["survived"].mean())

print("\nSurvival Rate by Passenger Class:")
print(df.groupby("pclass")["survived"].mean())

print("\nSurvival Rate by Sex and Passenger Class:")
print(
    df.groupby(["sex", "pclass"])["survived"]
    .mean()
)

# ============================================================
# 8. CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

numeric_df = df.select_dtypes(include=np.number)

correlation_matrix = numeric_df.corr()

print(correlation_matrix)

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig("analytics/correlation_heatmap.png", dpi=300)
plt.show()

# ============================================================
# 9. MULTIVARIATE VISUALIZATIONS
# ============================================================

# Chart 1 — Survival by Sex
plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="sex",
    y="survived"
)

plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.savefig("analytics/survival_by_sex.png", dpi=300)
plt.show()

print(
    "\nInterpretation 1: Survival rates differ substantially by sex, "
    "with female passengers showing a higher survival rate than male passengers."
)

# Chart 2 — Survival by Passenger Class
plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived"
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.savefig("analytics/survival_by_class.png", dpi=300)
plt.show()

print(
    "\nInterpretation 2: Survival rates vary across passenger classes, "
    "with passengers in higher classes generally having better survival outcomes."
)

# Chart 3 — Age Distribution by Survival
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="age",
    hue="survived",
    bins=30,
    kde=True
)

plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("analytics/age_distribution_survival.png", dpi=300)
plt.show()

print(
    "\nInterpretation 3: The age distributions of survivors and non-survivors "
    "show differences across passenger age groups."
)

# Chart 4 — Fare vs Age by Survival
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived"
)

plt.title("Fare vs Age by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.tight_layout()
plt.savefig("analytics/fare_vs_age_survival.png", dpi=300)
plt.show()

print(
    "\nInterpretation 4: Fare and age show variation across passengers, "
    "while survival status provides an additional dimension for comparison."
)

# ============================================================
# 10. STANDARDIZATION
# ============================================================

print("\n" + "=" * 60)
print("STANDARDIZATION")
print("=" * 60)

df["age_standardized"] = (
    (df["age"] - df["age"].mean())
    / df["age"].std()
)

df["fare_standardized"] = (
    (df["fare"] - df["fare"].mean())
    / df["fare"].std()
)

print("\nOriginal Age:")
print(df["age"].head())

print("\nStandardized Age:")
print(df["age_standardized"].head())

print("\nOriginal Fare:")
print(df["fare"].head())

print("\nStandardized Fare:")
print(df["fare_standardized"].head())

print("\nFinal dataset shape:", df.shape)

print("\nPART-A completed successfully.")