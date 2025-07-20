import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
file_path = "Delinquency_prediction_dataset.xlsx"  # Update if needed
df = pd.read_excel(file_path, sheet_name = "Delinquency_prediction_dataset")

# --- Summary 1: General Overview ---
print("\n🔍 Dataset Shape:", df.shape)
print("\n📋 Column Info:")
print(df.info())

print("\n📊 Descriptive Stats:")
print(df.describe(include='all'))

print("\n❓ Missing Values:")
print(df.isnull().sum())

# --- Summary 2: Outlier Detection ---
numeric_cols = df.select_dtypes(include='number').columns
print("\n📌 Numeric Columns:", list(numeric_cols))

# Boxplots for outlier visualization
for col in numeric_cols:
    plt.figure(figsize=(6, 2))
    sns.boxplot(x=df[col])
    plt.title(f'Outliers in {col}')
    plt.tight_layout()
    plt.show()

# --- Summary 3: Correlation with Delinquency (Assuming target is named 'Delinquent' or similar) ---
target_col = 'Delinquent'  # change this to your actual target column if needed
if target_col in df.columns:
    corr = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)
    print("\n🔗 Correlation with Delinquency:")
    print(corr)

    # Top 3 predictors
    top_predictors = corr.drop(target_col).abs().sort_values(ascending=False).head(3)
    print("\n🚀 Top 3 Potential Predictors of Delinquency:")
    print(top_predictors)

else:
    print("\n⚠️ Could not find a column named 'Delinquent'. Please check your target column name.")

