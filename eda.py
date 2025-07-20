import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the file (auto-load first sheet)
file_path = "Delinquency_prediction_dataset.xlsx"
df = pd.read_excel(file_path)

# Show summary
print("\n📋 Data Summary:")
print(df.info())
print("\n🔍 Missing Values:")
print(df.isnull().sum())

# Focus on numeric variables
numeric_cols = df.select_dtypes(include='number').columns
print("\n📊 Numeric Columns:")
print(numeric_cols)

# Correlation heatmap for all numeric variables
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Numeric Variables")
plt.tight_layout()
plt.show()

# Show distributions (to find skewed/interesting variables)
for col in numeric_cols:
    plt.figure(figsize=(5, 3))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()
