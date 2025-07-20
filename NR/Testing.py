import pandas as pd
import warnings

# Suppress warnings temporarily
warnings.simplefilter("ignore")

# Load the Excel file safely
file_path = "Delinquency_prediction_dataset.xlsx"  # Adjust if needed
xls = pd.ExcelFile(file_path)

# List available sheets
print("✅ Sheet names in file:")
print(xls.sheet_names)

# Load the first sheet directly (avoids name mismatch issues)
df = xls.parse(xls.sheet_names[0])
print("\n✅ First few rows of the sheet:")
print(df.head())
