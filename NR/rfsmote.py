import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_excel("Delinquency_prediction_dataset - Copy.xlsx", sheet_name="Delinquency_prediction_dataset")

# Feature engineering
month_cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
df['Total_Late_Payments'] = df[month_cols].apply(lambda x: (x == 1).sum(), axis=1)
df['Total_Missed_Payments'] = df[month_cols].apply(lambda x: (x == 2).sum(), axis=1)
df['OnTime_Ratio'] = df[month_cols].apply(lambda x: (x == 0).sum(), axis=1) / len(month_cols)
df['Avg_Payment_Code'] = df[month_cols].mean(axis=1)

# Define features and target
features = [
    'Age', 'Income', 'Credit_Score', 'Credit_Utilization',
    'Loan_Balance', 'Debt_to_Income_Ratio', 'Account_Tenure',
    'Total_Late_Payments', 'Total_Missed_Payments',
    'OnTime_Ratio', 'Avg_Payment_Code',
    'Employment_Status', 'Credit_Card_Type', 'Location'
]
target = 'Delinquent_Account'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessing
categorical_cols = ['Employment_Status', 'Credit_Card_Type', 'Location']
numeric_cols = list(set(features) - set(categorical_cols))

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Transform
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_bal, y_train_bal)

# Predict
y_proba = model.predict_proba(X_test_proc)[:, 1]
threshold = 0.3
y_pred = (y_proba >= threshold).astype(int)

# Evaluate
print(f"Using threshold = {threshold}")
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest with SMOTE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()