import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from step_2_code import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve
# Load dataset
df = pd.read_excel("Delinquency_prediction_dataset - Copy.xlsx", sheet_name="Delinquency_prediction_dataset")

# Feature engineering
month_cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
df['Total_Late_Payments'] = df[month_cols].apply(lambda x: (x == 1).sum(), axis=1)
df['Total_Missed_Payments'] = df[month_cols].apply(lambda x: (x == 2).sum(), axis=1)
df['OnTime_Ratio'] = df[month_cols].apply(lambda x: (x == 0).sum(), axis=1) / len(month_cols)
df['Avg_Payment_Code'] = df[month_cols].mean(axis=1)

# Select features and target
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

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Column types
categorical_cols = ['Employment_Status', 'Credit_Card_Type', 'Location']
numeric_cols = list(set(features) - set(categorical_cols))

# Preprocessing
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

# Preprocess the data
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

# Train XGBoost model
model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train_bal, y_train_bal)

# Predict probabilities
y_proba = model.predict_proba(X_test_proc)[:, 1]

# Threshold tuning
threshold = 0.2
y_pred = (y_proba >= threshold).astype(int)

# Evaluation
print(f"Using threshold = {threshold}")
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_proba):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost with SMOTE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall vs. Threshold")
plt.legend()
plt.grid(True)
plt.show()