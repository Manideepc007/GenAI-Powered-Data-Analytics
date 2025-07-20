import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from step_2_code import XGBClassifier
import numpy as np

# Load dataset
df = pd.read_excel("Delinquency_prediction_dataset - Copy.xlsx", sheet_name="Delinquency_prediction_dataset")

# Feature engineering
month_cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
df['Total_Late_Payments'] = df[month_cols].apply(lambda x: (x == 1).sum(), axis=1)
df['Total_Missed_Payments'] = df[month_cols].apply(lambda x: (x == 2).sum(), axis=1)
df['OnTime_Ratio'] = df[month_cols].apply(lambda x: (x == 0).sum(), axis=1) / len(month_cols)
df['Avg_Payment_Code'] = df[month_cols].mean(axis=1)

# Select features
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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Categorical and numeric columns
categorical_cols = ['Employment_Status', 'Credit_Card_Type', 'Location']
numeric_cols = list(set(features) - set(categorical_cols))

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Compute scale_pos_weight
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# XGBoost model
model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Train
model.fit(X_train_processed, y_train)

# Predict
y_pred = model.predict(X_test_processed)
y_proba = model.predict_proba(X_test_processed)[:, 1]

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Feature importance plot
feature_names = preprocessor.get_feature_names_out()
importances = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.show()
