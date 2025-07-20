import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve

# Load data
df = pd.read_excel("Delinquency_prediction_dataset.xlsx", sheet_name="Delinquency_prediction_dataset")

# Month columns (categorical with values: "On-time", "Late", "Missed")
month_cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']

# Feature engineering from month columns
df['Total_Late'] = df[month_cols].apply(lambda x: (x == 'Late').sum(), axis=1)
df['Total_Missed'] = df[month_cols].apply(lambda x: (x == 'Missed').sum(), axis=1)
df['Total_OnTime'] = df[month_cols].apply(lambda x: (x == 'On-time').sum(), axis=1)

# 3. 🔁 Feature Enrichment (NEW)
df['Missed_Last_Month'] = (df['Month_6'] == 'Missed').astype(int)
df['Late_Last_Month'] = (df['Month_6'] == 'Late').astype(int)
df['Has_Missed_Twice'] = (df['Total_Missed'] >= 2).astype(int)
df['Recent_Trouble'] = ((df['Month_6'] == 'Missed') | (df['Month_5'] == 'Missed')).astype(int)
df['Last_3_Months_Late'] = df[['Month_4', 'Month_5', 'Month_6']].apply(lambda x: (x == 'Late').sum(), axis=1)


# Select features
features = [
    'Age', 'Income', 'Credit_Score', 'Credit_Utilization',
    'Loan_Balance', 'Debt_to_Income_Ratio', 'Account_Tenure',
    'Employment_Status', 'Credit_Card_Type', 'Location',
    'Total_Late', 'Total_Missed', 'Total_OnTime',
    'Missed_Last_Month', 'Late_Last_Month',
    'Has_Missed_Twice', 'Recent_Trouble',
    'Last_3_Months_Late'
]
target = 'Delinquent_Account'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Identify column types
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

# Transform data
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_proc, y_train)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_sm, y_train_sm)

# Predict probabilities
y_proba = model.predict_proba(X_test_proc)[:, 1]

# 1. Get precision, recall, thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# 2. Calculate F1 for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)

# 3. Find threshold with highest F1
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Best threshold by F1: {best_threshold:.2f}")
print(f"Max F1 score: {best_f1:.4f}")

# Threshold tuning
threshold = 0.27
y_pred = (y_proba >= threshold).astype(int)

# Evaluation
print(f"Using threshold = {threshold}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

importances = model.feature_importances_
feature_names = preprocessor.get_feature_names_out()
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Show feature importances
importances = model.feature_importances_

# Get feature names after preprocessing
try:
    feature_names = preprocessor.get_feature_names_out()
except:
    # fallback if using older sklearn
    onehot_features = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([numeric_cols, onehot_features])

# Create and plot importances
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
feat_imp.head(20).plot(kind='bar')
plt.title("Top 20 Feature Importances")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest with SMOTE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision vs Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.show()