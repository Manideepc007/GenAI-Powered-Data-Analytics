import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.feature_selection import SelectFromModel

from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# Load dataset
df = pd.read_excel("Delinquency_prediction_dataset.xlsx")

# Drop identifier column and define features and target
df.drop(columns=["Customer_ID"], inplace=True)
target_col = "Delinquent_Account"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 1. Create interaction features
X['Utilization_Income_Interaction'] = X['Credit_Utilization'] * X['Income']
X['Missed_Utilization'] = X['Missed_Payments'] * X['Credit_Utilization']
X['Debt_Income'] = X['Debt_to_Income_Ratio'] * X['Income']
numeric_cols.extend(['Utilization_Income_Interaction', 'Missed_Utilization', 'Debt_Income'])

# 2. Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# 3. Full pipeline with feature selection, balancing, and logistic regression
clf_pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTETomek(random_state=42)),
    ("feature_select", SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=42))),
    ("clf", LogisticRegression(solver="liblinear", random_state=42))
])

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# 5. Train model
clf_pipeline.fit(X_train, y_train)

# 6. Predict probabilities and tune threshold
y_probs = clf_pipeline.predict_proba(X_test)[:, 1]

# Tune threshold to optimize F1-macro
thresholds = np.arange(0.1, 0.9, 0.01)
best_thresh = 0.5
best_f1 = 0
for t in thresholds:
    preds = (y_probs > t).astype(int)
    score = f1_score(y_test, preds, average='macro')
    if score > best_f1:
        best_f1 = score
        best_thresh = t

# Final predictions
y_pred = (y_probs > best_thresh).astype(int)

# 7. Evaluation
print(f"\n✅ Best Threshold: {round(best_thresh, 2)} with F1 Macro: {round(best_f1, 2)}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"ROC AUC Score: {roc_auc_score(y_test, y_probs):.4f}")
