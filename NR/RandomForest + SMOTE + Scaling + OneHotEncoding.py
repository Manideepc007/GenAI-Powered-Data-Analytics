import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load your dataset here
df = pd.read_excel("Delinquency_prediction_dataset - Copy.xlsx")

# Drop ID column and engineer features
df_model = df.drop(columns=['Customer_ID'])
month_cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
df_model['Total_On_Time'] = df_model[month_cols].apply(lambda x: (x == 0).sum(), axis=1)
df_model['Total_Late'] = df_model[month_cols].apply(lambda x: (x == 1).sum(), axis=1)
df_model['Total_Missed'] = df_model[month_cols].apply(lambda x: (x == 2).sum(), axis=1)
df_model.drop(columns=month_cols, inplace=True)

# Features and target
X = df_model.drop(columns='Delinquent_Account')
y = df_model['Delinquent_Account']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Column types
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

# Enhanced Pipeline
model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
])

# Fit the model
model_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

# Output metrics
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)
print(f"ROC AUC: {roc_auc:.2f}")
print(f"PR AUC: {pr_auc:.2f}")

threshold = 0.2
y_pred_thresh = (y_proba >= threshold).astype(int)

# Evaluate
conf_matrix_thresh = confusion_matrix(y_test, y_pred_thresh)
class_report_thresh = classification_report(y_test, y_pred_thresh)
roc_auc_thresh = roc_auc_score(y_test, y_proba)

print("Confusion Matrix (Threshold = 0.2):\n", conf_matrix_thresh)
print("\nClassification Report:\n", class_report_thresh)
print(f"\nROC AUC: {roc_auc_thresh:.2f}")