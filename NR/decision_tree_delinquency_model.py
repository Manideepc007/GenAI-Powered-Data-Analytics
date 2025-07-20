import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load dataset
df = pd.read_excel("Delinquency_prediction_dataset - Copy.xlsx", sheet_name="Delinquency_prediction_dataset")

# Feature engineering from month columns
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

# Column types
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

# Decision Tree model with class weights
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluate
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Plot decision tree
plt.figure(figsize=(16, 8))
plot_tree(model.named_steps['classifier'],
          feature_names=model.named_steps['preprocessor'].get_feature_names_out(),
          class_names=["Not Delinquent", "Delinquent"], filled=True)
plt.title("Decision Tree")
plt.show()
