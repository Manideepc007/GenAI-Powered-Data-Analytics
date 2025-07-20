import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_excel("Delinquency_prediction_dataset - Copy.xlsx", sheet_name="Delinquency_prediction_dataset")

# Feature engineering: monthly payment behavior
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

# Define categorical and numeric columns
categorical_cols = ['Employment_Status', 'Credit_Card_Type', 'Location']
numeric_cols = list(set(features) - set(categorical_cols))

# Define preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocess train and test data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

# Train Random Forest model
# model = RandomForestClassifier(class_weight='balanced', random_state=42)
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_balanced, y_train_balanced)


# Predict and evaluate
y_pred = model.predict(X_test_processed)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print(roc_auc_score(y_test, model.predict_proba(X_test_processed)[:, 1]))

print("Training label distribution after SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

importances = model.feature_importances_
feature_names = preprocessor.get_feature_names_out()
plt.barh(feature_names, importances)
plt.title("Feature Importance")
plt.show()