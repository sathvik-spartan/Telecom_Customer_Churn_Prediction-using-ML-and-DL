import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv('telecom_churn.csv')

# Fix negative values
df['calls_made'] = df['calls_made'].apply(lambda x: max(x, 0))
df['sms_sent'] = df['sms_sent'].apply(lambda x: max(x, 0))
df['data_used'] = df['data_used'].apply(lambda x: max(x, 0))

# Convert date_of_registration to datetime and extract features
df['date_of_registration'] = pd.to_datetime(df['date_of_registration'], errors='coerce')
df['registration_year'] = df['date_of_registration'].dt.year
df['registration_month'] = df['date_of_registration'].dt.month
df['registration_day'] = df['date_of_registration'].dt.day

df = df.drop(['date_of_registration'], axis=1)

# Drop rows with missing values in churn
df = df.dropna(subset=['churn'])

# Define target variable (y) and features (X)
y = df['churn']
X = df.drop(['churn'], axis=1)

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
ordinal_encoder = OrdinalEncoder()
X[categorical_cols] = ordinal_encoder.fit_transform(X[categorical_cols])

# Handle class imbalance using Random Over Sampling (ROS)
ros = RandomOverSampler()
X, y = ros.fit_resample(X, y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2529)

# Train Decision Tree model
dtc = DecisionTreeClassifier(random_state=2529)
dtc.fit(X_train, y_train)

# Make predictions
y_pred = dtc.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Visualization of Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(dtc, feature_names=X.columns, class_names=['No Churn', 'Churn'], filled=True, rounded=True, fontsize=8)
plt.show()
