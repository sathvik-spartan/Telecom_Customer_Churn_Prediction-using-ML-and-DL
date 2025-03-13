import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense #type: ignore
from tensorflow.keras.utils import to_categorical #type: ignore

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

# Define target and features
y = df['churn']
X = df.drop(['churn'], axis=1)

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
ordinal_encoder = OrdinalEncoder()
X[categorical_cols] = ordinal_encoder.fit_transform(X[categorical_cols])

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Handle class imbalance
ros = RandomOverSampler()
X, y = ros.fit_resample(X, y)

# Convert target variable to categorical
y = to_categorical(y)

# Reshape data for RNN (samples, time steps, features)
X = X.reshape(X.shape[0], 1, X.shape[1])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2529)

# Build RNN model
model = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(1, X.shape[2])),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("RNN Model Accuracy")
plt.show()
