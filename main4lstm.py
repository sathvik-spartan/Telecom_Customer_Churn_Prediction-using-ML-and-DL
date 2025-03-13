import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization #type: ignore
from tensorflow.keras.utils import to_categorical #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore

# Load dataset
df = pd.read_csv('telecom_churn.csv')

# Fix negative values efficiently
df[['calls_made', 'sms_sent', 'data_used']] = df[['calls_made', 'sms_sent', 'data_used']].clip(lower=0)

# Convert date_of_registration to datetime and extract features
df['date_of_registration'] = pd.to_datetime(df['date_of_registration'], errors='coerce')
df['registration_year'] = df['date_of_registration'].dt.year
df['registration_month'] = df['date_of_registration'].dt.month
df['registration_day'] = df['date_of_registration'].dt.day
df['days_since_registration'] = (pd.Timestamp.today() - df['date_of_registration']).dt.days
df = df.drop(['date_of_registration'], axis=1)

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric NaN with median
df.fillna(df.mode().iloc[0], inplace=True)  # Fill categorical NaN with mode

# Define target and features
y = df['churn']
X = df.drop(['churn'], axis=1)

# Encode categorical features using One-Hot Encoding
categorical_cols = X.select_dtypes(include=['object']).columns
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
categorical_data = ohe.fit_transform(X[categorical_cols])
categorical_df = pd.DataFrame(categorical_data, columns=ohe.get_feature_names_out(categorical_cols))
X = X.drop(categorical_cols, axis=1)
X = pd.concat([X, categorical_df], axis=1)

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE()
X, y = smote.fit_resample(X, y)

# Convert target variable to categorical
y = to_categorical(y)

# Reshape data for RNN (samples, time steps, features)
X = X.reshape(X.shape[0], 1, X.shape[1])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2529)

# Build improved LSTM model with Batch Normalization and more layers
model = Sequential([
    LSTM(128, return_sequences=True, activation='relu', input_shape=(1, X.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=True, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(32, return_sequences=False, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile model with optimized learning rate
optimizer = Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

# Train model
history = model.fit(X_train, y_train, epochs=75, batch_size=64, validation_data=(X_test, y_test), callbacks=callbacks)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Optimized LSTM Model Accuracy")
plt.show()
