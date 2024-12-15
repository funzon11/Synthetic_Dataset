import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import tensorflow as tf
from tensorboard.notebook import display
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, Dense
from keras.models import Model

data_raw = pd.read_csv("C:/PGDAI/Programming Language AI/CA70/BankFraud/Base.csv")
pd.set_option('display.max_columns', None)
numerical_columns = data_raw.select_dtypes(include=['number']).columns
numerical_columns = [col for col in data_raw.columns if data_raw[col].dtype in ['int64', 'float64'] and data_raw[col].dtype != 'bool']
print(numerical_columns)
n_features = 20
data1=data_raw[['fraud_bool','income','payment_type','zip_count_4w','velocity_6h','velocity_24h','velocity_4w','bank_branch_count_8w',
                'date_of_birth_distinct_emails_4w','credit_risk_score','email_is_free','has_other_cards','proposed_credit_limit',
                'foreign_request','device_fraud_count','month','name_email_similarity','prev_address_months_count',
                'customer_age','days_since_request','session_length_in_minutes','employment_status','housing_status']].copy()
# Initialize the StandardScaler
# Preprocess the data: Standardize the features
data_numeric = data1.select_dtypes(include=[np.number])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_numeric)

# Split into training and test sets
X_train, X_test = train_test_split(scaled_data, test_size=0.2, random_state=42)

# Reshape data for CNN (adding extra dimension for channels, as CNN requires 3D data)
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the CNN Autoencoder model
input_layer = Input(shape=(n_features, 1))

# Encoder: Conv1D, MaxPooling1D to downsample the data
encoded = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
encoded = MaxPooling1D(2, padding='same')(encoded)
encoded = Conv1D(32, 3, activation='relu', padding='same')(encoded)
encoded = MaxPooling1D(2, padding='same')(encoded)

# Decoder: UpSampling1D to reconstruct the data
decoded = Conv1D(32, 3, activation='relu', padding='same')(encoded)
decoded = UpSampling1D(2)(decoded)  # Upsample to the original size
decoded = Conv1D(64, 3, activation='relu', padding='same')(decoded)
decoded = UpSampling1D(2)(decoded)  # Upsample again

# Output layer (reconstruct the original input)
decoded_output = Conv1D(1, 3, activation='sigmoid', padding='same')(decoded)

# Compile the autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded_output)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder model
autoencoder.fit(X_train_reshaped, X_train_reshaped, epochs=20, batch_size=32, validation_data=(X_test_reshaped, X_test_reshaped))

# Use the encoder part to extract the encoded features (compressed representation)
encoder = Model(inputs=input_layer, outputs=encoded)
encoded_data = encoder.predict(X_test_reshaped)

# Print the shape of the encoded data (compressed features)
print("Encoded data shape:", encoded_data.shape)