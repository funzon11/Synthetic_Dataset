import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Create an imbalanced dataset using sklearn's make_classification
# Let's simulate a dataset with features related to transaction data.
n_samples = 10000  # Total samples
n_features = 10    # Number of features
n_informative = 5  # Number of informative features
n_classes = 2      # Binary classification: fraud vs non-fraud
weights = [0.95, 0.05]  # Imbalanced class distribution (95% non-fraud, 5% fraud)

# Generate the imbalanced dataset
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                            n_classes=n_classes, weights=weights, flip_y=0, random_state=42)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Apply SMOTE to generate synthetic fraud cases (minority class)
smote = SMOTE(sampling_strategy='minority', random_state=42)

# Fit SMOTE on the training data and generate synthetic data
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Step 4: Scaling the features for better performance with models
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# Step 5: Check the class distribution before and after applying SMOTE
print(f"Original class distribution: {np.bincount(y_train)}")
print(f"Resampled class distribution: {np.bincount(y_resampled)}")

# Step 6: Convert the resampled data to a DataFrame for easier inspection
df_resampled = pd.DataFrame(X_resampled_scaled, columns=[f'feature_{i+1}' for i in range(n_features)])
df_resampled['fraudulent'] = y_resampled

# Display the first few rows of the resampled data
print(df_resampled.head())

output_filename = 'synthetic_fraud_detection_data.csv'
df_resampled.to_csv(output_filename, index=False)

print(f"Resampled dataset saved to {output_filename}")
df = pd.read_csv(output_filename)

# Step 2: Convert to Excel and save as an XLSX file
xls_file = 'synthetic_fraud_detection_data.xlsx'  # Path for the output Excel file
df.to_excel(xls_file, index=False)

print(f"CSV file has been converted to Excel and saved as {xls_file}")

# Now, you can use the resampled data for model training