# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # For Random Forest
from sklearn.svm import SVC  # For SVM
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('path_to_your_file/kerala.csv')

# Inspect the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop or fill missing values (if any)
data.fillna(method='ffill', inplace=True)  # Forward fill for simplicity

# Define the feature columns and the target (assuming target is flood risk or rainfall level)
# Replace 'feature1', 'feature2', 'target' with actual column names
X = data[['feature1', 'feature2', 'feature3']]  # Features (rainfall, temperature, etc.)
y = data['target']  # Target (Flood risk: 0 = No flood, 1 = Flood)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling (especially important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model 1: Random Forest ---
# Create and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# --- Model 2: SVM ---
# Create and train the Support Vector Machine (SVM)
svm_model = SVC(kernel='linear')  # Use kernel='rbf' for non-linear
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_svm = svm_model.predict(X_test_scaled)

# Evaluate the SVM model
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))