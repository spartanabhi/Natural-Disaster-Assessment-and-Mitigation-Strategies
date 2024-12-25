# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # SVM model
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('kerala.csv')

# Inspect the first few rows
print(data.head())

# Check for missing values and handle them
print(data.isnull().sum())
data.fillna(method='ffill', inplace=True)  # Forward fill for simplicity

# Define the features and the target
# Replace 'feature1', 'feature2', 'target' with actual column names from your dataset
X = data[['JAN', 'FEB', 'MAR']]  # Features (replace with actual features)
y = data['FLOODS']  # Target (replace with the actual target column, e.g., flood risk)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Support Vector Machine (SVM) model
svm_model = SVC(kernel='linear')  # Use kernel='rbf' for non-linear, but 'linear' is simpler
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_svm = svm_model.predict(X_test_scaled)

# Evaluate the SVM model
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
