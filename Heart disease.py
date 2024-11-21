# -*- coding: utf-8 -*- 
"""
Created on Wed Nov 20 23:30:56 2024

@author: Giramata Suavis
"""

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Step 1: Set up an outputs folder
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Step 2: Load the dataset
file_path = "C:/Users/Giramata Suavis/Downloads/Predictive Analysis/data/Heart_Disease_Prediction.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Step 3: Inspect and clean the dataset
with open(os.path.join(output_folder, "dataset_info.txt"), "w") as f:
    f.write("Initial Dataset Info:\n")
    df.info(buf=f)

# Map 'Presence' and 'Absence' in the 'Heart Disease' column to numeric values
if 'Heart Disease' in df.columns:
    df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

# Replace missing values with column median
df.fillna(df.median(numeric_only=True), inplace=True)

# Use one-hot encoding for non-target categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Step 4: Feature and target separation
X = df.drop("Heart Disease", axis=1)  # Features
y = df["Heart Disease"]  # Target variable

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Build and evaluate models
# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Save Logistic Regression performance
log_report = classification_report(y_test, y_pred_log)
with open(os.path.join(output_folder, "logistic_regression_report.txt"), "w") as f:
    f.write("Logistic Regression Performance:\n")
    f.write(log_report)

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Save Random Forest performance
rf_report = classification_report(y_test, y_pred_rf)
with open(os.path.join(output_folder, "random_forest_report.txt"), "w") as f:
    f.write("Random Forest Performance:\n")
    f.write(rf_report)

# Support Vector Machine
svc_model = SVC(kernel='linear', random_state=42)
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)

# Save SVM performance
svc_report = classification_report(y_test, y_pred_svc)
with open(os.path.join(output_folder, "svm_report.txt"), "w") as f:
    f.write("SVM Performance:\n")
    f.write(svc_report)

# Step 8: Compare model accuracies and save plot
models = ["Logistic Regression", "Random Forest", "SVM"]
accuracies = [
    accuracy_score(y_test, y_pred_log),
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_svc),
]

plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(output_folder, "model_accuracy_comparison.png"))
plt.show()

# Step 9: Visualize feature importance for Random Forest and save the plot
importances = rf_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance (Random Forest)")
plt.savefig(os.path.join(output_folder, "random_forest_feature_importance.png"))
plt.show()
