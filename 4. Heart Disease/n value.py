from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Load the breast cancer dataset using pandas
data = pd.read_csv('4. Heart Disease/4. heart-disease.csv')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

# Preprocess the data by normalizing the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Loop through values of n_neighbors and find the best one
best_accuracy = 0
best_n_neighbors = 0
for n_neighbors in range(1, 51):
    # Train a KNN classifier on the training data
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = knn.predict(X_test)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Update the best accuracy and n_neighbors values if necessary
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_neighbors = n_neighbors

# Print the best n_neighbors and accuracy values
print(f"Best n_neighbors: {best_n_neighbors}")
print(f"Best accuracy: {best_accuracy}")
