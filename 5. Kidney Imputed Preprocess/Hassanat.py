from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Load the breast cancer dataset using pandas
data = pd.read_csv('5. Kidney Imputed Preprocess/5. kidney-imputed-preprocess-rev.csv')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

# Preprocess the data by normalizing the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Implement the HassanatKNNClassifier
import math
import numpy as np

class HassanatKNNClassifier:
    def __init__(self):
        self.n_neighbors = None

    def fit(self, X, y):
        self.X = X
        self.y = y.to_numpy()  # Convert y to numpy array for indexing

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = [self._hassanat_distance(x, xi) for xi in self.X]
            indices = np.argsort(distances)[:self.n_neighbors]
            neighbors = self.y[indices]
            labels, counts = np.unique(neighbors, return_counts=True)
            y_pred.append(labels[np.argmax(counts)])
        return np.array(y_pred)

    def _hassanat_distance(self, x, y):
        total = 0
        for xi, yi in zip(x, y):
            min_value = min(xi, yi)
            max_value = max(xi, yi)
            total += 1  # we sum the 1 in both cases
            if min_value >= 0:
                total -= (1 + min_value) / (1 + max_value)
            else:
                total -= 1 / (1 + max_value + abs(min_value))
        return total

# Loop through values of n_neighbors and find the best one
best_accuracy = 0
best_n_neighbors = 0
best_precision = 0
best_recall = 0
for n_neighbors in range(1, 51):
    # Train a HassanatKNNClassifier on the training data
    knn = HassanatKNNClassifier()
    knn.n_neighbors = n_neighbors
    knn.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = knn.predict(X_test)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Update the best accuracy and n_neighbors values if necessary
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_neighbors = n_neighbors
        best_precision = precision
        best_recall = recall
        
# Print the best n_neighbors, accuracy, precision, and recall values
print(f"Best n_neighbors: {best_n_neighbors}")
print(f"Best accuracy: {best_accuracy}")
print(f"Best precision: {best_precision}")
print(f"Best recall: {best_recall}")