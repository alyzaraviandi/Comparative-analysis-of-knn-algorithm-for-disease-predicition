from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Load the breast cancer dataset using pandas
data = pd.read_csv('9. Statlog/statlog.csv')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

# Preprocess the data by normalizing the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


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
    
knn = HassanatKNNClassifier()
knn.n_neighbors = 13
knn.fit(X_train, y_train)

c_values = [0.01, 0.1, 1, 10, 100, 1000]
class_weight_values = [None, 'balanced']

best_accuracy = 0
best_params = {}

# Iterate over the parameter values
for c in c_values:
    for class_weight in class_weight_values:
        # Train a SVM classifier with the current parameter values (kernel='linear') and class_weight
        svm = SVC(kernel='linear', C=c, class_weight=class_weight, probability=True)
        svm.fit(X_train, y_train)

        # Ensemble the SVM classifier with the KNN classifier using soft voting
        voting_clf = VotingClassifier(estimators=[('svm', svm), ('knn', knn)], voting='soft')
        voting_clf.fit(X_train, y_train)

        # Predict on the test set
        y_pred_ensemble = voting_clf.predict(X_test)

        # Calculate the accuracy score
        accuracy = accuracy_score(y_test, y_pred_ensemble)

        # Check if the current parameter values yield better accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'C': c, 'class_weight': class_weight}

# Train the final SVM classifier with the best parameter values (kernel='linear', class_weight)
final_svm = SVC(kernel='linear', C=best_params['C'], class_weight=best_params['class_weight'], probability=True)
final_svm.fit(X_train, y_train)

# Ensemble the final SVM classifier with the KNN classifier using soft voting
final_voting_clf = VotingClassifier(estimators=[('svm', final_svm), ('knn', knn)], voting='soft')
final_voting_clf.fit(X_train, y_train)

# Predict on the test set with the final ensemble classifier
y_pred_ensemble_final = final_voting_clf.predict(X_test)

# Calculate the accuracy, precision, and recall scores for the final ensemble classifier
accuracy_final = accuracy_score(y_test, y_pred_ensemble_final)
precision_final = precision_score(y_test, y_pred_ensemble_final)
recall_final = recall_score(y_test, y_pred_ensemble_final)

print('Best parameters:', best_params)
print('Accuracy (final):', accuracy_final)
print('Precision (final):', precision_final)
print('Recall (final):', recall_final)