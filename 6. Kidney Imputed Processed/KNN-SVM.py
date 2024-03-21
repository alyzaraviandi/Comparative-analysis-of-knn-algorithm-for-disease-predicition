from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

# Load the heart attack dataset
data = pd.read_csv('6. Kidney Imputed Processed/6. kidney-imputed-processed-rev.csv')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

# Preprocess the data by normalizing the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a KNN classifier on the training data
knn = KNeighborsClassifier(n_neighbors=1)

# Train a SVM classifier on the training data
svm = SVC(kernel='linear', C=1, probability=True)

# Ensemble both classifiers using voting
voting_clf = VotingClassifier(estimators=[('knn', knn), ('svm', svm)], voting='soft')
voting_clf.fit(X_train, y_train)

# Predict on the test set
y_pred_ensemble = voting_clf.predict(X_test)

# Calculate the accuracy, precision, and recall scores
accuracy = accuracy_score(y_test, y_pred_ensemble)
precision = precision_score(y_test, y_pred_ensemble)
recall = recall_score(y_test, y_pred_ensemble)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
