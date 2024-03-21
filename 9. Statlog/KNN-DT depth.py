from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the heart attack dataset
data = pd.read_csv('9. Statlog/statlog.csv')

# Separate the features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

# Preprocess the data by normalizing the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a KNN classifier on the training data
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train, y_train)

# Define the range of max depths to try
max_depths = [None] + list(range(1, 11))

# Define the range of criteria to try
criteria = ['gini', 'entropy']

# Initialize variables to store the best max depth, criterion, and accuracy
best_max_depth = None
best_criterion = ''
best_accuracy = 0

# Loop over the max depths and criteria and train a decision tree classifier for each combination
for max_depth in max_depths:
    for criterion in criteria:
        # Initialize the decision tree classifier
        dtc = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
        dtc.fit(X_train, y_train)
        
        # Ensemble both classifiers
        ensemble = VotingClassifier(estimators=[('knn', knn), ('dt', dtc)], voting='soft')
        ensemble.fit(X_train, y_train)

        # Make predictions using the ensemble classifier
        y_pred = ensemble.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Update the best max depth, criterion, and accuracy if a better combination is found
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_max_depth = max_depth
            best_criterion = criterion
            best_precision = precision
            best_recall = recall

# Print the best max depth, criterion, and accuracy
print('Best max depth:', best_max_depth)
print('Best criterion:', best_criterion)
print('Best accuracy:', best_accuracy)
print('Best precision:', best_precision)
print('Best recall:', best_recall)
