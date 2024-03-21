from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the heart attack dataset
data = pd.read_csv('1. Heart Attack/1. heart-attack.csv')

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
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# Define the initial number of estimators and the step size for decreasing it
n_estimators = 200
step_size = 10

# Define the range of max depth values
max_depths = range(1, 11)

# Initialize variables to store the best parameters and accuracy
best_n_estimators = 0
best_max_depth = 0
best_accuracy = 0
rfbest_n_estimators = 0
rfbest_max_depth = 0
rfbest_accuracy = 0

for n in range(n_estimators, 0, -step_size):
    for depth in max_depths:
        # Initialize the random forest classifier with the current number of estimators and max depth
        rf = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        # Define the voting classifier
        voting = VotingClassifier(estimators=[('knn', knn), ('rf', rf)], voting='soft')
        voting.fit(X_train, y_train)
        voting_pred = voting.predict(X_test)

        # Evaluate the accuracy of the classifier on the testing set
        accuracy = accuracy_score(y_test, voting_pred)
        rfaccuracy = accuracy_score(y_test, rf_pred)

        # Update the best parameters and accuracy if a better one is found
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_n_estimators = n
            best_max_depth = depth
        if rfaccuracy > rfbest_accuracy:
            rfbest_accuracy = rfaccuracy
            rfbest_n_estimators = n
            rfest_max_depth = depth

print("Best number of estimators:", best_n_estimators)
print("Best max depth:", best_max_depth)
print("Best accuracy:", best_accuracy)

print("Best number of estimators:", rfbest_n_estimators)
print("Best max depth:", rfbest_max_depth)
print("Best accuracy:", rfbest_accuracy)