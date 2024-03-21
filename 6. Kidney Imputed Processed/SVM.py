import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the Heart Attack Data Set
df = pd.read_csv('6. Kidney Imputed Processed/6. kidney-imputed-processed-rev.csv')

# Split the data set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

# Preprocess the data by normalizing the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a KNN classifier on the training data
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

c_values = [0.01, 0.1, 1, 10, 100, 1000]
gamma_values = [0.1, 1, 10, 100]
weight_values = [None, 'balanced']

best_accuracy = 0
best_params = {}

# Iterate over the parameter values
for c in c_values:
    for gamma in gamma_values:
        for weight in weight_values:
            # Train a SVM classifier with the current parameter values (kernel='rbf')
            svm = SVC(kernel='rbf', C=c, gamma=gamma, probability=True, class_weight=weight)
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
                best_params = {'C': c, 'gamma': gamma, 'class_weight': weight}

# Train the final SVM classifier with the best parameter values (kernel='rbf')
final_svm = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], probability=True, class_weight=best_params['class_weight'])
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
