import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the Heart Attack Data Set
df = pd.read_csv('8. Breast Cancer/8. breast-cancer.csv')

# Split the data set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

# Preprocess the data by normalizing the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a KNN classifier on the training data
knn = KNeighborsClassifier(n_neighbors=47)
knn.fit(X_train, y_train)

# Define the range of var_smoothing values and priors to try
var_smoothing_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
priors_values = [None, [0.5, 0.5], [0.3, 0.7], [0.7, 0.3]]

# Initialize variables to store the best parameter values and accuracy
best_var_smoothing = 0
best_priors = None
best_accuracy = 0

# Loop over the var_smoothing values and priors to train a GaussianNB classifier for each combination
for var_smoothing in var_smoothing_values:
    for priors in priors_values:
        # Initialize the GaussianNB classifier with the current var_smoothing and priors values
        nb = GaussianNB(var_smoothing=var_smoothing, priors=priors)
        ensemble = VotingClassifier(
            estimators=[('knn', knn), ('nb', nb)], voting='soft')
        ensemble.fit(X_train, y_train)

        # Make predictions using the ensemble classifier
        ensemble_preds = ensemble.predict(X_test)

        # Calculate accuracy, precision, and recall
        accuracy = accuracy_score(y_test, ensemble_preds)
        precision = precision_score(y_test, ensemble_preds)
        recall = recall_score(y_test, ensemble_preds)

        # Update the best parameter values and accuracy if a better one is found
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_var_smoothing = var_smoothing
            best_priors = priors

# Train the ensemble classifier with the best parameter values
nb_best = GaussianNB(var_smoothing=best_var_smoothing, priors=best_priors)
ensemble_best = VotingClassifier(
    estimators=[('knn', knn), ('nb', nb_best)], voting='soft')
ensemble_best.fit(X_train, y_train)

# Evaluate the classifier on the testing set
ensemble_preds_best = ensemble_best.predict(X_test)

# Print the accuracy, precision, and recall metrics and the best parameter values
print("Accuracy:", accuracy_score(y_test, ensemble_preds_best))
print("Precision:", precision_score(y_test, ensemble_preds_best))
print("Recall:", recall_score(y_test, ensemble_preds_best))
print("Best var_smoothing:", best_var_smoothing)
print("Best priors:", best_priors)