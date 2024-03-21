import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the Heart Attack Data Set
df = pd.read_csv("7. Pima Diabetes/7. pima-diabetes.csv")

# Split the data set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

# Preprocess the data by normalizing the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a KNN-Naive Bayes classifier ensemble using the training set
knn = KNeighborsClassifier(n_neighbors=20)
nb = GaussianNB()
ensemble = VotingClassifier(
    estimators=[('knn', knn), ('nb', nb)], voting='soft')
ensemble.fit(X_train, y_train)

# Evaluate the classifier on the testing set
ensemble_preds = ensemble.predict(X_test)

# Print the accuracy, precision, and recall metrics
print("Ensemble Classifier Metrics:")
print("Accuracy:", accuracy_score(y_test, ensemble_preds))
print("Precision:", precision_score(y_test, ensemble_preds))
print("Recall:", recall_score(y_test, ensemble_preds))
