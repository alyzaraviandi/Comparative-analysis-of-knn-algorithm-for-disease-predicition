from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the heart attack dataset
data = pd.read_csv('8. Breast Cancer/8. breast-cancer.csv')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

# Preprocess the data by normalizing the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a KNN classifier on the training data
knn = KNeighborsClassifier(n_neighbors=47)
knn.fit(X_train, y_train)

# Train a Decision Tree classifier on the training data
dt = DecisionTreeClassifier(max_depth=3, criterion='entropy')
dt.fit(X_train, y_train)

# Ensemble both classifiers
ensemble = VotingClassifier(
    estimators=[('knn', knn), ('dt', dt)], voting='soft')
ensemble.fit(X_train, y_train)

# Make predictions using the ensemble classifier
y_pred = ensemble.predict(X_test)

# Calculate the accuracy, precision, and recall scores
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
