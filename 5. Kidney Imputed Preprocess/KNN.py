from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Load the Heart Attack dataset
data = pd.read_csv('5. Kidney Imputed Preprocess/5. kidney-imputed-preprocess-rev.csv')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

# Preprocess the data by normalizing the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a KNN classifier on the training data [1, 3, 7, 8, 9, 10, 11]
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# Make predictions using the KNN classifier
y_pred = knn.predict(X_test)

# Calculate the accuracy, precision, and recall scores
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy:', accuracy*100)
print('Precision:', precision*100)
print('Recall:', recall*100)