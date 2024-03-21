import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the dataset
df = pd.read_csv('4. Heart Disease/4. heart-disease.csv')

# Split the data into features and target
X = df.drop(["target"], axis=1)
y = df["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the KNN and Random Forest models
knn = KNeighborsClassifier(n_neighbors=1)
rf = RandomForestClassifier(n_estimators=200, max_depth=1, random_state=42)

# Define the voting classifier
voting = VotingClassifier(estimators=[('knn', knn), ('rf', rf)], voting='soft')
voting.fit(X_train_scaled, y_train)

# Make predictions and evaluate performance
voting_pred = voting.predict(X_test_scaled)

voting_acc = accuracy_score(y_test, voting_pred)
voting_prec = precision_score(y_test, voting_pred)
voting_rec = recall_score(y_test, voting_pred)

print("Voting Accuracy: {:.3f}".format(voting_acc * 100))
print("Voting Precision: {:.3f}".format(voting_prec * 100))
print("Voting Recall: {:.3f}".format(voting_rec * 100))
