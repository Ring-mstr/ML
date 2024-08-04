import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
print("Features:\n", iris.feature_names)
print("Classes:\n", iris.target_names)
# Convert to DataFrame for better visualization
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
columns=iris['feature_names'] + ['target'])
print("\nFirst 5 rows of the dataset:\n", df.head())
# Plotting (optional)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Data Sepal Length vs Sepal Width')
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=42)
#Train the SVM classifier

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
# Train the classifier
svm_classifier.fit(X_train, y_train)
SVC(kernel='linear', random_state=42)
#Make predictions and evaluate the model:
# Make predictions
y_pred = svm_classifier.predict(X_test)
# Evaluate the model
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:\n", accuracy_score(y_test, y_pred))