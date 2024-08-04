import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('tennis.csv')

# Convert categorical variables to numerical values
le = LabelEncoder()
df['Outlook'] = le.fit_transform(df['Outlook'])
df['Temprature'] = le.fit_transform(df['Temprature'])
df['Humidity'] = le.fit_transform(df['Humidity'])
df['Wind'] = le.fit_transform(df['Wind'])
df['Play_Tennis'] = le.fit_transform(df['Play_Tennis'])

# Define the feature columns and the target column
X = df[['Outlook', 'Temprature', 'Humidity', 'Wind']]
y = df['Play_Tennis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the DecisionTreeClassifier with the ID3 criterion (entropy)
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Fit the model on the training data
clf.fit(X_train, y_train)

# Predicting a new sample
new_sample = [[0, 0, 0, 0]]  # Outlook:Sunny, Temprature:Hot, Humidity:High, Wind:Weak
prediction = clf.predict(new_sample)
print(f"Predicted class for the new sample: {prediction[0]}")

# Plotting the decision tree
plt.figure(figsize=(100, 5))
tree.plot_tree(clf, filled=True, feature_names=['Outlook', 'Temprature', 'Humidity', 'Wind'], class_names=['No', 'Yes'])
plt.show()