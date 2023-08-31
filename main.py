import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

# Set random seed for consistent results
np.random.seed(42)

# Generate random data
num_samples = 20
num_features = 2
X = np.random.rand(num_samples, num_features)
y = np.where(np.sum(X, axis=1) > 1, 1, -1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM
model = svm.SVC(C=1.0, kernel='rbf', gamma=0.5)
model.fit(X_train, y_train)

# Test the SVM
correct_predictions = np.sum(model.predict(X_test) == y_test)
accuracy = correct_predictions / len(y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")
