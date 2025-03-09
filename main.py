import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Example dataset: Customer data (Churn or Not Churn)
data = {
    'Age': [25, 34, 45, 23, 50, 60, 40, 35, 31, 60],
    'Salary': [50000, 60000, 80000, 45000, 100000, 120000, 75000, 65000, 55000, 115000],
    'Churn': [0, 0, 0, 0, 1, 1, 0, 0, 0, 1]  # 0 = Not Churn, 1 = Churn
}

# Convert to DataFrame
df = pd.DataFrame(data)

print(df[['Age', 'Salary']])
print("\n=====================\n")
# Feature columns (Age, Salary)
X = df[['Age', 'Salary']]

# Target column (Churn)
y = df['Churn']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
