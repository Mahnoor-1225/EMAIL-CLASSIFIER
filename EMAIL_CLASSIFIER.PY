import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv(r"C:\Users\mahno\OneDrive\Desktop\internship\emails.csv")

# Display basic information about the dataset
print("First 5 rows of the dataset:")
print(df.head())
print("\nData types of each column:")
print(df.dtypes)
print("\nClass distribution:")
print(df['Prediction'].value_counts())

# Ensure there are no missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Extract features (X) and target variable (y)
X = df.iloc[:, 1:3001]  # Features
y = df['Prediction']    # Target variable

# Verify if the target variable is binary or has more than 2 classes
print("\nUnique values in the target variable:")
print(y.unique())

# Scale the features to [0, 1] range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Use a smaller subset of data for quick experimentation
X_subset = X_scaled[:1000]
y_subset = y[:1000]

# Split data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X_subset, y_subset, test_size=0.25, random_state=42)

# Initialize models
models = {
    "MultinomialNB": MultinomialNB(alpha=1.9),
    "SVC": SVC(C=1.0, kernel='rbf', gamma='auto', max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=50, criterion='gini', n_jobs=-1),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    accuracy = accuracy_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred, average='macro', zero_division=0)
    recall = recall_score(test_y, y_pred, average='macro', zero_division=0)
    f1 = f1_score(test_y, y_pred, average='macro', zero_division=0)
    
    print(f"\n{name} Performance:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# Hyperparameter tuning for Random Forest using GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(train_x, train_y)
best_model = grid_search.best_estimator_

print("\nBest Random Forest Model:")
print(best_model)

# Evaluate the best model
y_pred_best = best_model.predict(test_x)
accuracy_best = accuracy_score(test_y, y_pred_best)
precision_best = precision_score(test_y, y_pred_best, average='macro', zero_division=0)
recall_best = recall_score(test_y, y_pred_best, average='macro', zero_division=0)
f1_best = f1_score(test_y, y_pred_best, average='macro', zero_division=0)

print("\nBest Random Forest Performance:")
print(f"Accuracy: {accuracy_best}")
print(f"Precision: {precision_best}")
print(f"Recall: {recall_best}")
print(f"F1 Score: {f1_best}")
