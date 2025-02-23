from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# 2. Split the Data into Training and Testing Sets
X = data.drop('target', axis=1) # features
y = data['target'] # target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Define the model (similar to Day 10's Random Forest)
model = XGBClassifier(random_state=42, n_jobs=-1)

# Define hyperparameters to search
param_grid = {
    'learning_rate': [0.01, 0.1],  # Step size for gradient descent (Day 3)
    'max_depth': [3, 5],           # From Day 10: deeper trees = more overfitting
    'n_estimators': [100, 200]     # Number of trees (Day 10: similar to Random Forest)
}

# Set up grid search (exhaustively try all combinations)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='f1_macro',  # From Day 6: F1-score for imbalanced classes
    cv=5,          # 5-fold cross-validation (Day 10)
    verbose=2
)

# Fit the model (this will take time)
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")