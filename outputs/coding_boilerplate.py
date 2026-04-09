# Python 3.10
# xgboost==1.7.6
# scikit-learn==1.3.0

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

SEED = 42

def load_data():
    data = load_breast_cancer()
    return data.data, data.target

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=SEED)

def create_model():
    # TODO [T1]: Initialize model
    # Bloom's Level: Apply
    # Difficulty: Easy  |  Expected lines: ~1
    # Description: Instantiate the XGBoost classifier. You should pass the predefined `SEED` as the `random_state` parameter to ensure reproducibility during execution.
    # Hints: 1. You only need to initialize the object here, training happens later.  2. Call the `XGBClassifier` constructor.
    # Expected behavior: Returns an un-fitted instance of XGBClassifier.
    
    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T1 not yet implemented")
    # >>> END YOUR CODE <<<

def train_model(model, X_train, y_train):
    # TODO [T2]: Train model
    # Bloom's Level: Apply
    # Difficulty: Easy  |  Expected lines: ~1
    # Description: Fit the provided XGBoost model to the training data. This process allows the model to learn the patterns mapping the features (`X_train`) to the target labels (`y_train`).
    # Hints: 1. The scikit-learn API uses a standard method name for training across all models.  2. Call the `.fit()` method on your model object.
    # Expected behavior: The model is fitted in-place. (Optionally return the model).
    
    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T2 not yet implemented")
    # >>> END YOUR CODE <<<

def predict(model, X_test):
    # TODO [T3]: Predict
    # Bloom's Level: Apply
    # Difficulty: Easy  |  Expected lines: ~1
    # Description: Use the trained model to generate class predictions on the unseen test dataset (`X_test`).
    # Hints: 1. We want discrete class labels (0 or 1), not probability scores.  2. Call the `.predict()` method on the model.
    # Expected behavior: Returns a 1D numpy array of predicted integer labels matching the number of rows in `X_test`.
    
    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T3 not yet implemented")
    # >>> END YOUR CODE <<<

def evaluate(y_true, y_pred):
    # TODO [T4]: Evaluate
    # Bloom's Level: Analyze
    # Difficulty: Easy  |  Expected lines: ~1
    # Description: Calculate the accuracy of the model by comparing the predicted labels against the actual ground truth labels from the test set.
    # Hints: 1. Accuracy is the ratio of correct predictions to total predictions.  2. Use the imported `accuracy_score` function.
    # Expected behavior: Returns a float between 0.0 and 1.0 representing the accuracy metric.
    
    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T4 not yet implemented")
    # >>> END YOUR CODE <<<

def tune_model():
    # TODO [T5]: Improve model
    # Bloom's Level: Create
    # Difficulty: Medium  |  Expected lines: ~10-15
    # Description: Perform a simple hyperparameter tuning search. Iterate over a list of `max_depth` values to find the one that yields the highest accuracy on the test set.
    # Hints: 1. Loop through possible depths (e.g., 3, 4, 5). 2. Train a new model for each depth and evaluate it.
    # Expected behavior: Returns a tuple containing the best depth and its corresponding accuracy score.
    
    # >>> YOUR CODE HERE <<<
    raise NotImplementedError("TODO T5 not yet implemented")
    # >>> END YOUR CODE <<<

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = create_model()
    train_model(model, X_train, y_train)

    preds = predict(model, X_test)
    acc = evaluate(y_test, preds)

    print("Accuracy:", acc)

if __name__ == "__main__":
    main()