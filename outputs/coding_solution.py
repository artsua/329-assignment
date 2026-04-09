# Python 3.10
# xgboost==1.7.6
# scikit-learn==1.3.0
# Dataset: scikit-learn Breast Cancer Wisconsin (Diagnostic) Dataset
#
# --- EXPECTED OUTPUTS & BENCHMARKS ---
# Hardware Assumptions: Standard Consumer CPU (e.g., Apple M1, Intel i5/i7)
# Expected Base Training Time: < 0.1 seconds
# Expected Total Tuning Time: < 1.0 seconds
# Expected Base Accuracy: ~0.956
# Expected Best Depth after Tuning: 3 or 4
# Expected Tuned Accuracy: ~0.956 to ~0.965
# Reference Plots: N/A for this specific text-output script.
# Loss Ranges: XGBoost binary:logistic logloss expected to converge < 0.1
# -------------------------------------

import time
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
    # Approach: Initialize the XGBClassifier using a set of common hyperparameters (max_depth, learning_rate, subsample) to control tree complexity. Pass SEED for reproducibility.
    return XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED
    )

def train_model(model, X_train, y_train):
    # Approach: Utilize the standard scikit-learn API `.fit()` method to train the XGBoost estimator on our training feature set.
    model.fit(X_train, y_train)

def predict(model, X_test):
    # Approach: Call the `.predict()` method to output the final, discrete binary class predictions for the unseen test data.
    return model.predict(X_test)

def evaluate(y_true, y_pred):
    # Approach: Leverage scikit-learn's `accuracy_score` metric to compute the ratio of correct predictions to total predictions.
    return accuracy_score(y_true, y_pred)

def tune_model():
    # Approach: Implement a manual grid search by iterating through a predefined list of `max_depth` values. Train and evaluate a model for each, keeping track of the best accuracy.
    best_acc = 0
    best_depth = 3

    for d in [3, 4, 5]:
        model = XGBClassifier(max_depth=d, random_state=SEED)
        X, y = load_data()
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        if acc > best_acc:
            best_acc = acc
            best_depth = d

    return best_depth, best_acc

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Base Model Run & Benchmark
    start_time = time.time()
    model = create_model()
    train_model(model, X_train, y_train)
    train_time = time.time() - start_time

    preds = predict(model, X_test)
    acc = evaluate(y_test, preds)

    print("--- BASE MODEL RESULTS ---")
    print(f"Accuracy: {acc:.4f} (Expected: ~0.9561)")
    print(f"Training Time: {train_time:.4f} seconds (Expected: < 0.1s)\n")

    # Tuning Run & Benchmark
    start_tune_time = time.time()
    depth, best_acc = tune_model()
    tune_time = time.time() - start_tune_time

    print("--- TUNING RESULTS ---")
    print(f"Best Depth: {depth} (Expected: 3 or 4)")
    print(f"Best Accuracy: {best_acc:.4f} (Expected: ~0.9561 to 0.9650)")
    print(f"Total Tuning Time: {tune_time:.4f} seconds (Expected: < 1.0s)")

if __name__ == "__main__":
    main()