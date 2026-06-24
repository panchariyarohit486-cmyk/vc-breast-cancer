# -*- coding: utf-8 -*-
"""
model_evaluation.py

DVC stage: model_evaluation
Deps    : src/model_evaluation.py, models/model.pkl, data/processed
Metrics : reports/metrics.json

Loads the trained model and the held-out test split, evaluates it,
and writes evaluation metrics to reports/metrics.json so `dvc metrics
show` / `dvc metrics diff` can track performance across experiments.
"""

import os
import json
import pickle
import pandas as pd
from pytest import param
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
with open('params.yaml') as f:
    params = yaml.safe_load(f)['model_evaluation']

PROCESSED_DATA_PATH = params['processed_data_path']
MODEL_PATH = params['model_path']
REPORTS_DIR = params['reports_dir']


# 1. Load model + test data

def load_model():
    """Loads the trained model saved by model_building.py."""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model


def load_test_data():
    """Loads the scaled, feature-engineered test split."""
    x_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'test.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'y_test.csv')).squeeze('columns')
    return x_test, y_test


# 2. Evaluate

def evaluate_model(model, x_test, y_test):
    """
    Computes accuracy, precision, recall, f1, and the confusion matrix
    on the held-out test set.

    Note: in this dataset label 0 = malignant, 1 = benign.
    """
    y_pred = model.predict(x_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    print(classification_report(y_test, y_pred, target_names=params['target_names']))
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 score:  {metrics['f1_score']:.4f}")

    return metrics


# 3. Save metrics

def save_metrics(metrics):
    """Writes evaluation metrics to reports/metrics.json."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)


# 4. Run as a DVC stage

def main():
    model = load_model()
    x_test, y_test = load_test_data()

    metrics = evaluate_model(model, x_test, y_test)
    save_metrics(metrics)

    print(f"\nMetrics saved to: {os.path.join(REPORTS_DIR, 'metrics.json')}")


if __name__ == "__main__":
    main()