# -*- coding: utf-8 -*-
"""
model_building.py

DVC stage: model_building
Deps    : src/model_building.py, data/processed
Outs    : models/model.pkl

Loads the scaled, feature-engineered train/test splits produced by
data_preprocessing.py and trains a Logistic Regression classifier,
saving it for the next pipeline stage (model_evaluation).
"""

import os
import pickle
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

with open('params.yaml') as f:
    params = yaml.safe_load(f)['model_building']

PROCESSED_DATA_PATH = params['processed_data_path']
MODEL_DIR           = params['model_dir']

# Model hyperparameters - hardcoded here (no params.yaml dependency).
# Tweak directly if you want to experiment.
MODEL_PARAMS = params['logistic_regression']


# 1. Load processed data

def load_data():
    """Loads the scaled, feature-engineered train split."""
    x_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'train.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'y_train.csv')).squeeze('columns')
    return x_train, y_train


# 2. Train

def train_model(x_train, y_train, params=MODEL_PARAMS):
    """Trains a Logistic Regression classifier with the given params."""
    model = LogisticRegression(
        C=params['C'],
        max_iter=params['max_iter'],
        random_state=params['random_state'],
        solver=params['solver']
    )
    model.fit(x_train, y_train)
    return model


# 3. Save model

def save_model(model):
    """Pickles the trained model to models/model.pkl."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)


# 4. Run as a DVC stage

def main():
    x_train, y_train = load_data()

    model = train_model(x_train, y_train)
    save_model(model)

    print(f"Model trained on {x_train.shape[0]} samples, {x_train.shape[1]} features.")
    print(f"Model saved to: {os.path.join(MODEL_DIR, 'model.pkl')}")


if __name__ == "__main__":
    main()