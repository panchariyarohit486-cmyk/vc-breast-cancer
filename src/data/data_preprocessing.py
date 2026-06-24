# -*- coding: utf-8 -*-
"""
data_preprocessing.py

DVC stage: data_preprocessing
Deps : src/data_preprocessing.py, data/raw
Outs : data/processed

Reads the train/test splits already produced by data_ingestion.py
(data/raw/train.csv, test.csv, y_train.csv, y_test.csv), applies
feature engineering and scaling, and writes the processed data +
the fitted scaler to data/processed/ for the next pipeline stage
(model training) to consume.
"""

import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml

with open('params.yaml') as f:
    params = yaml.safe_load(f)['data_preprocessing']



RAW_DATA_PATH       = params['raw_data_path']
PROCESSED_DATA_PATH = params['processed_data_path']


# load data split data

def load_data():
    """
    Loads the train/test feature and target CSVs written by
    data_ingestion.py. y_train/y_test are read as single-column
    DataFrames and squeezed into Series.
    """
    x_train = pd.read_csv(os.path.join(RAW_DATA_PATH, 'train.csv'))
    x_test = pd.read_csv(os.path.join(RAW_DATA_PATH, 'test.csv'))
    y_train = pd.read_csv(os.path.join(RAW_DATA_PATH, 'y_train.csv')).squeeze('columns')
    y_test = pd.read_csv(os.path.join(RAW_DATA_PATH, 'y_test.csv')).squeeze('columns')
    return x_train, x_test, y_train, y_test


# data quality checks

def check_data_quality(x_train, x_test, verbose=True):
    """Checks for missing values and duplicate rows in both splits."""
    report = {}
    for name, df in [('train', x_train), ('test', x_test)]:
        missing = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())
        report[name] = {'missing_values': missing, 'duplicate_rows': duplicates}
        if verbose:
            print(f"[{name}] missing values: {missing}, duplicate rows: {duplicates}")
    return report


#feature engineering

def engineer_features(x):
    """
    Adds engineered features based on relationships between the
    'mean', 'error' (standard error), and 'worst' measurements
    already present for each of the 10 base measurements (radius,
    texture, perimeter, area, smoothness, compactness, concavity,
    concave points, symmetry, fractal dimension).

    This is purely arithmetic on existing columns - no fitting
    involved - so it's safe to apply identically to x_train and
    x_test without any leakage risk.
    """
    df = x.copy()
    base_features = [
        'radius', 'texture', 'perimeter', 'area', 'smoothness',
        'compactness', 'concavity', 'concave points', 'symmetry',
        'fractal dimension'
    ]
    eps = params['epsilon']  # avoids divide-by-zero

    for feat in base_features:
        mean_col = f"mean {feat}"
        se_col = f"{feat} error"
        worst_col = f"worst {feat}"

        if mean_col in df.columns and worst_col in df.columns:
            df[f"{feat}_worst_to_mean_ratio"] = df[worst_col] / (df[mean_col] + eps)
            df[f"{feat}_worst_mean_diff"] = df[worst_col] - df[mean_col]

        if mean_col in df.columns and se_col in df.columns:
            df[f"{feat}_cv"] = df[se_col] / (df[mean_col] + eps)

    if 'mean area' in df.columns and 'mean perimeter' in df.columns:
        df['mean_shape_index'] = (df['mean perimeter'] ** 2) / (df['mean area'] + eps)

    if 'worst area' in df.columns and 'worst perimeter' in df.columns:
        df['worst_shape_index'] = (df['worst perimeter'] ** 2) / (df['worst area'] + eps)

    return df


# scaling features/ standardization

def scale_features(x_train, x_test):
    """
    Fits StandardScaler on x_train ONLY, then applies it to both
    x_train and x_test. Fitting on test data (or on combined data
    before the split) would leak test-set statistics into training.

    Returns:
        tuple: (x_train_scaled, x_test_scaled, scaler)
    """
    scaler = StandardScaler()

    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test), columns=x_test.columns, index=x_test.index
    )

    return x_train_scaled, x_test_scaled, scaler


# save outputs for next stage (model training)

def save_outputs(x_train, x_test, y_train, y_test, scaler):
    """Writes processed data + the fitted scaler to data/processed/."""
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    x_train.to_csv(os.path.join(PROCESSED_DATA_PATH, 'train.csv'), index=False)
    x_test.to_csv(os.path.join(PROCESSED_DATA_PATH, 'test.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DATA_PATH, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DATA_PATH, 'y_test.csv'), index=False)

    with open(os.path.join(PROCESSED_DATA_PATH, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)


# run the preprocessing pipeline

def main():
    x_train, x_test, y_train, y_test = load_data()
    check_data_quality(x_train, x_test)

    x_train = engineer_features(x_train)
    x_test = engineer_features(x_test)

    x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test)

    save_outputs(x_train_scaled, x_test_scaled, y_train, y_test, scaler)

    print(f"\nProcessed train shape: {x_train_scaled.shape}")
    print(f"Processed test shape:  {x_test_scaled.shape}")
    print(f"Saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()