import argparse
from collections import OrderedDict
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from typing import List
from xgboost import XGBRegressor

from src.defaults import TARGET_COLUMNS
from src.process_data import process_data
from src.model import ModelWrapper, METRICS_KEY, TARGETS_KEY

logging.basicConfig()


def train(
    database_path,
    target_columns: List[str],
    model_save_path: Path,
):
    prepared_data, categories_mapping = process_data(database_path)
    prepared_data = prepared_data.dropna(subset=target_columns)
    print(f"Input dataset size {len(prepared_data)}")

    data_train, data_test = train_test_split(prepared_data, test_size=0.4)
    y_train = data_train[target_columns]

    X_train = data_train.drop(columns=target_columns)
    y_test = data_test[target_columns]
    X_test = data_test.drop(columns=target_columns)

    xgbr = XGBRegressor(verbosity=1, num_target=len(target_columns))
    print("Train model:", xgbr)
    xgbr.fit(X_train, y_train)

    y_pred = np.array(xgbr.predict(X_test))
    y_test = np.array(y_test)
    print("Train finished:")
    targets_and_metrics = {TARGETS_KEY: [], METRICS_KEY: []}
    for i, column in enumerate(target_columns):
        y_t = y_test[:, i]
        y_p = y_pred[:, i]
        print(f"Metrics for target {column}:")
        print('MAE', mean_absolute_error(y_t, y_p))
        print('MSE', mean_squared_error(y_t, y_p))
        print('RMSE', mean_squared_error(y_t, y_p, squared=False))
        mape = mean_absolute_percentage_error(y_t, y_p)
        print('MAPE', mape)
        targets_and_metrics[TARGETS_KEY].append(column)
        targets_and_metrics[METRICS_KEY].append(mape)

    print("Features importance order")
    print(list(np.array(xgbr.feature_names_in_)[np.argsort(-xgbr.feature_importances_)]))
    ModelWrapper.save_model(model_save_path, xgbr, categories_mapping, targets_and_metrics)


def create_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-columns", type=str, nargs='+', default=TARGET_COLUMNS)
    parser.add_argument("--model-save-path", type=Path, required=True)
    parser.add_argument("--database-path", type=Path, required=True)
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    train(**vars(args))
