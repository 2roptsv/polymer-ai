import argparse
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from typing import List
from xgboost import XGBRegressor

from src.defaults import TARGET_COLUMNS
from src.process_data import process_data
from src.model import ModelWrapper

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

    y_pred = xgbr.predict(X_test)
    print("Train finished:")
    print('MAE', mean_absolute_error(y_test, y_pred))
    print('MSE', mean_squared_error(y_test, y_pred))
    print('RMSE', mean_squared_error(y_test, y_pred, squared=False))
    print('MAPE', mean_absolute_percentage_error(y_test, y_pred))

    print("Features importance order")
    print(list(np.array(xgbr.feature_names_in_)[np.argsort(-xgbr.feature_importances_)]))
    ModelWrapper.save_model(model_save_path, xgbr, categories_mapping)


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
