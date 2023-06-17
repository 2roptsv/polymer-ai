import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, Set
from xgboost import XGBRegressor

from src.process_data import DefaultSmilesFeaturizer


class ModelWrapper:
    @staticmethod
    def _model_path(checkpoint_path: Path):
        return checkpoint_path / (checkpoint_path.stem + ".txt")

    @staticmethod
    def _categories_mapping_path(checkpoint_path: Path):
        return checkpoint_path / (checkpoint_path.stem + "_cat_mapping.json")

    @staticmethod
    def _feature_names_path(checkpoint_path: Path):
        return checkpoint_path / (checkpoint_path.stem + "_feature_names.json")

    @staticmethod
    def save_model(
        checkpoint_path: Path,
        model: XGBRegressor,
        categories_mapping: Dict
    ):
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        model_checkpoint_path = ModelWrapper._model_path(checkpoint_path)
        print(f"Saving model at {model_checkpoint_path}")
        model.save_model(model_checkpoint_path)
        categories_mapping_dump_path = ModelWrapper._categories_mapping_path(checkpoint_path)
        print(f"Saving category mapping at {categories_mapping_dump_path}")
        with open(categories_mapping_dump_path, 'w') as f:
            json.dump(categories_mapping, f)
        feature_names_path = ModelWrapper._feature_names_path(checkpoint_path)
        print(f"Saving model input names at {feature_names_path}")
        with open(feature_names_path, 'w') as f:
            json.dump(list(model.feature_names_in_), f)

    def __init__(
        self,
        checkpoint_path: Path,
        smiles_featurizer: Callable[[str], np.ndarray] = DefaultSmilesFeaturizer()
    ):
        self._model = XGBRegressor()
        self._model.load_model(ModelWrapper._model_path(checkpoint_path))
        with open(ModelWrapper._categories_mapping_path(checkpoint_path), 'r') as f:
            self._categories_mapping = json.load(f)
        self._smiles_featurizer = smiles_featurizer
        with open(ModelWrapper._feature_names_path(checkpoint_path), 'r') as f:
            self._feature_names = json.load(f)

    def __call__(self, input_kwargs: Dict, smiles: str):
        model_names: Set[str] = set(self._feature_names)
        model_input = {}
        for key, value in input_kwargs:
            if key not in model_names:
                print(f"Passed input kwarg {key} not found in model input names. Check name or fix model train inputs.")
                continue
            model_input[key] = value

        not_found_names = set([name for name in model_names if not name.isnumeric() and name not in input_kwargs])
        print(f"Inputs {not_found_names} not found in keyword inputs. Defaulting to NaN")
        model_input.update({key: np.nan for key in not_found_names})

        for column_name, mapping in self._categories_mapping.items():
            if column_name in model_input:
                category_value = mapping.get(model_input[column_name])
                if category_value is None:
                    print(f"Value {model_input[column_name]} for categorical column {column_name} "
                          f"not found in mapping. Check if this value was present in train dataset."
                          f"Defaulting to NaN")
                    category_value = np.nan
                model_input[column_name] = category_value

        smiles = self._smiles_featurizer(smiles)
        if np.any(np.isnan(smiles)):
            print(f"Smiles featurizer failed to process {smiles}.")

        model_input.update({str(k): v for k, v in enumerate(smiles)})
        assert set(model_input.keys()) == model_names
        return self._model.predict(pd.DataFrame.from_dict([model_input]))[0]
