import deepchem as dc
import logging
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple

from src.defaults import CATEGORICAL_COLUMNS, COUPLED_COLUMNS, TRAIN_COLUMNS, UNNEEDED_COLUMNS

logging.getLogger('deepchem').setLevel(logging.ERROR)


def handle_coupled_columns(row, c_left, c_right):
    if '-' in str(row[c_left]) or ',' in str(row[c_left]):
        value = row[c_right]
    else:
        value = row[c_left]
    if '(s)' in str(value):
        value = value.split('(')[0]
    value = value.lstrip().lstrip('(').rstrip('(?)') if isinstance(value, str) else value
    if ',' in str(value):
        value = float(value.split(',')[0])
    if ' ' in str(value):
        value = float(value.split()[0])
    value = float(str(value).replace('..', '.'))
    return value


class DefaultSmilesFeaturizer:
    def __init__(self):
        self.featurizer = dc.feat.Mol2VecFingerprint()

    def __call__(self, smiles: str) -> np.ndarray:
        return self.featurizer.featurize([smiles])[0]


def process_data(
    database_path,
    smiles_column: str = "SMILES",
    smiles_featurizer: Callable[[str], np.ndarray] = DefaultSmilesFeaturizer(),
    categorical_columns: List[str] = CATEGORICAL_COLUMNS,
    coupled_columns: List[Tuple[str, str, str]] = COUPLED_COLUMNS,
    unneeded_columns: List[str] = UNNEEDED_COLUMNS,
    train_columns: List[str] = TRAIN_COLUMNS
) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(database_path.open(), delimiter='\t')
    df = df.drop(index=[0])
    df = df.drop(columns=unneeded_columns)

    for c_left, c_right, c_dest in coupled_columns:
        df[c_dest] = df.apply(lambda row: handle_coupled_columns(row, c_left, c_right), axis=1).astype('float')

    data = df[train_columns].copy()
    data = data.dropna(subset=[smiles_column])

    # 'polyethylene' and 'polyethylens' are the same class
    def a(row):
        return row["POLYMER CLASS"].rstrip('s')
    data["POLYMER CLASS"] = data.apply(a, axis=1)

    categories_mapping = {}
    for col in categorical_columns:
        cat = data[col].astype('category').cat
        categories_mapping[col] = {v: k for k, v in enumerate(cat.categories)}
        data[col] = cat.codes

    smiles_df = data[smiles_column].apply(smiles_featurizer).apply(pd.Series)
    data = pd.concat([data, smiles_df], axis=1)
    data = data.drop(columns=smiles_column)
    data = data.dropna(subset=[0])
    data = data.reset_index()

    return data, categories_mapping




