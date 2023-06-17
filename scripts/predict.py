import argparse
import deepchem
import json
import logging
from pathlib import Path

from src.model import ModelWrapper


logger = logging.getLogger('deepchem')
logger.disabled = True


def predict(
    keyword_inputs_json: str,
    smiles: str,
    model_checkpoint_path: Path,
):
    model = ModelWrapper(model_checkpoint_path)
    keyword_inputs = json.loads(keyword_inputs_json)
    print(model(keyword_inputs, smiles))


def create_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-checkpoint-path", type=Path, required=True)
    parser.add_argument("--smiles", type=str, required=True)
    parser.add_argument("--keyword-inputs-json", type=str, default='{}')
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    predict(**vars(args))