import importlib.resources
import json
import pickle
from pathlib import Path


def _load_json_resource(filename, path='scaner.data'):
    with importlib.resources.open_text(path, filename) as fh:
        return json.load(fh)


def load_labels():
    with importlib.resources.open_binary('scaner.data', 'sids_hadmids.pkl') as fh:
        return pickle.load(fh)


def load_annotations(prefix):
    """

    Args:
        prefix (str): in {test, train, val}

    Returns:

    """
    return _load_json_resource(f'{prefix}_hadm.json', 'scaner.data.annotations')


def load_guidelines():
    return _load_json_resource('section_guidelines.json')


def load_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def load_pickle(path: Path):
    with open(path, 'rb') as fh:
        sids, hadmids = pickle.load(fh)
    return sids, hadmids
