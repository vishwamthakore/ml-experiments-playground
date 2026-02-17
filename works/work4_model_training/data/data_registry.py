import pandas as pd
from data.base import BaseDataset
from data.sklearn_datasets import Iris, Wine

DATA_REGISTRY = {
    "iris": Iris,
    "wine": Wine
}

def load_dataset(name: str) -> BaseDataset:
    return DATA_REGISTRY[name]()


def get_datasets() -> list[str]:
    return DATA_REGISTRY.keys()