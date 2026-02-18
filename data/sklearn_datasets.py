from data.base import BaseDataset
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
import pandas as pd


def load_sklearn_dataset(load_func):
    X, y = load_func(return_X_y=True, as_frame=True)
    target_name = y.name if y.name else "target"
    X[target_name] = y
    return X


class Iris(BaseDataset):
    def load(self) -> pd.DataFrame:
        return load_sklearn_dataset(load_func=load_iris)


class Wine(BaseDataset):
    def load(self) -> pd.DataFrame:
        return load_sklearn_dataset(load_func=load_wine)

class BrestCancer(BaseDataset):
    def load(self) -> pd.DataFrame:
        return load_sklearn_dataset(load_func=load_breast_cancer)

class Digits(BaseDataset):
    def load(self) -> pd.DataFrame:
        return load_sklearn_dataset(load_func=load_digits)
