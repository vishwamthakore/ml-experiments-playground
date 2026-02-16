from data.base import BaseDataset
from sklearn.datasets import load_iris, load_wine
import pandas as pd


# def _sklearn_dataset

class Iris(BaseDataset):
    def load(self) -> pd.DataFrame:
        X, y = load_iris(return_X_y=True, as_frame=True)
        target_name = y.name if y.name else "target"
        X[target_name] = y
        return X


class Wine(BaseDataset):
    def load(self) -> pd.DataFrame:
        X, y = load_wine(return_X_y=True, as_frame= True)
        target_name = y.name if y.name else "target"
        X[target_name] = y
        return X