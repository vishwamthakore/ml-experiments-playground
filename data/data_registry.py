import pandas as pd
from data.base import BaseDataset
from data.sklearn_datasets import Iris, Wine, BrestCancer, Digits
import streamlit as st

DATA_REGISTRY = {
    "iris": Iris,
    "wine": Wine,
    "brest cancer": BrestCancer,
    "digits": Digits
}

@st.cache_data
def load_dataset(name: str) -> pd.DataFrame:
    return DATA_REGISTRY[name]().load()


def get_datasets() -> list[str]:
    return DATA_REGISTRY.keys()