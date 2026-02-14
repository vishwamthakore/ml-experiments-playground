import streamlit as st
import pandas as pd
from data.data_registry import get_datasets, load_dataset
from analysis.data_profile import profile_dataset
from utils import app_utils

# initiaize state variables
if "dataset" not in st.session_state:
    st.session_state.dataset = None
    st.session_state.dataset_name = None

st.markdown("## ML Experiments Playground")    

dataset_name = st.sidebar.selectbox(
    label="Select Dataset", options=get_datasets(), placeholder="select a dataset"
)

load_btn = st.sidebar.button(label="Load dataset")

# load dataset only on click
if load_btn:
    st.session_state.dataset = load_dataset(dataset_name).load()
    st.session_state.dataset_name = dataset_name


# if dataset is loaded, show it
if st.session_state.dataset is not None:
    df: pd.DataFrame = st.session_state.dataset
    st.success(f"Loaded dataset : {st.session_state.dataset_name}")
    
    st.markdown("#### Data preview")
    st.dataframe(data=df.head(10))

    target_column_name = df.columns[-1]

    profile = profile_dataset(df, target_col="target")

    app_utils.display_profile(profile)
    
