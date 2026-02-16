import streamlit as st
import pandas as pd
from data.data_registry import get_datasets, load_dataset
from analysis.data_profile import profile_dataset
from utils import app_utils

# initiaize state variables
if "dataset" not in st.session_state:
    st.session_state.dataset = None
    st.session_state.dataset_name = None
    st.session_state.profile = None

st.markdown("## ML Experiments Playground")    

dataset_name = st.sidebar.selectbox(
    label="Select Dataset", options=get_datasets(), placeholder="select a dataset"
)

load_btn = st.sidebar.button(label="Load dataset")

# load dataset only on click
if load_btn:
    st.session_state.dataset = load_dataset(dataset_name).load()
    st.session_state.dataset_name = dataset_name
    
    # calculate profile only when loading dataset
    df = st.session_state.dataset
    target_column_name = df.columns[-1]
    profile = profile_dataset(df, target_col=target_column_name)
    st.session_state.profile = profile


# if dataset is loaded, show it
if st.session_state.dataset is not None:
    df: pd.DataFrame = st.session_state.dataset
    profile = st.session_state.profile

    st.success(f"Loaded dataset : {st.session_state.dataset_name}")
    tab_overview, tab_visualization = st.tabs(["Overview", "Visualizations"])
    
    with tab_overview:
        st.markdown("#### Data preview")
        st.dataframe(data=df.head(10))
        app_utils.display_profile(profile)

    with tab_visualization:
        app_utils.display_visualizations(df=df, profile=profile)
    
