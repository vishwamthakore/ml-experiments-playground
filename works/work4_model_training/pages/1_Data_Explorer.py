import streamlit as st
import pandas as pd
from utils import app_utils

if "dataset" not in st.session_state or st.session_state.dataset is None:
    st.warning("Please load a dataset first.")
    st.stop()

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
    
