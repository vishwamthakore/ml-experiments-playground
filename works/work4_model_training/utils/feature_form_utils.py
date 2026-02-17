import streamlit as st
import pandas as pd
from pydantic import BaseModel

def display_feature_form(df, profile):

    # ----- target selection -----
    # Bring last column to start to make it default in selectbox
    default_target = df.columns[-1]
    columns = [default_target] + list(df.columns[:-1])

    target_col = st.selectbox(
        label="Select Target Column",
        options=columns,
    )

    # ----- feature selection -----
    feature_cols = st.multiselect(
        "Select Feature Columns",
        [c for c in df.columns if c != target_col],
        default=[c for c in df.columns if c != target_col],
    )

    # detect types
    numeric_cols = [c for c in profile["numeric_columns"] if c in feature_cols]
    categorical_cols = [c for c in profile["categorical_columns"] if c in feature_cols]

    # ----- categorical transformation -----
    cat_transform = st.selectbox(
        "Categorical Transformation", ["OneHotEncoder", "OrdinalEncoder", "None"]
    )

    # ----- numeric transformation -----
    num_transform = st.selectbox(
        "Numeric Transformation", ["None", "StandardScaler", "MinMaxScaler"]
    )

    # ----- split ratio -----
    test_size = st.slider(
        "Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05
    )

    submit = st.form_submit_button("Apply Transformations")

    form_data = {
        "target_col": target_col,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "cat_transform": cat_transform,
        "num_transform": num_transform,
        "test_size": test_size,
        "submit": submit,
    }
    return form_data
