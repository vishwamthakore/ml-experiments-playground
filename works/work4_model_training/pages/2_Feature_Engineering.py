import pandas as pd
import streamlit as st
from utils.feature_form_utils import display_feature_form
from sklearn.model_selection import train_test_split
from analysis.feature_engineering import FeatureEngineer
from utils import variable_utils

if "dataset" not in st.session_state or st.session_state.dataset is None:
    st.warning("Please load a dataset first.")
    st.stop()


st.success(f"Loaded dataset : {st.session_state.dataset_name}")

# Feature Engineering
df: pd.DataFrame = st.session_state.dataset
profile = st.session_state.profile

st.header("Feature Engineering")
with st.form("feature_form"):
    form_data = display_feature_form(df, profile)

if form_data["submit"]:
    # process feature engineering and save outputs only on submit
    st.session_state.fe_inputs = form_data

    target_col = form_data["target_col"]
    feature_cols = form_data["feature_cols"]
    numeric_cols = form_data["numeric_cols"]
    categorical_cols = form_data["categorical_cols"]
    cat_transform = form_data["cat_transform"]
    num_transform = form_data["num_transform"]
    test_size = form_data["test_size"]

    X = df[feature_cols]
    y = df[target_col]

    # ALWAYS split first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42
    )

    fe = FeatureEngineer(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        num_transform=num_transform,
        cat_transform=cat_transform
    )

    X_train_processed = fe.fit_transform(X_train)
    X_test_processed = fe.transform(X_test)

    # store in session
    st.session_state.fe = fe
    st.session_state.X_train = X_train_processed
    st.session_state.X_test = X_test_processed
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.X_test_unprocessed = X_test

    # Delete or clear the training variables on feature engineering form submission
    variable_utils.delete_model_state_variables()



# Display Feature Engineering Outputs    
if "X_train" in st.session_state:
    tab_fe_results, tab_fe_inputs = st.tabs(["Feature Engineering Results", "Feature Engineering Inputs"])
    
    with tab_fe_results:
        st.success("Transformations applied!")
        X_train_processed = st.session_state.X_train
        st.dataframe(X_train_processed.head())
    
    with tab_fe_inputs:
        fe_inputs = st.session_state.fe_inputs
        st.write(fe_inputs)

    

    