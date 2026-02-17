import streamlit as st
import pandas as pd
import numpy as np
from analysis.feature_engineering import FeatureEngineer
from utils.training_form_utils import display_model_selection_form, display_parameters_form
from utils.model_evaluation_utils import display_classification_metrics, display_regression_metrics
from analysis.model_training import get_model
from analysis.model_evaluation import evaluate_classification_model, evaluate_regression_model

if "dataset" not in st.session_state or st.session_state.dataset is None:
    st.warning("Please load a dataset first.")
    st.stop()


if "X_train" not in st.session_state:
    st.warning("Please complete feature engineering first")
    st.stop()


st.success(f"Loaded dataset : {st.session_state.dataset_name}")
st.success(f"Feature computation is completed.")

# Load state data
df: pd.DataFrame = st.session_state.dataset
profile = st.session_state.profile

fe: FeatureEngineer = st.session_state.fe
X_train = st.session_state.X_train
X_test = st.session_state.X_test
y_train = st.session_state.y_train
y_test = st.session_state.y_test
X_test_unprocessed = st.session_state.X_test_unprocessed


st.markdown("## Train Model")

model_name = display_model_selection_form(problem_type=profile["problem_type"])

with st.form("model_form"):
    form_data = display_parameters_form(model_name=model_name)


if form_data["submitted"]:
    model_name = form_data["model_name"]
    params = form_data["params"]
    model = get_model(model_name=model_name, params=params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.session_state["model_config"]=form_data
    st.session_state["model"] = model
    st.session_state["y_pred"] = y_pred
    st.success("Training complete!")


if "model" in st.session_state:
    tab_model_results, tab_model_config = st.tabs(["Model Results", "Model Config"])

    model_config = st.session_state["model_config"]
    model = st.session_state["model"]
    y_pred = st.session_state["y_pred"]

    with tab_model_config:
        st.write(model_config)

    with tab_model_results:
        st.markdown("##### Model Results")

        # Display predictions with along with unprocessed test data 
        test_df = X_test_unprocessed.copy()

        test_df["Actual"] = y_test
        test_df["Predicted"] = y_pred

        # Reorder columns before display
        columns = list(test_df.columns[-2:]) + list(test_df.columns[:-2])
        test_df = test_df[columns]
        st.dataframe(test_df.head(10))

        st.markdown("##### Metrics")

        if profile["problem_type"] == "classification":
            accuracy, precision, recall, f1, cm = evaluate_classification_model(y_test, y_pred)
            display_classification_metrics(y_test, accuracy, precision, recall, f1, cm)

        elif profile["problem_type"] == "regression":
            mse, rmse, mae, r2 = evaluate_regression_model(y_test, y_pred)
            display_regression_metrics(y_test, y_pred, mse, rmse, mae, r2)