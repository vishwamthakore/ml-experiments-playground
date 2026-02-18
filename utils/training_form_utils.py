import streamlit as st

def display_model_selection_form(problem_type: str) -> dict:
    # Select model type
    if problem_type == "classification":
        st.markdown("#### Problem Type : Classification")
        model_name = st.selectbox(
            "Choose Model",
            ["Logistic Regression", "Decision Tree", "Random Forest"]
        )

    elif problem_type == "regression":
        st.markdown("#### Problem Type : Regression")
        model_name = st.selectbox(
            "Choose Model",
            ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
        )
    
    return model_name

def display_parameters_form(model_name: str):
    # Select hyperparameters
    params = {}
    if model_name == "Logistic Regression":
        C = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
        max_iter = st.slider("Max Iterations", 100, 1000, 200)
        params["C"] = C
        params["max_iter"] = max_iter

    elif "Decision Tree" in model_name:
        max_depth = st.slider("Max Depth", 1, 30, 5)
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
        params["max_depth"] = max_depth
        params["min_samples_split"] = min_samples_split

    elif "Random Forest" in model_name:
        n_estimators = st.slider("Number of Trees", 5, 100, 20)
        max_depth = st.slider("Max Depth", 1, 30, 10)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth

    elif model_name == "Linear Regression":
        fit_intercept = st.checkbox("Fit Intercept", value=True)
        positive = st.checkbox("Force Positive Coefficients", value=False)
        params["fit_intercept"] = fit_intercept
        params["positive"] = positive

    # Submit form button
    submitted = st.form_submit_button("Train Model")

    form_data = {
        "model_name" : model_name,
        "params" : params,
        "submitted" : submitted
    }

    return form_data