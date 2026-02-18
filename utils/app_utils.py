import streamlit as st
import plotly.express as px
from utils import visualization_utils


def display_profile(profile):
    st.header("Dataset Overview")

    st.write("Rows:", profile["n_rows"])
    st.write("Columns:", profile["n_columns"])

    st.subheader("Column Types")
    st.dataframe(profile["dtypes"])

    st.subheader("Null Values")
    st.write("Total Nulls:", profile["total_nulls"])
    st.dataframe(profile["nulls_per_column"])

    st.subheader("Target Variable")
    st.write("Problem Type:", profile["problem_type"])
    st.write("Unique classes:", profile["target_unique_values"])
    st.dataframe(profile["target_distribution"])

    st.subheader("Numeric Features Summary")
    st.dataframe(profile["numeric_summary"])

    if "correlation" in profile:
        st.subheader("Correlation Matrix")
        st.dataframe(profile["correlation"])


def display_visualizations(df, profile):
    st.header("Target Distribution Plot")
    target_column_name = profile["target_column_name"]

    if profile["problem_type"] == "classification":
        st.markdown("#### Problem Type : Classification")
        fig = visualization_utils.get_bar_plot(df=df, column_name=target_column_name)
        st.plotly_chart(fig, use_container_width=True)

    elif profile["problem_type"] == "regression":
        st.markdown("#### Problem Type : Regression")
        fig = visualization_utils.get_histogram(df=df, column_name=target_column_name)
        st.plotly_chart(fig, use_container_width=True)

    st.header("Feature Distribution Plot")

    numeric_cols = profile["numeric_columns"]
    if len(numeric_cols) >= 1:
        selected_num = st.selectbox("Select numeric feature", numeric_cols)
        st.subheader("Numeric Feature Distribution")
        fig = visualization_utils.get_histogram(df=df, column_name=selected_num)
        st.plotly_chart(fig, use_container_width=True)

    categorical_cols = profile["categorical_columns"]

    if len(categorical_cols) >= 1:
        selected_cat = st.selectbox("Select categorical feature", categorical_cols)
        st.subheader("Categorical Column Distribution")
        fig = visualization_utils.get_bar_plot(df=df, column_name=selected_cat)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = profile["correlation"]
    fig = visualization_utils.get_correlation_heatmap(correlation_matrix=corr)
    st.plotly_chart(fig, use_container_width=True)


    if profile["problem_type"] == "classification":
        st.subheader("Feature vs Target")
        x_feature = st.selectbox("X axis feature", numeric_cols, key="x")
        y_feature = st.selectbox("Y axis feature", numeric_cols, key="y")
        
        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            color=df[target_column_name].astype(str),
            title="Feature Relationship by Class"
        )
        st.plotly_chart(fig, use_container_width=True)


def display_instructions():

    st.markdown(
    """
    ### Welcome 👋

    This application is an interactive machine learning lab.
    You can load a dataset, explore it, prepare features, train models, and evaluate results — all step by step.

    You do **not** need to write code. Just follow the pages from left to right.

    ---

    ### How to Use the App

    ##### Step 1 — Load a Dataset (You are here)

    * Use the **sidebar on the left**
    * Choose a dataset
    * Click **Load Dataset**

    After loading, the dataset size will appear at the top of the page.

    ---

    ##### Step 2 — Data Explorer

    Open **Data Explorer** from the sidebar.

    Here you can understand the dataset:

    * View rows and columns
    * See the target variable
    * Check class balance
    * View feature distributions
    * View correlation heatmap

    Purpose:
    Before training any model, you should first understand your data.

    ---

    ##### Step 3 — Feature Engineering

    Go to **Feature Engineering**.

    Here you will:

    * Select which columns to use
    * Choose transformations

    * Scale numeric columns
    * Encode categorical columns
    * Split data into train and test sets

    Then click **Apply Transformations**.

    Purpose:
    Models cannot learn directly from raw data.
    We prepare the data so models can understand it.

    ---

    ##### Step 4 — Model Training

    Open **Model Training**.

    You can:

    * Select a machine learning model
    * Adjust a few hyperparameters
    * Click **Train Model**

    The app will automatically:

    * Train the model
    * Make predictions
    * Evaluate performance

    ---

    ##### Step 5 — Evaluation

    After training you will see:

    For classification:

    * Accuracy
    * Precision / Recall / F1
    * Confusion Matrix

    For regression:

    * R² score
    * Error metrics
    * Actual vs Predicted plot

    Purpose:
    This tells you how good your model really is.

    ---

    ### Recommended Workflow

    Always follow this order:

    **Load Dataset → Explore → Feature Engineering → Train → Evaluate**

    If you change dataset or feature settings, you should retrain the model.

    ---

    You can experiment safely — nothing you do will break the data."""
    )