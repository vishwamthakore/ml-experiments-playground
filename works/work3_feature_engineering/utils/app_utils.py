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