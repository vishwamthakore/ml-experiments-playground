import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd

def display_classification_metrics(y_test, accuracy, precision, recall, f1, cm):
    st.subheader("Classification Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")
    col4.metric("F1 Score", f"{f1:.3f}")

    labels = np.unique(y_test)

    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count")
    )

    st.subheader("Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

def display_regression_metrics(y_test, y_pred, mse, rmse, mae, r2):
    st.subheader("Regression Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.3f}")
    col3.metric("MSE", f"{mse:.3f}")
    col4.metric("RMSE", f"{rmse:.3f}")

    results_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })

    fig = px.scatter(
        results_df,
        x="Actual",
        y="Predicted",
        trendline="ols"
    )

    fig.add_shape(
        type="line",
        x0=results_df["Actual"].min(),
        y0=results_df["Actual"].min(),
        x1=results_df["Actual"].max(),
        y1=results_df["Actual"].max(),
        line=dict(dash="dash", color="red")
    )

    st.subheader("Actual vs Predicted")
    st.plotly_chart(fig, use_container_width=True)