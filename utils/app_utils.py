import streamlit as st


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
