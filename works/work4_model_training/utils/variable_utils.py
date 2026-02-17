import streamlit as st

def delete_model_state_variables():
    if "model_config" in st.session_state:
        del st.session_state["model_config"]
        del st.session_state["model"]
        del st.session_state["y_pred"]


def delete_fe_state_variables():
    if "fe_inputs" in st.session_state:
        del st.session_state.fe_inputs
        del st.session_state.fe
        del st.session_state.X_train
        del st.session_state.X_test
        del st.session_state.y_train
        del st.session_state.y_test
        del st.session_state.X_test_unprocessed
