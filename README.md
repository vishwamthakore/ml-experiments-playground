# 🧠 ML Experiments Playground

An interactive Streamlit application for exploring datasets, applying feature engineering, and training machine learning models — all from a clean, modular interface.

Built as a hands-on experimentation layer for structured ML workflows.

> ## 🚀 Live Application
> [▶ Open ML Experiments Playground](https://fvzycrudy4iq8gzgt944lj.streamlit.app/)


## 📌 Project Overview

ML Experiments Playground allows you to:

- 📊 Explore datasets with automatic profiling and visualizations
- 🧩 Apply configurable feature engineering pipelines
- 🤖 Train classification and regression models
- 📈 Evaluate models with metrics and visual outputs

The goal of this project is to create a clean, organized, and extensible ML experimentation environment while learning proper project structuring and state management in Streamlit.

---

## 🚀 Usage Flow (User Perspective)

### 1️⃣ Load Dataset
- Select a built-in dataset (e.g., Iris)
- Dataset is loaded into session state

### 2️⃣ Explore Data
- View dataset preview (`df.head()`)
- Check shape and column types
- View summary statistics
- Identify numerical vs categorical columns

### 3️⃣ Apply Transformations
- Apply transformations to:
  - All numerical columns
  - All categorical columns
- Clean and prepare data for modeling

### 4️⃣ Visualize
- Generate visualizations:
  - Histograms
  - Correlation matrix
  - Scatter plots
  - Target distribution

### 5️⃣ Run ML Experiment
- Select:
  - Target column
  - Model
- Train model
- View performance metrics

---

## 🧠 Developer Notes

### State Management

The app uses Streamlit's `st.session_state` to manage:

- Loaded dataset
- Transformed dataset
- Selected target column
- Selected model
- Experiment results

Example state variables:

```python
st.session_state["dataset"]
st.session_state["dataset_name"]
st.session_state["profile"]

st.session_state["fe_inputs"]
st.session_state["X_train"]
st.session_state["X_test"]
st.session_state["y_train"]
st.session_state["y_test"]

st.session_state["model_config"]
st.session_state["model"]
st.session_state["y_pred"]
```

The variables of the next step must be deleted, if there are any changes in the previous step.
For example, if user uploads/selects a new dataset the feature engineering variables and training variables should be deleted.

### Design Decisions

🔹  Multi-Page Architecture

    Instead of a single-page application, the app is structured using Streamlit’s multipage system for:

    - Clean separation of concerns
    - Better scalability
    - Improved code organization

🔹 Sidebar Handling
    Each page shares the same sidebar logic via reusable functions to ensure consistent state across pages.

🔹 Simplified Transformations

    To avoid cluttered UI:
    - One transformation option for all numerical columns
    - One transformation option for all categorical columns

## 🛠️ Tech Stack

- Pythony
- Streamlit
- Pandas
- Scikit-learn
- Plotl

## 🎯 Future Improvements

- CSV upload support
- Experiment tracking history
- Model comparison view
- Download trained models
- Feature importance visualization

## 🛠️ Tech Stack

Python
Streamlit
Pandas
Scikit-learn
Matplotlib / Seaborn

## 🎯 Future Improvements

- CSV upload support
- Experiment tracking history
- Model comparison view
- Download trained models
- Feature importance visualization