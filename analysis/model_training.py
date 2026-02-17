from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def get_model(model_name: str, params: dict):
    if model_name == "Logistic Regression":
        model = LogisticRegression(**params)

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(**params)

    elif model_name == "Random Forest":
        model = RandomForestClassifier(**params)

    elif model_name == "Linear Regression":
        model = LinearRegression(**params)

    elif model_name == "Decision Tree Regressor":
        model = DecisionTreeRegressor(**params)

    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor(**params)

    return model
