import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class FeatureEngineer:
    def __init__(self,
                numeric_cols,
                categorical_cols,
                num_transform="None",
                cat_transform="OneHotEncoder"):

        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.num_transform = num_transform
        self.cat_transform = cat_transform

        self.num_scaler = None
        self.cat_encoder = None

    # ---------------- FIT ----------------
    def fit(self, df: pd.DataFrame):

        # ----- numeric -----
        if self.num_transform == "StandardScaler":
            self.num_scaler = StandardScaler()
            self.num_scaler.fit(df[self.numeric_cols])

        elif self.num_transform == "MinMaxScaler":
            self.num_scaler = MinMaxScaler()
            self.num_scaler.fit(df[self.numeric_cols])

        # ----- categorical -----
        if self.cat_transform == "OneHotEncoder":
            self.cat_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
            self.cat_encoder.fit(df[self.categorical_cols])

        elif self.cat_transform == "OrdinalEncoder":
            self.cat_encoder = OrdinalEncoder()
            self.cat_encoder.fit(df[self.categorical_cols])

    # ---------------- TRANSFORM ----------------
    def transform(self, df: pd.DataFrame):

        df = df.copy()

        # ---- numeric transform ----
        if self.num_scaler is not None and len(self.numeric_cols) > 0:
            scaled = self.num_scaler.transform(df[self.numeric_cols])
            df[self.numeric_cols] = scaled

        # ---- categorical transform ----
        if self.cat_encoder is not None and len(self.categorical_cols) > 0:

            encoded = self.cat_encoder.transform(df[self.categorical_cols])

            # get new column names for one hot
            if self.cat_transform == "OneHotEncoder":
                new_cols = self.cat_encoder.get_feature_names_out(self.categorical_cols)
                encoded_df = pd.DataFrame(encoded, columns=new_cols, index=df.index)

                df = df.drop(columns=self.categorical_cols)
                df = pd.concat([df, encoded_df], axis=1)

            else:  # Ordinal
                df[self.categorical_cols] = encoded

        return df

    # ---------------- FIT TRANSFORM ----------------
    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)
