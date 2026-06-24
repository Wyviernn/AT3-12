import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib


class AcademicPredictor:
    def __init__(self, target_column='Software_Engineering_Final'):
        self.target_column = target_column
        self.pipeline = None
        self.feature_columns = None

    def load_data(self, path):
        df = pd.read_csv(path)
        return df

    def clean_data(self, df):
        df = df.copy()
        # Normalize string 'NaN' to actual NaN
        df.replace('NaN', np.nan, inplace=True)

        # Identify numeric columns (exclude student name and any non-numeric)
        numeric_cols = [c for c in df.columns if c != 'Student_Name']

        # Coerce to numeric where possible
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Detect and null-out invalid scores: score < 0 or > 100
        for col in numeric_cols:
            df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan

        return df

    def prepare_features(self, df):
        df = df.copy()
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not in dataframe")

        X = df.drop(columns=[self.target_column, 'Student_Name'], errors='ignore')
        y = df[self.target_column]
        self.feature_columns = list(X.columns)
        return X, y

    def train(self, df, test_size=0.2, random_state=42):
        df = self.clean_data(df)
        X, y = self.prepare_features(df)

        # Drop rows where target is missing
        mask = y.notna()
        X = X[mask]
        y = y[mask]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Pipeline: impute median -> scale -> linear regression
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ])

        # Fit
        self.pipeline.fit(X_train, y_train)

        # Evaluate
        preds = self.pipeline.predict(X_test)
        rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))

        # Refit on full dataset for final predictions
        self.pipeline.fit(X, y)

        return {'rmse': float(rmse)}

    def predict_student(self, df, student_name):
        if self.pipeline is None:
            raise RuntimeError('Model not trained. Call train() first.')

        df_clean = self.clean_data(df)
        row = df_clean[df_clean['Student_Name'].str.strip().str.lower() == student_name.strip().lower()]
        if row.empty:
            raise ValueError(f"Student '{student_name}' not found in data")

        X_row = row.drop(columns=[self.target_column, 'Student_Name'], errors='ignore')
        # Ensure same feature columns ordering
        X_row = X_row.reindex(columns=self.feature_columns)
        pred = self.pipeline.predict(X_row)
        return float(pred[0])

    def save_pipeline(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'pipeline': self.pipeline, 'features': self.feature_columns}, path)

    def load_pipeline(self, path):
        data = joblib.load(path)
        self.pipeline = data['pipeline']
        self.feature_columns = data.get('features')
