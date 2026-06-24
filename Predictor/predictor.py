import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
import importlib.util

# Load cleaning.clean_data from cleaning.py if available, else define fallback
_cleaning_path = os.path.join(os.path.dirname(__file__), 'cleaning.py')
if os.path.exists(_cleaning_path):
    spec = importlib.util.spec_from_file_location('cleaning', _cleaning_path)
    _clean_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_clean_mod)
    clean_data = _clean_mod.clean_data
else:
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.replace('NaN', np.nan, inplace=True)
        numeric_cols = [c for c in df.columns if c != 'Student_Name']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan
        return df


class UniversalPredictor:
    """Predict any missing column using other columns as features."""

    def __init__(self):
        self.models = {}
        self.feature_sets = {}
        self.metrics = {}

    def load_data(self, path):
        df = pd.read_csv(path)
        return df

    # cleaning delegated to Predictor/cleaning.py via clean_data()

    def train_models(self, df):
        """Train a model for each column that has missing values."""
        df = clean_data(df)

        missing_cols = df.columns[df.isna().any()].tolist()
        if 'Student_Name' in missing_cols:
            missing_cols.remove('Student_Name')

        for target_col in missing_cols:
            feature_cols = [c for c in df.columns if c not in [target_col, 'Student_Name']]

            X = df[feature_cols].copy()
            y = df[target_col].copy()

            mask = y.notna()
            X = X[mask]
            y = y[mask]

            if len(X) < 3:
                continue

            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('lr', LinearRegression())
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

            cv_scores = cross_val_score(pipeline, X, y, cv=3, scoring='r2')
            cv_mean = float(cv_scores.mean())

            pipeline.fit(X, y)

            self.models[target_col] = pipeline
            self.feature_sets[target_col] = feature_cols
            self.metrics[target_col] = {
                'rmse': rmse,
                'cv_r2_score': cv_mean,
                'train_size': len(X),
                'test_size': len(X_test)
            }

    def predict_missing_values(self, df, student_name):
        """Predict all missing values for a specific student."""
        df = clean_data(df)

        row = df[df['Student_Name'].str.strip().str.lower() == student_name.strip().lower()]
        if row.empty:
            return None

        row = row.iloc[0]
        predictions = {}

        for col in df.columns:
            if col in ['Student_Name']:
                continue
            if pd.isna(row[col]):
                if col in self.models:
                    features = self.feature_sets[col]
                    X_input = row[features].values.reshape(1, -1)
                    pred = self.models[col].predict(X_input)[0]
                    predictions[col] = float(pred)

        return predictions if predictions else None

    def get_all_missing_students(self, df):
        df = clean_data(df)
        mask = df.drop(columns=['Student_Name']).isna().any(axis=1)
        return df[mask]['Student_Name'].tolist()
