import pandas as pd
import numpy as np
from logging import Logger
from .utils.logger import setup_logger

class FeatureEngineer:
    """
    Performs feature engineering on a PRD-mapped DataFrame:
    - Date features: day, week, month, quarter, day-of-week
    - Lag features: previous day/week demand
    - Moving averages: 7-day and 30-day
    """

    def __init__(self, logger: Logger = None, lag_days=[1, 7], ma_windows=[7, 30]):
        self.logger = logger if logger else setup_logger("feature_engineering")
        self.lag_days = lag_days
        self.ma_windows = ma_windows

    def add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Adding date features...")
        df['day'] = df['Date'].dt.day
        df['week'] = df['Date'].dt.isocalendar().week
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['day_of_week'] = df['Date'].dt.dayofweek
        self.logger.info("Date features added.")
        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Adding lag features: {self.lag_days}")
        df_sorted = df.sort_values(['Product Name', 'Date']).copy()
        for lag in self.lag_days:
            df_sorted[f'Units_Sold_lag_{lag}'] = df_sorted.groupby('Product Name')['Units Sold'].shift(lag)
        self.logger.info("Lag features added.")
        return df_sorted

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Adding moving average features: {self.ma_windows}")
        df_sorted = df.sort_values(['Product Name', 'Date']).copy()
        for window in self.ma_windows:
            df_sorted[f'Units_Sold_ma_{window}'] = df_sorted.groupby('Product Name')['Units Sold'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        self.logger.info("Moving average features added.")
        return df_sorted

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps to the DataFrame.
        """
        df_fe = self.add_date_features(df)
        df_fe = self.add_lag_features(df_fe)
        df_fe = self.add_moving_averages(df_fe)

        # Optional: fill NaN in lag features with 0
        lag_cols = [col for col in df_fe.columns if 'lag' in col or 'ma' in col]
        df_fe[lag_cols] = df_fe[lag_cols].fillna(0)

        self.logger.info("Feature engineering completed.")
        return df_fe


# # -----------------------
# # Test Run (Standalone)
# # -----------------------
# if __name__ == "__main__":
#     from src.column_mapper import ColumnMapper
#     from src.schema_validator import SchemaValidator
#     from src.utils.logger import setup_logger

#     logger = setup_logger("feature_engineering_test")
#     logger.info("Testing FeatureEngineer...")

#     # Load and map dataset
#     df_raw = pd.read_csv("data/raw/data.csv")
#     mapper = ColumnMapper(logger=logger)
#     df_prd = mapper.map_inventory_dataset(df_raw)

#     # Validate schema
#     validator = SchemaValidator(logger=logger)
#     df_validated = validator.validate(df_prd)

#     # Apply feature engineering
#     fe = FeatureEngineer(logger=logger)
#     df_fe = fe.transform(df_validated)

#     print(df_fe.head())
