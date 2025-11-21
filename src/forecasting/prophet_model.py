import pandas as pd
import numpy as np
from prophet import Prophet
from logging import Logger
from pathlib import Path
import joblib
from src.utils.logger import setup_logger

class ProphetForecaster:
    def __init__(
        self, 
        logger: Logger = None, 
        forecast_horizon: int = 4,  # number of periods ahead (weeks/days depending on freq)
        freq: str = 'W',            # 'W' for weekly, 'D' for daily
        convert_to_daily: bool = True,
        changepoint_prior_scale: float = 0.05  # smoothness parameter
    ):
        self.logger = logger if logger else setup_logger("prophet_forecaster")
        self.forecast_horizon = forecast_horizon
        self.freq = freq
        self.convert_to_daily = convert_to_daily
        self.changepoint_prior_scale = changepoint_prior_scale
        self.models = {}

    def fit(self, df: pd.DataFrame, product_col='Product Name', date_col='Date', target_col='Units Sold', category_col= 'Category'):
        self.logger.info(f"Starting aggregation with freq='{self.freq}' for Prophet models...")

        # Aggregate based on frequency
        if self.freq == 'W':
            df['Period'] = df[date_col].dt.to_period('W').apply(lambda r: r.start_time)
        elif self.freq == 'D':
            df['Period'] = df[date_col]
        elif self.freq == 'M':
            df['Period'] = df[date_col].dt.to_period('M').apply(lambda r: r.start_time)
        else:
            raise ValueError(f"Unsupported frequency: {self.freq}")

        df_agg = df.groupby([product_col, 'Period'])[target_col].sum().reset_index()
        df_agg.rename(columns={'Period': 'ds', target_col: 'y'}, inplace=True)

        # Store product -> category mapping if category_col is provided
        if category_col and category_col in df.columns:
            self.product_category_map = df[[product_col, category_col]].drop_duplicates().set_index(product_col)[category_col].to_dict()
        else:
            self.product_category_map = {}

        products = df_agg[product_col].unique()
        self.logger.info(f"Training Prophet on {len(products)} products (aggregated)")

        for product in products:
            df_product = df_agg[df_agg[product_col] == product][['ds','y']]
            if df_product.empty:
                self.logger.warning(f"No data for product {product}, skipping.")
                continue

            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=self.changepoint_prior_scale
            )

            try:
                model.fit(df_product)
                self.models[product] = model
                self.logger.info(f"Trained Prophet model for {product} successfully.")
            except Exception as e:
                self.logger.error(f"Failed to fit model for {product}: {e}")

        self.logger.info("All Prophet models trained.")

    def predict(self) -> pd.DataFrame:
        self.logger.info(f"Generating forecasts for {self.forecast_horizon} periods ahead")
        forecasts = []

        for product, model in self.models.items():
            future = model.make_future_dataframe(periods=self.forecast_horizon, freq=self.freq)
            forecast = model.predict(future)

            # Clip negative predictions
            forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))
            forecast['yhat_lower'] = forecast['yhat_lower'].apply(lambda x: max(x, 0))
            forecast['yhat_upper'] = forecast['yhat_upper'].apply(lambda x: max(x, 0))

            # Convert weekly to daily if needed
            if self.freq == 'W' and self.convert_to_daily:
                predicted_daily = forecast['yhat'] / 7
            else:
                predicted_daily = forecast['yhat']

            forecast_product = pd.DataFrame({
                'Product Name': product,
                'Date': forecast['ds'],
                'Predicted Units': forecast['yhat'],
                'Predicted Units Daily': predicted_daily,
                'Lower Bound': forecast['yhat_lower'],
                'Upper Bound': forecast['yhat_upper']
            })

            forecasts.append(forecast_product)

        df_forecast = pd.concat(forecasts, ignore_index=True)
        df_forecast.sort_values(['Product Name', 'Date'], inplace=True)
        df_forecast.reset_index(drop=True, inplace=True)

        if hasattr(self, 'product_category_map') and self.product_category_map:
            df_forecast['Category'] = df_forecast['Product Name'].map(self.product_category_map)

        self.logger.info(f"Forecast generation complete. Total rows: {len(df_forecast)}")
        return df_forecast

    def evaluate_mape(self, df_actual: pd.DataFrame, df_forecast: pd.DataFrame, product_col='Product Name', target_col='Units Sold') -> float:
        # Aggregate actuals same as forecast frequency
        if self.freq == 'W':
            df_actual['Period'] = df_actual['Date'].dt.to_period('W').apply(lambda r: r.start_time)
        elif self.freq == 'D':
            df_actual['Period'] = df_actual['Date']
        elif self.freq == 'M':
            df_actual['Period'] = df_actual['Date'].dt.to_period('M').apply(lambda r: r.start_time)
        else:
            raise ValueError(f"Unsupported frequency: {self.freq}")

        df_agg = df_actual.groupby([product_col, 'Period'])[target_col].sum().reset_index()
        df_agg.rename(columns={'Period': 'Date'}, inplace=True)

        merged = pd.merge(df_agg, df_forecast, on=[product_col, 'Date'], how='inner')
        merged = merged[merged[target_col] > 0]  # ignore zero actuals

        if merged.empty:
            self.logger.warning("No overlapping data for MAPE evaluation.")
            return np.nan

        mape = (np.abs(merged[target_col] - merged['Predicted Units']) / merged[target_col]).mean() * 100
        self.logger.info(f"MAPE ({self.freq}-aggregated): {mape:.2f}%")
        return mape

    def save_models(self, path: str = "models/prophet_models.pkl"):
        Path(Path(path).parent).mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(self.models, path)
            self.logger.info(f"Saved all Prophet models into a single file at {path}")
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")


# # -----------------------
# # Test Run
# # -----------------------
# if __name__ == "__main__":
#     from src.column_mapper import ColumnMapper
#     from src.schema_validator import SchemaValidator
#     from src.feature_engineering import FeatureEngineer

#     logger = setup_logger("prophet_weekly_test")
#     logger.info("Testing improved ProphetForecaster...")

#     df_raw = pd.read_csv("data/raw/data.csv")
#     mapper = ColumnMapper(logger=logger)
#     df_prd = mapper.map_inventory_dataset(df_raw)

#     validator = SchemaValidator(logger=logger)
#     df_validated = validator.validate(df_prd)

#     fe = FeatureEngineer(logger=logger)
#     df_fe = fe.transform(df_validated)

#     forecaster = ProphetForecaster(logger=logger, forecast_horizon=4)
#     forecaster.fit(df_fe)
#     df_forecast = forecaster.predict()
#     forecaster.evaluate_mape(df_fe, df_forecast)
#     forecaster.save_models()
#     print(df_forecast.head())
