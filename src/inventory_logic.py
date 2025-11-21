import pandas as pd
import numpy as np
from logging import Logger
from src.utils.logger import setup_logger

class InventoryManager:
    def __init__(self, logger: Logger = None, service_level: float = 0.95, holding_cost_pct: float = 0.2, order_cost: float = 50):
        """
        Parameters:
        - service_level: desired service level for stockout prevention (e.g., 0.95)
        - holding_cost_pct: percentage of unit cost as holding cost per period
        - order_cost: fixed cost per order
        """
        self.logger = logger if logger else setup_logger("inventory_manager")
        self.service_level = service_level
        self.z = self._service_level_to_z(service_level)
        self.holding_cost_pct = holding_cost_pct
        self.order_cost = order_cost

    @staticmethod
    def _service_level_to_z(service_level: float) -> float:
        """Convert service level to z-score (normal distribution)"""
        from scipy.stats import norm
        return norm.ppf(service_level)

    def calculate_inventory_metrics(self, df_forecast: pd.DataFrame, df_current: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate inventory metrics: ROP, EOQ, Days to Stockout, Overstock Risk.

        df_forecast: DataFrame with columns ['Product Name', 'Date', 'Predicted Units']
        df_current: DataFrame with current inventory info, must include
            ['Product Name', 'Stock on Hand', 'Unit Cost', 'Supplier Lead Time']
        """
        self.logger.info("Starting inventory calculations...")

        # Clip negative forecasts
        df_forecast['Predicted Units'] = df_forecast['Predicted Units'].clip(lower=0)

        # Aggregate weekly forecast to daily average per product
        df_avg_demand = df_forecast.groupby('Product Name')['Predicted Units Daily'].mean().reset_index()
        df_avg_demand.rename(columns={'Predicted Units Daily': 'Avg Daily Demand'}, inplace=True)

        # Merge with current inventory
        df = pd.merge(df_current, df_avg_demand, on='Product Name', how='left')
        df['Lead Time'] = df['Supplier Lead Time'].fillna(7)  # default to 7 days if missing

        # Safety Stock
        df['Demand Std'] = df_forecast.groupby('Product Name')['Predicted Units'].std().reindex(df['Product Name']).values
        df['Safety Stock'] = self.z * df['Demand Std'] * np.sqrt(df['Lead Time'])

        # Reorder Point (ROP)
        df['ROP'] = (df['Avg Daily Demand'] * df['Lead Time']) + df['Safety Stock']

        # Economic Order Quantity (EOQ)
        df['Holding Cost'] = df['Unit Cost'] * self.holding_cost_pct
        df['EOQ'] = np.sqrt((2 * df['Avg Daily Demand'] * 30 * self.order_cost) / df['Holding Cost'])  # monthly demand

        # Days to Stockout
        df['Days to Stockout'] = df['Stock on Hand'] / df['Avg Daily Demand']
        df['Days to Stockout'] = df['Days to Stockout'].clip(lower=0)  # prevent negative or huge values

        # Stockout Warning
        df['Stockout Warning'] = (df['Days to Stockout'] < df['Lead Time']).astype(bool)

        # Overstock Risk
        df['Overstock Risk'] = ((df['Stock on Hand'] + df['EOQ']) - df['Avg Daily Demand'] * 30) / (df['Avg Daily Demand'] * 30)
        df['Overstock Risk'] = df['Overstock Risk'].apply(lambda x: 'High' if x > 0.25 else 'Normal')

        self.logger.info("Inventory calculations completed")
        return df[['Product Name', 'Date', 'Stock on Hand', 'Avg Daily Demand', 'Safety Stock', 'ROP', 'EOQ',
                   'Days to Stockout', 'Stockout Warning', 'Overstock Risk']]

# # -----------------------
# # Test Run
# # -----------------------
# if __name__ == "__main__":
#     from src.forecasting.prophet_model import ProphetForecaster
#     from src.column_mapper import ColumnMapper
#     from src.schema_validator import SchemaValidator
#     from src.feature_engineering import FeatureEngineer
#     from src.utils.logger import setup_logger

#     logger = setup_logger("inventory_logic_test")
#     logger.info("Testing InventoryManager...")

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

#     inv_manager = InventoryManager(logger=logger)
#     df_inventory_metrics = inv_manager.calculate_inventory_metrics(df_forecast, df_fe)
#     print(df_inventory_metrics.head())
