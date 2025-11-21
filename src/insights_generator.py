import pandas as pd
from logging import Logger
from src.utils.logger import setup_logger

class InsightsGenerator:
    """
    Generate actionable business insights from inventory and forecast data.
    Focuses on stockouts, overstock risk, and demand trends.
    """

    def __init__(self, logger: Logger = None):
        self.logger = logger if logger else setup_logger("insights_generator")

    def generate_insights(self, df_inventory: pd.DataFrame, df_forecast: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            df_inventory: Output from InventoryManager containing ROP, EOQ, safety stock, stockout warnings
            df_forecast: Forecast output from ProphetForecaster

        Returns:
            df_insights: DataFrame with business insights per product
        """
        self.logger.info("Generating business-ready insights...")

        # Convert forecast to weekly periods and aggregate
        df_forecast['Week'] = df_forecast['Date'].dt.to_period('W').apply(lambda r: r.start_time)
        df_weekly = df_forecast.groupby(['Product Name', 'Week']).agg({
            'Predicted Units': 'mean',
            'Predicted Units Daily': 'mean'
        }).reset_index()
        df_weekly.rename(columns={
            'Predicted Units': 'Avg Weekly Forecast',
            'Predicted Units Daily': 'Avg Daily Forecast'
        }, inplace=True)

        # Merge inventory metrics
        df_merged = pd.merge(
            df_weekly,
            df_inventory[['Product Name', 'ROP', 'Safety Stock', 'Days to Stockout', 'Stockout Warning', 'Overstock Risk']],
            on='Product Name',
            how='left'
        )

        # Handle NaNs in Days to Stockout
        df_merged['Days to Stockout'] = df_merged['Days to Stockout'].fillna(999)

        # Flags for actionable insights
        df_merged['Stockout Risk'] = df_merged['Stockout Warning'] & (df_merged['Days to Stockout'] < df_merged['ROP'])
        df_merged['Overstock Risk Flag'] = df_merged['Overstock Risk'] == 'High'

        # Summarize per product
        df_summary = df_merged.groupby('Product Name').agg({
            'Avg Weekly Forecast': 'mean',
            'Avg Daily Forecast': 'mean',
            'Stockout Risk': 'sum',  # count of weeks at risk
            'Overstock Risk Flag': 'sum',
            'Days to Stockout': 'min'
        }).reset_index()

        # Generate human-readable insights
        def insight_text(row):
            if row['Stockout Risk'] > 0:
                return "⚠️ Stockout risk in upcoming weeks. Prioritize ordering."
            elif row['Overstock Risk Flag'] > 0:
                return "📦 Overstock risk detected. Consider reducing orders."
            else:
                return "✅ Inventory levels healthy."

        df_summary['Insight'] = df_summary.apply(insight_text, axis=1)

        # Optional: remove internal flags before returning
        df_summary.drop(columns=['Stockout Risk', 'Overstock Risk Flag'], inplace=True)

        self.logger.info("Insights generation complete and business-ready")
        return df_summary


# # Test block
# if __name__ == "__main__":
#     from src.column_mapper import ColumnMapper
#     from src.schema_validator import SchemaValidator
#     from src.feature_engineering import FeatureEngineer
#     from src.forecasting.prophet_model import ProphetForecaster
#     from src.inventory_logic import InventoryManager
#     from src.utils.logger import setup_logger

#     logger = setup_logger("insights_generator_test")
#     logger.info("Testing updated InsightsGenerator with business-ready outputs...")

#     # Load raw data
#     df_raw = pd.read_csv("data/raw/data.csv")

#     # Column mapping
#     mapper = ColumnMapper(logger=logger)
#     df_mapped = mapper.map_inventory_dataset(df_raw)

#     # Schema validation
#     validator = SchemaValidator(logger=logger)
#     df_validated = validator.validate(df_mapped)

#     # Feature engineering
#     fe = FeatureEngineer(logger=logger)
#     df_fe = fe.transform(df_validated)

#     # Forecasting
#     forecaster = ProphetForecaster(logger=logger, forecast_horizon=4)
#     forecaster.fit(df_fe)
#     df_forecast = forecaster.predict()

#     # Inventory calculation
#     inventory_manager = InventoryManager(logger=logger)
#     df_inventory = inventory_manager.calculate_inventory_metrics(df_forecast, df_fe)

#     # Generate insights
#     generator = InsightsGenerator(logger=logger)
#     df_insights = generator.generate_insights(df_inventory, df_forecast)

#     # Show top 10 insights
#     print(df_insights.head(10))

#     # Debugging info
#     print("\nUnique Insight messages:", df_insights['Insight'].unique())
#     print("\nSummary statistics:")
#     print(df_insights.describe(include='all'))
