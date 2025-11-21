import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from logging import Logger
from src.utils.logger import setup_logger

sns.set(style="whitegrid")  # clean chart style

class ChartGenerator:
    """
    Generate business-ready charts from forecast, inventory, and insights data.
    Outputs PNGs suitable for PDF reports or dashboards.
    """

    def __init__(self, output_dir="output/charts", logger: Logger = None):
        self.logger = logger if logger else setup_logger("chart_generator")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_sales_forecast(self, df_forecast: pd.DataFrame, df_inventory: pd.DataFrame, product_name: str):
        """Sales forecast vs actual stock for a single product"""
        self.logger.info(f"Generating Sales Forecast chart for {product_name}")
        df_product_forecast = df_forecast[df_forecast['Product Name'] == product_name].copy()
        df_product_inventory = df_inventory[df_inventory['Product Name'] == product_name].copy()

        plt.figure(figsize=(12,6))
        plt.plot(df_product_forecast['Date'], df_product_forecast['Predicted Units'], label='Weekly Forecast', color='blue')
        plt.plot(df_product_forecast['Date'], df_product_forecast['Predicted Units Daily'], label='Daily Avg Forecast', color='cyan', linestyle='--')
        plt.scatter(df_product_inventory['Date'], df_product_inventory['Stock on Hand'], label='Stock on Hand', color='orange')
        if 'ROP' in df_product_inventory.columns:
            plt.axhline(df_product_inventory['ROP'].mean(), color='red', linestyle=':', label='Reorder Point (ROP)')
        plt.title(f"Sales Forecast vs Stock - {product_name}")
        plt.xlabel("Date")
        plt.ylabel("Units")
        plt.legend()
        plt.tight_layout()
        file_path = self.output_dir / f"{product_name}_SalesForecast.png"
        plt.savefig(file_path)
        plt.close()
        self.logger.info(f"Chart saved: {file_path}")

    def plot_days_to_stockout(self, df_inventory: pd.DataFrame, product_name: str):
        """Days to stockout trend with stockout warning highlight"""
        self.logger.info(f"Generating Days to Stockout chart for {product_name}")
        df_product = df_inventory[df_inventory['Product Name'] == product_name].copy()
        plt.figure(figsize=(12,6))
        plt.plot(df_product['Date'], df_product['Days to Stockout'], marker='o', color='green')
        plt.fill_between(df_product['Date'], 0, df_product['Days to Stockout'],
                         color='red', alpha=0.2, where=df_product['Stockout Warning'])
        plt.title(f"Days to Stockout Trend - {product_name}")
        plt.xlabel("Date")
        plt.ylabel("Days to Stockout")
        plt.tight_layout()
        file_path = self.output_dir / f"{product_name}_DaysToStockout.png"
        plt.savefig(file_path)
        plt.close()
        self.logger.info(f"Chart saved: {file_path}")

    def plot_category_trends(self, df_forecast: pd.DataFrame, category_col='Category'):
        """Aggregated category-level forecast trends"""
        self.logger.info(f"Generating category-level trends chart")
        df_category = df_forecast.groupby([category_col, 'Date'])['Predicted Units'].sum().reset_index()
        plt.figure(figsize=(12,6))
        sns.lineplot(data=df_category, x='Date', y='Predicted Units', hue=category_col, marker='o')
        plt.title("Category-Level Forecast Trends")
        plt.xlabel("Date")
        plt.ylabel("Total Predicted Units")
        plt.legend(title=category_col)
        plt.tight_layout()
        file_path = self.output_dir / "Category_Trends.png"
        plt.savefig(file_path)
        plt.close()
        self.logger.info(f"Chart saved: {file_path}")

    def generate_all_charts(self, df_forecast: pd.DataFrame, df_inventory: pd.DataFrame, products=None, category_col='Category'):
        """Generate charts for all products and category trends"""
        self.logger.info("Starting chart generation for all products...")
        if products is None:
            products = df_forecast['Product Name'].unique()
        for product in products:
            self.plot_sales_forecast(df_forecast, df_inventory, product)
            self.plot_days_to_stockout(df_inventory, product)
        self.plot_category_trends(df_forecast, category_col)
        self.logger.info("All charts generated successfully")

# # -----------------------
# # Test Run
# # -----------------------
# if __name__ == "__main__":
#     from src.column_mapper import ColumnMapper
#     from src.schema_validator import SchemaValidator
#     from src.feature_engineering import FeatureEngineer
#     from src.forecasting.prophet_model import ProphetForecaster
#     from src.inventory_logic import InventoryManager
#     from src.utils.logger import setup_logger

#     logger = setup_logger("chart_generator_test")
#     logger.info("Testing ChartGenerator...")

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

#     # Chart generation
#     from src.chart_generator import ChartGenerator
#     chart_gen = ChartGenerator(logger=logger)
#     chart_gen.generate_all_charts(df_forecast, df_inventory)

#     logger.info("ChartGenerator test run complete. Check 'output/charts/' for generated charts.")
