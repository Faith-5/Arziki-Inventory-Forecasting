import pandas as pd
from pathlib import Path
from logging import Logger
from src.utils.logger import setup_logger
from src.column_mapper import ColumnMapper
from src.schema_validator import SchemaValidator
from src.feature_engineering import FeatureEngineer
from src.forecasting.prophet_model import ProphetForecaster
from src.inventory_logic import InventoryManager
from src.insights_generator import InsightsGenerator
from src.chart_generator import ChartGenerator
import json

class ArzikiPipeline:
    """
    End-to-end ML & Analytics pipeline for Arziki.
    Steps:
    1. Column mapping → PRD schema
    2. Schema validation
    3. Feature engineering
    4. Forecasting (Prophet)
    5. Inventory calculation
    6. Insights generation
    7. Charts generation
    8. Export JSON + PNGs
    """

    def __init__(self, output_dir="output", logger: Logger = None, forecast_horizon: int = 4):
        self.logger = logger if logger else setup_logger("arziki_pipeline")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.forecast_horizon = forecast_horizon

        # Initialize modules
        self.mapper = ColumnMapper(logger=self.logger)
        self.validator = SchemaValidator(logger=self.logger)
        self.fe = FeatureEngineer(logger=self.logger)
        self.forecaster = ProphetForecaster(logger=self.logger, forecast_horizon=self.forecast_horizon)
        self.inventory_manager = InventoryManager(logger=self.logger)
        self.insights_generator = InsightsGenerator(logger=self.logger)
        self.chart_generator = ChartGenerator(output_dir=self.output_dir / "charts", logger=self.logger)

    def run(self, input_csv: str):
        self.logger.info(f"Starting pipeline for input: {input_csv}")
        
        # Step 1: Load and map columns
        df_raw = pd.read_csv(input_csv)
        df_mapped = self.mapper.map_inventory_dataset(df_raw)

        # Step 2: Validate schema
        df_validated = self.validator.validate(df_mapped)

        # Step 3: Feature engineering
        df_fe = self.fe.transform(df_validated)

        # Step 4: Forecasting
        self.forecaster.fit(df_fe)
        df_forecast = self.forecaster.predict()

        mape_value = self.forecaster.evaluate_mape(df_fe, df_forecast)
        self.logger.info(f"Forecasting MAPE ({self.forecaster.freq}-aggregated): {mape_value:.2f}%")

        # Save combined models for this CSV/session
        self.forecaster.save_models(path=self.output_dir / "prophet_models.pkl")

        # Step 5: Inventory calculations
        df_inventory = self.inventory_manager.calculate_inventory_metrics(df_forecast, df_fe)

        # Step 6: Insights generation
        df_insights = self.insights_generator.generate_insights(df_inventory, df_forecast)

        # Step 7: Chart generation
        self.chart_generator.generate_all_charts(df_forecast, df_inventory)

        # Step 8: Export JSON
        forecast_path = self.output_dir / "forecast.json"
        insights_path = self.output_dir / "insights.json"
        df_forecast.to_json(forecast_path, orient="records", date_format="iso")
        df_insights.to_json(insights_path, orient="records", date_format="iso")

        self.logger.info(f"Pipeline complete. Outputs saved to {self.output_dir}")
        return {
            "forecast": forecast_path,
            "insights": insights_path,
            "charts_dir": str(self.output_dir / "charts")
        }


# # -----------------------
# # CLI / Test Run
# # -----------------------
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Run Arziki ML & Analytics Pipeline")
#     parser.add_argument("--file", type=str, required=True, help="Path to input CSV file")
#     args = parser.parse_args()

#     logger = setup_logger("arziki_pipeline_test")
#     pipeline = ArzikiPipeline(logger=logger, forecast_horizon=4)
#     outputs = pipeline.run(args.file)

#     logger.info("Pipeline test run complete. Summary of outputs:")
#     logger.info(outputs)
