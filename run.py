import argparse
from pathlib import Path
from src.utils.logger import setup_logger
from src.pipeline import ArzikiPipeline
import pandas as pd

def main():
    # -----------------------
    # CLI arguments
    # -----------------------
    parser = argparse.ArgumentParser(description="Run Arziki ML & Analytics Pipeline")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to input CSV file containing supermarket inventory/sales data"
    )
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=4,
        help="Forecast horizon in weeks (default: 4)"
    )
    parser.add_argument(
        "--products",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of product names to filter and forecast"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save forecasts, insights, and charts"
    )
    args = parser.parse_args()

    # -----------------------
    # Setup logger
    # -----------------------
    logger = setup_logger("arziki_run")
    logger.info("Starting Arziki pipeline run...")

    # -----------------------
    # Run pipeline
    # -----------------------
    pipeline = ArzikiPipeline(
        output_dir=args.output_dir,
        logger=logger,
        forecast_horizon=args.forecast_horizon
    )

    outputs = pipeline.run(args.file)
    logger.info("Pipeline run complete")

    # -----------------------
    # Optional: Filter by products
    # -----------------------
    if args.products:
        logger.info(f"Filtering outputs for products: {args.products}")
        # Forecast JSON
        df_forecast = pd.read_json(outputs["forecast"])
        df_forecast_filtered = df_forecast[df_forecast["Product Name"].isin(args.products)]
        df_forecast_filtered.to_json(Path(args.output_dir) / "forecast_filtered.json", orient="records", date_format="iso")
        # Insights JSON
        df_insights = pd.read_json(outputs["insights"])
        df_insights_filtered = df_insights[df_insights["Product Name"].isin(args.products)]
        df_insights_filtered.to_json(Path(args.output_dir) / "insights_filtered.json", orient="records", date_format="iso")
        logger.info("Filtered outputs saved")

    logger.info(f"All outputs available in: {Path(args.output_dir).resolve()}")
    logger.info("Charts folder: " + str(Path(args.output_dir) / "charts"))

if __name__ == "__main__":
    main()
