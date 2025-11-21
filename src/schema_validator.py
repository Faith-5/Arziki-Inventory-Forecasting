import pandas as pd
import numpy as np
from logging import Logger
from .utils.logger import setup_logger

class SchemaValidator:
    """
    Validates a DataFrame against the PRD schema:
    Ensures all required columns exist, correct data types, non-negative numeric values,
    and proper Date format.
    """

    def __init__(self, logger: Logger = None):
        self.logger = logger if logger else setup_logger("schema_validator")
        self.required_columns = [
            "Product Name",
            "Category",
            "Date",
            "Units Sold",
            "Stock on Hand",
            "Unit Cost",
            "Selling Price",
            "Supplier Lead Time"
        ]
        self.numeric_columns = [
            "Units Sold",
            "Stock on Hand",
            "Unit Cost",
            "Selling Price",
            "Supplier Lead Time"
        ]

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate DataFrame against PRD schema.
        Returns the validated DataFrame or raises ValueError with details.
        """
        self.logger.info("Starting schema validation...")

        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            msg = f"Schema validation failed: Missing columns: {missing_cols}"
            self.logger.error(msg)
            raise ValueError(msg)
        self.logger.info("All required columns exist")

        # Check numeric columns for non-negative values
        for col in self.numeric_columns:
            if not np.issubdtype(df[col].dtype, np.number):
                msg = f"Column '{col}' must be numeric, found {df[col].dtype}"
                self.logger.error(msg)
                raise ValueError(msg)
            if (df[col] < 0).any():
                msg = f"Column '{col}' contains negative values"
                self.logger.error(msg)
                raise ValueError(msg)
        self.logger.info("All numeric columns are valid and non-negative")

        # Check Date column
        if not np.issubdtype(df['Date'].dtype, np.datetime64):
            try:
                df['Date'] = pd.to_datetime(df['Date'], errors='raise')
                self.logger.info("Date column successfully converted to datetime")
            except Exception as e:
                msg = f"Date column cannot be converted to datetime: {e}"
                self.logger.error(msg)
                raise ValueError(msg)

        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            self.logger.warning(f"Data contains missing values:\n{missing_values[missing_values > 0]}")
        else:
            self.logger.info("No missing values detected")

        self.logger.info("Schema validation completed successfully")
        return df


# # -----------------------
# # Test Run (Standalone)
# # -----------------------
# if __name__ == "__main__":
#     from src.column_mapper import ColumnMapper

#     logger = setup_logger("schema_validator_test")
#     logger.info("Testing SchemaValidator...")

#     # Load sample dataset
#     try:
#         df_raw = pd.read_csv("data/raw/data.csv")
#         logger.info(f"Loaded raw dataset with shape: {df_raw.shape}")
#     except FileNotFoundError:
#         logger.error("Sample CSV not found at 'data/raw/data.csv'")
#         exit(1)

#     # Map columns
#     mapper = ColumnMapper(logger=logger)
#     df_prd = mapper.map_inventory_dataset(df_raw)

#     # Validate schema
#     validator = SchemaValidator(logger=logger)
#     df_validated = validator.validate(df_prd)

#     print(df_validated.head())
