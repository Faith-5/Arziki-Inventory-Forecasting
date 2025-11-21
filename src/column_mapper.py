import pandas as pd
import numpy as np
from .utils.logger import setup_logger

class ColumnMapper:
    """
    Maps any input CSV into the standardized PRD schema:
    Product Name, Category, Date, Units Sold, Stock on Hand, Unit Cost, Selling Price, Supplier Lead Time
    """

    def __init__(self, logger=None):
        self.logger = logger if logger else setup_logger("column_mapper")
        self.prd_columns = [
            "Product Name",
            "Category",
            "Date",
            "Units Sold",
            "Stock on Hand",
            "Unit Cost",
            "Selling Price",
            "Supplier Lead Time"
        ]

    def map_inventory_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps Inventory Optimisation Dataset columns to PRD schema.
        """
        try:
            self.logger.info("Starting column mapping...")

            # If CSV already has all PRD columns, return it directly
            if all(col in df.columns for col in self.prd_columns):
                self.logger.info("CSV already matches PRD schema. No mapping required.")
                return df.copy()

            df_mapped = pd.DataFrame()
            df_mapped['Product Name'] = df['Product Name'] if 'Product Name' in df.columns else df['Product ID']
            df_mapped['Category'] = df['Category']
            df_mapped['Date'] = pd.to_datetime(df['Date'])
            df_mapped['Units Sold'] = df['Units Sold']
            df_mapped['Stock on Hand'] = df['Inventory Level']

            # Simulate Unit Cost (60% of Selling Price)
            df_mapped['Unit Cost'] = df['Price'] * 0.6

            # Selling Price (apply discount if available)
            if 'Discount' in df.columns:
                df_mapped['Selling Price'] = df['Price'] * (1 - df['Discount'].fillna(0).clip(upper=1))
            else:
                df_mapped['Selling Price'] = df['Price']

            # Deterministic lead time per product
            np.random.seed(42)
            lead_time_map = {
                name: np.random.randint(3, 15)
                for name in df_mapped['Product Name'].unique()
            }
            df_mapped['Supplier Lead Time'] = df_mapped['Product Name'].map(lead_time_map)

            self.logger.info(f"Column mapping complete. Mapped columns: {list(df_mapped.columns)}")
            return df_mapped

        except KeyError as e:
            msg = f"Missing expected column in dataset: {e}"
            self.logger.error(msg)
            raise ValueError(msg)
        except Exception as e:
            self.logger.error(f"Unexpected error during column mapping: {e}")
            raise e

    def map_to_prd(self, df: pd.DataFrame, custom_mapping: dict = None) -> pd.DataFrame:
        """
        Generic mapping function that allows user to provide custom column mapping:
        custom_mapping = {'Product Name': 'ProductID', 'Units Sold': 'Qty_Sold', ...}
        """
        df_mapped = pd.DataFrame()
        mapping = custom_mapping if custom_mapping else {col: col for col in self.prd_columns}

        for prd_col in self.prd_columns:
            source_col = mapping.get(prd_col)
            if source_col in df.columns:
                df_mapped[prd_col] = df[source_col]
            else:
                # Handle missing columns by simulating reasonable defaults
                if prd_col == "Unit Cost":
                    df_mapped[prd_col] = df['Price'] * 0.6 if 'Price' in df.columns else 0
                elif prd_col == "Supplier Lead Time":
                    np.random.seed(42)
                    df_mapped[prd_col] = np.random.randint(3, 15, size=len(df))
                elif prd_col == "Selling Price":
                    df_mapped[prd_col] = df['Price'] if 'Price' in df.columns else 0
                else:
                    df_mapped[prd_col] = 0  # Default placeholder

        # Ensure Date column is datetime
        if 'Date' in df_mapped.columns:
            df_mapped['Date'] = pd.to_datetime(df_mapped['Date'], errors='coerce')

        self.logger.info("Generic mapping to PRD schema completed.")
        return df_mapped


# # -----------------------
# # Test Run (Standalone)
# # -----------------------
# if __name__ == "__main__":
#     logger = setup_logger("column_mapper_test")
#     logger.info("Test run for ColumnMapper starting...")

#     # Load sample dataset (update path as needed)
#     try:
#         df_raw = pd.read_csv("data/raw/data.csv")
#         logger.info(f"Loaded raw dataset with shape: {df_raw.shape}")
#     except FileNotFoundError:
#         logger.error("Sample CSV not found at 'data/raw/data.csv'")
#         exit(1)

#     mapper = ColumnMapper(logger=logger)

#     # Map Inventory Optimisation Dataset
#     df_prd = mapper.map_inventory_dataset(df_raw)
#     logger.info(f"Mapped DataFrame shape: {df_prd.shape}")
#     logger.info(f"Mapped DataFrame columns: {list(df_prd.columns)}")

#     print(df_prd.head())
