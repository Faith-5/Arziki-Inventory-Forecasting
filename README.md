# Arziki ML + Analytics Engine

## Overview

The **Arziki ML + Analytics Engine** is a fully modular, deterministic pipeline designed to process supermarket inventory and sales data, generate demand forecasts, compute inventory metrics, provide actionable insights, and produce charts for visualization.

It is built to **fit any CSV uploaded by stores or supermarkets**, standardizes the data to a fixed schema (PRD), and produces outputs ready for AI engineer integration, frontend dashboards, or reporting systems.

---

## Features

1. **Data Standardization**
   * Maps any raw CSV to the **PRD schema**:

     ```
     Product Name, Category, Date, Units Sold, Stock on Hand, Unit Cost, Selling Price, Supplier Lead Time
     ```
   * Simulates missing cost/lead time values where necessary.

2. **Schema Validation**
   * Ensures required columns exist.
   * Checks numeric ranges and date formats.
   * Logs errors for missing or invalid data.

3. **Feature Engineering**
   * Adds derived features for forecasting:
     * Day, week, month, quarter, day of week
     * Lag features (previous day/week demand)
     * Moving averages (7-day, 30-day)

4. **Demand Forecasting**
   * Weekly forecasts per product using **Prophet**.
   * Generates **daily forecasts** by dividing weekly predictions by 7.
   * Saves prediction intervals (upper/lower bounds).
   * Evaluates **MAPE** to ensure high accuracy.

5. **Inventory Logic**
   * Computes:
     * Avg daily demand
     * Safety stock
     * Reorder Point (ROP)
     * Economic Order Quantity (EOQ)
     * Days to stockout
     * Stockout and overstock risk flags

6. **Insights Generator**
   * Provides actionable messages per product:
     * ⚠️ Stockout risk
     * 📦 Overstock risk
     * ✅ Healthy inventory
   * Includes weekly and daily forecasts for visualization.

7. **Chart Generation**
   * Generates:
     * Daily forecast trends per product
     * Days to stockout per product
     * Category-level sales trends

---

## Folder Structure

````

/arziki-ml-engine
│
├── data/
│   └── sample_data/            # Sample dataset for testing
│
├── src/
│   ├── schema_validator.py
│   ├── column_mapper.py
│   ├── feature_engineering.py
│   ├── forecasting/
│   │   ├── prophet_model.py
│   ├── utils/
│   │   ├── logger.py
│   ├── inventory_logic.py
│   ├── insights_generator.py
│   ├── chart_generator.py
│   └── pipeline.py        # Main runner
│
├── output/
│
├── app.py              # Streamlit application
├── requirements.txt
├── README.md
└── run.py                 # CLI: run the pipeline

````

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
````

### 2. Run the Pipeline

```bash
python run.py --file data/sample_data/data.csv
```

Optional arguments:

* `--forecast_horizon <int>`: Number of weeks to forecast (default: 4)
* `--products <list>`: Filter pipeline for specific product names
* `--output_dir <path>`: Custom directory to save outputs (default: `output`)

Example:

```bash
python run.py --file data/sample_data/output.csv --forecast_horizon 6 --products P0001 P0002 --output_dir output_weekly
```

### 3. Output

* **Forecasts**: `output/forecast.json`
* **Insights**: `output/insights.json`
* **Charts**: `output/charts/`
* **Logs**: `output/logs/`

If products are filtered, filtered outputs are saved as `forecast_filtered.json` and `insights_filtered.json`.

---

## Retraining

To retrain models with new data:

1. Replace or add new CSVs in `data/raw/`.
2. Run the pipeline again using `run.py`.
3. Updated models will be saved in `models/prophet_model.pkl`.

> Note: Forecast models are **weekly Prophet models** per product. Daily forecasts are derived internally for inventory logic.

---

## Success Metrics

* **Low MAPE** on weekly forecasts (17% or lower in test runs)
* Accurate inventory metrics (ROP, EOQ, Days to Stockout)
* Correct insights per product
* Charts match forecast and inventory trends

---

## Example Output

### Insight Sample

| Product Name | Avg Weekly Forecast | Avg Daily Forecast | Days to Stockout | Insight                                                  |
| ------------ | ------------------- | ------------------ | ---------------- | -------------------------------------------------------- |
| P0010        | 4595                | 656                | 0.076            | ⚠️ Stockout risk in upcoming weeks. Prioritize ordering. |

### Forecast Sample

| Product Name | Date       | Predicted Units | Predicted Units Daily | Lower Bound | Upper Bound | Category  | Week       |
| ------------ | ---------- | --------------- | --------------------- | ----------- | ----------- | --------- | ---------- |
| P0001        | 2021-12-27 | 3370            | 481                   | 2452        | 4267        | Furniture | 2021-12-27 |

---