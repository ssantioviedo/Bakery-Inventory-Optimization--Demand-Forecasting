# Bakery Inventory Optimization — Demand Forecasting

## Project Overview
This project applies a full data science workflow to the inventory optimization problem, from raw transactional data to actionable business recommendations. The pipeline includes:

## Installation & Reproducibility

To install all required Python packages and ensure reproducibility, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

- **Goal:** Transform sales data into accurate, actionable demand forecasts for smarter inventory decisions.
- **Key Features:**
   - Data wrangling and feature engineering to convert transactions into daily material demand
   - Exploratory data analysis (EDA) to identify key drivers and patterns
   - Implementation and comparison of multiple time series forecasting models (ETS, SARIMAX, Prophet, Naive)
   - Robust model evaluation using expanding-window backtesting and multiple error metrics (MAE, WAPE, sMAPE)
   - Probabilistic Monte Carlo simulation to generate reorder point (ROP) recommendations for any lead time and service level
   - Clean, reproducible code and clear visualizations

## Workflow
1. **Setup & Data Loading:**
   - Load and inspect the bakery sales dataset.
2. **Data Preprocessing & Aggregation:**
   - Parse timestamps, aggregate sales to daily counts per SKU, and prepare a continuous time series.
3. **Exploratory Data Analysis (EDA):**
   - Identify top-selling products and visualize demand patterns.
4. **Feature Engineering:**
   - Map product sales to estimated flour usage (kg/unit) and compute daily flour demand.
5. **Time Series Construction:**
   - Aggregate daily flour demand, fill missing days, and prepare data for forecasting models.
6. **Forecasting & Model Evaluation:**
   - Train and compare multiple models (ETS, SARIMAX, Prophet, Naive) using expanding-window backtests (MAE, WAPE, sMAPE).
7. **Inventory Recommendations:**
   - Use Monte Carlo simulation to generate a probabilistic ROP table for a chosen lead time (e.g., 4 days).
8. **Conclusion:**
   - Summarize business impact and actionable recommendations.

## Main Result: Reorder Point (ROP) Table

The notebook produces a clear, visually emphasized ROP table. This table answers the core business question: **How much of a key input should be stocked to meet demand over the lead time?**

- The table provides recommended stock levels for different service levels (mean, 95%, 98%, 99%) over a 4-day lead time.
- Higher service levels require more safety stock, helping you avoid stockouts at the cost of holding more inventory.

## Example Output
| Confidence Level | Recommended Stock (kg) | Safety Stock (vs Mean) |
|------------------|------------------------|------------------------|
| Mean (50%)       |        39.0            |         0.0            |
| 95%              |        50.3            |         11.3           |
| 98%              |        53.4            |         14.4           |
| 99%              |        55.4            |         16.5           |


## Business Impact
- **Reduces waste** by optimizing stock levels.
- **Prevents lost sales** by minimizing stockouts.
- **Provides a reproducible, data-driven pipeline** for inventory management.

---
