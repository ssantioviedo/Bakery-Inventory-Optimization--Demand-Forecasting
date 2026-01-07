# 🚀 Quick Start Guide

## Run Locally (2 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the app
streamlit run app.py

# 3. Open browser to http://localhost:8501
```

---

## Deploy to Streamlit Cloud (3 minutes)

### Prerequisites:
- GitHub account
- This repository pushed to GitHub

### Steps:

1. **Go to:** [share.streamlit.io](https://share.streamlit.io)

2. **Click:** "New app"

3. **Select:**
   - Repository: `your-username/edinburgh_bakery`
   - Branch: `main`
   - Main file path: `app.py`

4. **Click "Deploy"** and wait ~2 minutes

5. **Get your link:** `https://your-username-edinburgh-bakery.streamlit.app`

6. **Share on LinkedIn!** 🎉

---

## Dashboard Sections

### 📊 Overview
- Key metrics at a glance
- Project summary and features

### 🔍 EDA (Exploratory Data Analysis)
- Top-selling products by count and weight
- Product summary table

### 📈 Historical Demand
- Time series visualization of flour usage
- Basic statistics

### 🤖 Model Evaluation
- Compare 4 forecasting models
- Adjustable backtesting parameters
- Model accuracy metrics (MAE, MAPE, WAPE, sMAPE)

### 📦 Inventory Recommendations
- Probabilistic demand forecast (Monte Carlo)
- Adjustable lead time and service levels
- Reorder point (ROP) recommendations table
- Business interpretation guide

---

## Troubleshooting

**App won't start:**
```bash
pip install --upgrade streamlit prophet statsmodels scikit-learn pandas numpy
```

**Data not loading:**
- Ensure `bread basket.csv` is in the same folder as `app.py`
- Check file path in `load_data()` function

**Streamlit Cloud deployment fails:**
- Make sure all dependencies are in `requirements.txt`
- Check that `app.py` is in the root directory
- No API keys or secrets should be hardcoded

---

## Share Your Project

### LinkedIn Post Template:

```
🥖 Excited to share my latest data science project: Bakery Inventory Optimization

📊 What it does:
• Analyzes bakery sales transactions to forecast daily flour demand
• Compares 4 time-series models (ETS, SARIMAX, Prophet, Seasonal Naive)
• Generates probabilistic safety stock recommendations using Monte Carlo

🔧 Tech Stack:
• Python (pandas, scikit-learn, prophet, statsmodels)
• Interactive Dashboard: Streamlit
• Deployed on Streamlit Cloud (free!)

🎯 Business Impact:
✓ Reduces inventory waste
✓ Prevents lost sales from stockouts
✓ Data-driven reorder point recommendations

👉 Try the interactive dashboard: [LINK]
📂 Code on GitHub: [LINK]

#DataScience #Analytics #Python #Streamlit #TimeSeries #InventoryOptimization
```

---

## Next Steps

- [ ] Run locally and test all features
- [ ] Push to GitHub
- [ ] Deploy to Streamlit Cloud
- [ ] Share on LinkedIn
- [ ] Get feedback from network
- [ ] Extend: Add more datasets, features, or deploy with FastAPI backend

---

**Questions?** Check the [Streamlit Docs](https://docs.streamlit.io) or [Streamlit Community](https://discuss.streamlit.io)
