# 📱 Deployment Guide for LinkedIn Portfolio

## Overview

Your Streamlit app is **production-ready** and can be deployed for **free** on Streamlit Cloud. Here's everything you need to know.

---

## Option 1: Streamlit Cloud (Recommended for Portfolio) ⭐

### Why Streamlit Cloud?
- ✅ **Free** forever (generous free tier)
- ✅ **Zero setup** — no servers, databases, or infrastructure
- ✅ **Auto-deploy** from GitHub (push → automatic update)
- ✅ **Professional URL** — perfect for LinkedIn
- ✅ **Fastest to deploy** — 3 minutes
- ✅ **Best for portfolios** — shows modern web skills

### Step-by-Step Deployment:

#### Step 1: Initialize Git Repository
```bash
cd c:\Users\Santi\Desktop\edinburgh_bakery
git init
git add .
git commit -m "Initial commit: Bakery inventory optimization dashboard"
```

#### Step 2: Push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/edinburgh_bakery.git
git branch -M main
git push -u origin main
```

#### Step 3: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Sign in with GitHub
4. Fill in:
   - **Repository:** `your-username/edinburgh_bakery`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click "Deploy"
6. Wait 2-3 minutes for deployment
7. Your app is now live! 🎉

**Your URL will be:** `https://your-username-edinburgh-bakery.streamlit.app`

#### Step 4: Share on LinkedIn
```
🥖 Just deployed my Bakery Inventory Optimization dashboard!

This interactive app demonstrates:
• Time-series forecasting (Prophet, SARIMAX, ETS)
• Monte Carlo simulation for safety stock
• Expanding-window backtesting
• Modern Python + Streamlit architecture

Try it here: [URL]
Code: [GitHub Link]

#DataScience #Python #Streamlit #Analytics
```

---

## Option 2: Heroku (Professional Alternative)

### Pros:
- More customizable
- Good for production apps
- Better for high traffic

### Cons:
- Costs money ($5-50/month)
- More setup required

### Quick Setup:
1. Create `Procfile` in root:
   ```
   web: streamlit run app.py --logger.level=error
   ```

2. Create `runtime.txt`:
   ```
   python-3.9.16
   ```

3. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

---

## Option 3: Docker + AWS/Azure (Enterprise)

### For large-scale deployments:
- Docker containerization
- AWS ECS / Azure Container Instances
- Custom domains, SSL certificates
- Higher cost but maximum control

**Note:** This is overkill for a portfolio project unless you're targeting enterprise roles.

---

## What's Included in Your App

### Files:
- ✅ `app.py` — Main Streamlit application (500+ lines)
- ✅ `bread basket.csv` — Raw data
- ✅ `requirements.txt` — Dependencies (updated with Streamlit)
- ✅ `.streamlit/config.toml` — Theme customization (#D21E54 color scheme)
- ✅ `notebook.ipynb` — Original Jupyter notebook for reference
- ✅ `.gitignore` — Git configuration
- ✅ `README.md` — Comprehensive documentation
- ✅ `QUICKSTART.md` — Quick start guide

### Features:
1. **Interactive EDA** — Explore top products and patterns
2. **Historical Demand** — Visualize 162 days of flour usage
3. **Model Backtesting** — Compare 4 forecasting models with customizable parameters
4. **Monte Carlo Simulation** — Generate probabilistic demand forecasts
5. **ROP Recommendations** — Safety stock table for different service levels (50%, 95%, 98%, 99%)
6. **Professional UI** — Custom color scheme matching your original design

---

## After Deployment

### Marketing Your Project:

**LinkedIn Post:**
```
🥖 Exciting news! I just launched an interactive analytics dashboard 
for my Bakery Inventory Optimization project.

This dashboard demonstrates:

📊 Data Pipeline
→ Raw transactions → Aggregated daily demand
→ 162 days of bakery data analyzed

🤖 Machine Learning
→ Prophet, SARIMAX, ETS, Seasonal Naive models
→ Expanding-window backtesting for robust evaluation

📈 Probabilistic Forecasting
→ Monte Carlo simulation (10,000 scenarios)
→ ROP recommendations for 50%, 95%, 98%, 99% service levels

🛠️ Tech Stack
→ Python (pandas, scikit-learn, statsmodels, prophet)
→ Streamlit (interactive web dashboard)
→ Deployed on Streamlit Cloud (free, auto-updated from GitHub)

The app is fully interactive — try adjusting lead times, simulation counts, 
and service levels in real-time.

🔗 [Live Dashboard Link]
📂 [GitHub Repo Link]

This project showcases full-stack data science skills: data wrangling, EDA, 
statistical modeling, web app development, and cloud deployment.

#DataScience #Python #Streamlit #TimeSeries #Analytics #Portfolio
```

### GitHub README Tips:
- Add a badge showing Streamlit deployment status
- Include a GIF or screenshot of the dashboard
- Mention the free Streamlit Cloud deployment (shows cost-consciousness)
- Link to your live app prominently

### Personal Branding:
- Pin this project on your GitHub profile
- Mention the live dashboard link in your LinkedIn headline
- Create a project case study (Medium article, blog post)
- Include the dashboard link in your CV/resume

---

## Maintenance & Updates

### To update your deployed app:
```bash
# Make changes locally
git add .
git commit -m "Update forecasting parameters"
git push origin main

# Streamlit Cloud automatically detects changes and redeploys!
# Check status at https://share.streamlit.io/your-username/edinburgh-bakery
```

### Monitoring:
- View app logs on Streamlit Cloud dashboard
- Monitor usage stats (free tier shows ~1GB/month data transfer)

---

## Cost Breakdown

| Platform | Cost | Pros | Cons |
|----------|------|------|------|
| **Streamlit Cloud** | FREE | Simple, fast, auto-deploy | Limited customization |
| **Heroku** | $5-50/mo | Flexible, professional | Setup required |
| **AWS/Azure** | Pay-as-go | Maximum control | Complex setup |

**Recommendation:** Start with **Streamlit Cloud** (free), upgrade if you outgrow it.

---

## Troubleshooting Deployment

### App won't load:
- Check `requirements.txt` has all packages
- Verify file paths are relative (not absolute)
- View logs on Streamlit Cloud dashboard

### Data not loading:
- Ensure `bread basket.csv` is in repository root
- Check `.gitignore` doesn't exclude CSV
- Verify path in `load_data()` is correct

### Slow performance:
- Streamlit Cloud free tier uses 2GB RAM
- 10,000 simulations might take 30-60 seconds (acceptable for demo)
- Reduce `n_sims` slider maximum if needed

### Custom domain:
- Available on Streamlit Community Cloud Pro (paid)
- Alternative: Use GitHub Pages redirect

---

## Next Steps

1. **Today:** Deploy to Streamlit Cloud
2. **Tomorrow:** Share on LinkedIn with thoughtful write-up
3. **This week:** Reach out to 2-3 connections with personalized message linking to your dashboard
4. **Next week:** Write a case study blog post about the project
5. **Future:** Extend with additional features (compare forecasts, export ROP table, etc.)

---

## Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Community Forum](https://discuss.streamlit.io)
- [Prophet Documentation](https://facebook.github.io/prophet/)

---

**You're all set!** Your portfolio project is production-ready. Deploy it, share it, and build your data science profile! 🚀
