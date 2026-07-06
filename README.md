# Sales Forecasting & Demand Intelligence System
## Internship Project — Week 3 & 4

A production-grade end-to-end sales forecasting system built on the Superstore dataset.

## Project Structure
```
├-- analysis.ipynb   # Complete Jupyter Notebook (8 tasks)
├-- train.csv        # Superstore dataset
├-- vgsales.csv      # Supplementary dataset
├-- app.py           # Streamlit interactive dashboard
├-- requirements.txt # Python dependencies
├-- summary.docx     # Executive business report
└-- charts/          # All visualization PNG exports
```

## Models Built
| Model | Purpose |
|-------|---------|
| SARIMA | Statistical seasonal forecasting |
| Facebook Prophet | Industry-grade forecasting |
| XGBoost | ML-based time series regression |

## Techniques Used
- Time Series Decomposition (Additive, period=12)
- Augmented Dickey-Fuller stationarity test
- Isolation Forest anomaly detection
- Z-Score rolling anomaly detection
- K-Means clustering + PCA visualization
- Elbow Method for optimal cluster count

## Setup
```bash
pip install -r requirements.txt
jupyter notebook analysis.ipynb
streamlit run app.py
```
