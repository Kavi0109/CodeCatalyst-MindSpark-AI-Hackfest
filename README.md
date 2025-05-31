AI-Powered Demand Forecasting & Recommendation System - The CodeCatalyst

This repository contains the full solution for the AI Hackathon 2025 challenge: an intelligent forecasting and product recommendation system built on simulated fashion e-commerce datasets.

📌 Project Overview

This system combines time-series forecasting, sentiment analysis, and trend detection to provide actionable product recommendations and demand insights. The final insights are presented using an interactive Power BI dashboard.

🧱 Core Components

1. Product Demand Forecasting
   • Model: XGBoost Regressor
   • Input: sales_data.csv
   • Output: product_demand_forecast.json
   • Forecasts next 7 days of unit demand per product.

2. Category & Attribute Forecasting
   • Input: Aggregated sales by category + season + color
   • Output: category_and_attribute_demand_forecast.json
   • Forecasts demand over the next 30 days.

3. Customer Sentiment Analysis
   • Input: customer_feedback.csv
   • Tools: TextBlob + VADER + Rating adjustment
   • Output: customer_feedback_sentiment_enriched.csv
   • Sentiment score (0 to 1) for each product

4. Search Trend Growth Analysis
   • Input: search_trends.csv
   • Output: query_growth.csv
   • Captures keyword frequency growth & assigns it to products

5. Final Scoring Model
   • Merges forecast, sentiment, and trend growth
   • Formula: rank*score = 0.4 * forecast + 0.3 \_ sentiment + 0.3 \* trend
   • Output: final_product_insights.csv

📁 File Structure

├── data/
│ ├── product_catalog.csv
│ ├── sales_data.csv
│ ├── search_trends.csv
│ └── customer_feedback.csv
├── outputs/
│ ├── product_demand_forecast.json
│ ├── category_and_attribute_demand_forecast.json
│ ├── customer_feedback_sentiment_enriched.csv
│ ├── query_growth.csv
│ └── final_product_insights.csv
├── notebooks/
│ ├── sentiment_analysis.py
│ ├── trend_analysis.py
│ ├── demand_forecasting.py
│ └── final_model.py
├── dashboard/
│ └── PowerBI_Dashboard.pbix
└── README.md

🚀 How to Run Locally 1. Clone this repo 2. Install dependencies:

pip install -r requirements.txt

    3.	Run scripts in order:
    •	sentiment_analysis.py
    •	trend_analysis.py
    •	demand_forecasting.py
    •	final_model.py
    4.	Open Power BI dashboard in dashboard/PowerBI_Dashboard.pbix

📊 Deliverables
• Forecast JSON files
• Categorized query trends
• Final product insight CSV
• Power BI dashboard for executive insights

✅ Outcome

An end-to-end pipeline that unifies customer behavior, historical sales, and trend analysis into a single scoring model, powering smarter recommendations and supply chain decisions.

⸻

Made with 💡 for Hackathon 2025
