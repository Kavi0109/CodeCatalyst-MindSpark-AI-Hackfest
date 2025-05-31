import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# 1. Load all data
# -------------------------------
catalog = pd.read_csv("product_catalog.csv")
sentiment = pd.read_csv("customer_feedback_sentiment_enriched.csv")
trend = pd.read_csv("query_growth.csv")

with open("product_demand_forecast.json") as f:
    product_forecast = json.load(f)

# -------------------------------
# 2. Preprocess product metadata
# -------------------------------
season_keywords = ["summer", "winter", "fall", "spring"]
color_keywords = [
    "black",
    "white",
    "navy",
    "red",
    "blue",
    "green",
    "yellow",
    "pink",
    "beige",
    "gray",
]


def extract_modifier(text, keywords):
    text = str(text).lower()
    for word in keywords:
        if word in text:
            return word
    return "unknown"


catalog["season"] = catalog["modifiers"].apply(
    lambda x: extract_modifier(x, season_keywords)
)
catalog["color"] = catalog["modifiers"].apply(
    lambda x: extract_modifier(x, color_keywords)
)


# -------------------------------
# 3. Match trend to title
# -------------------------------
def map_trend(title):
    title = str(title).lower()
    for _, row in trend.iterrows():
        if row["query"].lower() in title:
            return row["growthRate"]
    return 0


catalog["trend_growth_rate"] = catalog["title"].apply(map_trend)

# -------------------------------
# 4. Merge other signals
# -------------------------------
forecast_df = pd.DataFrame(product_forecast["forecast"])
sentiment["sentiment_score"] = sentiment["final_sentiment"].map(
    {"positive": 1, "neutral": 0.5, "negative": 0}
)
catalog = catalog.merge(
    sentiment[["productId", "sentiment_score"]], on="productId", how="left"
)
catalog = catalog.merge(
    sentiment[["productId", "final_sentiment"]], on="productId", how="left"
)

# -------------------------------
# 5. Normalize & Score
# -------------------------------
scaler = MinMaxScaler()
catalog[["norm_forecast", "norm_sentiment", "norm_trend"]] = scaler.fit_transform(
    catalog[["forecastedQuantity", "final_sentiment", "trend_growth_rate"]].fillna(0)
)

catalog["rank_score"] = (
    0.4 * catalog["norm_forecast"]
    + 0.3 * catalog["norm_sentiment"]
    + 0.3 * catalog["norm_trend"]
)

# -------------------------------
# 6. Output Final Insights
# -------------------------------
final_df = catalog[
    [
        "productId",
        "title",
        "category",
        "season",
        "color",
        "forecastedQuantity",
        "final_sentiment",
        "trend_growth_rate",
        "rank_score",
    ]
].sort_values(by="rank_score", ascending=False)

final_df.to_csv("final_product_insights.csv", index=False)
print("✅ final_product_insights.csv generated")
