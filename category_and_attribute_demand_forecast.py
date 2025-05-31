import pandas as pd
from xgboost import XGBRegressor
from datetime import timedelta
import json

# Load datasets
sales = pd.read_csv("sales_data.csv")
catalog = pd.read_csv("product_catalog.csv")
sales["timestamp"] = pd.to_datetime(sales["timestamp"])
sales["quantitySold"] = sales["quantitySold"].astype(int)


# Extract modifiers
def extract_modifier(text, keywords):
    text = str(text).lower()
    for word in keywords:
        if word in text:
            return word
    return "unknown"


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

catalog["season"] = catalog["modifiers"].apply(
    lambda x: extract_modifier(x, season_keywords)
)
catalog["color"] = catalog["modifiers"].apply(
    lambda x: extract_modifier(x, color_keywords)
)

# Merge sales + catalog
merged = pd.merge(
    sales,
    catalog[["productId", "category", "season", "color"]],
    on="productId",
    how="left",
)

# Forecast config
forecast_start = pd.to_datetime("2025-06-01")
forecast_days = 30
forecast_end = forecast_start + timedelta(days=forecast_days - 1)

# Forecast loop
output = []
grouped = (
    merged.groupby(["category", "season", "color", "timestamp"])["quantitySold"]
    .sum()
    .reset_index()
)

for key, group in grouped.groupby(["category", "season", "color"]):
    df = group.rename(columns={"timestamp": "ds", "quantitySold": "y"})
    df = df.set_index("ds").asfreq("D").fillna(0).reset_index()
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["lag1"] = df["y"].shift(1)
    df["rolling7"] = df["y"].rolling(7).mean()
    df.dropna(inplace=True)

    if len(df) < 15:
        continue

    model = XGBRegressor(n_estimators=20, max_depth=3, verbosity=0)
    model.fit(df[["dayofweek", "month", "lag1", "rolling7"]], df["y"])

    preds = []
    last_known = df.iloc[-1].copy()
    recent = df["y"][-7:].tolist()

    for _ in range(forecast_days):
        next_day = last_known["ds"] + timedelta(days=1)
        next_input = {
            "dayofweek": next_day.dayofweek,
            "month": next_day.month,
            "lag1": last_known["y"],
            "rolling7": sum(recent[-7:]) / len(recent[-7:]),
        }
        pred = model.predict(pd.DataFrame([next_input]))[0]
        pred = max(0, int(pred))
        preds.append(pred)
        recent.append(pred)
        last_known = {"ds": next_day, "y": pred}

    output.append(
        {
            "category": key[0],
            "season": key[1],
            "color": key[2],
            "forecastedQuantity": int(sum(preds)),
        }
    )

# Save output
forecast_json = {
    "periodStart": forecast_start.strftime("%Y-%m-%d"),
    "periodEnd": forecast_end.strftime("%Y-%m-%d"),
    "forecast": output,
}

with open("category_and_attribute_demand_forecast.json", "w") as f:
    json.dump(forecast_json, f, indent=2)

print("✅ Saved: category_and_attribute_demand_forecast.json")
