import pandas as pd
from xgboost import XGBRegressor
from datetime import timedelta
import json

# ------------------------
# 1. Load & Preprocess
# ------------------------
sales = pd.read_csv("sales_data.csv")
sales["timestamp"] = pd.to_datetime(sales["timestamp"])
sales["quantitySold"] = sales["quantitySold"].astype(int)

agg = sales.groupby(["productId", "timestamp"]).sum().reset_index()

# ------------------------
# 2. Forecast Settings
# ------------------------
forecast_start = pd.to_datetime("2025-06-01")
forecast_days = 7
forecast_end = forecast_start + timedelta(days=forecast_days - 1)
product_output = []

# ------------------------
# 3. Forecast Loop
# ------------------------
for product_id in agg["productId"].unique():
    product_df = agg[agg["productId"] == product_id].copy()
    product_df = product_df.set_index("timestamp").asfreq("D").fillna(0).reset_index()

    # Features
    product_df["dayofweek"] = product_df["timestamp"].dt.dayofweek
    product_df["month"] = product_df["timestamp"].dt.month
    product_df["lag1"] = product_df["quantitySold"].shift(1)
    product_df["rolling7"] = product_df["quantitySold"].rolling(7).mean()
    product_df.dropna(inplace=True)

    if len(product_df) < 15:
        continue  # skip short history

    features = ["dayofweek", "month", "lag1", "rolling7"]
    model = XGBRegressor(n_estimators=20, max_depth=3, verbosity=0)
    model.fit(product_df[features], product_df["quantitySold"])

    # Forecast
    preds = []
    last_known = product_df.iloc[-1].copy()
    recent_sales = product_df["quantitySold"][-7:].tolist()

    for _ in range(forecast_days):
        next_day = last_known["timestamp"] + timedelta(days=1)
        next_input = {
            "dayofweek": next_day.dayofweek,
            "month": next_day.month,
            "lag1": last_known["quantitySold"],
            "rolling7": sum(recent_sales[-7:]) / len(recent_sales[-7:]),
        }

        pred = model.predict(pd.DataFrame([next_input]))[0]
        pred = max(0, int(pred))  # no negatives
        preds.append(pred)

        # update loop
        new_row = {"timestamp": next_day, "quantitySold": pred}
        recent_sales.append(pred)
        last_known = new_row
        product_df = pd.concat([product_df, pd.DataFrame([new_row])], ignore_index=True)

    product_output.append(
        {"productId": product_id, "forecastedQuantity": int(sum(preds))}
    )

# ------------------------
# 4. Save Output
# ------------------------
forecast_json = {
    "periodStart": forecast_start.strftime("%Y-%m-%d"),
    "periodEnd": forecast_end.strftime("%Y-%m-%d"),
    "forecast": product_output,
}

with open("product_demand_forecast.json", "w") as f:
    json.dump(forecast_json, f, indent=2)

print("✅ Forecast saved to product_demand_forecast.json")
