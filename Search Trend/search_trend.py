import pandas as pd
import re
import json
from datetime import datetime

# -----------------------------------
# Configurable Time Periods
# -----------------------------------
period_start = "2025-03-01"
period_end = "2025-03-31"
previous_period_start = "2024-03-01"
previous_period_end = "2024-03-31"

# -----------------------------------
# Load and Clean Data
# -----------------------------------
df = pd.read_csv("search_trends.csv")


def clean_query(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["query"] = df["query"].apply(clean_query)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["frequency"] = df["frequency"].astype(int)

# -----------------------------------
# Filter Periods
# -----------------------------------
df_current = df[(df["timestamp"] >= period_start) & (df["timestamp"] <= period_end)]
df_previous = df[
    (df["timestamp"] >= previous_period_start)
    & (df["timestamp"] <= previous_period_end)
]

# -----------------------------------
# Aggregate Frequencies
# -----------------------------------
current_freq = df_current.groupby("query")["frequency"].sum()
previous_freq = df_previous.groupby("query")["frequency"].sum()

# -----------------------------------
# Compute Growth
# -----------------------------------
growth_list = []
for query in current_freq.index:
    cur = current_freq.get(query, 0)
    prev = previous_freq.get(query, 0)
    growth = ((cur - prev) / (prev + 1)) * 100
    growth_list.append({"searchTerm": query, "growthRate": round(growth, 2)})

df_growth = pd.DataFrame(growth_list).sort_values(by="growthRate", ascending=False)
top_10 = df_growth.head(10)

# -----------------------------------
# Categorization Logic
# -----------------------------------
category_keywords = {
    "fashion": [
        "dress",
        "tank",
        "top",
        "shirt",
        "tshirt",
        "trench",
        "coat",
        "blazer",
        "jacket",
        "sweater",
        "hoodie",
        "jeans",
        "denim",
        "shorts",
        "skirt",
        "trousers",
        "pants",
        "cargo",
        "joggers",
        "legging",
        "palazzo",
        "jumpsuit",
        "romper",
        "bodysuit",
        "co-ord",
        "matching set",
        "overalls",
        "bra",
        "sports bra",
        "camisole",
        "corset",
        "bralette",
        "kurta",
        "saree",
        "lehenga",
        "salwar",
        "kaftan",
        "cape",
        "shrug",
    ],
    "footwear": [
        "boots",
        "shoes",
        "heels",
        "flats",
        "sneakers",
        "sandals",
        "loafers",
        "flipflops",
        "mules",
        "slippers",
    ],
    "accessories": [
        "necklace",
        "ring",
        "earrings",
        "bracelet",
        "watch",
        "belt",
        "cap",
        "hat",
        "sunglasses",
        "goggles",
        "scarf",
        "beanie",
        "bag",
        "purse",
        "wallet",
        "backpack",
        "handbag",
    ],
    "electronics": [
        "phone",
        "charger",
        "laptop",
        "usb",
        "adapter",
        "tablet",
        "monitor",
        "keyboard",
        "mouse",
        "earphones",
        "headphones",
        "airpods",
        "earbuds",
        "speaker",
        "smartwatch",
    ],
    "home": [
        "curtain",
        "lamp",
        "sofa",
        "blanket",
        "pillow",
        "rug",
        "mattress",
        "towel",
        "cushion",
        "bedsheet",
        "duvet",
        "comforter",
        "quilt",
        "vase",
        "mirror",
    ],
}


def categorize_query(query):
    for category, keywords in category_keywords.items():
        if any(word in query for word in keywords):
            return category
    return "misc"


df_growth["category"] = df_growth["searchTerm"].apply(categorize_query)
top_10["category"] = top_10["searchTerm"].apply(categorize_query)

# -----------------------------------
# Export Outputs
# -----------------------------------
# JSON for top 10
json_output = {
    "periodStart": period_start,
    "periodEnd": period_end,
    "searchTerms": top_10.to_dict(orient="records"),
}
with open("top_trending_keywords.json", "w") as f:
    json.dump(json_output, f, indent=2)

# CSVs
df_growth.to_csv("all_trending_keywords_categorized.csv", index=False)
top_10.to_csv("top_10_trending_keywords_categorized.csv", index=False)

print("✅ All files exported:")
print("- top_trending_keywords.json")
print("- top_10_trending_keywords_categorized.csv")
print("- all_trending_keywords_categorized.csv")
