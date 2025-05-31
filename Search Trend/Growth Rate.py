import pandas as pd
import re

# -------------------------------------
# Load the original file
# -------------------------------------
df = pd.read_csv("search_trends.csv")


# -------------------------------------
# Clean and normalize the queries
# -------------------------------------
def clean_query(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["query"] = df["query"].apply(clean_query)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["frequency"] = df["frequency"].astype(int)

# -------------------------------------
# Define periods
# -------------------------------------
period_start = "2025-03-01"
period_end = "2025-03-31"
previous_period_start = "2024-03-01"
previous_period_end = "2024-03-31"

# -------------------------------------
# Filter data for each period
# -------------------------------------
df_current = df[(df["timestamp"] >= period_start) & (df["timestamp"] <= period_end)]
df_previous = df[
    (df["timestamp"] >= previous_period_start)
    & (df["timestamp"] <= previous_period_end)
]

# -------------------------------------
# Aggregate frequency and capture timestamps
# -------------------------------------
current_agg = (
    df_current.groupby("query")
    .agg(
        frequency_current=("frequency", "sum"),
        timestamp_current=("timestamp", lambda x: x.min()),
    )
    .reset_index()
)

previous_agg = (
    df_previous.groupby("query")
    .agg(
        frequency_previous=("frequency", "sum"),
        timestamp_previous=("timestamp", lambda x: x.min()),
    )
    .reset_index()
)

# -------------------------------------
# Merge and calculate growth rate
# -------------------------------------
merged = pd.merge(current_agg, previous_agg, on="query", how="outer").fillna(
    {"frequency_current": 0, "frequency_previous": 0}
)

merged["growthRate"] = (merged["frequency_current"] - merged["frequency_previous"]) / (
    merged["frequency_previous"] + 1
)
merged["growthRate"] = merged["growthRate"].round(2)

# -------------------------------------
# Category assignment
# -------------------------------------
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


merged["category"] = merged["query"].apply(categorize_query)

# -------------------------------------
# Export final file
# -------------------------------------
merged.to_csv("trend_growth_with_timestamps.csv", index=False)
print("✅ File exported: trend_growth_with_timestamps.csv")
