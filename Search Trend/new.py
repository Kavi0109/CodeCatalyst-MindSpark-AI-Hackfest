import pandas as pd

# Load files
raw = pd.read_csv("search_trends.csv")
summary = pd.read_csv("all_trending_keywords_categorized.csv")


# Clean raw queries
def clean_query(q):
    import re

    q = str(q).lower()
    q = re.sub(r"[^\w\s]", "", q)
    return re.sub(r"\s+", " ", q).strip()


raw["query"] = raw["query"].apply(clean_query)
summary["searchTerm"] = summary["searchTerm"].apply(clean_query)

# Merge for daily entries
daily = raw.merge(
    summary[["searchTerm", "growthRate", "category"]],
    left_on="query",
    right_on="searchTerm",
    how="left",
)

daily = daily.drop(columns=["searchTerm"])
daily["periodType"] = "daily"

# Prepare summary entries (no timestamp)
summary_rows = summary.copy()
summary_rows["timestamp"] = pd.NaT
summary_rows["frequency"] = pd.NA
summary_rows["periodType"] = "summary"
summary_rows = summary_rows.rename(columns={"searchTerm": "query"})

# Combine
combined = pd.concat([daily, summary_rows], ignore_index=True)
combined.to_csv("combined_trend_data.csv", index=False)

print("✅ Exported: combined_trend_data.csv")
