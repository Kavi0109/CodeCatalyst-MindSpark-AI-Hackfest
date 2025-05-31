import pandas as pd
import re
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Downloads (only needed once)
nltk.download("vader_lexicon")
nltk.download("punkt")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("customer_feedback.csv")
df.dropna(subset=["commentText"], inplace=True)
df["commentText"] = df["commentText"].astype(str)


# -------------------------------
# Text Normalization
# -------------------------------
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Collapse whitespace
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # Reduce letter spam (e.g., goooood)
    return text


df["cleaned_text"] = df["commentText"].apply(normalize_text)

# -------------------------------
# Sentiment Scoring
# -------------------------------
vader = SentimentIntensityAnalyzer()


def score_textblob(text):
    return TextBlob(text).sentiment.polarity  # Range [-1.0, 1.0]


def score_vader(text):
    return vader.polarity_scores(text)["compound"]  # Range [-1.0, 1.0]


df["sentiment_textblob"] = df["cleaned_text"].apply(score_textblob)
df["sentiment_vader"] = df["cleaned_text"].apply(score_vader)


# -------------------------------
# Rule-Based Sentiment Labeling with Rating Awareness
# -------------------------------
def classify_sentiment(row):
    rating = row["rating"]
    vader_score = row["sentiment_vader"]

    if pd.notnull(rating):
        if rating <= 2:
            return "negative"
        elif rating == 3:
            if vader_score >= 0.5:
                return "positive"
            elif vader_score <= -0.5:
                return "negative"
            else:
                return "neutral"
        elif rating >= 4:
            if vader_score <= -0.4:
                return "negative"
            elif vader_score >= 0.4:
                return "positive"
            else:
                return "neutral"
    else:
        if vader_score >= 0.4:
            return "positive"
        elif vader_score <= -0.4:
            return "negative"
        else:
            return "neutral"


df["final_sentiment"] = df.apply(classify_sentiment, axis=1)

# -------------------------------
# Optional: Export
# -------------------------------
df.to_csv("customer_feedback_sentiment_enriched.csv", index=False)
print(
    "✅ Sentiment analysis complete. Output saved to 'customer_feedback_sentiment_enriched.csv'"
)
