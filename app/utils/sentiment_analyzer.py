# utils/sentiment_analyzer.py
from transformers import pipeline
import re

def get_sentiment_label_by_score(sentiment_score):
    """
    Map sentiment score to custom sentiment labels based on the provided ranges.
    """
    if sentiment_score <= -0.35:
        return "Bearish"
    elif -0.35 < sentiment_score <= -0.15:
        return "Somewhat-Bearish"
    elif -0.15 < sentiment_score < 0.15:
        return "Neutral"
    elif 0.15 <= sentiment_score < 0.35:
        return "Somewhat-Bullish"
    else:
        return "Bullish"

def aggregate_sentiments(sentiments):
    """
    Aggregate sentiments to calculate overall sentiment score, mood, and sentiment counts
    """
    score_sum = 0
    mood_count = {
        "Bearish": 0,
        "Somewhat-Bearish": 0,
        "Neutral": 0,
        "Somewhat-Bullish": 0,
        "Bullish": 0,
    }

    # Sum sentiment scores and count each sentiment label
    for url, sentiment_score, sentiment_label in sentiments:
        score_sum += sentiment_score
        mood_count[sentiment_label] += 1

    # Calculate overall sentiment score
    overall_score = score_sum / len(sentiments) if sentiments else 0

    # Find the sentiment label with the most occurrences
    final_mood = get_sentiment_label_by_score(overall_score)

    return overall_score, final_mood, mood_count

def analyze_sentiment(summary_text, sentiment_pipeline = pipeline("sentiment-analysis")):
    """
    Use a sentiment analysis pipeline to analyze the sentiment of the summary text.
    Adjust sentiment score for negative sentiment.
    """
    sentiment_result = sentiment_pipeline(summary_text)
    
    # Extract sentiment score and label (POSITIVE/NEGATIVE/NEUTRAL)
    sentiment_score = sentiment_result[0]["score"]
    sentiment_label_hf = sentiment_result[0]["label"]

    # Adjust the sentiment score for negative sentiment (making it negative)
    if sentiment_label_hf == "NEGATIVE":
        sentiment_score = -sentiment_score

    return sentiment_score, get_sentiment_label_by_score(sentiment_score)
