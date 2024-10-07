from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import json
import os

from ..utils.summarizer import summarize_article
from ..utils.sentiment_analyzer import aggregate_sentiments, analyze_sentiment
from ..utils.text_extractor import extract_text_from_url
from ..models.bart_model import bart_tokenizer, bart_model

# router = APIRouter()

class NewsArticleFeed(BaseModel):
    feed: list

# File path for storing the result
RESULT_FILE_PATH = "analyzed_feed_results.json"

# Function to write results to a local JSON file
def write_to_json(result_data):
    with open(RESULT_FILE_PATH, "w") as file:
        json.dump(result_data, file, indent=4)

# Function to load results from the local JSON file
def load_from_json():
    if os.path.exists(RESULT_FILE_PATH):
        with open(RESULT_FILE_PATH, "r") as file:
            return json.load(file)
    else:
        return {"message": "No results found."}


@router.post("/analyze_feed/")
async def analyze_feed(feed: NewsArticleFeed, model_name: str = "bart"):
    sentiments = []
    
    # Choose model based on the parameter
    if model_name == "bart":
        tokenizer = bart_tokenizer
        model = bart_model
    else:
        raise HTTPException(status_code=400, detail="Invalid model name")

    # Process each article
    for article in feed.feed[:10]:
        article_text = extract_text_from_url(article["url"])
        summary = summarize_article(article_text, tokenizer, model)
        sentiment_score, sentiment_label = analyze_sentiment(summary)
        sentiments.append((article["url"], sentiment_score, sentiment_label))

    # Aggregate sentiment scores, labels, and counts
    overall_score, final_mood, sentiment_counts = aggregate_sentiments(sentiments)

    # Add a timestamp for when the feed was analyzed
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Result data with timestamp
    result_data = {
        "timestamp": timestamp,
        "overall_sentiment_score": overall_score,
        "overall_sentiment_label": final_mood,
        "sentiment_counts": sentiment_counts,
        "details": sentiments
    }

    # Save the result to JSON locally
    write_to_json(result_data)

    return result_data

# Fetch results from the JSON file
@router.get("/fetch_results/")
async def fetch_results():
    result_data = load_from_json()
    return result_data
