import os
import json
import logging
from datetime import datetime, timedelta, timezone
import requests
from fastapi import APIRouter, BackgroundTasks
from dotenv import load_dotenv
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from pymongo import MongoClient

from ..utils.text_extractor import extract_text_from_url
from ..utils.sentiment_analyzer import aggregate_sentiments, analyze_sentiment, get_sentiment_label_by_score

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["market_pulse"]
news_collection = db["news_data"]
analyzed_news_collection = db["analyzed_news_data"]
summary_collection = db["summary_data"]

# Get Alpha Vantage API key from environment variable
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

if not ALPHA_VANTAGE_API_KEY or not MONGO_URI:
    raise EnvironmentError("Missing ALPHA_VANTAGE_API_KEY or MONGO_URI in environment variables")

# Check if GPU is available, else fallback to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the BART tokenizer and model, with GPU/CPU handling
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)

router = APIRouter()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AV_JSON_DIR_PATH = os.path.join(BASE_DIR, '..', 'av_news_files')
NEWS_JSON_DIR_PATH = os.path.join(BASE_DIR, '..', 'news_analysis_files')
ANALYZED_JSON_DIR_PATH = os.path.join(BASE_DIR, '..', 'analyzed_files')
SUMMARY_JSON_FILE = os.path.join(BASE_DIR, '..', 'analyzed_feed_results.json')

# Ensure directories exist
for path in [NEWS_JSON_DIR_PATH, ANALYZED_JSON_DIR_PATH, AV_JSON_DIR_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

def fetch_news_for_day(date: datetime, day_idx):
    time_from = date.strftime("%Y%m%dT0000")
    time_to = date.strftime("%Y%m%dT2359")

    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&time_from={time_from}&time_to={time_to}&limit=1000&apikey={ALPHA_VANTAGE_API_KEY}"

    logger.info(f"Fetching news for {date.strftime('%Y-%m-%d')} from Alpha Vantage")

    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json().get("feed", [])
        av_json_filename = f"av_{day_idx}.json"
        av_json_file_path = os.path.join(AV_JSON_DIR_PATH, av_json_filename)

        logger.info(f"Saving AV news articles to {av_json_file_path}")

        with open(av_json_file_path, "w") as file:
            json.dump(news_data, file, indent=4)

        # Store in MongoDB
        news_collection.insert_many(news_data)
        return news_data
    else:
        logger.error(f"Failed to fetch news for {date.strftime('%Y-%m-%d')}, status code: {response.status_code}")
        return []

def analyze_articles(articles, day_idx):
    """
    Analyze the fetched articles using the BART model and save the output.
    """
    results = []
    for article in articles[:100]:
        article_text = extract_text_from_url(article["url"])
        if not article_text:
            continue

        # Use BART model to generate summary
        inputs = bart_tokenizer([article_text], return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs['input_ids'].to(device)
        summary_ids = bart_model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        sentiment_score, sentiment_label = analyze_sentiment(summary)

        article_data = {
            "title": article.get("title", "No title"),
            "summary": summary,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "url": article.get("url", "#"),
            "date": article.get("time_published", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        }

        results.append(article_data)

    # Save the analyzed results to a JSON file, rotating between 7 files
    analyzed_json_filename = f"analyzed_{day_idx}.json"
    analyzed_json_file_path = os.path.join(ANALYZED_JSON_DIR_PATH, analyzed_json_filename)

    logger.info(f"Saving analyzed articles to {analyzed_json_file_path}")

    with open(analyzed_json_file_path, "w") as file:
        json.dump(results, file, indent=4)

    # Store analyzed results in MongoDB
    analyzed_news_collection.insert_many(results)

    return results

def save_summary(articles_summary):
    """
    Save the summary data to a JSON file with timestamp, overall sentiment score, label, and sentiment counts.
    """
    logger.info(f"Saving sentiment summary to {SUMMARY_JSON_FILE}")
    with open(SUMMARY_JSON_FILE, "w") as file:
        json.dump(articles_summary, file, indent=4)
    # Store summary in MongoDB
    summary_collection.insert_many(articles_summary)

# ------------------- FastAPI Endpoints ----------------------

@router.post("/trigger_async_fetch_and_analyze/")
async def trigger_fetch_and_analyze(background_tasks: BackgroundTasks):
    """
    Trigger the fetch_and_analyze function asynchronously when called.
    """
    background_tasks.add_task(fetch_and_analyze)
    return {"message": "Fetch and analyze started in the background."}

async def fetch_and_analyze():
    """
    Fetch news for the last 7 days from AlphaVantage, analyze each day, and generate summary.
    """
    today = datetime.now(timezone.utc)
    summary_data = []

    # Process news for the last 7 days
    for i in range(7):
        day = today - timedelta(days=i)
        day_idx = i % 7  # Circular overwrite using mod 7

        # Fetch the news for a single day
        articles = fetch_news_for_day(day, day_idx)

        if not articles:
            continue

        # Analyze the articles and save the results
        analyzed_results = analyze_articles(articles, day_idx)

        # Create overall sentiment summary for the day
        sentiment_counts = {
            "Bearish": 0,
            "Somewhat-Bearish": 0,
            "Neutral": 0,
            "Somewhat-Bullish": 0,
            "Bullish": 0,
        }

        total_sentiment_score = 0
        num_articles = len(analyzed_results)

        for analyzed_result in analyzed_results:
            sentiment_label = analyzed_result["sentiment_label"]
            sentiment_score = analyzed_result["sentiment_score"]
            total_sentiment_score += sentiment_score
            sentiment_counts[sentiment_label] += 1

        # Calculate overall sentiment score and label
        overall_sentiment_score = total_sentiment_score / num_articles if num_articles > 0 else 0
        overall_sentiment_label = get_sentiment_label_by_score(overall_sentiment_score)

        # Store the day's summary
        summary_data.append({
            "timestamp": day.strftime("%Y-%m-%dT%H:%M:%S"),
            "overall_sentiment_score": overall_sentiment_score,
            "overall_sentiment_label": overall_sentiment_label,
            "sentiment_counts": sentiment_counts
        })

    # Save the summary to a JSON file
    save_summary(summary_data)

# ------------------- Fetch Summary and Today's Analysis ----------------------

@router.get("/fetch_summary/")
def fetch_summary():
    summaries = list(summary_collection.find().sort("timestamp", -1).limit(7))
    if summaries:
        return {"results": summaries}
    else:
        return {"error": "No summary data found."}

@router.get("/fetch_today_analysis/")
def fetch_today_analysis():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_data = list(analyzed_news_collection.find({"date": today}))
    if today_data:
        return {"results": today_data}
    else:
        return {"error": "No analysis available for today."}
