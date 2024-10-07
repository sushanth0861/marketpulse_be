import json

# Load JSON data from a file
def load_feed_data(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

# Sum the overall_sentiment_score and count sentiment labels
def sum_and_count_sentiments(file_path):
    data = load_feed_data(file_path)
    
    total_sentiment_score = 0
    sentiment_count = {
        "Bearish": 0,
        "Somewhat-Bearish": 0,
        "Neutral": 0,
        "Somewhat-Bullish": 0,
        "Bullish": 0
    }
    feed_count = 0

    for article in data["feed"][:10]:
        total_sentiment_score += article["overall_sentiment_score"]
        
        sentiment_label = article["overall_sentiment_label"]
        if sentiment_label in sentiment_count:
            sentiment_count[sentiment_label] += 1
        
        feed_count +=1
    
    # Determine the sentiment with the maximum count
    max_sentiment_label = max(sentiment_count, key=sentiment_count.get)
    
    return total_sentiment_score, sentiment_count, max_sentiment_label, feed_count

# Example usage
if __name__ == "__main__":
    file_path = '10324.json'  # Replace with your file path
    total_score, sentiment_count, max_sentiment_label, feed_count = sum_and_count_sentiments(file_path)
    print(total_score, sentiment_count, max_sentiment_label, feed_count)
    print(f"Average Sentiment Score: {total_score/feed_count}")
    print(f"Sentiment Counts: {sentiment_count}")
    print(f"Most Frequent Sentiment: {max_sentiment_label}")
