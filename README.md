# MarketPulse Backend

MarketPulse is a backend service built with FastAPI to provide market sentiment analysis and financial data for various sectors, leveraging MongoDB for data storage and Alpha Vantage for financial insights.

## Features
- Fetches and processes live market news and provides a sentiment (range from Bearish to Bullish).
- RESTful API endpoints for retrieving market sentiment data.
- Integration with Alpha Vantage for real-time financial news data.
- Utilizes a BART-based language model for generating summaries of financial news articles.
- Leverages large language models (LLMs) from Hugging Face’s Transformers library to perform AI-driven sentiment analysis.
- By utilizing these advanced models, MarketPulse effectively processes financial news, extracting insights and generating sentiment scores to gauge market mood dynamically.

## Live Deployment
- Access the live frontend application at: [marketpulse-fe.vercel.app](https://marketpulse-fe.vercel.app)
- Access the frontend repo at: [github.com/sushanth0861/marketpulse_fe](https://github.com/sushanth0861/marketpulse_fe)

## Prerequisites
- Python 3.8+
- MongoDB
- Alpha Vantage API Key

## Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sushanth0861/marketpulse_be.git
   cd marketpulse_be
   ```

2. **Set Up Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**  
   Create a `.env` file and add:
   ```env
   MONGODB_URI=mongodb://localhost:27017/marketpulse_db
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
   SECRET_KEY=your_secret_key
   ```

## Running the Application
1. **Start the FastAPI Server:**
   ```bash
   uvicorn app.main:app --reload
   ```
   Access the API documentation at `http://127.0.0.1:8000/docs`.

## Key Endpoints
- **POST /trigger_async_fetch_and_analyze/**: Starts asynchronous fetching and analysis of market data.
- **GET /fetch_summary/**: Retrieves sentiment summaries for the past 7 days.
- **GET /fetch_today_analysis/**: Fetches today’s sentiment analysis.
