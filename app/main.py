
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
import asyncio

from .routes.analyze import router as analyze_router, fetch_and_analyze

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Setup APScheduler for cron job scheduling
scheduler = BackgroundScheduler()
frontend_origin = os.getenv("FRONTEND_ORIGIN")

def cron_fetch_and_analyze():
    """
    Function to run fetch_and_analyze periodically via a cron job.
    """
    asyncio.run(fetch_and_analyze())

# Schedule the cron job (runs daily at midnight)
scheduler.add_job(cron_fetch_and_analyze, CronTrigger(hour=0, minute=0))

# Define lifespan events
async def lifespan(app: FastAPI):
    # Startup event
    logger.info("Starting up the FastAPI app and APScheduler...")
    scheduler.start()

    yield  # Application runs here

    # Shutdown event
    logger.info("Shutting down the FastAPI app and APScheduler...")
    scheduler.shutdown()

# Create the FastAPI app with a lifespan context
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],  # Specify the frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Include the router
app.include_router(analyze_router)

@app.get("/")
def read_root():
    return {"message": "Financial Sentiment Analysis API"}
