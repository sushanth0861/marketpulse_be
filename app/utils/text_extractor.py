# utils/text_extractor.py

import requests
from bs4 import BeautifulSoup
from fastapi import HTTPException

def extract_text_from_url(url):
    """
    Extract text content from a given URL by scraping paragraphs.
    Raises an HTTPException if there is an issue with fetching or parsing the URL.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        article_text = "\n".join([para.get_text() for para in paragraphs])
        return article_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching URL content: {str(e)}")
