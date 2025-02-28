import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv("SECRET_KEY")
    SESSION_TYPE = os.getenv("SESSION_TYPE")
    URL = os.getenv("url")
    LLAMA_MODEL_NAME = "llama3:8b"
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    # DATABASE_SERVICE_URL = os.getenv("DATABASE_SERVICE_URL")
    # CHATBOT_SERVICE_URL = "http://127.0.0.1:5000"
    # SCRAPER_SERVICE_URL = "http://127.0.0.1:5000"
    # DATABASE_SERVICE_URL = "http://127.0.0.1:5000"
    