
from flask import Blueprint, request
from scrape.controller.scrapeController import scrapeController
from config import Config

scraper_bp = Blueprint("scraper", __name__)

@scraper_bp.route("/scrape", methods=["POST"])
def scrape():
    # data = request.get_json()
    url = Config.URL
    # print(f"Config URL {Config.URL}")
    # print(f"url {url}")

    return scrapeController.scrape(url)
