import requests
from flask import jsonify
from scrape.service.scrapeService import scrapeService
from config import Config

scrapeService = scrapeService()

class scrapeController:
    def scrape(url):
        if not url:
            return jsonify({"error": "URL is required"}), 400

        scrapedData = scrapeService.scrapeChatbotData(url)
        if "error" in scrapedData:
            return jsonify({"error": "Scraper failed", "details": scrapedData}), 500

        print(f"scraped Data: {scrapedData}")

        # databaseServiceUrl = Config.DATABASE_SERVICE_URL 
        # dbResponse = requests.post(databaseServiceUrl, json={"data": scrapedData})
        # print(f"db Response: {dbResponse.text}") 

        # Check if the response is JSON before calling .json()
        # try:
        #     dbResponseJson = dbResponse.json()
        # except requests.exceptions.JSONDecodeError:
        #     return jsonify({
        #         "error": "Invalid response from database service",
        #         "status_code": dbResponse.status_code,
        #         "response": dbResponse.text
        #     }), 500

        return jsonify({"scrapedData": scrapedData})
