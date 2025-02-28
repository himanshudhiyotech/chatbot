
import requests
import json
from bs4 import BeautifulSoup

class scrapeService:
    def scrapeChatbotData(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return {"error": f"Failed to fetch data. Status code: {response.status_code}"}

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find all <h2> and <p> tags (modify this if needed)
        # questions = soup.find_all(["h1", "h2", "h4"])
        # answers = soup.find_all(["p", "ul", "a", "span", "div"])


        # # Extract text
        # scraped_data = []
        # for q, a in zip(questions, answers):
        #     scraped_data.append({
        #         "question": q.text.strip(),
        #         "answer": a.text.strip()
        #     })

        scraped_data = []
        faq_sections_h2 = soup.find_all("div", class_="container")  # Modify this if needed

        for section in faq_sections_h2:
            question_h2 = section.find("h2", class_ ="title")
            answer_div = section.find("div")

            if question_h2 and answer_div:
                scraped_data.append({
                    "question": question_h2.get_text(strip=True),
                    "answer": answer_div.get_text(strip=True)
                })

        if not scraped_data:
            return {"error": "No FAQ data found on the page"}
        
        with open("scraped_data.json", "w", encoding="utf-8") as json_file:
            json.dump(scraped_data, json_file, indent=4)

        return scraped_data
