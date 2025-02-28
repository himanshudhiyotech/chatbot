from flask import Flask, request, jsonify, session, render_template
import requests
from flask_session import Session
from config import Config  # Import the Config class

# importing my routes or blueprints
from scrape.routes.scrapeRoutes import scraper_bp
# from data.routes.dataRoutes import data_bp
from chat.routes.chatRoutes import chat_bp

# importing database 
from data.database.database import initializeDb

app = Flask(__name__)

# Load config settings from config.py
app.config.from_object(Config)

# Initialize database connection
initializeDb(app)

# Initialize session
Session(app)

# Register the blueprint
app.register_blueprint(scraper_bp, url_prefix='/scrape')
# app.register_blueprint(data_bp, url_prefix='/store')
app.register_blueprint(chat_bp, url_prefix='/chat')


# @app.route("/", methods=["GET", "POST"])
# def home():
#     return "Server is running!"

# @app.route("/scrape", methods=["GET", "POST"])
# def scrape():
#     response = requests.get("http://127.0.0.1:5000/scrape")
#     return response.json()

# @app.route("/store", methods=["GET","POST"])
# def store():
#     data = {"name": "Alice", "email": "alice@example.com", "phoneNo": "9876543210", "query": "Sample query"}
#     response = requests.post("http://127.0.0.1:5000/store", json=data)
#     return response.json()

# @app.route("/chat", methods=["POST"])
# def chat():
#     response = requests.post("http://127.0.0.1:5000/chat", json={"message": "Hello"})
#     return response.json()


if __name__ == '__main__':
    app.run(port=5000, debug=True)














# @app.route("/chat", methods=["POST"])
# def chatbot():
#     input_text = request.form["message"].strip()
    
#     # Forward user message to Chatbot Service
#     chatbot_response = requests.post(f"{CHATBOT_SERVICE_URL}/chat", json={"message": input_text})
#     response_data = chatbot_response.json()

#     session["chat_history"].append({"sender": "user", "text": input_text})
#     session["chat_history"].append({"sender": "bot", "text": response_data["response"]})
#     session.modified = True

#     return render_template("chat.html", chat_history=session["chat_history"])

# @app.route("/scrape", methods=["GET"])
# def scrape():
#     # Forward request to Scraper Service
#     scrape_response = requests.get(f"{SCRAPER_SERVICE_URL}/scrape")
#     return jsonify(scrape_response.json())


# @app.route("/store", methods=["POST"])
# def store_user():
#     data = request.json.get("data")
    
#     new_user = User(
#         name=data["name"],
#         email=data["email"],
#         phoneNo=data["phoneNo"],
#         query_description=data["query"]
#     )
#     db.session.add(new_user)
#     db.session.commit()

#     return jsonify({"message": "User stored successfully!"})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)

