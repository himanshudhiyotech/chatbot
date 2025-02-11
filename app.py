from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import json
import re

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Needed for session management

# Configure MySQL database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:Sarita&2007@localhost:3306/chatbotdata'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Define a User model
class User(db.Model):
    __tablename__ = 'userDetail'
    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phoneNo = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    query_description = db.Column(db.String(100))

# Create the database tables
with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        print(f"Error creating database tables: {str(e)}")
  
def load_dataset():
    try:
        with open("D:/Coding/DhiyoTech/Project/dataset.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print("Dataset file not found.")
        return {}
    except json.JSONDecodeError:
        print("Error decoding the dataset JSON.")
        return {}
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {}

dataset = load_dataset()

# print(type(dataset)) 

def respondcategories():
    category_list = []
    for category in dataset.keys():
        category_list.append(f"{category}<br>")
    return "\n".join(category_list)

def respondsubcategories(query_category_c):
    query_category = query_category_c.replace(" ", "").lower()
    for category in dataset.keys():
        category_c = category.lower().replace(" ", "")
        if query_category == category_c:
            sub_category = ["Choose from the following SubCategories: "]
            for subcategory in dataset[category].keys():
                sub_category.append(f"{subcategory}<br>")
            sub_category.append("Back <br>")
            sub_category.append("Stop <br>")
            return "\n".join(sub_category)
    return "Sorry, I am unable to understand your choosen category. <br> Please rephrase your question, or type BACK to select/change query categories."

def respondquestions(query_subcategory):
    query_subcategory = query_subcategory.replace(" ", "").lower()
    for category in dataset.keys(): 
        for subcategory in dataset[category]:  
            if query_subcategory == subcategory.replace(" ", "").lower():
                questions = ["Choose from the following questions: "]
                for question_entry in dataset[category][subcategory]:  
                    questions.append(f"{question_entry['question']}<br>")
                questions.append("Back <br>")
                questions.append("Thank You <br>")
                questions.append("Stop <br>")
                return "\n".join(questions)
    return "Sorry, I am unable to understand your choosen subcategory. <br> Please rephrase your question, or type BACK to select/change query categories."

def respond(query_category, query_subcategory, input_text):
    query_category = query_category.replace(" ", "").lower()
    query_subcategory = query_subcategory.replace(" ", "").lower()
    for category in dataset.keys():
        for subcategory in dataset[category].keys():
            for entry in dataset[category][subcategory]: 
                if input_text.lower() in entry["question"].lower():
                    return entry["answer"]
    return "Sorry, I don't have an answer for that. <br> Please rephrase your question, or type BACK to select/change query categories."

phone_regex = r'^(?:\+91|)[1-9][0-9]{9}$'
def validate_phone_number(phone):
    if re.match(phone_regex, phone):
        return True
    return False

email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
def validate_email(email):
    if re.match(email_regex, email):
        return True
    return False

# Chatbot route
@app.route("/", methods=["GET"])
def index():
    session["chat_history"] = []
    session["chat_history"].append({"sender": "bot", "text": "Type HELLO"})
    session["step"] = "greet" 
    session["query_category"] = ""
    session["query_subcategory"] = ""
    return render_template("chat.html", chat_history=session["chat_history"])

@app.route("/chat", methods=["POST"])
def chatbot():
    input_text = request.form["message"].strip()
    query_category = ""
    query_subcategory = ""
    if not input_text:
        return redirect(url_for("index"))

    # Initialize chat history in session if not already present
    if "chat_history" not in session:
        session["chat_history"] = []
        session["step"] = "greet"
    response = ""

    # Handle conversation steps
    if session["step"] == "greet":
        response = "Hello! What is your name?"
        session["step"] = "ask_name"

    elif session["step"] == "ask_name":
        session["name"] = input_text
        response = f"Nice to meet you, {input_text}! Please enter your email."
        session["step"] = "ask_email"

    elif session["step"] == "ask_email":
        if validate_email(input_text):
            session["email"] = input_text
            response = "Thanks! Now, enter your phone number."
            session["step"] = "ask_phone"
        else:
            response = "Please enter a valid email address."

    elif session["step"] == "ask_phone":
        if validate_phone_number(input_text):
            session["phoneNo"] = input_text
            response = respondcategories()
            session["step"] = "ask_query_category"
        else:
            response = "Please enter a valid phone number."

    elif session["step"] == "ask_query_category":
        session["query_category"] = input_text
        input_text = input_text.replace(" ", "")
        # Store user details in the database
        new_user = User(
            name=session["name"],
            email=session["email"],
            phoneNo=session["phoneNo"],
            query_description=session["query_category"]
        )
        db.session.add(new_user)
        db.session.commit()
        response = respondsubcategories(input_text)
        session["step"] = "ask_query_subcategory"

    elif session["step"] == "ask_query_category_again":
        session["query_category"] = input_text
        input_text = input_text.replace(" ", "")
        response = respondsubcategories(input_text)
        session["step"] = "ask_query_subcategory"

    elif session["step"] == "ask_query_subcategory":
        session["query_subcategory"] = input_text
        input_text = input_text.replace(" ", "")
        if input_text.lower() == "back":
            session["step"] = "ask_query_category_again"
            session["query_subcategory"] = ""
            session["query_category"] = ""
            response = respondcategories()
        elif input_text.lower() == 'stop':
            session['step'] = "completed"
            response = "Are you satisfied with the response? Type Yes or No"
        else:
            response = respondsubcategories(input_text)
            # print(f"Response from respondquestions: {response}")
            session["step"] = "ask_query"

    elif session["step"] == "ask_query":
        session["query"] = input_text
        query_category = session["query_category"]
        query_subcategory = session["query_subcategory"]
        if input_text.lower() == "back":
            session["step"] = "ask_query_subcategory"
            response = respondsubcategories(input_text)
        elif input_text.lower() == 'thankyou':
            response = "Thank you! If you have more questions, type MORE and feel free to ask or type STOP to end conversation."
            session["step"] = "more"
        elif input_text.lower() == 'stop':
            session['step'] = "completed"
            response = "Are you satisfied with the response? Type Yes or No"
        else:
            response = respond(query_category, query_subcategory, input_text)
            session["step"] = "ask_query"

    elif session["step"] == "more":
        session["again"] = input_text
        if input_text.lower() == "more":
            session["step"] = "ask_query_category_again"
            response = respondcategories()
        elif input_text.lower() == "stop":
            response = "Are you satisfied with the response? Type Yes or No"
        else: 
            response = "Thank you! If you have more questions, type MORE and feel free to ask or type STOP to end conversation."
    
    elif session["step"] == "completed":
        session["again"] = input_text
        if input_text.lower().replace(" ", "") != "stop":
            session["step"] = "ask_query_category_again"
            response = respondcategories()
        else: 
            response = "Thank you! If you have more questions, type MORE and feel free to ask."

    else:
        response = "Error!"

    # Store messages in chat history
    session["chat_history"].append({"sender": "user", "text": input_text})
    session["chat_history"].append({"sender": "bot", "text": response})
    session.modified = True
    # Pass chat history to template to display it
    return render_template("chat.html", chat_history=session["chat_history"])

if __name__ == "__main__":
    app.run(debug=True)


