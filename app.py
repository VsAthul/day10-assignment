from flask import Flask, render_template, request, jsonify
from database import db, save_conversation, get_all_conversations
from retriever import retrieve_relevant_chunks
from ai_response import generate_ai_response

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///conversations.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()


@app.route("/")
def index():
    conversations = get_all_conversations()
    return render_template("index.html", conversations=conversations)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    relevant_chunks = retrieve_relevant_chunks(user_message)
    ai_response = generate_ai_response(user_message, relevant_chunks)

    save_conversation(user_message, ai_response)

    return jsonify({"response": ai_response})


if __name__ == "__main__":
    app.run(debug=True)