from flask import Flask, render_template, request, jsonify, session
from database import db, create_session, get_all_sessions, get_session, delete_session, delete_all_sessions, save_conversation, get_session_conversations

from retriever import retrieve_relevant_chunks
from ai_response import generate_ai_response

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///conversations.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "bankdoc-secret-key"   # needed for flask session cookie

db.init_app(app)

with app.app_context():
    db.create_all()


# ── Pages ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    sessions = get_all_sessions()

    # Pick the active session from cookie, or the most recent one, or create one
    active_id = flask_session_get()
    if active_id:
        active = get_session(active_id)
        if not active:          # was deleted
            active_id = None

    if not active_id:
        if sessions:
            active_id = sessions[0]["id"]   # most recent
        else:
            new = create_session()
            active_id = new.id
            sessions = get_all_sessions()

    set_flask_session(active_id)

    conversations = get_session_conversations(active_id)
    return render_template(
        "index.html",
        sessions=sessions,
        active_session_id=active_id,
        conversations=conversations,
    )


# ── Chat ───────────────────────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    data       = request.get_json()
    user_msg   = (data.get("message") or "").strip()
    session_id = data.get("session_id") or flask_session_get()

    if not user_msg:
        return jsonify({"error": "Message cannot be empty."}), 400
    if not session_id:
        return jsonify({"error": "No active session."}), 400

    relevant_chunks = retrieve_relevant_chunks(user_msg)
    ai_response     = generate_ai_response(user_msg, relevant_chunks)
    convo           = save_conversation(int(session_id), user_msg, ai_response)

    # Return updated session title (changes after first message)
    sess = get_session(int(session_id))
    return jsonify({
        "response":      ai_response,
        "session_title": sess.title if sess else "",
        "created_at":    convo.created_at.strftime("%H:%M"),
    })


# ── Session API ────────────────────────────────────────────────────────────────

@app.route("/sessions", methods=["GET"])
def list_sessions():
    return jsonify(get_all_sessions())


@app.route("/sessions/new", methods=["POST"])
def new_session():
    sess = create_session()
    set_flask_session(sess.id)
    return jsonify(sess.to_dict())


@app.route("/sessions/<int:session_id>", methods=["GET"])
def load_session(session_id):
    sess = get_session(session_id)
    if not sess:
        return jsonify({"error": "Session not found."}), 404
    set_flask_session(session_id)
    convos = get_session_conversations(session_id)
    return jsonify({"session": sess.to_dict(), "conversations": convos})


@app.route("/sessions/<int:session_id>", methods=["DELETE"])
def remove_session(session_id):
    ok = delete_session(session_id)
    if not ok:
        return jsonify({"error": "Session not found."}), 404

    # If we just deleted the active session, clear cookie
    if flask_session_get() == session_id:
        clear_flask_session()

    return jsonify({"deleted": session_id})


@app.route("/sessions/all", methods=["DELETE"])
def remove_all_sessions():
    count = delete_all_sessions()
    clear_flask_session()
    return jsonify({"deleted_count": count})


# ── Flask session helpers (stores active session id in cookie) ─────────────────

def flask_session_get():
    return session.get("active_session_id")

def set_flask_session(session_id):
    session["active_session_id"] = session_id

def clear_flask_session():
    session.pop("active_session_id", None)


if __name__ == "__main__":
    app.run(debug=True)