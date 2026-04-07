from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class ChatSession(db.Model):
    """Represents a named chat session (conversation thread)."""

    __tablename__ = "chat_sessions"

    id         = db.Column(db.Integer, primary_key=True)
    title      = db.Column(db.String(200), nullable=False, default="New Chat")
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    conversations = db.relationship(
        "Conversation",
        backref="session",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="Conversation.created_at",
    )

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "title":      self.title,
            "created_at": self.created_at.strftime("%b %d, %Y · %H:%M"),
        }


class Conversation(db.Model):
    """Stores individual message pairs within a chat session."""

    __tablename__ = "conversations"

    id           = db.Column(db.Integer, primary_key=True, nullable=False)
    session_id   = db.Column(db.Integer, db.ForeignKey("chat_sessions.id"), nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    ai_response  = db.Column(db.Text, nullable=False)
    created_at   = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "session_id":   self.session_id,
            "user_message": self.user_message,
            "ai_response":  self.ai_response,
            "created_at":   self.created_at.strftime("%H:%M"),
        }


# ── Session helpers ────────────────────────────────────────────────────────────

def create_session(title: str = "New Chat") -> ChatSession:
    session = ChatSession(title=title, created_at=datetime.utcnow())
    db.session.add(session)
    db.session.commit()
    return session


def get_all_sessions() -> list[dict]:
    sessions = ChatSession.query.order_by(ChatSession.created_at.desc()).all()
    return [s.to_dict() for s in sessions]


def get_session(session_id: int) -> ChatSession | None:
    return ChatSession.query.get(session_id)


def delete_session(session_id: int) -> bool:
    session = ChatSession.query.get(session_id)
    if not session:
        return False
    db.session.delete(session)
    db.session.commit()
    return True


def delete_all_sessions() -> int:
    count = ChatSession.query.count()
    ChatSession.query.delete()
    db.session.commit()
    return count


# ── Conversation helpers ───────────────────────────────────────────────────────

def save_conversation(session_id: int, user_message: str, ai_response: str) -> Conversation:
    """Save a message pair under the given session. Auto-titles session from first message."""
    session = ChatSession.query.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found.")

    # Title the session from the first user message (truncated to 40 chars)
    if session.title == "New Chat" and not session.conversations:
        session.title = (user_message[:40] + "…") if len(user_message) > 40 else user_message

    convo = Conversation(
        session_id=session_id,
        user_message=user_message,
        ai_response=ai_response,
        created_at=datetime.utcnow(),
    )
    db.session.add(convo)
    db.session.commit()
    return convo


def get_session_conversations(session_id: int) -> list[dict]:
    convos = (
        Conversation.query
        .filter_by(session_id=session_id)
        .order_by(Conversation.created_at.asc())
        .all()
    )
    return [c.to_dict() for c in convos]