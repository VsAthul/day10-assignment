from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Conversation(db.Model):
    """SQLAlchemy model for storing chat conversations."""

    __tablename__ = "conversations"

    id = db.Column(db.Integer, primary_key=True, nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    ai_response = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_message": self.user_message,
            "ai_response": self.ai_response,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }


def save_conversation(user_message: str, ai_response: str) -> Conversation:
    """Save a user message and AI response to the database."""
    conversation = Conversation(
        user_message=user_message,
        ai_response=ai_response,
        created_at=datetime.utcnow(),
    )
    db.session.add(conversation)
    db.session.commit()
    return conversation


def get_all_conversations() -> list[dict]:
    """Retrieve all conversations ordered by creation time."""
    conversations = Conversation.query.order_by(Conversation.created_at.asc()).all()
    return [c.to_dict() for c in conversations]
