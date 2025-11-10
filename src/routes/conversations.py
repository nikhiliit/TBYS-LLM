"""Conversation management routes."""

from flask import Blueprint, jsonify, request

from database import db_manager

conversations_bp = Blueprint('conversations', __name__)


@conversations_bp.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get list of conversations."""
    try:
        conversations = db_manager.get_conversations()
        return jsonify(conversations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@conversations_bp.route('/api/conversations', methods=['POST'])
def create_conversation():
    """Create a new conversation."""
    try:
        data = request.json or {}
        title = data.get('title', 'New Conversation')

        conversation_id = db_manager.create_conversation(title)
        return jsonify({"id": conversation_id, "title": title})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@conversations_bp.route('/api/conversations/<int:conversation_id>', methods=['GET'])
def get_conversation(conversation_id: int):
    """Get conversation with messages."""
    try:
        conversation = db_manager.get_conversation(conversation_id)
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404

        return jsonify(conversation)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@conversations_bp.route('/api/conversations/<int:conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id: int):
    """Delete a conversation."""
    try:
        deleted = db_manager.delete_conversation(conversation_id)
        if not deleted:
            return jsonify({"error": "Conversation not found"}), 404

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
