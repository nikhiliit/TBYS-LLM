"""Health check routes."""

from flask import Blueprint, jsonify

from models import Qwen3Manager, Qwen3VLManager

health_bp = Blueprint('health', __name__)

# Global model instances (initialized in main app)
text_manager: Qwen3Manager = None
vl_manager: Qwen3VLManager = None
current_model_type = 'qwen3-text'


@health_bp.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "text_model_loaded": text_manager is not None and text_manager.model is not None,
        "vl_model_loaded": vl_manager is not None and vl_manager.model is not None,
        "current_model": current_model_type
    })
