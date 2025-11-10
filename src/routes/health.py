"""Health check routes."""

from flask import Blueprint, jsonify

from ..models import Qwen3Manager, Qwen3VLManager

health_bp = Blueprint('health', __name__)

# Model instances will be set by the main app
_models = {}


def init_health_models(text_mgr, vl_mgr):
    """Initialize model instances for the health blueprint."""
    _models['text_manager'] = text_mgr
    _models['vl_manager'] = vl_mgr
    _models['current_model_type'] = 'qwen3-text'


@health_bp.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "text_model_loaded": _models.get('text_manager') is not None and _models['text_manager'].model is not None,
        "vl_model_loaded": _models.get('vl_manager') is not None and _models['vl_manager'].model is not None,
        "current_model": _models.get('current_model_type', 'qwen3-text')
    })
