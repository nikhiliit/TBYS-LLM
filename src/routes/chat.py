"""Chat functionality routes."""

import json
from typing import List

from flask import Blueprint, request, Response, stream_with_context, jsonify

from ..database import db_manager
from ..models import Qwen3Manager, Qwen3VLManager, StreamingGenerator

chat_bp = Blueprint('chat', __name__)

# Model instances will be set by the main app
_models = {}


def init_chat_models(text_mgr, vl_mgr):
    """Initialize model instances for the chat blueprint."""
    _models['text_manager'] = text_mgr
    _models['vl_manager'] = vl_mgr
    _models['current_model_type'] = 'qwen3-text'


def handle_vl_chat(
    prompt: str,
    conversation_id: int,
    pdf_images: List,
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float
):
    """Handle vision-language chat with PDF images."""
    if not vl_manager or not vl_manager.model:
        yield json.dumps({"error": "VL model not loaded"}) + "\n"
        return

    try:
        # Process with VL model
        result = vl_manager.generate_response(
            prompt=prompt,
            images=pdf_images,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking
        )

        if not result or "full_response" not in result:
            yield json.dumps({"error": "Failed to generate response"}) + "\n"
            return

        response_text = result["full_response"]

        # Save to database
        db_manager.save_message(conversation_id, "assistant", response_text)

        # Stream response
        yield f"data: {json.dumps({'content': response_text})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@chat_bp.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint with streaming support."""

    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    conversation_id = data.get('conversation_id', 0)
    enable_thinking = data.get('enable_thinking', True)
    max_new_tokens = data.get('max_new_tokens', 2048)
    temperature = data.get('temperature', 0.6)
    pdf_images = data.get('pdf_images', [])

    # Create conversation if needed
    if conversation_id <= 0:
        conversation_id = db_manager.create_conversation(
            prompt[:50] + "..." if len(prompt) > 50 else prompt
        )

    # Save user message
    db_manager.save_message(conversation_id, "user", prompt)

    # Handle VL model for PDF images
    if pdf_images and _models.get('vl_manager') and _models['vl_manager'].model:
        _models['current_model_type'] = 'qwen3-vl'

        def generate():
            yield from handle_vl_chat(
                prompt, conversation_id, pdf_images,
                enable_thinking, max_new_tokens, temperature
            )

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    # Handle text model
    elif _models.get('text_manager') and _models['text_manager'].model:
        _models['current_model_type'] = 'qwen3-text'

        def generate():
            try:
                generator = StreamingGenerator(
                    _models['text_manager'], prompt, conversation_id,
                    enable_thinking, max_new_tokens, temperature, pdf_images
                )

                for chunk in generator.generate_stream():
                    yield chunk

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    else:
        return jsonify({"error": "No model loaded"}), 503
