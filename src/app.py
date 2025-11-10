"""Main Flask application for Qwen Agent."""

import argparse
import os
from pathlib import Path

from flask import Flask, render_template

from .config import Config
from .database import db_manager
from .models import Qwen3Manager, Qwen3VLManager
from .routes import chat_bp, conversations_bp, health_bp
from .routes.chat import init_chat_models
from .routes.health import init_health_models
from .utils import cleanup_old_conversations


def create_app(config_class=Config) -> Flask:
    """Create and configure the Flask application.

    Args:
        config_class: Configuration class to use

    Returns:
        Configured Flask application
    """
    app = Flask(
        __name__,
        template_folder=config_class.get_template_folder(),
        static_folder=config_class.get_static_folder()
    )

    # Configure app
    app.config.from_object(config_class)

    # Register blueprints
    app.register_blueprint(chat_bp)
    app.register_blueprint(conversations_bp)
    app.register_blueprint(health_bp)

    # Global model instances
    global text_manager, vl_manager, current_model_type
    text_manager = None
    vl_manager = None
    current_model_type = 'qwen3-text'

    @app.route('/')
    def index():
        """Serve the main web interface."""
        return render_template('index.html')

    return app


def load_models(app: Flask, args: argparse.Namespace) -> bool:
    """Load and initialize the AI models.

    Args:
        app: Flask application instance
        args: Command line arguments

    Returns:
        True if models loaded successfully, False otherwise
    """
    global text_manager, vl_manager, current_model_type

    success = True

    # Load text model
    if not args.skip_text:
        print("🔄 Loading text model...")
        text_manager = Qwen3Manager(str(Config.TEXT_MODEL_PATH))

        if text_manager.load_local_model():
            print("✅ Text model loaded successfully!")
            current_model_type = 'qwen3-text'
        else:
            print("❌ Failed to load text model!")
            success = False

    # Load VL model
    if not args.skip_vl and Path(Config.VL_MODEL_PATH).exists():
        print("🔄 Loading VL model...")
        vl_manager = Qwen3VLManager(Config.VL_MODEL_PATH)

        if vl_manager.load_local_model():
            print("✅ VL model loaded successfully!")
        else:
            print("❌ Failed to load VL model!")
            success = False

    # Initialize model instances in blueprints
    init_chat_models(text_manager, vl_manager)
    init_health_models(text_manager, vl_manager)

    return success


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='Qwen Agent Web UI')
    parser.add_argument('--host', default=Config.HOST,
                       help=f'Host to bind to (default: {Config.HOST})')
    parser.add_argument('--port', type=int, default=Config.PORT,
                       help=f'Port to bind to (default: {Config.PORT})')
    parser.add_argument('--skip-text', action='store_true',
                       help='Skip loading text model')
    parser.add_argument('--skip-vl', action='store_true',
                       help='Skip loading VL model')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage instead of MPS/CUDA')

    args = parser.parse_args()

    # Force CPU if requested
    if args.cpu:
        os.environ['QWEN_FORCE_CPU'] = '1'

    # Create Flask app
    app = create_app()

    # Load models
    if not load_models(app, args):
        print("❌ Failed to load required models!")
        return

    # Cleanup old conversations
    deleted_count = cleanup_old_conversations()
    if deleted_count > 0:
        print(f"🧹 Cleaned up {deleted_count} old conversations")

    print(f"💡 VL model will load automatically when selected in UI")
    print(f"🚀 Starting web UI on http://{args.host}:{args.port}")

    # Start server
    app.run(host=args.host, port=args.port, debug=Config.DEBUG, threaded=True)


if __name__ == '__main__':
    main()
