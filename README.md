# TBYS-LLM

Think it before you speak - A web interface for Qwen language models.

## Features

- Real-time streaming responses
- Thinking mode with collapsible reasoning sections
- PDF upload and analysis
- Conversation management with SQLite database
- Modern responsive UI
- Support for text and vision-language models

## Project Structure

```
├── src/
│   ├── app.py                    # Main Flask application
│   ├── config.py                 # Configuration settings
│   ├── database.py               # Database operations
│   ├── models/                   # Model management
│   │   ├── qwen_manager.py       # Text model manager
│   │   ├── vl_manager.py         # Vision-language model manager
│   │   └── streaming.py          # Streaming generator
│   ├── routes/                   # Flask routes
│   │   ├── chat.py               # Chat endpoints
│   │   ├── conversations.py      # Conversation management
│   │   └── health.py             # Health check
│   └── utils/                    # Utilities
│       └── cleanup.py            # Cleanup utilities
├── templates/
│   └── index.html                # Web interface
├── static/                       # Static assets
├── models/                       # Text model files
├── models_vl/                    # Vision-language model files
├── tests/                        # Test directory
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── run.py                        # Entry point
└── README.md                     # This file
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nikhiliit/TBYS-LLM.git
   cd TBYS-LLM
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download models:**
   ```bash
   # Text model - downloads to models/ directory
   python -c "from src.models.qwen_manager import Qwen3Manager; m = Qwen3Manager(); m.download_model()"

   # Vision-Language model should be placed in models_vl/ directory
   # Download from: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
   ```

## Usage

Run the application:

```bash
python run.py
```

Options:
- `--host HOST`: Server host (default: 0.0.0.0)
- `--port PORT`: Server port (default: 8080)
- `--skip-text`: Skip loading text model
- `--enable-vl`: Enable loading VL model (VL is skipped by default)
- `--cpu`: Force CPU usage

## Configuration

Environment variables:

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8080)
- `DB_PATH`: Database file path (default: qwen3_chat.db)

## API Endpoints

- `GET /`: Main web interface
- `GET /api/health`: Health check
- `GET /api/conversations`: List conversations
- `POST /api/conversations`: Create conversation
- `GET /api/conversations/<id>`: Get conversation
- `DELETE /api/conversations/<id>`: Delete conversation
- `POST /api/chat`: Send chat message (streaming)

## Model Support

- **Text Model (Qwen3-0.6B)**: Standard text generation with thinking mode
- **Vision-Language Model (Qwen3-VL)**: Text + image understanding, PDF analysis
