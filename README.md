# VoxCraft - Multi-Language Voice Studio

A web-based text-to-speech application powered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), supporting 10 languages with three modes: Preset Voice, Voice Design, and Voice Clone.

![VoxCraft Setup](https://img.shields.io/badge/Qwen3--TTS-Powered-c8a44e) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-Apache%202.0-green)

## Features

- **Preset Voice** — 9 premium speakers (Vivian, Serena, Dylan, etc.) with optional style control
- **Voice Design** — Create custom voices by describing them in natural language
- **Voice Clone** — Clone any voice from just 3 seconds of audio
- **Web Setup UI** — First-launch setup wizard with model selection and loading progress
- **10 Languages** — Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

## Requirements

| Item | Requirement |
|------|------------|
| Python | 3.10 or higher |
| GPU (recommended) | NVIDIA GPU with CUDA, or Apple Silicon (MPS) |
| VRAM | ~4 GB for 0.6B models, ~8 GB for 1.7B models |
| Disk | ~15 GB for all three 1.7B models (auto-downloaded on first use) |

> CPU-only mode is supported but will be significantly slower.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/TTS-Ver1.git
cd TTS-Ver1
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# macOS / Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### 3. Install PyTorch

Choose the correct version for your hardware:

**NVIDIA GPU (Windows / Linux):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Apple Silicon (Mac M1/M2/M3/M4):**
```bash
pip install torch
```

**CPU only (no GPU):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> For other CUDA versions or platforms, see: https://pytorch.org/get-started/locally/

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. (Optional) Install FlashAttention for faster inference

Only available on Linux with NVIDIA GPU:

```bash
pip install flash-attn --no-build-isolation
```

## Usage

### Start the server

```bash
python run.py
```

The server starts instantly (models are loaded later). Open your browser:

```
http://localhost:8000
```

### First-time setup

On first launch, you will see a **Setup Wizard**:

1. **Select modes** — Choose which modes to enable (Preset Voice / Voice Design / Voice Clone)
2. **Select model size** — `0.6B` (faster, less VRAM) or `1.7B` (better quality)
3. **Click "Start Loading Models"** — Models are downloaded from HuggingFace automatically
4. **Wait for loading** — Progress is shown for each model. First download may take several minutes
5. **Done** — The main interface appears automatically when all models are ready

Your settings are saved to `config/user_settings.json`. On next launch, models load automatically.

### Reconfigure

Click the **gear icon** (top-right) in the main interface to reopen the setup wizard and change modes or model size.

## Advanced Configuration

For advanced users, environment variables override the web UI settings:

```bash
# Disable specific modes
ENABLE_PRESET=0 python run.py
ENABLE_CLONE=0 python run.py

# Use 0.6B models instead of 1.7B
MODEL_SIZE=0.6B python run.py

# Point to local model paths (skip download)
CUSTOM_VOICE_MODEL=/path/to/local/model python run.py
VOICE_DESIGN_MODEL=/path/to/local/model python run.py
VOICE_CLONE_MODEL=/path/to/local/model python run.py
```

## Project Structure

```
TTS-Ver1/
├── app/
│   ├── config.py        # Configuration, device detection, config persistence
│   ├── main.py          # FastAPI server, API endpoints
│   ├── tts_engine.py    # TTS model wrapper, loading/unloading
│   └── static/
│       └── index.html   # Web UI (single-page app)
├── config/
│   └── user_settings.json  # Saved user settings (auto-created)
├── uploads/             # Temporary audio uploads (auto-cleaned)
├── requirements.txt
├── run.py               # Server entry point
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/setup/status` | GET | Model loading status and configuration |
| `/api/setup/load` | POST | Save config and start loading models |
| `/api/capabilities` | GET | Currently loaded model capabilities |
| `/api/tts` | POST | Generate speech with preset voice |
| `/api/voice-design` | POST | Generate speech with voice design |
| `/api/voice-clone` | POST | Generate speech with voice cloning |
| `/api/voice-register` | POST | Register a voice for reuse |
| `/api/voice-clone-registered` | POST | Generate speech with registered voice |

## Troubleshooting

### Models download is slow

Models are hosted on HuggingFace. If download is slow, try setting a mirror:

```bash
# For users in mainland China
export HF_ENDPOINT=https://hf-mirror.com
python run.py
```

### Out of memory

- Switch to `0.6B` models in the setup wizard
- Enable fewer modes (e.g., only Preset Voice)
- Close other GPU-intensive applications

### `torch.cuda.is_available()` returns False on Windows

Make sure you installed the CUDA version of PyTorch (Step 3), not the CPU version. Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, reinstall PyTorch with CUDA support.

### SoX warning on startup

The message `SoX could not be found` is harmless — it comes from an optional dependency and does not affect functionality.

## License

This project uses [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) models, which are licensed under Apache 2.0.
