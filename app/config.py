"""TTS 應用設定"""

import json
import os
import torch

# --- Config 持久化 ---
CONFIG_DIR = "config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "user_settings.json")

# 模型 ID 對照表
MODEL_IDS = {
    "preset": {
        "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    },
    "design": {
        "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    },
    "clone": {
        "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    },
}

# 模型屬性名稱對照
MODEL_ATTR = {
    "preset": "custom_model",
    "design": "design_model",
    "clone":  "clone_model",
}


def load_user_config() -> dict | None:
    """讀取使用者設定，不存在回傳 None"""
    if not os.path.exists(CONFIG_FILE):
        return None
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_user_config(cfg: dict):
    """儲存使用者設定到 JSON"""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def get_model_id(mode: str, model_size: str) -> str:
    """根據模式和大小回傳 HuggingFace model ID"""
    return MODEL_IDS[mode][model_size]


# --- 環境變數覆蓋（進階用戶） ---

# 模式啟用（環境變數覆蓋：ENABLE_PRESET=0 可停用）
ENABLE_PRESET = os.getenv("ENABLE_PRESET", "1") == "1"
ENABLE_DESIGN = os.getenv("ENABLE_DESIGN", "1") == "1"
ENABLE_CLONE  = os.getenv("ENABLE_CLONE", "1") == "1"

# 模型大小選擇（"0.6B" 或 "1.7B"，環境變數：MODEL_SIZE=0.6B）
MODEL_SIZE = os.getenv("MODEL_SIZE", "1.7B")

# 模型路徑（可用環境變數指定本地路徑）
CUSTOM_VOICE_MODEL = os.getenv("CUSTOM_VOICE_MODEL", f"Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-CustomVoice")
VOICE_DESIGN_MODEL = os.getenv("VOICE_DESIGN_MODEL", f"Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-VoiceDesign")
VOICE_CLONE_MODEL  = os.getenv("VOICE_CLONE_MODEL",  f"Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-Base")

# 裝置設定
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

# 語言與音色對照表（1.7B-CustomVoice 預設音色）
LANGUAGE_SPEAKERS = {
    "Chinese": ["vivian", "serena", "uncle_fu", "dylan", "eric"],
    "English": ["ryan", "aiden"],
    "Japanese": ["ono_anna"],
    "Korean": ["sohee"],
}

# 音色顯示名稱
SPEAKER_DISPLAY_NAMES = {
    "vivian": "Vivian", "serena": "Serena", "uncle_fu": "Uncle Fu",
    "dylan": "Dylan", "eric": "Eric", "ryan": "Ryan", "aiden": "Aiden",
    "ono_anna": "Ono Anna", "sohee": "Sohee",
}

# VoiceDesign 支援的語言
VOICE_DESIGN_LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

# Voice Clone 支援的語言（與 VoiceDesign 相同）
VOICE_CLONE_LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

# 上傳設定
UPLOAD_DIR = "uploads"
MAX_AUDIO_SIZE_MB = 10

# Server 設定
HOST = "0.0.0.0"
PORT = 8000
