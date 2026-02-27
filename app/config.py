"""TTS 應用設定"""

import torch

# 模型設定
CUSTOM_VOICE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
VOICE_CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

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
