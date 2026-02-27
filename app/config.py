"""TTS 應用設定"""

import torch

# 模型設定
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
# 升級時改為: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# 裝置設定：優先 MPS (Apple Silicon)，否則 CUDA，最後 CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32  # MPS 對 bfloat16 支援有限
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

# 語言與音色對照表（0.6B-CustomVoice 預設音色，模型使用小寫名稱）
LANGUAGE_SPEAKERS = {
    "Chinese": ["vivian", "serena", "uncle_fu", "dylan", "eric"],
    "English": ["ryan", "aiden"],
    "Japanese": ["ono_anna"],
    "Korean": ["sohee"],
}

# 音色顯示名稱（用於前端）
SPEAKER_DISPLAY_NAMES = {
    "vivian": "Vivian", "serena": "Serena", "uncle_fu": "Uncle Fu",
    "dylan": "Dylan", "eric": "Eric", "ryan": "Ryan", "aiden": "Aiden",
    "ono_anna": "Ono Anna", "sohee": "Sohee",
}

# Server 設定
HOST = "0.0.0.0"
PORT = 8000
