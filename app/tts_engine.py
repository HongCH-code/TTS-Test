"""TTS 模型封裝 - 負責模型載入與語音生成"""

import io
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from app.config import MODEL_NAME, DTYPE, LANGUAGE_SPEAKERS


class TTSEngine:
    def __init__(self):
        self.model = None
        self.sample_rate = None

    def load_model(self):
        """載入 TTS 模型（啟動時呼叫一次）"""
        print(f"正在載入模型: {MODEL_NAME}")
        print(f"精度: {DTYPE}")

        # 先用 CPU 載入確保穩定，MPS 支援待後續測試
        self.model = Qwen3TTSModel.from_pretrained(MODEL_NAME, dtype=DTYPE)

        print(f"模型載入完成，裝置: {self.model.device}")

        speakers = self.model.get_supported_speakers()
        if speakers:
            print(f"支援音色: {speakers}")

    def get_speakers(self) -> dict:
        """回傳語言與音色對照表"""
        return LANGUAGE_SPEAKERS

    def get_languages(self) -> list:
        """回傳支援的語言列表"""
        return list(LANGUAGE_SPEAKERS.keys())

    def generate(self, text: str, language: str, speaker: str) -> bytes:
        """生成語音，回傳 WAV 格式的 bytes"""
        if not self.model:
            raise RuntimeError("模型尚未載入")

        if language not in LANGUAGE_SPEAKERS:
            raise ValueError(f"不支援的語言: {language}")

        if speaker not in LANGUAGE_SPEAKERS[language]:
            raise ValueError(
                f"語言 {language} 不支援音色 {speaker}，"
                f"可用: {LANGUAGE_SPEAKERS[language]}"
            )

        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
        )

        self.sample_rate = sr

        # 轉換為 WAV bytes
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, wavs[0], sr, format="WAV")
        wav_buffer.seek(0)
        return wav_buffer.read()


# 全域單例
engine = TTSEngine()
