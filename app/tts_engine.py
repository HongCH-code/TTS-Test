"""TTS 模型封裝 - 負責模型載入與語音生成"""

import io
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from app.config import (
    CUSTOM_VOICE_MODEL, VOICE_DESIGN_MODEL, DTYPE,
    LANGUAGE_SPEAKERS, VOICE_DESIGN_LANGUAGES,
)


class TTSEngine:
    def __init__(self):
        self.custom_model = None
        self.design_model = None

    def load_model(self):
        """載入 TTS 模型（啟動時呼叫）"""
        # CustomVoice 模型（預設音色 + instruct 控制）
        print(f"正在載入 CustomVoice 模型: {CUSTOM_VOICE_MODEL}")
        self.custom_model = Qwen3TTSModel.from_pretrained(
            CUSTOM_VOICE_MODEL, dtype=DTYPE
        )
        print(f"CustomVoice 載入完成，裝置: {self.custom_model.device}")
        speakers = self.custom_model.get_supported_speakers()
        if speakers:
            print(f"支援音色: {speakers}")

        # VoiceDesign 模型（自定義音色）
        print(f"正在載入 VoiceDesign 模型: {VOICE_DESIGN_MODEL}")
        self.design_model = Qwen3TTSModel.from_pretrained(
            VOICE_DESIGN_MODEL, dtype=DTYPE
        )
        print(f"VoiceDesign 載入完成，裝置: {self.design_model.device}")

    def get_speakers(self) -> dict:
        return LANGUAGE_SPEAKERS

    def get_languages(self) -> list:
        return list(LANGUAGE_SPEAKERS.keys())

    def get_design_languages(self) -> list:
        return VOICE_DESIGN_LANGUAGES

    def _to_wav_bytes(self, wavs, sr) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        buf.seek(0)
        return buf.read()

    def generate_custom(self, text: str, language: str, speaker: str,
                        instruct: str = None) -> bytes:
        """使用預設音色生成語音，可選 instruct 風格控制"""
        if not self.custom_model:
            raise RuntimeError("CustomVoice 模型尚未載入")

        if language not in LANGUAGE_SPEAKERS:
            raise ValueError(f"不支援的語言: {language}")

        if speaker not in LANGUAGE_SPEAKERS[language]:
            raise ValueError(
                f"語言 {language} 不支援音色 {speaker}，"
                f"可用: {LANGUAGE_SPEAKERS[language]}"
            )

        kwargs = dict(text=text, language=language, speaker=speaker)
        if instruct:
            kwargs["instruct"] = instruct

        wavs, sr = self.custom_model.generate_custom_voice(**kwargs)
        return self._to_wav_bytes(wavs, sr)

    def generate_design(self, text: str, language: str,
                        instruct: str) -> bytes:
        """使用自然語言描述產生自定義音色"""
        if not self.design_model:
            raise RuntimeError("VoiceDesign 模型尚未載入")

        if language not in VOICE_DESIGN_LANGUAGES:
            raise ValueError(f"VoiceDesign 不支援語言: {language}")

        if not instruct or not instruct.strip():
            raise ValueError("請提供音色描述")

        wavs, sr = self.design_model.generate_voice_design(
            text=text, language=language, instruct=instruct,
        )
        return self._to_wav_bytes(wavs, sr)


# 全域單例
engine = TTSEngine()
