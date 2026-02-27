"""TTS 模型封裝 - 負責模型載入與語音生成"""

import io
import uuid
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from app.config import (
    CUSTOM_VOICE_MODEL, VOICE_DESIGN_MODEL, VOICE_CLONE_MODEL, DTYPE,
    LANGUAGE_SPEAKERS, VOICE_DESIGN_LANGUAGES, VOICE_CLONE_LANGUAGES,
)


class TTSEngine:
    def __init__(self):
        self.custom_model = None
        self.design_model = None
        self.clone_model = None
        self.voice_registry = {}  # {voice_id: {"name": str, "prompt": list}}

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

        # Base 模型（語音複製）
        print(f"正在載入 VoiceClone 模型: {VOICE_CLONE_MODEL}")
        self.clone_model = Qwen3TTSModel.from_pretrained(
            VOICE_CLONE_MODEL, dtype=DTYPE
        )
        print(f"VoiceClone 載入完成，裝置: {self.clone_model.device}")

    def get_speakers(self) -> dict:
        return LANGUAGE_SPEAKERS

    def get_languages(self) -> list:
        return list(LANGUAGE_SPEAKERS.keys())

    def get_design_languages(self) -> list:
        return VOICE_DESIGN_LANGUAGES

    def get_clone_languages(self) -> list:
        return VOICE_CLONE_LANGUAGES

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

    def generate_clone(self, text: str, language: str,
                       ref_audio_path: str, ref_text: str = None,
                       x_vector_only: bool = False) -> bytes:
        """使用參考音頻複製語音"""
        if not self.clone_model:
            raise RuntimeError("VoiceClone 模型尚未載入")

        if language not in VOICE_CLONE_LANGUAGES:
            raise ValueError(f"VoiceClone 不支援語言: {language}")

        # ICL 模式需要 ref_text，x_vector_only 模式不需要
        if not x_vector_only and (not ref_text or not ref_text.strip()):
            raise ValueError("請提供參考音頻的文字稿（或啟用僅聲紋模式）")

        kwargs = dict(
            text=text, language=language,
            ref_audio=ref_audio_path,
            x_vector_only_mode=x_vector_only,
        )
        if ref_text and ref_text.strip():
            kwargs["ref_text"] = ref_text

        wavs, sr = self.clone_model.generate_voice_clone(**kwargs)
        return self._to_wav_bytes(wavs, sr)

    # --- Voice Registration (memory-only cache) ---

    def register_voice(self, name: str, ref_audio_path: str,
                       ref_text: str = None,
                       x_vector_only: bool = False) -> str:
        """註冊音色：預計算 voice_clone_prompt，回傳 voice_id"""
        if not self.clone_model:
            raise RuntimeError("VoiceClone 模型尚未載入")

        kwargs = dict(
            ref_audio=ref_audio_path,
            x_vector_only_mode=x_vector_only,
        )
        if ref_text and ref_text.strip():
            kwargs["ref_text"] = ref_text

        prompt_items = self.clone_model.create_voice_clone_prompt(**kwargs)
        voice_id = uuid.uuid4().hex[:8]
        self.voice_registry[voice_id] = {"name": name, "prompt": prompt_items}
        return voice_id

    def get_registered_voices(self) -> list:
        """回傳已註冊音色列表"""
        return [{"id": vid, "name": v["name"]}
                for vid, v in self.voice_registry.items()]

    def delete_registered_voice(self, voice_id: str):
        """刪除已註冊音色"""
        self.voice_registry.pop(voice_id, None)

    def generate_with_registered(self, text: str, language: str,
                                 voice_id: str) -> bytes:
        """使用已註冊音色生成語音（跳過音頻處理，直接用快取 prompt）"""
        if not self.clone_model:
            raise RuntimeError("VoiceClone 模型尚未載入")

        if voice_id not in self.voice_registry:
            raise ValueError("找不到已註冊的音色")

        if language not in VOICE_CLONE_LANGUAGES:
            raise ValueError(f"VoiceClone 不支援語言: {language}")

        prompt_items = self.voice_registry[voice_id]["prompt"]
        wavs, sr = self.clone_model.generate_voice_clone(
            text=text, language=language, voice_clone_prompt=prompt_items,
        )
        return self._to_wav_bytes(wavs, sr)


# 全域單例
engine = TTSEngine()
