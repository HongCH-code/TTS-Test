"""FastAPI 應用入口 - Qwen3-TTS 多國語音播放器"""

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.tts_engine import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load_model()
    yield


app = FastAPI(title="VoxCraft - Multi-Language Voice Studio", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str
    language: str
    speaker: str
    instruct: Optional[str] = None


class VoiceDesignRequest(BaseModel):
    text: str
    language: str
    instruct: str


@app.post("/api/tts")
async def generate_tts(req: TTSRequest):
    """使用預設音色生成語音"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="文字不能為空")
    if len(req.text) > 500:
        raise HTTPException(status_code=400, detail="文字長度不能超過 500 字")

    try:
        wav_bytes = engine.generate_custom(
            req.text, req.language, req.speaker, req.instruct
        )
        return Response(content=wav_bytes, media_type="audio/wav")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失敗: {str(e)}")


@app.post("/api/voice-design")
async def generate_voice_design(req: VoiceDesignRequest):
    """使用自然語言描述產生自定義音色"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="文字不能為空")
    if len(req.text) > 500:
        raise HTTPException(status_code=400, detail="文字長度不能超過 500 字")
    if not req.instruct.strip():
        raise HTTPException(status_code=400, detail="請提供音色描述")

    try:
        wav_bytes = engine.generate_design(
            req.text, req.language, req.instruct
        )
        return Response(content=wav_bytes, media_type="audio/wav")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失敗: {str(e)}")


@app.get("/api/speakers")
async def get_speakers():
    return engine.get_speakers()


@app.get("/api/languages")
async def get_languages():
    return engine.get_languages()


@app.get("/api/design-languages")
async def get_design_languages():
    return engine.get_design_languages()


# 靜態檔案（前端）
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
