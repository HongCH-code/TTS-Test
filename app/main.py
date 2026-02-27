"""FastAPI 應用入口 - Qwen3-TTS 多國語音播放器"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.tts_engine import engine
from app.config import HOST, PORT


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用啟動時載入模型"""
    engine.load_model()
    yield


app = FastAPI(title="Qwen3-TTS 多國語音播放器", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str
    language: str
    speaker: str


@app.post("/api/tts")
async def generate_tts(req: TTSRequest):
    """生成語音"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="文字不能為空")
    if len(req.text) > 500:
        raise HTTPException(status_code=400, detail="文字長度不能超過 500 字")

    try:
        wav_bytes = engine.generate(req.text, req.language, req.speaker)
        return Response(content=wav_bytes, media_type="audio/wav")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失敗: {str(e)}")


@app.get("/api/speakers")
async def get_speakers():
    """取得語言與音色對照表"""
    return engine.get_speakers()


@app.get("/api/languages")
async def get_languages():
    """取得支援的語言列表"""
    return engine.get_languages()


# 靜態檔案（前端）
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
