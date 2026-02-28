"""FastAPI 應用入口 - Qwen3-TTS 多國語音播放器"""

import os
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.tts_engine import engine
from app.config import (
    UPLOAD_DIR, MAX_AUDIO_SIZE_MB, CONFIG_DIR, DEVICE,
    load_user_config, save_user_config, get_model_id,
)


def auto_load_from_config(cfg: dict):
    """根據已儲存的設定，背景載入模型"""
    # 支援新格式 model_sizes dict，兼容舊格式 model_size 單一值
    model_sizes = cfg.get("model_sizes", None)
    fallback_size = cfg.get("model_size", "1.7B")
    custom_paths = cfg.get("custom_paths", {})

    for mode in ("preset", "design", "clone"):
        if cfg.get(mode, False):
            size = model_sizes.get(mode, fallback_size) if model_sizes else fallback_size
            # 環境變數指定的路徑優先，其次是 custom_paths，最後是預設
            model_id = custom_paths.get(mode) or get_model_id(mode, size)
            engine.load_single_model(mode, model_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    # 如果有已儲存的設定，背景自動載入（不阻塞啟動）
    saved = load_user_config()
    if saved:
        threading.Thread(
            target=auto_load_from_config, args=(saved,), daemon=True
        ).start()
    yield


app = FastAPI(title="VoxCraft - Multi-Language Voice Studio", lifespan=lifespan)


# --- Setup API ---

class SetupLoadRequest(BaseModel):
    preset: bool = False
    design: bool = False
    clone: bool = False
    model_sizes: dict = {"preset": "1.7B", "design": "1.7B", "clone": "1.7B"}


@app.get("/api/setup/status")
async def get_setup_status():
    """回傳設定狀態 + 各模型載入狀態"""
    saved = load_user_config()
    model_status = engine.get_status()

    # 兼容舊格式：若無 model_sizes，從 model_size 建構
    default_sizes = {"preset": "1.7B", "design": "1.7B", "clone": "1.7B"}
    if saved:
        fallback = saved.get("model_size", "1.7B")
        model_sizes = saved.get("model_sizes", {m: fallback for m in default_sizes})
    else:
        model_sizes = default_sizes

    models = {}
    for mode in ("preset", "design", "clone"):
        enabled = saved.get(mode, False) if saved else False
        models[mode] = {
            "enabled": enabled,
            "status": model_status[mode]["status"],
            "model_id": model_status[mode]["model_id"],
            "error": model_status[mode]["error"],
        }

    return {
        "configured": saved is not None,
        "models": models,
        "model_sizes": model_sizes,
        "device": DEVICE,
    }


@app.post("/api/setup/load")
async def setup_load(req: SetupLoadRequest):
    """儲存設定並開始背景載入模型"""
    for mode, size in req.model_sizes.items():
        if size not in ("0.6B", "1.7B"):
            raise HTTPException(status_code=400, detail=f"{mode} 的 model_size 必須為 0.6B 或 1.7B")

    if not any([req.preset, req.design, req.clone]):
        raise HTTPException(status_code=400, detail="請至少啟用一個模式")

    # 儲存 config
    cfg = {
        "preset": req.preset,
        "design": req.design,
        "clone": req.clone,
        "model_sizes": req.model_sizes,
        "custom_paths": {
            "preset": None,
            "design": None,
            "clone": None,
        },
    }
    save_user_config(cfg)

    # 在背景 thread 卸載不需要的 + 載入需要的
    def background_load():
        for mode in ("preset", "design", "clone"):
            if not getattr(req, mode):
                engine.unload_model(mode)

        for mode in ("preset", "design", "clone"):
            if getattr(req, mode):
                size = req.model_sizes.get(mode, "1.7B")
                model_id = get_model_id(mode, size)
                # 如果已經載入相同模型，跳過
                current_id = engine.model_ids.get(mode)
                if engine.model_status[mode] == "ready" and current_id == model_id:
                    continue
                # 如果載入了不同模型，先卸載
                if engine.model_status[mode] == "ready":
                    engine.unload_model(mode)
                engine.load_single_model(mode, model_id)

    threading.Thread(target=background_load, daemon=True).start()

    return {"ok": True}


# --- Existing API ---

class TTSRequest(BaseModel):
    text: str
    language: str
    speaker: str
    instruct: Optional[str] = None


class VoiceDesignRequest(BaseModel):
    text: str
    language: str
    instruct: str


class RegisteredCloneRequest(BaseModel):
    text: str
    language: str
    voice_id: str


@app.get("/api/capabilities")
async def get_capabilities():
    """回傳目前啟用的功能"""
    return engine.get_capabilities()


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


@app.post("/api/voice-clone")
async def generate_voice_clone(
    text: str = Form(...),
    language: str = Form(...),
    ref_text: Optional[str] = Form(None),
    x_vector_only: bool = Form(False),
    ref_audio: UploadFile = File(...),
):
    """使用參考音頻複製語音"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="文字不能為空")
    if len(text) > 500:
        raise HTTPException(status_code=400, detail="文字長度不能超過 500 字")

    # 驗證檔案類型
    allowed = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    ext = os.path.splitext(ref_audio.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"不支援的音頻格式，請上傳: {', '.join(allowed)}"
        )

    # 讀取並存暫存檔
    content = await ref_audio.read()
    if len(content) > MAX_AUDIO_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400, detail=f"音頻檔案不能超過 {MAX_AUDIO_SIZE_MB}MB"
        )

    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(content)

        wav_bytes = engine.generate_clone(
            text, language, tmp_path, ref_text, x_vector_only
        )
        return Response(content=wav_bytes, media_type="audio/wav")
    except ValueError as e:
        print(f"[VoiceClone ValueError] {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成失敗: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/voice-register")
async def register_voice(
    name: str = Form(...),
    ref_text: Optional[str] = Form(None),
    x_vector_only: bool = Form(False),
    ref_audio: UploadFile = File(...),
):
    """註冊音色：預計算 prompt 快取在記憶體中"""
    if not name.strip():
        raise HTTPException(status_code=400, detail="請提供音色名稱")

    allowed = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    ext = os.path.splitext(ref_audio.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"不支援的音頻格式，請上傳: {', '.join(allowed)}"
        )

    content = await ref_audio.read()
    if len(content) > MAX_AUDIO_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400, detail=f"音頻檔案不能超過 {MAX_AUDIO_SIZE_MB}MB"
        )

    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(content)

        voice_id = engine.register_voice(
            name.strip(), tmp_path, ref_text, x_vector_only
        )
        return {"voice_id": voice_id, "name": name.strip()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"註冊失敗: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/api/registered-voices")
async def get_registered_voices():
    """回傳已註冊音色列表"""
    return engine.get_registered_voices()


@app.delete("/api/registered-voices/{voice_id}")
async def delete_registered_voice(voice_id: str):
    """刪除已註冊音色"""
    engine.delete_registered_voice(voice_id)
    return {"ok": True}


@app.post("/api/voice-clone-registered")
async def generate_voice_clone_registered(req: RegisteredCloneRequest):
    """使用已註冊音色生成語音"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="文字不能為空")
    if len(req.text) > 500:
        raise HTTPException(status_code=400, detail="文字長度不能超過 500 字")

    try:
        wav_bytes = engine.generate_with_registered(
            req.text, req.language, req.voice_id
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


@app.get("/api/clone-languages")
async def get_clone_languages():
    return engine.get_clone_languages()


# 靜態檔案（前端）
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
