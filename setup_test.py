"""環境驗證腳本 - 檢查 Qwen3-TTS 所需的環境是否正確設置"""

import sys

def check_python_version():
    v = sys.version_info
    print(f"[1/4] Python 版本: {v.major}.{v.minor}.{v.micro}", end=" ")
    if v.major == 3 and v.minor >= 10:
        print("✓")
        return True
    else:
        print("✗ (建議 3.10+)")
        return False

def check_torch_mps():
    print("[2/4] PyTorch MPS 後端:", end=" ")
    try:
        import torch
        print(f"PyTorch {torch.__version__}", end=" ")
        if torch.backends.mps.is_available():
            print("MPS 可用 ✓")
            # 簡單測試 MPS 運算
            x = torch.ones(3, 3, device="mps")
            y = x * 2
            assert y.sum().item() == 18.0
            print("       MPS 運算測試通過 ✓")
            return True
        else:
            print("MPS 不可用 ✗")
            if torch.backends.mps.is_built():
                print("       MPS 已編譯但不可用（可能是 macOS 版本問題）")
            return False
    except Exception as e:
        print(f"錯誤: {e}")
        return False

def check_qwen_tts():
    print("[3/4] qwen-tts 套件:", end=" ")
    try:
        import qwen_tts
        print(f"版本 {getattr(qwen_tts, '__version__', '未知')} ✓")
        return True
    except ImportError as e:
        print(f"未安裝 ✗ ({e})")
        return False

def check_fastapi():
    print("[4/4] FastAPI + Uvicorn:", end=" ")
    try:
        import fastapi
        import uvicorn
        print(f"FastAPI {fastapi.__version__}, Uvicorn {uvicorn.__version__} ✓")
        return True
    except ImportError as e:
        print(f"未安裝 ✗ ({e})")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Qwen3-TTS 環境驗證")
    print("=" * 50)
    results = [
        check_python_version(),
        check_torch_mps(),
        check_qwen_tts(),
        check_fastapi(),
    ]
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    if all(results):
        print(f"全部通過 ({passed}/{total}) ✓")
    else:
        print(f"通過 {passed}/{total}，部分項目需要修正")
