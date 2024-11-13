from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from transcriber import AudioTranscriber
import threading
import os

app = FastAPI()


# 确保static目录存在
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 全局变量存储转录器实例
transcriber = None
transcriber_lock = threading.Lock()

@app.get("/")
async def read_root():
    """返回主页"""
    return FileResponse("static/index.html")

@app.post("/start")
async def start_transcription():
    """开始转录"""
    global transcriber
    with transcriber_lock:
        try:
            if transcriber is None:
                transcriber = AudioTranscriber(show_volume=True)
            
            if not transcriber.is_running:
                transcriber.start()
                return {"status": "success", "message": "转录已开始"}
            else:
                return {"status": "warning", "message": "转录已在运行中"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
async def stop_transcription():
    """停止转录"""
    global transcriber
    with transcriber_lock:
        try:
            if transcriber and transcriber.is_running:
                transcriber.stop()
                return {"status": "success", "message": "转录已停止"}
            else:
                return {"status": "warning", "message": "转录未在运行"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """获取转录状态"""
    global transcriber
    with transcriber_lock:
        if transcriber is None:
            return {"status": "stopped", "message": "转录器未初始化"}
        return {
            "status": "running" if transcriber.is_running else "stopped",
            "message": "转录正在运行" if transcriber.is_running else "转录已停止"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    #uvicorn.run(app, host="0.0.0.0", port=8000)
