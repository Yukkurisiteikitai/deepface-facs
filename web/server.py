"""
FACS Web Server - ローカルネットワーク経由でスマホから顔分析
"""
import asyncio
import base64
import json
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from facs import FACSAnalyzer
from facs.core.enums import AnalysisMode
from facs.recording import FACSRecorder


class FACSWebServer:
    """FACS Webサーバー"""
    
    def __init__(self, recordings_dir: str = "./recordings"):
        self.app = FastAPI(title="FACS Analyzer")
        self.analyzer = FACSAnalyzer(mode=AnalysisMode.REALTIME)
        self.recordings_dir = Path(recordings_dir)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_recorders: dict[str, FACSRecorder] = {}
        
        self._setup_routes()
    
    def _setup_routes(self):
        """ルートを設定"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            return self._get_index_html()
        
        @self.app.websocket("/ws/analyze")
        async def websocket_analyze(websocket: WebSocket):
            await websocket.accept()
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            recorder: Optional[FACSRecorder] = None
            is_recording = False
            frame_count = 0
            
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message["type"] == "frame":
                        # Base64画像をデコード
                        image_data = message["data"].split(",")[1]
                        image_bytes = base64.b64decode(image_data)
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # 分析
                            result = self.analyzer.analyze(frame)
                            result.frame_number = frame_count
                            frame_count += 1
                            
                            # 記録中なら保存
                            if is_recording and recorder:
                                recorder.record_frame(result)
                            
                            # 結果を送信
                            response = {
                                "type": "result",
                                "data": result.to_dict(),
                                "recording": is_recording,
                                "frame_count": recorder.frame_count if recorder else 0,
                            }
                            await websocket.send_text(json.dumps(response))
                    
                    elif message["type"] == "start_recording":
                        if not is_recording:
                            width = message.get("width", 640)
                            height = message.get("height", 480)
                            fps = message.get("fps", 30)
                            
                            recorder = FACSRecorder(str(self.recordings_dir), session_id)
                            recorder.start(
                                fps=fps,
                                width=width,
                                height=height,
                                source="mobile_web",
                                description=message.get("description", "")
                            )
                            is_recording = True
                            self.active_recorders[session_id] = recorder
                            
                            await websocket.send_text(json.dumps({
                                "type": "recording_started",
                                "session_id": session_id
                            }))
                    
                    elif message["type"] == "stop_recording":
                        if is_recording and recorder:
                            metadata = recorder.stop()
                            is_recording = False
                            
                            if session_id in self.active_recorders:
                                del self.active_recorders[session_id]
                            
                            await websocket.send_text(json.dumps({
                                "type": "recording_stopped",
                                "session_id": session_id,
                                "total_frames": metadata.total_frames,
                                "duration": metadata.duration_sec
                            }))
                            recorder = None
            
            except WebSocketDisconnect:
                # 切断時に記録中なら停止
                if is_recording and recorder:
                    recorder.stop()
                    if session_id in self.active_recorders:
                        del self.active_recorders[session_id]
        
        @self.app.get("/recordings")
        async def list_recordings():
            """記録一覧を取得"""
            recordings = []
            for meta_file in self.recordings_dir.glob("*_meta.json"):
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    meta["id"] = meta_file.stem.replace("_meta", "")
                    recordings.append(meta)
            return sorted(recordings, key=lambda x: x["created_at"], reverse=True)
        
        @self.app.get("/recordings/{recording_id}")
        async def get_recording(recording_id: str):
            """記録データを取得"""
            data_file = self.recordings_dir / f"{recording_id}.jsonl"
            if not data_file.exists():
                return {"error": "Recording not found"}
            
            frames = []
            with open(data_file, "r") as f:
                for line in f:
                    if line.strip():
                        frames.append(json.loads(line))
            return {"frames": frames}
    
    def _get_index_html(self) -> str:
        """メインページのHTML"""
        return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>FACS Analyzer</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 10px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        .video-container {
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 15px;
        }
        #video {
            width: 100%;
            display: block;
            transform: scaleX(-1);
        }
        #canvas {
            display: none;
        }
        .overlay {
            position: absolute;
            top: 10px;
            left: 10px;
            right: 10px;
            display: flex;
            justify-content: space-between;
            pointer-events: none;
        }
        .status {
            background: rgba(0,0,0,0.7);
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        .status.recording {
            background: rgba(255,0,0,0.8);
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 15px;
        }
        button {
            padding: 15px 30px;
            font-size: 1.1em;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }
        #startBtn {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }
        #stopBtn {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            color: white;
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        button:active:not(:disabled) {
            transform: scale(0.95);
        }
        .results {
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
        }
        .result-section {
            margin-bottom: 15px;
        }
        .result-section h3 {
            color: #667eea;
            margin-bottom: 8px;
            font-size: 1em;
        }
        .emotion-bar {
            display: flex;
            align-items: center;
            margin: 5px 0;
        }
        .emotion-name {
            width: 80px;
            font-size: 0.9em;
        }
        .bar-container {
            flex: 1;
            height: 20px;
            background: #0f3460;
            border-radius: 10px;
            overflow: hidden;
        }
        .bar {
            height: 100%;
            transition: width 0.3s;
            border-radius: 10px;
        }
        .bar.positive { background: linear-gradient(90deg, #11998e, #38ef7d); }
        .bar.negative { background: linear-gradient(90deg, #eb3349, #f45c43); }
        .bar.neutral { background: linear-gradient(90deg, #f7971e, #ffd200); }
        .confidence {
            width: 50px;
            text-align: right;
            font-size: 0.9em;
        }
        .facs-code {
            font-family: monospace;
            background: #0f3460;
            padding: 10px;
            border-radius: 8px;
            word-break: break-all;
        }
        .au-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .au-tag {
            background: #667eea;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.85em;
        }
        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .metric {
            background: #0f3460;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.8em;
            color: #888;
        }
        .connection-status {
            text-align: center;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 8px;
        }
        .connection-status.connected {
            background: rgba(17, 153, 142, 0.3);
        }
        .connection-status.disconnected {
            background: rgba(235, 51, 73, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 FACS Analyzer</h1>
        
        <div class="connection-status disconnected" id="connectionStatus">
            接続中...
        </div>
        
        <div class="video-container">
            <video id="video" autoplay playsinline></video>
            <canvas id="canvas"></canvas>
            <div class="overlay">
                <div class="status" id="status">準備中</div>
                <div class="status" id="fps">-- FPS</div>
            </div>
        </div>
        
        <div class="controls">
            <button id="startBtn" disabled>⏺ 記録開始</button>
            <button id="stopBtn" disabled>⏹ 記録停止</button>
        </div>
        
        <div class="results">
            <div class="result-section">
                <h3>📊 感情分析</h3>
                <div id="emotions"></div>
            </div>
            
            <div class="result-section">
                <h3>📈 Valence / Arousal</h3>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="valence">0.00</div>
                        <div class="metric-label">Valence (快/不快)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="arousal">0.00</div>
                        <div class="metric-label">Arousal (覚醒度)</div>
                    </div>
                </div>
            </div>
            
            <div class="result-section">
                <h3>🔢 FACSコード</h3>
                <div class="facs-code" id="facsCode">-</div>
            </div>
            
            <div class="result-section">
                <h3>🎯 検出AU</h3>
                <div class="au-list" id="auList"></div>
            </div>
        </div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusEl = document.getElementById('status');
        const fpsEl = document.getElementById('fps');
        const connectionStatus = document.getElementById('connectionStatus');
        
        let ws = null;
        let isRecording = false;
        let frameCount = 0;
        let lastTime = performance.now();
        let fps = 0;
        
        // WebSocket接続
        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws/analyze`);
            
            ws.onopen = () => {
                connectionStatus.textContent = '✅ 接続済み';
                connectionStatus.className = 'connection-status connected';
                startBtn.disabled = false;
                startCamera();
            };
            
            ws.onclose = () => {
                connectionStatus.textContent = '❌ 切断 - 再接続中...';
                connectionStatus.className = 'connection-status disconnected';
                startBtn.disabled = true;
                stopBtn.disabled = true;
                setTimeout(connect, 2000);
            };
            
            ws.onerror = (err) => {
                console.error('WebSocket error:', err);
            };
            
            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                handleMessage(message);
            };
        }
        
        // カメラ起動
        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: 'user',
                        width: { ideal: 640 },
                        height: { ideal: 480 }
                    },
                    audio: false
                });
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    statusEl.textContent = '分析中';
                    sendFrames();
                };
            } catch (err) {
                console.error('Camera error:', err);
                statusEl.textContent = 'カメラエラー';
            }
        }
        
        // フレーム送信ループ
        function sendFrames() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ctx.drawImage(video, 0, 0);
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                
                ws.send(JSON.stringify({
                    type: 'frame',
                    data: imageData
                }));
                
                // FPS計算
                frameCount++;
                const now = performance.now();
                if (now - lastTime >= 1000) {
                    fps = frameCount;
                    frameCount = 0;
                    lastTime = now;
                    fpsEl.textContent = `${fps} FPS`;
                }
            }
            
            // 15FPSで送信
            setTimeout(sendFrames, 66);
        }
        
        // メッセージ処理
        function handleMessage(message) {
            if (message.type === 'result') {
                updateResults(message.data);
                if (message.recording) {
                    statusEl.textContent = `⏺ 記録中 (${message.frame_count}f)`;
                    statusEl.className = 'status recording';
                } else {
                    statusEl.textContent = '分析中';
                    statusEl.className = 'status';
                }
            } else if (message.type === 'recording_started') {
                isRecording = true;
                startBtn.disabled = true;
                stopBtn.disabled = false;
            } else if (message.type === 'recording_stopped') {
                isRecording = false;
                startBtn.disabled = false;
                stopBtn.disabled = true;
                alert(`記録完了!\\n${message.total_frames}フレーム, ${message.duration.toFixed(1)}秒`);
            }
        }
        
        // 結果表示を更新
        function updateResults(data) {
            // 感情
            const emotionsEl = document.getElementById('emotions');
            if (data.emotions && data.emotions.length > 0) {
                emotionsEl.innerHTML = data.emotions.slice(0, 4).map(e => {
                    const barClass = e.confidence > 0.5 ? 'positive' : 'neutral';
                    return `
                        <div class="emotion-bar">
                            <span class="emotion-name">${e.emotion}</span>
                            <div class="bar-container">
                                <div class="bar ${barClass}" style="width: ${e.confidence * 100}%"></div>
                            </div>
                            <span class="confidence">${(e.confidence * 100).toFixed(0)}%</span>
                        </div>
                    `;
                }).join('');
            }
            
            // Valence/Arousal
            const valenceEl = document.getElementById('valence');
            const arousalEl = document.getElementById('arousal');
            valenceEl.textContent = (data.valence >= 0 ? '+' : '') + data.valence.toFixed(2);
            arousalEl.textContent = (data.arousal >= 0 ? '+' : '') + data.arousal.toFixed(2);
            valenceEl.style.color = data.valence >= 0 ? '#38ef7d' : '#f45c43';
            
            // FACSコード
            document.getElementById('facsCode').textContent = data.facs_code || '-';
            
            // AU一覧
            const auListEl = document.getElementById('auList');
            if (data.active_aus && data.active_aus.length > 0) {
                auListEl.innerHTML = data.active_aus.map(au => 
                    `<span class="au-tag">AU${au.au}: ${au.name}</span>`
                ).join('');
            } else {
                auListEl.innerHTML = '<span style="color: #888">検出なし</span>';
            }
        }
        
        // 記録開始
        startBtn.onclick = () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'start_recording',
                    width: canvas.width,
                    height: canvas.height,
                    fps: 15,
                    description: 'Mobile recording'
                }));
            }
        };
        
        // 記録停止
        stopBtn.onclick = () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'stop_recording'
                }));
            }
        };
        
        // 起動
        connect();
    </script>
</body>
</html>
"""
    
    def get_local_ip(self) -> str:
        """ローカルIPアドレスを取得"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, use_https: bool = False):
        """サーバーを起動"""
        import uvicorn
        
        local_ip = self.get_local_ip()
        protocol = "https" if use_https else "http"
        
        print("\n" + "=" * 50)
        print("🎭 FACS Web Server")
        print("=" * 50)
        print(f"\n📱 スマホからアクセス:")
        print(f"   {protocol}://{local_ip}:{port}")
        print(f"\n💻 このPCからアクセス:")
        print(f"   {protocol}://localhost:{port}")
        
        if not use_https:
            print(f"\n⚠️ HTTPではカメラアクセスが制限される場合があります")
        
        print("\n" + "=" * 50 + "\n")
        
        if use_https:
            cert_dir = Path(__file__).parent / "certs"
            cert_file, key_file = generate_ssl_cert(cert_dir)
            if cert_file and key_file:
                uvicorn.run(
                    self.app, host=host, port=port,
                    ssl_certfile=cert_file, ssl_keyfile=key_file,
                    log_level="info"
                )
                return
            print("❌ HTTPS起動に失敗。HTTPで起動します。")
        
        uvicorn.run(self.app, host=host, port=port, log_level="info")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FACS Web Server")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port number")
    parser.add_argument("--recordings", "-r", default="./recordings", help="Recordings directory")
    args = parser.parse_args()
    
    server = FACSWebServer(recordings_dir=args.recordings)
    server.run(port=args.port)


if __name__ == "__main__":
    main()
