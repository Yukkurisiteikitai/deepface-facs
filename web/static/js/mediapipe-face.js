/**
 * MediaPipe Face Landmarker - フロントエンドのみで完結する実装
 */

class MediaPipeFaceDetector {
    constructor(options = {}) {
        this.faceLandmarker = null;
        this.isInitialized = false;
        
        this.options = {
            numFaces: options.numFaces || 1,
            minDetectionConfidence: options.minDetectionConfidence || 0.5,
            minPresenceConfidence: options.minPresenceConfidence || 0.5,
            minTrackingConfidence: options.minTrackingConfidence || 0.5,
            outputFaceBlendshapes: true,
            ...options
        };
        
        this.onError = options.onError || console.error;
    }
    
    async initialize() {
        if (this.isInitialized) return;
        
        try {
            const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest');
            const { FaceLandmarker, FilesetResolver } = vision;
            
            const filesetResolver = await FilesetResolver.forVisionTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
            );
            
            this.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
                baseOptions: {
                    modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                    delegate: 'GPU'
                },
                runningMode: 'VIDEO',
                numFaces: this.options.numFaces,
                minFaceDetectionConfidence: this.options.minDetectionConfidence,
                minFacePresenceConfidence: this.options.minPresenceConfidence,
                minTrackingConfidence: this.options.minTrackingConfidence,
                outputFaceBlendshapes: true
            });
            
            this.isInitialized = true;
            console.log('MediaPipe Face Landmarker initialized');
        } catch (error) {
            this.onError('Failed to initialize MediaPipe:', error);
            throw error;
        }
    }
    
    detect(source, timestamp = performance.now()) {
        if (!this.isInitialized || !this.faceLandmarker) {
            throw new Error('Face Landmarker not initialized');
        }
        return this.processResults(this.faceLandmarker.detectForVideo(source, timestamp));
    }
    
    processResults(results) {
        const processed = {
            faceLandmarks: [],
            faceBlendshapes: [],
            timestamp: Date.now()
        };
        
        if (results.faceLandmarks) {
            processed.faceLandmarks = results.faceLandmarks.map(landmarks => 
                landmarks.map(point => ({ x: point.x, y: point.y, z: point.z }))
            );
        }
        
        if (results.faceBlendshapes) {
            processed.faceBlendshapes = results.faceBlendshapes.map(blendshapes => {
                const shapes = {};
                blendshapes.categories.forEach(cat => {
                    shapes[cat.categoryName] = cat.score;
                });
                return shapes;
            });
        }
        
        return processed;
    }
    
    close() {
        if (this.faceLandmarker) {
            this.faceLandmarker.close();
            this.faceLandmarker = null;
        }
        this.isInitialized = false;
    }
}


/**
 * FACS分析エンジン（フロントエンドのみ）
 */
class FACSAnalyzer {
    constructor() {
        // Blendshape -> AU マッピング
        this.blendshapeToAU = {
            'browInnerUp': { au: 'AU1', weight: 1.0 },
            'browOuterUpLeft': { au: 'AU2', weight: 0.5 },
            'browOuterUpRight': { au: 'AU2', weight: 0.5 },
            'browDownLeft': { au: 'AU4', weight: 0.5 },
            'browDownRight': { au: 'AU4', weight: 0.5 },
            'eyeWideLeft': { au: 'AU5', weight: 0.5 },
            'eyeWideRight': { au: 'AU5', weight: 0.5 },
            'cheekSquintLeft': { au: 'AU6', weight: 0.5 },
            'cheekSquintRight': { au: 'AU6', weight: 0.5 },
            'eyeSquintLeft': { au: 'AU7', weight: 0.5 },
            'eyeSquintRight': { au: 'AU7', weight: 0.5 },
            'noseSneerLeft': { au: 'AU9', weight: 0.5 },
            'noseSneerRight': { au: 'AU9', weight: 0.5 },
            'mouthUpperUpLeft': { au: 'AU10', weight: 0.5 },
            'mouthUpperUpRight': { au: 'AU10', weight: 0.5 },
            'mouthSmileLeft': { au: 'AU12', weight: 0.5 },
            'mouthSmileRight': { au: 'AU12', weight: 0.5 },
            'mouthDimpleLeft': { au: 'AU14', weight: 0.5 },
            'mouthDimpleRight': { au: 'AU14', weight: 0.5 },
            'mouthFrownLeft': { au: 'AU15', weight: 0.5 },
            'mouthFrownRight': { au: 'AU15', weight: 0.5 },
            'mouthLowerDownLeft': { au: 'AU16', weight: 0.5 },
            'mouthLowerDownRight': { au: 'AU16', weight: 0.5 },
            'mouthPressLeft': { au: 'AU17', weight: 0.5 },
            'mouthPressRight': { au: 'AU17', weight: 0.5 },
            'mouthPucker': { au: 'AU18', weight: 1.0 },
            'mouthStretchLeft': { au: 'AU20', weight: 0.5 },
            'mouthStretchRight': { au: 'AU20', weight: 0.5 },
            'mouthFunnel': { au: 'AU22', weight: 1.0 },
            'mouthRollUpper': { au: 'AU23', weight: 1.0 },
            'mouthRollLower': { au: 'AU24', weight: 1.0 },
            'jawOpen': { au: 'AU26', weight: 1.0 },
            'jawForward': { au: 'AU29', weight: 1.0 },
            'cheekPuff': { au: 'AU34', weight: 1.0 },
            'eyeBlinkLeft': { au: 'AU45', weight: 0.5 },
            'eyeBlinkRight': { au: 'AU45', weight: 0.5 },
            'eyeLookUpLeft': { au: 'AU61', weight: 0.5 },
            'eyeLookUpRight': { au: 'AU61', weight: 0.5 },
            'eyeLookDownLeft': { au: 'AU64', weight: 0.5 },
            'eyeLookDownRight': { au: 'AU64', weight: 0.5 },
            'eyeLookInLeft': { au: 'AU65', weight: 0.5 },
            'eyeLookInRight': { au: 'AU65', weight: 0.5 },
            'eyeLookOutLeft': { au: 'AU66', weight: 0.5 },
            'eyeLookOutRight': { au: 'AU66', weight: 0.5 }
        };
        
        // AU名称
        this.auNames = {
            'AU1': '眉内側挙上', 'AU2': '眉外側挙上', 'AU4': '眉下制',
            'AU5': '上瞼挙上', 'AU6': '頬挙上', 'AU7': '瞼緊張',
            'AU9': '鼻しわ', 'AU10': '上唇挙上', 'AU12': '口角挙上（笑顔）',
            'AU14': 'えくぼ', 'AU15': '口角下制', 'AU16': '下唇下制',
            'AU17': '顎挙上', 'AU18': '口すぼめ', 'AU20': '唇伸展',
            'AU22': '口漏斗', 'AU23': '唇緊張', 'AU24': '唇圧迫',
            'AU26': '顎下制', 'AU29': '顎突出', 'AU34': '頬膨張',
            'AU45': '瞬き', 'AU61': '視線上', 'AU64': '視線下',
            'AU65': '視線内', 'AU66': '視線外'
        };
        
        // 感情パターン
        this.emotionPatterns = {
            'happy': { aus: ['AU6', 'AU12'], threshold: 0.3 },
            'sad': { aus: ['AU1', 'AU4', 'AU15'], threshold: 0.25 },
            'angry': { aus: ['AU4', 'AU5', 'AU7', 'AU23'], threshold: 0.3 },
            'surprise': { aus: ['AU1', 'AU2', 'AU5', 'AU26'], threshold: 0.3 },
            'fear': { aus: ['AU1', 'AU2', 'AU4', 'AU5', 'AU20'], threshold: 0.25 },
            'disgust': { aus: ['AU9', 'AU10', 'AU17'], threshold: 0.3 },
            'contempt': { aus: ['AU12', 'AU14'], threshold: 0.3 }
        };
    }
    
    /**
     * Blendshapesから完全な分析結果を生成
     */
    analyze(blendshapes) {
        const actionUnits = this.blendshapesToAUs(blendshapes);
        const facsCode = this.generateFACSCode(actionUnits);
        const emotion = this.estimateEmotion(actionUnits);
        const { valence, arousal } = this.calculateValenceArousal(actionUnits);
        
        return {
            actionUnits,
            facsCode,
            emotion,
            valence,
            arousal
        };
    }
    
    /**
     * BlendshapesをAction Unitsに変換
     */
    blendshapesToAUs(blendshapes) {
        const aus = {};
        
        for (const [bsName, mapping] of Object.entries(this.blendshapeToAU)) {
            if (blendshapes[bsName] !== undefined) {
                const value = blendshapes[bsName] * mapping.weight;
                const auCode = mapping.au;
                aus[auCode] = Math.max(aus[auCode] || 0, value);
            }
        }
        
        // 閾値フィルタリング
        const filtered = {};
        for (const [au, value] of Object.entries(aus)) {
            if (value > 0.05) {
                filtered[au] = value;
            }
        }
        
        return filtered;
    }
    
    /**
     * FACSコードを生成
     */
    generateFACSCode(aus) {
        if (Object.keys(aus).length === 0) return '---';
        
        const sorted = Object.entries(aus)
            .filter(([, v]) => v >= 0.2)
            .sort((a, b) => {
                const numA = parseInt(a[0].replace('AU', ''));
                const numB = parseInt(b[0].replace('AU', ''));
                return numA - numB;
            });
        
        if (sorted.length === 0) return '---';
        
        const codes = sorted.map(([au, intensity]) => {
            let level;
            if (intensity >= 0.8) level = 'E';
            else if (intensity >= 0.6) level = 'D';
            else if (intensity >= 0.4) level = 'C';
            else if (intensity >= 0.3) level = 'B';
            else level = 'A';
            return `${au}${level}`;
        });
        
        return codes.join('+');
    }
    
    /**
     * 感情を推定
     */
    estimateEmotion(aus) {
        let bestEmotion = 'neutral';
        let bestScore = 0.2;
        
        for (const [emotion, pattern] of Object.entries(this.emotionPatterns)) {
            let score = 0;
            let count = 0;
            
            for (const au of pattern.aus) {
                if (aus[au]) {
                    score += aus[au];
                    count++;
                }
            }
            
            if (count > 0) {
                const avgScore = score / pattern.aus.length;
                if (avgScore > bestScore && avgScore >= pattern.threshold) {
                    bestScore = avgScore;
                    bestEmotion = emotion;
                }
            }
        }
        
        return {
            name: bestEmotion,
            confidence: Math.min(bestScore * 1.5, 1.0)
        };
    }
    
    /**
     * Valence/Arousalを計算
     */
    calculateValenceArousal(aus) {
        // Valence: ポジティブ - ネガティブ
        const positive = (aus['AU6'] || 0) + (aus['AU12'] || 0);
        const negative = (aus['AU4'] || 0) + (aus['AU15'] || 0) + (aus['AU9'] || 0);
        let valence = (positive - negative) / 2;
        valence = Math.max(-1, Math.min(1, valence));
        
        // Arousal: 活性度
        const arousalAUs = [
            aus['AU1'] || 0,
            aus['AU2'] || 0,
            aus['AU5'] || 0,
            aus['AU26'] || 0,
            aus['AU20'] || 0
        ];
        let arousal = arousalAUs.reduce((a, b) => a + b, 0) / arousalAUs.length;
        arousal = Math.max(0, Math.min(1, arousal));
        
        return {
            valence: Math.round(valence * 1000) / 1000,
            arousal: Math.round(arousal * 1000) / 1000
        };
    }
    
    /**
     * AU名を取得
     */
    getAUName(auCode) {
        return this.auNames[auCode] || '';
    }
}


/**
 * リアルタイム顔分析（フロントエンドのみ）
 */
class RealtimeFaceAnalyzer {
    constructor(options = {}) {
        this.detector = new MediaPipeFaceDetector(options);
        this.facsAnalyzer = new FACSAnalyzer();
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.isRunning = false;
        this.animationId = null;
        
        this.drawLandmarks = options.drawLandmarks !== false;
        this.drawBlendshapes = options.drawBlendshapes !== false;
        this.drawConnections = options.drawConnections !== false;
        
        this.onFrame = options.onFrame || null;
        this.onAnalysis = options.onAnalysis || null;
        this.onError = options.onError || console.error;
        
        this.lastFps = 0;
        this.frameCount = 0;
        this.lastFpsUpdate = 0;
    }
    
    async start(videoElement, canvasElement) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        
        await this.detector.initialize();
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
            });
            
            this.video.srcObject = stream;
            await this.video.play();
            
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            
            this.isRunning = true;
            this.lastFpsUpdate = performance.now();
            this.processFrame();
        } catch (error) {
            this.onError('Failed to access camera:', error);
            throw error;
        }
    }
    
    stop() {
        this.isRunning = false;
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        if (this.video && this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
        }
        
        this.detector.close();
    }
    
    async processFrame() {
        if (!this.isRunning) return;
        
        const startTime = performance.now();
        
        try {
            const results = this.detector.detect(this.video, startTime);
            
            // Canvas描画
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.drawImage(this.video, 0, 0);
            
            let analysis = null;
            
            if (results.faceLandmarks.length > 0) {
                // ランドマーク描画
                if (this.drawLandmarks) {
                    this.drawFaceLandmarks(results.faceLandmarks[0]);
                }
                
                // FACS分析
                if (results.faceBlendshapes.length > 0) {
                    analysis = this.facsAnalyzer.analyze(results.faceBlendshapes[0]);
                    
                    if (this.drawBlendshapes) {
                        this.drawBlendshapePanel(results.faceBlendshapes[0]);
                    }
                    
                    if (this.onAnalysis) {
                        this.onAnalysis(analysis);
                    }
                }
            }
            
            // FPS計算と表示
            this.frameCount++;
            const now = performance.now();
            if (now - this.lastFpsUpdate >= 1000) {
                this.lastFps = Math.round(this.frameCount * 1000 / (now - this.lastFpsUpdate));
                this.frameCount = 0;
                this.lastFpsUpdate = now;
            }
            this.drawFPS();
            
            if (this.onFrame) {
                this.onFrame(results, analysis);
            }
        } catch (error) {
            this.onError('Frame processing error:', error);
        }
        
        this.animationId = requestAnimationFrame(() => this.processFrame());
    }
    
    drawFaceLandmarks(landmarks) {
        // ポイント描画
        this.ctx.fillStyle = '#00FF00';
        landmarks.forEach(point => {
            const x = point.x * this.canvas.width;
            const y = point.y * this.canvas.height;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 1, 0, 2 * Math.PI);
            this.ctx.fill();
        });
        
        // 接続線描画
        if (this.drawConnections) {
            this.drawFaceConnections(landmarks);
        }
    }
    
    drawFaceConnections(landmarks) {
        const connections = [
            // 顔輪郭
            [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
            // 左目
            [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362],
            // 右目
            [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33],
            // 左眉
            [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
            // 右眉
            [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
            // 唇外側
            [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61],
            // 唇内側
            [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78],
            // 鼻
            [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
        ];
        
        this.ctx.strokeStyle = 'rgba(0, 255, 255, 0.5)';
        this.ctx.lineWidth = 1;
        
        connections.forEach(connection => {
            this.ctx.beginPath();
            connection.forEach((index, i) => {
                if (index < landmarks.length) {
                    const x = landmarks[index].x * this.canvas.width;
                    const y = landmarks[index].y * this.canvas.height;
                    i === 0 ? this.ctx.moveTo(x, y) : this.ctx.lineTo(x, y);
                }
            });
            this.ctx.stroke();
        });
    }
    
    drawBlendshapePanel(blendshapes) {
        const important = [
            'browInnerUp', 'browDownLeft', 'eyeBlinkLeft', 'eyeWideLeft',
            'cheekSquintLeft', 'noseSneerLeft', 'mouthSmileLeft', 'mouthFrownLeft',
            'jawOpen', 'mouthPucker'
        ];
        
        const panelHeight = important.length * 14 + 25;
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
        this.ctx.fillRect(8, 8, 200, panelHeight);
        
        this.ctx.fillStyle = '#00FFFF';
        this.ctx.font = 'bold 11px monospace';
        this.ctx.fillText('Blendshapes', 12, 20);
        
        this.ctx.font = '10px monospace';
        important.forEach((name, i) => {
            const value = blendshapes[name] || 0;
            const y = 35 + i * 14;
            const shortName = name.replace(/([A-Z])/g, ' $1').trim().substring(0, 12);
            
            this.ctx.fillStyle = '#999';
            this.ctx.fillText(shortName, 12, y);
            
            // バー
            const barWidth = Math.max(0, value * 50);
            this.ctx.fillStyle = value > 0.5 ? '#00FF00' : value > 0.2 ? '#FFFF00' : '#444';
            this.ctx.fillRect(100, y - 8, barWidth, 10);
            
            // 値
            this.ctx.fillStyle = '#FFF';
            this.ctx.fillText((value * 100).toFixed(0) + '%', 160, y);
        });
    }
    
    drawFPS() {
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        this.ctx.fillRect(this.canvas.width - 55, 8, 47, 20);
        this.ctx.fillStyle = '#00FF00';
        this.ctx.font = 'bold 12px monospace';
        this.ctx.fillText(`${this.lastFps} FPS`, this.canvas.width - 50, 22);
    }
}

// グローバルエクスポート
window.MediaPipeFaceDetector = MediaPipeFaceDetector;
window.FACSAnalyzer = FACSAnalyzer;
window.RealtimeFaceAnalyzer = RealtimeFaceAnalyzer;
