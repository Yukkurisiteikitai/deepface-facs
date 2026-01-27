/**
 * 画面視線追跡モジュール (Screen Gaze Tracker)
 * 
 * Webカメラの顔ランドマークから画面上の視線位置を推定
 * - キャリブレーションによる個人適応
 * - 頭の動きの補正
 * - マウスカーソル代替機能
 */

class ScreenGazeTracker {
    constructor(options = {}) {
        this.config = {
            // 画面サイズ（自動検出）
            screenWidth: window.innerWidth,
            screenHeight: window.innerHeight,
            
            // キャリブレーション設定
            calibrationPoints: 9,           // キャリブレーションポイント数
            samplesPerPoint: 30,            // 各ポイントで取得するサンプル数
            
            // スムージング
            smoothingFactor: 0.3,           // 低いほど滑らか（0-1）
            historySize: 5,                 // 移動平均のサイズ
            
            // 精度向上
            headPoseCorrection: true,       // 頭の傾き補正
            blinkFilter: true,              // 瞬き中は無視
            
            ...options
        };
        
        // キャリブレーションデータ
        this.calibration = {
            isCalibrated: false,
            points: [],                     // キャリブレーションポイント
            mappingCoefficients: null,      // 座標変換係数
            eyeToScreenMatrix: null         // 変換行列
        };
        
        // 現在の視線位置
        this.currentGaze = {
            x: this.config.screenWidth / 2,
            y: this.config.screenHeight / 2,
            confidence: 0,
            timestamp: 0
        };
        
        // 履歴（スムージング用）
        this.gazeHistory = [];
        
        // 頭部姿勢
        this.headPose = {
            pitch: 0,   // 上下
            yaw: 0,     // 左右
            roll: 0     // 傾き
        };
        
        // 基準となる目の位置
        this.referenceEye = {
            leftIris: null,
            rightIris: null,
            leftEyeCenter: null,
            rightEyeCenter: null
        };
        
        // キャリブレーション中のデータ収集
        this.calibrationSession = {
            isActive: false,
            currentPoint: 0,
            samples: [],
            targetPosition: null
        };
        
        // コールバック
        this.onGazeUpdate = options.onGazeUpdate || null;
        this.onCalibrationProgress = options.onCalibrationProgress || null;
    }
    
    /**
     * 顔ランドマークから視線を更新
     * @param {Array} landmarks - MediaPipeの顔ランドマーク
     * @param {Object} blendshapes - Blendshapes
     */
    update(landmarks, blendshapes = {}) {
        if (!landmarks || landmarks.length < 478) return null;
        
        // 瞬き検出（瞬き中は更新しない）
        if (this.config.blinkFilter && this.isBlinking(blendshapes)) {
            return this.currentGaze;
        }
        
        // 虹彩位置を取得
        const leftIris = landmarks[468];    // 左目虹彩中心
        const rightIris = landmarks[473];   // 右目虹彩中心
        
        // 目の中心を計算
        const leftEyeCenter = this.calculateEyeCenter(landmarks, 'left');
        const rightEyeCenter = this.calculateEyeCenter(landmarks, 'right');
        
        // 頭部姿勢を更新
        if (this.config.headPoseCorrection) {
            this.updateHeadPose(landmarks);
        }
        
        // 視線ベクトルを計算
        const gazeVector = this.calculateGazeVector(
            leftIris, rightIris, 
            leftEyeCenter, rightEyeCenter
        );
        
        // キャリブレーション中の場合
        if (this.calibrationSession.isActive) {
            this.collectCalibrationSample(gazeVector, {
                leftIris, rightIris, leftEyeCenter, rightEyeCenter
            });
            return this.currentGaze;
        }
        
        // 画面座標に変換
        let screenPosition;
        if (this.calibration.isCalibrated) {
            screenPosition = this.gazeToScreen(gazeVector);
        } else {
            // キャリブレーション前は単純なマッピング
            screenPosition = this.simpleMapping(gazeVector);
        }
        
        // スムージング
        screenPosition = this.smoothGaze(screenPosition);
        
        // 範囲制限
        screenPosition.x = Math.max(0, Math.min(this.config.screenWidth, screenPosition.x));
        screenPosition.y = Math.max(0, Math.min(this.config.screenHeight, screenPosition.y));
        
        // 更新
        this.currentGaze = {
            x: screenPosition.x,
            y: screenPosition.y,
            confidence: this.calculateConfidence(gazeVector),
            timestamp: performance.now(),
            raw: gazeVector
        };
        
        // コールバック
        if (this.onGazeUpdate) {
            this.onGazeUpdate(this.currentGaze);
        }
        
        return this.currentGaze;
    }
    
    /**
     * 目の中心を計算
     */
    calculateEyeCenter(landmarks, side) {
        const indices = side === 'left' ? 
            { inner: 362, outer: 263, upper: 386, lower: 374 } :
            { inner: 133, outer: 33, upper: 159, lower: 145 };
        
        return {
            x: (landmarks[indices.inner].x + landmarks[indices.outer].x) / 2,
            y: (landmarks[indices.upper].y + landmarks[indices.lower].y) / 2,
            z: (landmarks[indices.inner].z + landmarks[indices.outer].z) / 2
        };
    }
    
    /**
     * 視線ベクトルを計算
     */
    calculateGazeVector(leftIris, rightIris, leftCenter, rightCenter) {
        // 各目の視線オフセット（虹彩位置 - 目の中心）
        const leftOffset = {
            x: leftIris.x - leftCenter.x,
            y: leftIris.y - leftCenter.y
        };
        
        const rightOffset = {
            x: rightIris.x - rightCenter.x,
            y: rightIris.y - rightCenter.y
        };
        
        // 両目の平均
        const avgOffset = {
            x: (leftOffset.x + rightOffset.x) / 2,
            y: (leftOffset.y + rightOffset.y) / 2
        };
        
        // 頭部姿勢補正
        if (this.config.headPoseCorrection) {
            avgOffset.x -= this.headPose.yaw * 0.1;
            avgOffset.y -= this.headPose.pitch * 0.1;
        }
        
        return avgOffset;
    }
    
    /**
     * 頭部姿勢を更新
     */
    updateHeadPose(landmarks) {
        // 顔の主要ポイント
        const nose = landmarks[1];
        const leftEye = landmarks[33];
        const rightEye = landmarks[263];
        const chin = landmarks[152];
        const forehead = landmarks[10];
        
        // Yaw（左右）: 両目の中点と鼻の水平位置差
        const eyeCenter = (leftEye.x + rightEye.x) / 2;
        this.headPose.yaw = (nose.x - eyeCenter) * 2;
        
        // Pitch（上下）: 鼻と顎・額の垂直位置関係
        const faceHeight = chin.y - forehead.y;
        const noseRelative = (nose.y - forehead.y) / faceHeight;
        this.headPose.pitch = (noseRelative - 0.5) * 2;
        
        // Roll（傾き）: 両目の傾き
        this.headPose.roll = Math.atan2(
            rightEye.y - leftEye.y,
            rightEye.x - leftEye.x
        );
    }
    
    /**
     * 瞬き検出
     */
    isBlinking(blendshapes) {
        const leftBlink = blendshapes.eyeBlinkLeft || 0;
        const rightBlink = blendshapes.eyeBlinkRight || 0;
        return (leftBlink + rightBlink) / 2 > 0.5;
    }
    
    /**
     * 単純な視線→画面マッピング（キャリブレーション前）
     */
    simpleMapping(gazeVector) {
        // 視線オフセットを画面座標に線形マッピング
        // 視線が右を向く(x>0) → カーソルは右へ
        const sensitivity = 8000;
        
        return {
            x: this.config.screenWidth / 2 + gazeVector.x * sensitivity,
            y: this.config.screenHeight / 2 + gazeVector.y * sensitivity
        };
    }
    
    /**
     * キャリブレーション済みの視線→画面変換
     */
    gazeToScreen(gazeVector) {
        if (!this.calibration.mappingCoefficients) {
            return this.simpleMapping(gazeVector);
        }
        
        const coef = this.calibration.mappingCoefficients;
        
        // 多項式回帰モデル（2次）
        const gx = gazeVector.x;
        const gy = gazeVector.y;
        
        const screenX = coef.x.a * gx * gx + coef.x.b * gy * gy + 
                       coef.x.c * gx * gy + coef.x.d * gx + 
                       coef.x.e * gy + coef.x.f;
        
        const screenY = coef.y.a * gx * gx + coef.y.b * gy * gy + 
                       coef.y.c * gx * gy + coef.y.d * gx + 
                       coef.y.e * gy + coef.y.f;
        
        return { x: screenX, y: screenY };
    }
    
    /**
     * スムージング（移動平均）
     */
    smoothGaze(position) {
        this.gazeHistory.push(position);
        
        if (this.gazeHistory.length > this.config.historySize) {
            this.gazeHistory.shift();
        }
        
        // 加重移動平均
        let sumX = 0, sumY = 0, weightSum = 0;
        
        this.gazeHistory.forEach((pos, i) => {
            const weight = i + 1;
            sumX += pos.x * weight;
            sumY += pos.y * weight;
            weightSum += weight;
        });
        
        // 指数移動平均も適用
        const smoothed = {
            x: sumX / weightSum,
            y: sumY / weightSum
        };
        
        return {
            x: this.currentGaze.x * (1 - this.config.smoothingFactor) + 
               smoothed.x * this.config.smoothingFactor,
            y: this.currentGaze.y * (1 - this.config.smoothingFactor) + 
               smoothed.y * this.config.smoothingFactor
        };
    }
    
    /**
     * 信頼度を計算
     */
    calculateConfidence(gazeVector) {
        // 視線が極端な位置にないか
        const magnitude = Math.sqrt(gazeVector.x ** 2 + gazeVector.y ** 2);
        const positionConfidence = Math.max(0, 1 - magnitude * 10);
        
        // 履歴の安定性
        if (this.gazeHistory.length < 3) return positionConfidence * 0.5;
        
        const recent = this.gazeHistory.slice(-3);
        let variance = 0;
        for (let i = 1; i < recent.length; i++) {
            const dx = recent[i].x - recent[i-1].x;
            const dy = recent[i].y - recent[i-1].y;
            variance += dx * dx + dy * dy;
        }
        variance /= recent.length - 1;
        
        const stabilityConfidence = Math.max(0, 1 - variance / 10000);
        
        return (positionConfidence + stabilityConfidence) / 2;
    }
    
    // ==================== キャリブレーション ====================
    
    /**
     * キャリブレーションを開始
     */
    startCalibration() {
        this.calibrationSession = {
            isActive: true,
            currentPoint: 0,
            samples: [],
            allData: []
        };
        
        this.calibration.points = this.generateCalibrationPoints();
        
        return this.calibration.points[0];
    }
    
    /**
     * キャリブレーションポイントを生成（3x3グリッド）
     */
    generateCalibrationPoints() {
        const points = [];
        const margin = 50;
        const cols = 3;
        const rows = 3;
        
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                points.push({
                    x: margin + (this.config.screenWidth - 2 * margin) * col / (cols - 1),
                    y: margin + (this.config.screenHeight - 2 * margin) * row / (rows - 1)
                });
            }
        }
        
        return points;
    }
    
    /**
     * キャリブレーションサンプルを収集
     */
    collectCalibrationSample(gazeVector, eyeData) {
        this.calibrationSession.samples.push({
            gaze: gazeVector,
            eye: eyeData,
            target: this.calibration.points[this.calibrationSession.currentPoint]
        });
        
        // 十分なサンプルが集まったら次のポイントへ
        if (this.calibrationSession.samples.length >= this.config.samplesPerPoint) {
            // このポイントのデータを保存
            const targetPoint = this.calibration.points[this.calibrationSession.currentPoint];
            const avgGaze = this.averageSamples(this.calibrationSession.samples);
            
            this.calibrationSession.allData.push({
                target: targetPoint,
                gaze: avgGaze
            });
            
            this.calibrationSession.samples = [];
            this.calibrationSession.currentPoint++;
            
            // 進捗コールバック
            if (this.onCalibrationProgress) {
                this.onCalibrationProgress({
                    current: this.calibrationSession.currentPoint,
                    total: this.calibration.points.length,
                    progress: this.calibrationSession.currentPoint / this.calibration.points.length
                });
            }
            
            // 全ポイント完了
            if (this.calibrationSession.currentPoint >= this.calibration.points.length) {
                this.finishCalibration();
            }
        }
    }
    
    /**
     * サンプルの平均を計算
     */
    averageSamples(samples) {
        let sumX = 0, sumY = 0;
        samples.forEach(s => {
            sumX += s.gaze.x;
            sumY += s.gaze.y;
        });
        return {
            x: sumX / samples.length,
            y: sumY / samples.length
        };
    }
    
    /**
     * キャリブレーション完了
     */
    finishCalibration() {
        this.calibrationSession.isActive = false;
        
        // 回帰係数を計算
        this.calibration.mappingCoefficients = this.computeRegressionCoefficients(
            this.calibrationSession.allData
        );
        
        this.calibration.isCalibrated = true;
        
        console.log('Calibration complete:', this.calibration.mappingCoefficients);
    }
    
    /**
     * 回帰係数を計算（最小二乗法）
     */
    computeRegressionCoefficients(data) {
        // 簡易的な多項式回帰（実用的には正規方程式を解く）
        // ここでは線形回帰 + オフセットの近似
        
        let sumGX = 0, sumGY = 0, sumSX = 0, sumSY = 0;
        let sumGX2 = 0, sumGY2 = 0;
        let sumGXSX = 0, sumGYSX = 0, sumGXSY = 0, sumGYSY = 0;
        const n = data.length;
        
        data.forEach(d => {
            sumGX += d.gaze.x;
            sumGY += d.gaze.y;
            sumSX += d.target.x;
            sumSY += d.target.y;
            sumGX2 += d.gaze.x * d.gaze.x;
            sumGY2 += d.gaze.y * d.gaze.y;
            sumGXSX += d.gaze.x * d.target.x;
            sumGYSX += d.gaze.y * d.target.x;
            sumGXSY += d.gaze.x * d.target.y;
            sumGYSY += d.gaze.y * d.target.y;
        });
        
        // 線形回帰係数（簡易版）
        const avgGX = sumGX / n;
        const avgGY = sumGY / n;
        const avgSX = sumSX / n;
        const avgSY = sumSY / n;
        
        const varGX = sumGX2 / n - avgGX * avgGX;
        const varGY = sumGY2 / n - avgGY * avgGY;
        
        const covGXSX = sumGXSX / n - avgGX * avgSX;
        const covGYSX = sumGYSX / n - avgGY * avgSX;
        const covGXSY = sumGXSY / n - avgGX * avgSY;
        const covGYSY = sumGYSY / n - avgGY * avgSY;
        
        // 係数（簡易的な線形モデル）
        const slopeX_gx = varGX > 0.0001 ? covGXSX / varGX : 5000;
        const slopeX_gy = varGY > 0.0001 ? covGYSX / varGY : 0;
        const slopeY_gx = varGX > 0.0001 ? covGXSY / varGX : 0;
        const slopeY_gy = varGY > 0.0001 ? covGYSY / varGY : 5000;
        
        return {
            x: {
                a: 0, b: 0, c: 0,
                d: slopeX_gx,
                e: slopeX_gy,
                f: avgSX - slopeX_gx * avgGX - slopeX_gy * avgGY
            },
            y: {
                a: 0, b: 0, c: 0,
                d: slopeY_gx,
                e: slopeY_gy,
                f: avgSY - slopeY_gx * avgGX - slopeY_gy * avgGY
            }
        };
    }
    
    /**
     * 現在のキャリブレーションポイントを取得
     */
    getCurrentCalibrationPoint() {
        if (!this.calibrationSession.isActive) return null;
        return this.calibration.points[this.calibrationSession.currentPoint];
    }
    
    /**
     * キャリブレーションをリセット
     */
    resetCalibration() {
        this.calibration = {
            isCalibrated: false,
            points: [],
            mappingCoefficients: null
        };
        this.calibrationSession = {
            isActive: false,
            currentPoint: 0,
            samples: [],
            allData: []
        };
    }
    
    /**
     * 現在の視線位置を取得
     */
    getGaze() {
        return this.currentGaze;
    }
}

// グローバルエクスポート
window.ScreenGazeTracker = ScreenGazeTracker;
