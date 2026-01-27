/**
 * 瞳孔計測（Pupillometry）モジュール
 * 
 * 最新研究（2024-2026）に基づく瞳孔反応分析
 * - 感情価（Valence）の分離
 * - 嫌悪反応の特異的縮動パターン
 * - 恐怖の時間的ダイナミクス
 * - 認知負荷の推定
 */

class Pupillometry {
    constructor(options = {}) {
        // 設定
        this.config = {
            baselineWindowMs: 500,      // ベースライン計算のウィンドウ（ms）
            responseLagMs: 200,         // 反応遅延（ms）
            historySize: 90,            // 履歴サイズ（約3秒@30fps）
            samplingRate: 30,           // サンプリングレート（fps）
            ...options
        };
        
        // 瞳孔径の履歴
        this.history = {
            left: [],
            right: [],
            timestamps: []
        };
        
        // ベースライン
        this.baseline = {
            left: null,
            right: null,
            isCalibrated: false,
            calibrationSamples: [],
            targetSamples: 60  // 約2秒分
        };
        
        // 反応分析用
        this.responseAnalysis = {
            lastStimulus: null,
            dilationOnset: null,
            peakDilation: 0,
            peakTime: 0,
            responseLatency: 0,
            duration: 0
        };
        
        // 統計値
        this.statistics = {
            mean: { left: 0, right: 0 },
            std: { left: 0, right: 0 },
            min: { left: 1, right: 1 },
            max: { left: 0, right: 0 }
        };
    }
    
    /**
     * 瞳孔データを更新
     * @param {Object} irisData - 虹彩/瞳孔データ { left: {diameter, center}, right: {...} }
     * @param {number} timestamp - タイムスタンプ
     */
    update(irisData, timestamp = performance.now()) {
        const leftDiameter = this.estimatePupilDiameter(irisData.left);
        const rightDiameter = this.estimatePupilDiameter(irisData.right);
        
        // 履歴に追加
        this.history.left.push(leftDiameter);
        this.history.right.push(rightDiameter);
        this.history.timestamps.push(timestamp);
        
        // 古いデータを削除
        while (this.history.left.length > this.config.historySize) {
            this.history.left.shift();
            this.history.right.shift();
            this.history.timestamps.shift();
        }
        
        // ベースラインキャリブレーション
        if (!this.baseline.isCalibrated) {
            this.updateCalibration(leftDiameter, rightDiameter);
        }
        
        // 統計値更新
        this.updateStatistics();
        
        return this.analyze(timestamp);
    }
    
    /**
     * 瞳孔径を推定（虹彩データから）
     */
    estimatePupilDiameter(eyeData) {
        if (!eyeData || !eyeData.irisPoints) {
            return eyeData?.diameter || 0;
        }
        
        // 虹彩の直径から瞳孔径を推定
        // 通常、瞳孔は虹彩の約30-60%
        const irisDiameter = eyeData.diameter || 0;
        
        // 明るさ情報があれば光反射を考慮
        const lightCompensation = eyeData.brightness ? 
            this.compensateForLight(eyeData.brightness) : 1.0;
        
        return irisDiameter * lightCompensation;
    }
    
    /**
     * 光反射の補正
     * PLR (Pupil Light Reflex) の影響を軽減
     */
    compensateForLight(brightness) {
        // 明るいと瞳孔は縮小するため、補正係数を適用
        // 0.5-1.5の範囲で補正
        const normalizedBrightness = Math.max(0, Math.min(1, brightness));
        return 1.0 + (0.5 - normalizedBrightness) * 0.5;
    }
    
    /**
     * キャリブレーション更新
     */
    updateCalibration(leftDiameter, rightDiameter) {
        this.baseline.calibrationSamples.push({
            left: leftDiameter,
            right: rightDiameter
        });
        
        if (this.baseline.calibrationSamples.length >= this.baseline.targetSamples) {
            const leftValues = this.baseline.calibrationSamples.map(s => s.left);
            const rightValues = this.baseline.calibrationSamples.map(s => s.right);
            
            this.baseline.left = this.calculateMean(leftValues);
            this.baseline.right = this.calculateMean(rightValues);
            this.baseline.isCalibrated = true;
            
            console.log('Pupillometry calibration complete:', {
                leftBaseline: this.baseline.left.toFixed(4),
                rightBaseline: this.baseline.right.toFixed(4)
            });
        }
    }
    
    /**
     * 統計値を更新
     */
    updateStatistics() {
        if (this.history.left.length < 5) return;
        
        const recentLeft = this.history.left.slice(-30);
        const recentRight = this.history.right.slice(-30);
        
        this.statistics.mean.left = this.calculateMean(recentLeft);
        this.statistics.mean.right = this.calculateMean(recentRight);
        this.statistics.std.left = this.calculateStd(recentLeft);
        this.statistics.std.right = this.calculateStd(recentRight);
        this.statistics.min.left = Math.min(...recentLeft);
        this.statistics.min.right = Math.min(...recentRight);
        this.statistics.max.left = Math.max(...recentLeft);
        this.statistics.max.right = Math.max(...recentRight);
    }
    
    /**
     * 瞳孔分析を実行
     */
    analyze(currentTime) {
        if (this.history.left.length < 10) {
            return null;
        }
        
        const avgDiameter = (this.statistics.mean.left + this.statistics.mean.right) / 2;
        
        // ベースラインからの変化量
        const baselineChange = this.baseline.isCalibrated ?
            this.calculateBaselineChange() : { left: 0, right: 0, average: 0 };
        
        // 瞳孔反応の分類
        const response = this.classifyResponse(baselineChange.average);
        
        // 時間的ダイナミクスの分析
        const dynamics = this.analyzeTemporalDynamics();
        
        // 感情価の推定（研究に基づく）
        const emotionalState = this.estimateEmotionalState(response, dynamics);
        
        // 認知負荷の推定
        const cognitiveLoad = this.estimateCognitiveLoad();
        
        return {
            // 基本計測値
            diameter: {
                left: this.statistics.mean.left,
                right: this.statistics.mean.right,
                average: avgDiameter
            },
            
            // ベースラインからの変化
            change: baselineChange,
            
            // 瞳孔反応タイプ
            response: response,
            
            // 時間的特性
            dynamics: dynamics,
            
            // 推定される感情状態
            emotional: emotionalState,
            
            // 認知負荷
            cognitiveLoad: cognitiveLoad,
            
            // キャリブレーション状態
            calibration: {
                isCalibrated: this.baseline.isCalibrated,
                progress: Math.min(1, this.baseline.calibrationSamples.length / this.baseline.targetSamples),
                baseline: this.baseline.isCalibrated ? 
                    (this.baseline.left + this.baseline.right) / 2 : null
            }
        };
    }
    
    /**
     * ベースラインからの変化を計算
     */
    calculateBaselineChange() {
        if (!this.baseline.isCalibrated) {
            return { left: 0, right: 0, average: 0 };
        }
        
        const leftChange = (this.statistics.mean.left - this.baseline.left) / this.baseline.left;
        const rightChange = (this.statistics.mean.right - this.baseline.right) / this.baseline.right;
        
        return {
            left: leftChange,
            right: rightChange,
            average: (leftChange + rightChange) / 2
        };
    }
    
    /**
     * 瞳孔反応を分類
     * 研究に基づく：散大/縮動/維持
     */
    classifyResponse(change) {
        const dilationThreshold = 0.05;    // 5%以上の増加 = 散大
        const constrictionThreshold = -0.03; // 3%以上の減少 = 縮動
        
        if (change > dilationThreshold) {
            return {
                type: 'dilation',
                magnitude: change,
                label: '散大',
                description: '交感神経優位：覚醒、興奮、認知的努力、恐怖など'
            };
        } else if (change < constrictionThreshold) {
            return {
                type: 'constriction',
                magnitude: Math.abs(change),
                label: '縮動',
                description: '副交感神経優位：嫌悪反応、知覚的防衛、リラックスなど'
            };
        } else {
            return {
                type: 'stable',
                magnitude: Math.abs(change),
                label: '安定',
                description: 'ベースライン維持'
            };
        }
    }
    
    /**
     * 時間的ダイナミクスを分析
     * 恐怖vs悲しみの区別に使用
     */
    analyzeTemporalDynamics() {
        if (this.history.left.length < 30) {
            return { latency: 0, duration: 0, trend: 'stable' };
        }
        
        const recent = this.history.left.slice(-30);
        const older = this.history.left.slice(-60, -30);
        
        if (older.length < 10) {
            return { latency: 0, duration: 0, trend: 'stable' };
        }
        
        const recentMean = this.calculateMean(recent);
        const olderMean = this.calculateMean(older);
        const change = recentMean - olderMean;
        
        // トレンド判定
        let trend = 'stable';
        if (change > 0.01) trend = 'increasing';
        else if (change < -0.01) trend = 'decreasing';
        
        // 変化の速度（反応潜時の指標）
        const velocity = this.calculateChangeVelocity();
        
        return {
            trend: trend,
            velocity: velocity,
            // 遅延反応（恐怖の特徴）vs 即時反応
            isDelayed: velocity < 0.02,
            // 持続時間の長さ（悲しみの特徴）
            isProlonged: this.isProlongedResponse()
        };
    }
    
    /**
     * 変化速度を計算
     */
    calculateChangeVelocity() {
        if (this.history.left.length < 10) return 0;
        
        const recent = this.history.left.slice(-10);
        let totalChange = 0;
        
        for (let i = 1; i < recent.length; i++) {
            totalChange += Math.abs(recent[i] - recent[i-1]);
        }
        
        return totalChange / recent.length;
    }
    
    /**
     * 持続的な反応かどうか
     */
    isProlongedResponse() {
        if (this.history.left.length < 60) return false;
        
        const threshold = 0.03;
        const baselineDeviation = this.calculateBaselineChange().average;
        
        // 一定期間以上ベースラインから外れているか
        return Math.abs(baselineDeviation) > threshold;
    }
    
    /**
     * 感情状態を推定
     * 研究に基づく瞳孔反応パターン
     */
    estimateEmotionalState(response, dynamics) {
        const state = {
            arousal: 0,           // 覚醒度 (0-1)
            valence: 0,           // 感情価 (-1 to 1)
            dominantEmotion: null,
            confidence: 0
        };
        
        // 覚醒度：瞳孔変化の絶対量
        state.arousal = Math.min(1, response.magnitude * 10);
        
        // 感情価の推定
        if (response.type === 'constriction') {
            // 縮動 = 嫌悪反応の可能性
            state.valence = -0.5;
            state.dominantEmotion = 'disgust';
            state.confidence = Math.min(0.8, response.magnitude * 15);
        } else if (response.type === 'dilation') {
            if (dynamics.isDelayed) {
                // 遅延散大 = 恐怖の可能性
                state.valence = -0.3;
                state.dominantEmotion = 'fear';
                state.confidence = Math.min(0.7, response.magnitude * 10);
            } else if (dynamics.isProlonged) {
                // 持続的散大 = 悲しみの可能性
                state.valence = -0.4;
                state.dominantEmotion = 'sadness';
                state.confidence = Math.min(0.6, response.magnitude * 8);
            } else {
                // 即時散大 = 興奮/興味の可能性
                state.valence = 0.2;
                state.dominantEmotion = 'interest';
                state.confidence = Math.min(0.6, response.magnitude * 8);
            }
        } else {
            state.dominantEmotion = 'neutral';
            state.confidence = 0.5;
        }
        
        return state;
    }
    
    /**
     * 認知負荷を推定
     * LC-NE系の活性化指標
     */
    estimateCognitiveLoad() {
        if (!this.baseline.isCalibrated) {
            return { level: 'unknown', value: 0 };
        }
        
        const change = this.calculateBaselineChange().average;
        const velocity = this.calculateChangeVelocity();
        
        // 認知負荷指標：変化量と変動性の組み合わせ
        const loadIndex = Math.abs(change) * 0.7 + velocity * 0.3;
        
        let level = 'low';
        if (loadIndex > 0.1) level = 'high';
        else if (loadIndex > 0.05) level = 'medium';
        
        return {
            level: level,
            value: Math.min(1, loadIndex * 10),
            description: level === 'high' ? 
                '高い認知的努力が検出されました' :
                level === 'medium' ? 
                '中程度の認知負荷' : 
                '低い認知負荷（リラックス状態）'
        };
    }
    
    /**
     * 刺激イベントを記録（実験用）
     */
    markStimulus(type, timestamp = performance.now()) {
        this.responseAnalysis.lastStimulus = {
            type: type,
            timestamp: timestamp,
            baselineAtEvent: this.calculateBaselineChange().average
        };
    }
    
    /**
     * リセット
     */
    reset() {
        this.history = { left: [], right: [], timestamps: [] };
        this.baseline = {
            left: null,
            right: null,
            isCalibrated: false,
            calibrationSamples: [],
            targetSamples: 60
        };
    }
    
    // ユーティリティ関数
    calculateMean(arr) {
        if (arr.length === 0) return 0;
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }
    
    calculateStd(arr) {
        if (arr.length < 2) return 0;
        const mean = this.calculateMean(arr);
        const squareDiffs = arr.map(v => Math.pow(v - mean, 2));
        return Math.sqrt(this.calculateMean(squareDiffs));
    }
}

// グローバルエクスポート
window.Pupillometry = Pupillometry;
