/**
 * マイクロサッカード検出モジュール
 * 
 * 研究に基づく微小眼球運動分析：
 * - 潜在的注意（Covert Attention）の方向検出
 * - 脅威刺激に対する抑制（Freezing Response）
 * - 注意の集中度測定
 */

class MicrosaccadeDetector {
    constructor(options = {}) {
        this.config = {
            // 検出パラメータ
            velocityThreshold: 0.01,      // 速度閾値（正規化座標/フレーム）
            amplitudeMin: 0.001,          // 最小振幅
            amplitudeMax: 0.03,           // 最大振幅（これ以上はサッカード）
            minDurationMs: 10,            // 最小持続時間
            maxDurationMs: 100,           // 最大持続時間
            
            // 分析ウィンドウ
            analysisWindowMs: 1000,       // 分析ウィンドウ（1秒）
            historySize: 150,             // 履歴サイズ（約5秒@30fps）
            
            // サンプリング
            samplingRate: 30,
            
            ...options
        };
        
        // 視線位置の履歴
        this.gazeHistory = [];
        
        // 検出されたマイクロサッカード
        this.microsaccades = [];
        
        // 速度履歴
        this.velocityHistory = [];
        
        // 抑制状態の追跡
        this.inhibitionState = {
            isInhibited: false,
            startTime: null,
            baseline: 0
        };
        
        // 統計
        this.statistics = {
            rate: 0,              // 発生率（回/秒）
            avgAmplitude: 0,
            avgDirection: 0,      // 平均方向（ラジアン）
            directionBias: null   // 方向の偏り
        };
        
        // ベースライン
        this.baseline = {
            isCalibrated: false,
            normalRate: 1.5,      // 通常は1-2回/秒
            samples: [],
            calibrationTime: 5000 // 5秒
        };
        
        this.startTime = performance.now();
    }
    
    /**
     * フレームごとの更新
     * @param {Object} gazePosition - 視線位置 {x, y}（正規化座標 0-1）
     * @param {number} timestamp - タイムスタンプ
     */
    update(gazePosition, timestamp = performance.now()) {
        if (!gazePosition || gazePosition.x === undefined) {
            return null;
        }
        
        // 履歴に追加
        this.gazeHistory.push({
            x: gazePosition.x,
            y: gazePosition.y,
            timestamp: timestamp
        });
        
        // 古いデータを削除
        const cutoff = timestamp - this.config.historySize * (1000 / this.config.samplingRate);
        this.gazeHistory = this.gazeHistory.filter(g => g.timestamp > cutoff);
        
        // 速度を計算
        const velocity = this.calculateVelocity();
        if (velocity) {
            this.velocityHistory.push({
                ...velocity,
                timestamp: timestamp
            });
        }
        
        // 古い速度データを削除
        this.velocityHistory = this.velocityHistory.filter(v => v.timestamp > cutoff);
        
        // マイクロサッカード検出
        this.detectMicrosaccades(timestamp);
        
        // 古いマイクロサッカードを削除
        const msCutoff = timestamp - 10000; // 10秒保持
        this.microsaccades = this.microsaccades.filter(m => m.timestamp > msCutoff);
        
        // 統計更新
        this.updateStatistics(timestamp);
        
        // 抑制状態の更新
        this.updateInhibitionState(timestamp);
        
        // キャリブレーション
        if (!this.baseline.isCalibrated && 
            timestamp - this.startTime > this.baseline.calibrationTime) {
            this.calibrate();
        }
        
        return this.analyze(timestamp);
    }
    
    /**
     * 視線の速度を計算
     */
    calculateVelocity() {
        if (this.gazeHistory.length < 2) return null;
        
        const current = this.gazeHistory[this.gazeHistory.length - 1];
        const previous = this.gazeHistory[this.gazeHistory.length - 2];
        
        const dt = (current.timestamp - previous.timestamp) / 1000; // 秒
        if (dt <= 0) return null;
        
        const dx = current.x - previous.x;
        const dy = current.y - previous.y;
        
        const speed = Math.sqrt(dx * dx + dy * dy) / dt;
        const direction = Math.atan2(dy, dx);
        
        return {
            speed: speed,
            dx: dx / dt,
            dy: dy / dt,
            direction: direction
        };
    }
    
    /**
     * マイクロサッカードを検出
     */
    detectMicrosaccades(timestamp) {
        if (this.velocityHistory.length < 5) return;
        
        const recent = this.velocityHistory.slice(-10);
        
        // 速度のピークを検出
        for (let i = 2; i < recent.length - 2; i++) {
            const prev = recent[i - 1];
            const curr = recent[i];
            const next = recent[i + 1];
            
            // ピーク検出（前後より大きい）
            if (curr.speed > prev.speed && curr.speed > next.speed) {
                // 閾値チェック
                if (curr.speed > this.config.velocityThreshold) {
                    // 振幅を計算（移動距離）
                    const amplitude = this.calculateMovementAmplitude(i, recent);
                    
                    // マイクロサッカードの振幅範囲内かチェック
                    if (amplitude >= this.config.amplitudeMin && 
                        amplitude <= this.config.amplitudeMax) {
                        
                        // 重複チェック
                        const isDuplicate = this.microsaccades.some(
                            m => Math.abs(m.timestamp - curr.timestamp) < 100
                        );
                        
                        if (!isDuplicate) {
                            this.microsaccades.push({
                                timestamp: curr.timestamp,
                                amplitude: amplitude,
                                direction: curr.direction,
                                peakVelocity: curr.speed,
                                // 方向をカテゴリ化
                                directionCategory: this.categorizeDirection(curr.direction)
                            });
                        }
                    }
                }
            }
        }
    }
    
    /**
     * 移動の振幅を計算
     */
    calculateMovementAmplitude(peakIndex, velocityData) {
        // ピーク周辺の移動量を積分
        let totalDx = 0;
        let totalDy = 0;
        
        const start = Math.max(0, peakIndex - 2);
        const end = Math.min(velocityData.length, peakIndex + 3);
        
        for (let i = start; i < end; i++) {
            const v = velocityData[i];
            const dt = 1 / this.config.samplingRate;
            totalDx += v.dx * dt;
            totalDy += v.dy * dt;
        }
        
        return Math.sqrt(totalDx * totalDx + totalDy * totalDy);
    }
    
    /**
     * 方向をカテゴリ化
     */
    categorizeDirection(radians) {
        // -π to π を 8方向に分類
        const degrees = (radians * 180 / Math.PI + 360) % 360;
        
        if (degrees < 22.5 || degrees >= 337.5) return 'right';
        if (degrees < 67.5) return 'down-right';
        if (degrees < 112.5) return 'down';
        if (degrees < 157.5) return 'down-left';
        if (degrees < 202.5) return 'left';
        if (degrees < 247.5) return 'up-left';
        if (degrees < 292.5) return 'up';
        return 'up-right';
    }
    
    /**
     * 統計を更新
     */
    updateStatistics(timestamp) {
        const windowStart = timestamp - this.config.analysisWindowMs;
        const recentMS = this.microsaccades.filter(m => m.timestamp > windowStart);
        
        // 発生率（回/秒）
        this.statistics.rate = recentMS.length / (this.config.analysisWindowMs / 1000);
        
        if (recentMS.length > 0) {
            // 平均振幅
            this.statistics.avgAmplitude = this.calculateMean(
                recentMS.map(m => m.amplitude)
            );
            
            // 方向の偏りを計算
            this.statistics.directionBias = this.calculateDirectionBias(recentMS);
        }
    }
    
    /**
     * 方向の偏りを計算（潜在的注意の方向）
     */
    calculateDirectionBias(microsaccades) {
        if (microsaccades.length < 3) return null;
        
        // 方向ベクトルの平均
        let sumX = 0;
        let sumY = 0;
        
        microsaccades.forEach(m => {
            sumX += Math.cos(m.direction) * m.amplitude;
            sumY += Math.sin(m.direction) * m.amplitude;
        });
        
        const avgX = sumX / microsaccades.length;
        const avgY = sumY / microsaccades.length;
        
        const magnitude = Math.sqrt(avgX * avgX + avgY * avgY);
        const direction = Math.atan2(avgY, avgX);
        
        // 偏りが有意かどうか
        const isSignificant = magnitude > 0.001;
        
        return {
            x: avgX,
            y: avgY,
            magnitude: magnitude,
            direction: direction,
            directionCategory: this.categorizeDirection(direction),
            isSignificant: isSignificant,
            interpretation: isSignificant ? 
                `潜在的注意が「${this.categorizeDirection(direction)}」方向に向いています` :
                '明確な注意の偏りはありません'
        };
    }
    
    /**
     * 抑制状態を更新（Freezing Response）
     */
    updateInhibitionState(timestamp) {
        const recentRate = this.statistics.rate;
        
        // ベースラインの50%以下に低下 = 抑制
        const threshold = this.baseline.isCalibrated ? 
            this.baseline.normalRate * 0.5 : 0.75;
        
        if (recentRate < threshold) {
            if (!this.inhibitionState.isInhibited) {
                // 抑制開始
                this.inhibitionState = {
                    isInhibited: true,
                    startTime: timestamp,
                    baseline: this.baseline.normalRate
                };
            }
        } else {
            if (this.inhibitionState.isInhibited) {
                // 抑制終了
                this.inhibitionState.isInhibited = false;
            }
        }
    }
    
    /**
     * キャリブレーション
     */
    calibrate() {
        if (this.statistics.rate > 0) {
            this.baseline.normalRate = this.statistics.rate;
            this.baseline.isCalibrated = true;
            
            console.log('Microsaccade calibration complete:', {
                normalRate: this.baseline.normalRate.toFixed(2)
            });
        }
    }
    
    /**
     * 分析結果を生成
     */
    analyze(timestamp) {
        // 発生率の変化
        const rateChange = this.baseline.isCalibrated ?
            (this.statistics.rate - this.baseline.normalRate) / this.baseline.normalRate : 0;
        
        // 抑制の深さ
        const inhibitionDepth = this.inhibitionState.isInhibited ?
            1 - (this.statistics.rate / this.baseline.normalRate) : 0;
        
        // 抑制の持続時間
        const inhibitionDuration = this.inhibitionState.isInhibited ?
            timestamp - this.inhibitionState.startTime : 0;
        
        return {
            // 基本統計
            statistics: {
                rate: this.statistics.rate,
                avgAmplitude: this.statistics.avgAmplitude,
                count: this.microsaccades.filter(
                    m => m.timestamp > timestamp - 1000
                ).length
            },
            
            // 方向分析（潜在的注意）
            attention: {
                bias: this.statistics.directionBias,
                covertDirection: this.statistics.directionBias?.directionCategory || 'center',
                interpretation: this.statistics.directionBias?.interpretation || '分析中...'
            },
            
            // 抑制状態（すくみ反応）
            inhibition: {
                isInhibited: this.inhibitionState.isInhibited,
                depth: inhibitionDepth,
                duration: inhibitionDuration,
                interpretation: this.interpretInhibition(inhibitionDepth, inhibitionDuration)
            },
            
            // 発生率の変化
            rateChange: {
                value: rateChange,
                description: this.describeRateChange(rateChange)
            },
            
            // 最近のイベント
            recentEvents: this.microsaccades.slice(-5).map(m => ({
                timestamp: m.timestamp,
                direction: m.directionCategory,
                amplitude: m.amplitude
            })),
            
            // キャリブレーション状態
            calibration: {
                isCalibrated: this.baseline.isCalibrated,
                normalRate: this.baseline.normalRate
            }
        };
    }
    
    /**
     * 抑制状態の解釈
     */
    interpretInhibition(depth, duration) {
        if (!this.inhibitionState.isInhibited) {
            return '通常状態';
        }
        
        if (depth > 0.7) {
            return '強い抑制（脅威検出/すくみ反応の可能性）';
        } else if (depth > 0.4) {
            return '中程度の抑制（注意の集中/警戒状態）';
        } else {
            return '軽度の抑制';
        }
    }
    
    /**
     * 発生率変化の説明
     */
    describeRateChange(change) {
        if (change < -0.5) return '大幅に減少（抑制/すくみ反応）';
        if (change < -0.25) return 'やや減少';
        if (change > 0.5) return '大幅に増加（リバウンド/興奮）';
        if (change > 0.25) return 'やや増加';
        return '安定';
    }
    
    /**
     * 刺激イベントを記録（抑制分析用）
     */
    markStimulus(type = 'unknown', timestamp = performance.now()) {
        // 刺激後の抑制を追跡するための記録
        this.lastStimulus = {
            type: type,
            timestamp: timestamp,
            preRate: this.statistics.rate
        };
    }
    
    /**
     * リセット
     */
    reset() {
        this.gazeHistory = [];
        this.microsaccades = [];
        this.velocityHistory = [];
        this.inhibitionState = {
            isInhibited: false,
            startTime: null,
            baseline: 0
        };
        this.baseline.isCalibrated = false;
        this.startTime = performance.now();
    }
    
    // ユーティリティ
    calculateMean(arr) {
        if (arr.length === 0) return 0;
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }
}

// グローバルエクスポート
window.MicrosaccadeDetector = MicrosaccadeDetector;
