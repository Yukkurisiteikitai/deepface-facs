/**
 * 瞬き分析モジュール
 * 
 * 研究に基づく瞬き特性分析：
 * - 瞬目率（Blink Rate）と不安の相関
 * - 瞬目振幅（Blink Amplitude）と状態不安
 * - 瞬目同期（Blink Synchronization）と没入感
 * - ドーパミン系活性の推定
 */

class BlinkAnalyzer {
    constructor(options = {}) {
        this.config = {
            // 瞬き検出閾値
            blinkThreshold: 0.3,          // 目の開き具合がこれ以下で瞬きと判定
            minBlinkDurationMs: 50,       // 最小瞬き持続時間
            maxBlinkDurationMs: 400,      // 最大瞬き持続時間
            
            // 分析ウィンドウ
            rateWindowMs: 60000,          // 瞬目率計算のウィンドウ（1分）
            amplitudeWindowSize: 10,      // 振幅計算に使用する直近の瞬き数
            
            // サンプリング
            samplingRate: 30,             // fps
            
            ...options
        };
        
        // 瞬きイベントの履歴
        this.blinkHistory = [];
        
        // 現在の瞬き状態
        this.currentBlink = {
            isBlinking: false,
            startTime: null,
            minOpenness: 1,
            startOpenness: 0
        };
        
        // 連続的な目の開き具合の履歴
        this.opennessHistory = [];
        
        // 統計
        this.statistics = {
            rate: 0,                // 瞬目率（回/分）
            avgAmplitude: 0,        // 平均振幅
            avgDuration: 0,         // 平均持続時間
            rhythmRegularity: 0     // リズムの規則性
        };
        
        // 基準値（個人差対応）
        this.baseline = {
            isCalibrated: false,
            normalRate: 15,         // 通常の瞬目率（回/分）
            normalAmplitude: 0.8,   // 通常の振幅
            samples: [],
            calibrationDuration: 30000  // 30秒
        };
        
        this.startTime = performance.now();
    }
    
    /**
     * フレームごとの更新
     * @param {number} eyeOpenness - 目の開き具合 (0-1)
     * @param {number} timestamp - タイムスタンプ
     */
    update(eyeOpenness, timestamp = performance.now()) {
        // 履歴に追加
        this.opennessHistory.push({
            openness: eyeOpenness,
            timestamp: timestamp
        });
        
        // 古いデータを削除（直近5秒分を保持）
        const cutoff = timestamp - 5000;
        this.opennessHistory = this.opennessHistory.filter(h => h.timestamp > cutoff);
        
        // 瞬き検出
        this.detectBlink(eyeOpenness, timestamp);
        
        // 古い瞬きイベントを削除
        const rateCutoff = timestamp - this.config.rateWindowMs;
        this.blinkHistory = this.blinkHistory.filter(b => b.endTime > rateCutoff);
        
        // 統計更新
        this.updateStatistics(timestamp);
        
        // キャリブレーション
        if (!this.baseline.isCalibrated && 
            timestamp - this.startTime > this.baseline.calibrationDuration) {
            this.calibrate();
        }
        
        return this.analyze(timestamp);
    }
    
    /**
     * 瞬きを検出
     */
    detectBlink(eyeOpenness, timestamp) {
        const wasBlinking = this.currentBlink.isBlinking;
        const isNowBlinking = eyeOpenness < this.config.blinkThreshold;
        
        if (!wasBlinking && isNowBlinking) {
            // 瞬き開始
            this.currentBlink = {
                isBlinking: true,
                startTime: timestamp,
                minOpenness: eyeOpenness,
                startOpenness: this.opennessHistory.length > 1 ? 
                    this.opennessHistory[this.opennessHistory.length - 2].openness : 1
            };
        } else if (wasBlinking && isNowBlinking) {
            // 瞬き継続中
            this.currentBlink.minOpenness = Math.min(
                this.currentBlink.minOpenness, 
                eyeOpenness
            );
        } else if (wasBlinking && !isNowBlinking) {
            // 瞬き終了
            const duration = timestamp - this.currentBlink.startTime;
            
            // 有効な瞬きかチェック
            if (duration >= this.config.minBlinkDurationMs && 
                duration <= this.config.maxBlinkDurationMs) {
                
                // 振幅を計算（開始時の開き具合 - 最小値）
                const amplitude = this.currentBlink.startOpenness - this.currentBlink.minOpenness;
                
                this.blinkHistory.push({
                    startTime: this.currentBlink.startTime,
                    endTime: timestamp,
                    duration: duration,
                    amplitude: amplitude,
                    minOpenness: this.currentBlink.minOpenness,
                    // 瞬きの深さ（完全に閉じたか）
                    isComplete: this.currentBlink.minOpenness < 0.1,
                    // 瞬きの速さ
                    velocity: amplitude / duration
                });
            }
            
            this.currentBlink.isBlinking = false;
        }
    }
    
    /**
     * 統計を更新
     */
    updateStatistics(timestamp) {
        const recentBlinks = this.blinkHistory.filter(
            b => b.endTime > timestamp - 60000  // 直近1分
        );
        
        // 瞬目率（回/分）
        this.statistics.rate = recentBlinks.length;
        
        if (recentBlinks.length > 0) {
            // 平均振幅
            this.statistics.avgAmplitude = this.calculateMean(
                recentBlinks.map(b => b.amplitude)
            );
            
            // 平均持続時間
            this.statistics.avgDuration = this.calculateMean(
                recentBlinks.map(b => b.duration)
            );
            
            // リズムの規則性（瞬き間隔の標準偏差）
            if (recentBlinks.length > 2) {
                const intervals = [];
                for (let i = 1; i < recentBlinks.length; i++) {
                    intervals.push(recentBlinks[i].startTime - recentBlinks[i-1].endTime);
                }
                const meanInterval = this.calculateMean(intervals);
                const stdInterval = this.calculateStd(intervals);
                // 変動係数（低いほど規則的）
                this.statistics.rhythmRegularity = meanInterval > 0 ? 
                    1 - Math.min(1, stdInterval / meanInterval) : 0;
            }
        }
    }
    
    /**
     * キャリブレーション
     */
    calibrate() {
        if (this.statistics.rate > 0) {
            this.baseline.normalRate = this.statistics.rate;
            this.baseline.normalAmplitude = this.statistics.avgAmplitude || 0.8;
            this.baseline.isCalibrated = true;
            
            console.log('Blink calibration complete:', {
                normalRate: this.baseline.normalRate,
                normalAmplitude: this.baseline.normalAmplitude.toFixed(3)
            });
        }
    }
    
    /**
     * 分析結果を生成
     */
    analyze(timestamp) {
        // 瞬目率の変化
        const rateChange = this.baseline.isCalibrated ?
            (this.statistics.rate - this.baseline.normalRate) / this.baseline.normalRate : 0;
        
        // 振幅の変化
        const amplitudeChange = this.baseline.isCalibrated ?
            (this.statistics.avgAmplitude - this.baseline.normalAmplitude) / this.baseline.normalAmplitude : 0;
        
        // 不安指標の計算（研究に基づく）
        const anxietyIndicator = this.calculateAnxietyIndicator(rateChange, amplitudeChange);
        
        // 集中度の計算
        const focusLevel = this.calculateFocusLevel(timestamp);
        
        // ドーパミン活性の推定
        const dopamineActivity = this.estimateDopamineActivity();
        
        return {
            // 基本統計
            statistics: {
                rate: this.statistics.rate,
                rateLabel: this.getRateLabel(),
                avgAmplitude: this.statistics.avgAmplitude,
                avgDuration: this.statistics.avgDuration,
                rhythmRegularity: this.statistics.rhythmRegularity
            },
            
            // 現在の状態
            currentState: {
                isBlinking: this.currentBlink.isBlinking,
                timeSinceLastBlink: this.getTimeSinceLastBlink(timestamp)
            },
            
            // 相対的変化
            changes: {
                rate: rateChange,
                amplitude: amplitudeChange,
                rateDescription: this.describeRateChange(rateChange),
                amplitudeDescription: this.describeAmplitudeChange(amplitudeChange)
            },
            
            // 心理状態推定
            psychological: {
                anxiety: anxietyIndicator,
                focus: focusLevel,
                dopamineActivity: dopamineActivity
            },
            
            // 最近の瞬きパターン
            pattern: this.analyzePattern(),
            
            // キャリブレーション状態
            calibration: {
                isCalibrated: this.baseline.isCalibrated,
                normalRate: this.baseline.normalRate
            }
        };
    }
    
    /**
     * 不安指標を計算
     * 研究：不安が高いと瞬目率は増加し、振幅は浅くなる
     */
    calculateAnxietyIndicator(rateChange, amplitudeChange) {
        // 瞬目率増加 + 振幅減少 = 不安の兆候
        let score = 0;
        
        // 瞬目率の増加（正の変化）= 不安
        if (rateChange > 0.2) {
            score += Math.min(0.5, rateChange * 0.5);
        }
        
        // 振幅の減少（負の変化）= 不安
        if (amplitudeChange < -0.1) {
            score += Math.min(0.5, Math.abs(amplitudeChange) * 0.5);
        }
        
        // リズムの不規則性
        if (this.statistics.rhythmRegularity < 0.3) {
            score += 0.2;
        }
        
        return {
            level: score > 0.5 ? 'high' : score > 0.25 ? 'moderate' : 'low',
            value: Math.min(1, score),
            indicators: {
                elevatedRate: rateChange > 0.2,
                shallowBlinks: amplitudeChange < -0.1,
                irregularRhythm: this.statistics.rhythmRegularity < 0.3
            }
        };
    }
    
    /**
     * 集中度を計算
     */
    calculateFocusLevel(timestamp) {
        const timeSinceLast = this.getTimeSinceLastBlink(timestamp);
        
        // 長時間瞬きがない = 集中している可能性
        // 一般的に3-4秒以上瞬きがないと集中状態
        const focusFromBlinkGap = Math.min(1, timeSinceLast / 5000);
        
        // 瞬目率が低い = 集中
        const focusFromRate = this.baseline.isCalibrated ?
            Math.max(0, 1 - this.statistics.rate / this.baseline.normalRate / 1.5) : 0.5;
        
        const focusScore = focusFromBlinkGap * 0.6 + focusFromRate * 0.4;
        
        return {
            level: focusScore > 0.7 ? 'high' : focusScore > 0.4 ? 'moderate' : 'low',
            value: focusScore,
            timeSinceLastBlink: timeSinceLast
        };
    }
    
    /**
     * ドーパミン活性を推定
     * 研究：ドーパミン系は瞬目率と相関
     */
    estimateDopamineActivity() {
        if (!this.baseline.isCalibrated) {
            return { level: 'unknown', value: 0.5 };
        }
        
        const rateRatio = this.statistics.rate / this.baseline.normalRate;
        
        // 瞬目率が高い = ドーパミン活性が高い
        // （ただし極端に高い場合は異常）
        let activityLevel = Math.min(1, rateRatio);
        
        return {
            level: activityLevel > 0.8 ? 'high' : activityLevel > 0.5 ? 'normal' : 'low',
            value: activityLevel,
            description: activityLevel > 0.8 ? 
                '高いドーパミン活性（興奮状態）' :
                activityLevel < 0.5 ?
                '低いドーパミン活性（疲労/抑うつの可能性）' :
                '正常範囲'
        };
    }
    
    /**
     * 瞬きパターンを分析
     */
    analyzePattern() {
        const recent = this.blinkHistory.slice(-10);
        
        if (recent.length < 3) {
            return { type: 'insufficient_data', description: 'データ不足' };
        }
        
        // 完全な瞬き vs 不完全な瞬きの割合
        const completeRatio = recent.filter(b => b.isComplete).length / recent.length;
        
        // 速い瞬き vs 遅い瞬きの割合
        const avgVelocity = this.calculateMean(recent.map(b => b.velocity));
        
        // パターン判定
        if (completeRatio > 0.8 && this.statistics.rhythmRegularity > 0.6) {
            return {
                type: 'normal',
                description: '正常な瞬きパターン',
                completeRatio: completeRatio,
                regularity: this.statistics.rhythmRegularity
            };
        } else if (completeRatio < 0.5) {
            return {
                type: 'shallow',
                description: '浅い瞬きが多い（緊張/不安の可能性）',
                completeRatio: completeRatio
            };
        } else if (this.statistics.rhythmRegularity < 0.3) {
            return {
                type: 'irregular',
                description: '不規則な瞬き',
                regularity: this.statistics.rhythmRegularity
            };
        } else if (avgVelocity > 0.005) {
            return {
                type: 'rapid',
                description: '素早い瞬きが多い',
                avgVelocity: avgVelocity
            };
        }
        
        return { type: 'normal', description: '通常パターン' };
    }
    
    /**
     * 最後の瞬きからの経過時間
     */
    getTimeSinceLastBlink(timestamp) {
        if (this.blinkHistory.length === 0) {
            return timestamp - this.startTime;
        }
        return timestamp - this.blinkHistory[this.blinkHistory.length - 1].endTime;
    }
    
    /**
     * 瞬目率のラベル
     */
    getRateLabel() {
        const rate = this.statistics.rate;
        if (rate < 10) return '低い';
        if (rate > 25) return '高い';
        return '正常';
    }
    
    /**
     * 瞬目率変化の説明
     */
    describeRateChange(change) {
        if (change > 0.3) return '顕著に増加（緊張/興奮）';
        if (change > 0.15) return 'やや増加';
        if (change < -0.3) return '顕著に減少（集中/疲労）';
        if (change < -0.15) return 'やや減少';
        return '安定';
    }
    
    /**
     * 振幅変化の説明
     */
    describeAmplitudeChange(change) {
        if (change < -0.2) return '浅い瞬き（緊張の可能性）';
        if (change > 0.2) return '深い瞬き（リラックス）';
        return '通常';
    }
    
    /**
     * リセット
     */
    reset() {
        this.blinkHistory = [];
        this.opennessHistory = [];
        this.currentBlink = {
            isBlinking: false,
            startTime: null,
            minOpenness: 1,
            startOpenness: 0
        };
        this.baseline.isCalibrated = false;
        this.startTime = performance.now();
    }
    
    // ユーティリティ
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
window.BlinkAnalyzer = BlinkAnalyzer;
