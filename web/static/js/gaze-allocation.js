/**
 * 視線配分（Gaze Allocation）分析モジュール
 * 
 * 研究に基づく感情認識時の視線パターン：
 * - 感情別ヒートマップ分析
 * - AOI（Area of Interest）ベースの分析
 * - 視線滞留時間（Dwell Time）の計測
 */

class GazeAllocationAnalyzer {
    constructor(options = {}) {
        this.config = {
            // AOI（関心領域）の定義
            aoiSettings: {
                eyeWeight: 1.0,
                noseWeight: 0.5,
                mouthWeight: 0.8
            },
            
            // 滞留判定
            dwellThreshold: 100,          // 滞留と判定する最小時間（ms）
            fixationRadius: 0.02,         // 固視と判定する最大移動量
            
            // 履歴
            historySize: 300,             // 約10秒@30fps
            
            ...options
        };
        
        // AOI定義（顔の領域）
        this.aois = {
            leftEye: { x: 0.35, y: 0.35, width: 0.15, height: 0.1, label: '左目' },
            rightEye: { x: 0.5, y: 0.35, width: 0.15, height: 0.1, label: '右目' },
            eyes: { x: 0.35, y: 0.35, width: 0.3, height: 0.1, label: '目' },
            nose: { x: 0.4, y: 0.45, width: 0.2, height: 0.15, label: '鼻' },
            mouth: { x: 0.35, y: 0.6, width: 0.3, height: 0.15, label: '口' },
            eyebrows: { x: 0.3, y: 0.28, width: 0.4, height: 0.08, label: '眉' },
            forehead: { x: 0.3, y: 0.15, width: 0.4, height: 0.15, label: '額' },
            chin: { x: 0.35, y: 0.75, width: 0.3, height: 0.1, label: '顎' }
        };
        
        // 視線履歴
        this.gazeHistory = [];
        
        // 固視イベント
        this.fixations = [];
        
        // 現在の固視
        this.currentFixation = null;
        
        // AOI滞留統計
        this.aoiStatistics = {};
        this.resetAOIStatistics();
        
        // 感情別の典型的視線パターン（研究に基づく）
        this.emotionGazePatterns = {
            happy: { eyes: 0.91, nose: 0.04, mouth: 0.05 },
            sad: { eyes: 0.78, nose: 0.10, mouth: 0.12 },
            angry: { eyes: 0.45, eyebrows: 0.25, mouth: 0.20, nose: 0.10 },
            fear: { eyes: 0.50, mouth: 0.35, nose: 0.15 },
            disgust: { eyes: 0.65, nose: 0.20, mouth: 0.15 },
            surprise: { eyes: 0.60, mouth: 0.30, eyebrows: 0.10 },
            neutral: { eyes: 0.70, nose: 0.15, mouth: 0.15 }
        };
        
        this.startTime = performance.now();
    }
    
    /**
     * AOI統計をリセット
     */
    resetAOIStatistics() {
        this.aoiStatistics = {};
        for (const aoiName of Object.keys(this.aois)) {
            this.aoiStatistics[aoiName] = {
                totalDwellTime: 0,
                fixationCount: 0,
                visits: 0,
                percentage: 0
            };
        }
        this.aoiStatistics.other = {
            totalDwellTime: 0,
            fixationCount: 0,
            visits: 0,
            percentage: 0
        };
    }
    
    /**
     * フレームごとの更新
     * @param {Object} gazePosition - 視線位置 {x, y}（顔の相対座標 0-1）
     * @param {number} timestamp - タイムスタンプ
     */
    update(gazePosition, timestamp = performance.now()) {
        if (!gazePosition) return null;
        
        // 履歴に追加
        this.gazeHistory.push({
            x: gazePosition.x,
            y: gazePosition.y,
            timestamp: timestamp,
            aoi: this.getAOI(gazePosition)
        });
        
        // 古いデータを削除
        const cutoff = timestamp - (this.config.historySize * 33);
        this.gazeHistory = this.gazeHistory.filter(g => g.timestamp > cutoff);
        
        // 固視の検出・更新
        this.updateFixation(gazePosition, timestamp);
        
        // AOI統計を更新
        this.updateAOIStatistics(timestamp);
        
        return this.analyze(timestamp);
    }
    
    /**
     * 視線位置がどのAOIに属するか判定
     */
    getAOI(position) {
        for (const [name, aoi] of Object.entries(this.aois)) {
            if (position.x >= aoi.x && position.x <= aoi.x + aoi.width &&
                position.y >= aoi.y && position.y <= aoi.y + aoi.height) {
                return name;
            }
        }
        return 'other';
    }
    
    /**
     * 固視の更新
     */
    updateFixation(position, timestamp) {
        if (!this.currentFixation) {
            // 新しい固視を開始
            this.currentFixation = {
                startTime: timestamp,
                x: position.x,
                y: position.y,
                aoi: this.getAOI(position),
                samples: [position]
            };
            return;
        }
        
        // 現在の位置が固視範囲内かチェック
        const dx = position.x - this.currentFixation.x;
        const dy = position.y - this.currentFixation.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < this.config.fixationRadius) {
            // 固視継続
            this.currentFixation.samples.push(position);
            // 中心を更新（移動平均）
            this.currentFixation.x = (this.currentFixation.x * 0.9 + position.x * 0.1);
            this.currentFixation.y = (this.currentFixation.y * 0.9 + position.y * 0.1);
        } else {
            // 固視終了
            const duration = timestamp - this.currentFixation.startTime;
            
            if (duration >= this.config.dwellThreshold) {
                // 有効な固視として記録
                this.fixations.push({
                    startTime: this.currentFixation.startTime,
                    endTime: timestamp,
                    duration: duration,
                    x: this.currentFixation.x,
                    y: this.currentFixation.y,
                    aoi: this.currentFixation.aoi,
                    sampleCount: this.currentFixation.samples.length
                });
                
                // AOI統計を更新
                const aoiName = this.currentFixation.aoi;
                if (this.aoiStatistics[aoiName]) {
                    this.aoiStatistics[aoiName].totalDwellTime += duration;
                    this.aoiStatistics[aoiName].fixationCount++;
                }
            }
            
            // 新しい固視を開始
            this.currentFixation = {
                startTime: timestamp,
                x: position.x,
                y: position.y,
                aoi: this.getAOI(position),
                samples: [position]
            };
        }
        
        // 古い固視を削除
        const fixationCutoff = timestamp - 30000; // 30秒
        this.fixations = this.fixations.filter(f => f.endTime > fixationCutoff);
    }
    
    /**
     * AOI統計を更新
     */
    updateAOIStatistics(timestamp) {
        // 直近の視線データから割合を計算
        const recentGaze = this.gazeHistory.slice(-90); // 約3秒
        
        if (recentGaze.length === 0) return;
        
        // AOIごとのカウント
        const counts = {};
        for (const aoi of Object.keys(this.aoiStatistics)) {
            counts[aoi] = 0;
        }
        
        recentGaze.forEach(g => {
            counts[g.aoi] = (counts[g.aoi] || 0) + 1;
        });
        
        // パーセンテージを計算
        const total = recentGaze.length;
        for (const aoi of Object.keys(this.aoiStatistics)) {
            this.aoiStatistics[aoi].percentage = counts[aoi] / total;
        }
    }
    
    /**
     * 分析結果を生成
     */
    analyze(timestamp) {
        // 現在の視線配分
        const distribution = this.getDistribution();
        
        // 感情パターンとのマッチング
        const emotionMatch = this.matchEmotionPattern(distribution);
        
        // 視線遷移分析
        const transitions = this.analyzeTransitions();
        
        // スキャンパス特性
        const scanPath = this.analyzeScanPath();
        
        // 鼻の「アンカーポイント」としての使用
        const noseAnchor = this.analyzeNoseAnchor();
        
        return {
            // 現在の視線配分
            distribution: distribution,
            
            // 感情パターンマッチング
            emotionMatch: emotionMatch,
            
            // AOI詳細統計
            aoiStatistics: this.aoiStatistics,
            
            // 視線遷移
            transitions: transitions,
            
            // スキャンパス特性
            scanPath: scanPath,
            
            // 鼻のアンカー機能
            noseAnchor: noseAnchor,
            
            // 現在の固視
            currentFixation: this.currentFixation ? {
                aoi: this.currentFixation.aoi,
                duration: timestamp - this.currentFixation.startTime,
                position: { x: this.currentFixation.x, y: this.currentFixation.y }
            } : null,
            
            // 最近の固視
            recentFixations: this.fixations.slice(-10).map(f => ({
                aoi: f.aoi,
                duration: f.duration,
                position: { x: f.x, y: f.y }
            }))
        };
    }
    
    /**
     * 視線配分を取得
     */
    getDistribution() {
        const dist = {};
        for (const [aoi, stats] of Object.entries(this.aoiStatistics)) {
            if (aoi !== 'other') {
                dist[aoi] = stats.percentage;
            }
        }
        
        // 主要領域をまとめる
        return {
            detailed: dist,
            summary: {
                eyes: (dist.leftEye || 0) + (dist.rightEye || 0) + (dist.eyes || 0),
                nose: dist.nose || 0,
                mouth: dist.mouth || 0,
                eyebrows: dist.eyebrows || 0,
                other: this.aoiStatistics.other?.percentage || 0
            }
        };
    }
    
    /**
     * 感情パターンとマッチング
     */
    matchEmotionPattern(distribution) {
        const summary = distribution.summary;
        let bestMatch = { emotion: 'neutral', score: 0, confidence: 0 };
        
        for (const [emotion, pattern] of Object.entries(this.emotionGazePatterns)) {
            let score = 0;
            let count = 0;
            
            for (const [region, expected] of Object.entries(pattern)) {
                const actual = summary[region] || 0;
                // 類似度を計算（差分の逆数）
                const diff = Math.abs(actual - expected);
                score += 1 - Math.min(1, diff * 2);
                count++;
            }
            
            const avgScore = count > 0 ? score / count : 0;
            
            if (avgScore > bestMatch.score) {
                bestMatch = {
                    emotion: emotion,
                    score: avgScore,
                    confidence: Math.min(1, avgScore * 1.2)
                };
            }
        }
        
        return {
            ...bestMatch,
            interpretation: this.interpretGazePattern(distribution.summary, bestMatch.emotion)
        };
    }
    
    /**
     * 視線パターンの解釈
     */
    interpretGazePattern(summary, matchedEmotion) {
        const eyeRatio = summary.eyes || 0;
        const noseRatio = summary.nose || 0;
        const mouthRatio = summary.mouth || 0;
        
        let interpretation = '';
        
        if (eyeRatio > 0.8) {
            interpretation = '目に強く注目しています（幸福の認識パターン）';
        } else if (noseRatio > 0.2) {
            interpretation = '鼻に注目しています（嫌悪の認識パターン）';
        } else if (mouthRatio > 0.3) {
            interpretation = '口に注目しています（恐怖/驚きの認識パターン）';
        } else if (eyeRatio > 0.6 && mouthRatio > 0.2) {
            interpretation = '目と口を交互に見ています（恐怖/怒りの認識パターン）';
        } else {
            interpretation = '通常の視線配分';
        }
        
        return interpretation;
    }
    
    /**
     * 視線遷移を分析
     */
    analyzeTransitions() {
        if (this.fixations.length < 2) {
            return { patterns: [], dominantPath: null };
        }
        
        // AOI間の遷移をカウント
        const transitions = {};
        
        for (let i = 1; i < this.fixations.length; i++) {
            const from = this.fixations[i - 1].aoi;
            const to = this.fixations[i].aoi;
            
            if (from !== to) {
                const key = `${from}->${to}`;
                transitions[key] = (transitions[key] || 0) + 1;
            }
        }
        
        // ソートして上位を取得
        const sorted = Object.entries(transitions)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(([path, count]) => ({ path, count }));
        
        return {
            patterns: sorted,
            dominantPath: sorted.length > 0 ? sorted[0].path : null,
            totalTransitions: Object.values(transitions).reduce((a, b) => a + b, 0)
        };
    }
    
    /**
     * スキャンパス特性を分析
     */
    analyzeScanPath() {
        const recentFixations = this.fixations.slice(-20);
        
        if (recentFixations.length < 3) {
            return { type: 'insufficient_data', coverage: 0 };
        }
        
        // 訪問したAOIの種類
        const visitedAOIs = new Set(recentFixations.map(f => f.aoi));
        const coverage = visitedAOIs.size / Object.keys(this.aois).length;
        
        // 平均固視時間
        const avgDuration = this.calculateMean(recentFixations.map(f => f.duration));
        
        // スキャンパスの長さ（総移動距離）
        let totalDistance = 0;
        for (let i = 1; i < recentFixations.length; i++) {
            const dx = recentFixations[i].x - recentFixations[i - 1].x;
            const dy = recentFixations[i].y - recentFixations[i - 1].y;
            totalDistance += Math.sqrt(dx * dx + dy * dy);
        }
        
        // パターン判定
        let type = 'exploratory';
        if (coverage < 0.3 && avgDuration > 500) {
            type = 'focused';  // 特定の領域に集中
        } else if (coverage > 0.6 && avgDuration < 200) {
            type = 'scanning';  // 広く素早くスキャン
        } else if (totalDistance < 0.1 * recentFixations.length) {
            type = 'fixated';  // ほぼ固定
        }
        
        return {
            type: type,
            coverage: coverage,
            avgFixationDuration: avgDuration,
            pathLength: totalDistance,
            visitedAOIs: Array.from(visitedAOIs),
            interpretation: this.interpretScanPath(type)
        };
    }
    
    /**
     * スキャンパスの解釈
     */
    interpretScanPath(type) {
        switch (type) {
            case 'focused':
                return '特定の領域に集中して注目しています';
            case 'scanning':
                return '広い範囲を素早くスキャンしています';
            case 'fixated':
                return 'ほぼ一点を見つめています';
            case 'exploratory':
            default:
                return '通常の探索的な視線パターン';
        }
    }
    
    /**
     * 鼻の「アンカーポイント」機能を分析
     * 研究：怒りや恐怖の認識時、目と口の間を移動する際に鼻が中継点となる
     */
    analyzeNoseAnchor() {
        if (this.fixations.length < 5) {
            return { isAnchor: false, usage: 0 };
        }
        
        // 鼻を経由した目↔口の遷移をカウント
        let noseAsAnchorCount = 0;
        let totalRelevantTransitions = 0;
        
        for (let i = 2; i < this.fixations.length; i++) {
            const f1 = this.fixations[i - 2];
            const f2 = this.fixations[i - 1];
            const f3 = this.fixations[i];
            
            // 目→鼻→口 または 口→鼻→目 のパターン
            const isEyeNoseMouth = 
                (f1.aoi.includes('Eye') || f1.aoi === 'eyes') &&
                f2.aoi === 'nose' &&
                f3.aoi === 'mouth';
            
            const isMouthNoseEye = 
                f1.aoi === 'mouth' &&
                f2.aoi === 'nose' &&
                (f3.aoi.includes('Eye') || f3.aoi === 'eyes');
            
            if (isEyeNoseMouth || isMouthNoseEye) {
                noseAsAnchorCount++;
            }
            
            // 目と口の間の移動をカウント
            const involvesBoth = 
                ((f1.aoi.includes('Eye') || f1.aoi === 'eyes') && f3.aoi === 'mouth') ||
                (f1.aoi === 'mouth' && (f3.aoi.includes('Eye') || f3.aoi === 'eyes'));
            
            if (involvesBoth) {
                totalRelevantTransitions++;
            }
        }
        
        const anchorUsage = totalRelevantTransitions > 0 ?
            noseAsAnchorCount / totalRelevantTransitions : 0;
        
        return {
            isAnchor: anchorUsage > 0.3,
            usage: anchorUsage,
            count: noseAsAnchorCount,
            interpretation: anchorUsage > 0.3 ?
                '鼻が視線のアンカーポイントとして機能しています（怒り/恐怖の認識パターン）' :
                '鼻はアンカーとして使用されていません'
        };
    }
    
    /**
     * 顔のランドマークからAOIを更新
     */
    updateAOIsFromLandmarks(landmarks) {
        // MediaPipe Face Meshのランドマークから各領域の位置を更新
        // これにより個人の顔の形状に適応
        
        if (!landmarks || landmarks.length < 468) return;
        
        // 目の領域
        const leftEyeCenter = landmarks[468]; // 左目虹彩中心
        const rightEyeCenter = landmarks[473]; // 右目虹彩中心
        
        // 鼻
        const noseTip = landmarks[1];
        
        // 口
        const mouthTop = landmarks[13];
        const mouthBottom = landmarks[14];
        
        // AOIを更新
        this.aois.leftEye.x = leftEyeCenter.x - 0.075;
        this.aois.leftEye.y = leftEyeCenter.y - 0.05;
        
        this.aois.rightEye.x = rightEyeCenter.x - 0.075;
        this.aois.rightEye.y = rightEyeCenter.y - 0.05;
        
        this.aois.nose.x = noseTip.x - 0.1;
        this.aois.nose.y = noseTip.y - 0.075;
        
        this.aois.mouth.x = (mouthTop.x + mouthBottom.x) / 2 - 0.15;
        this.aois.mouth.y = mouthTop.y - 0.02;
        this.aois.mouth.height = (mouthBottom.y - mouthTop.y) + 0.05;
    }
    
    /**
     * リセット
     */
    reset() {
        this.gazeHistory = [];
        this.fixations = [];
        this.currentFixation = null;
        this.resetAOIStatistics();
        this.startTime = performance.now();
    }
    
    // ユーティリティ
    calculateMean(arr) {
        if (arr.length === 0) return 0;
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }
}

// グローバルエクスポート
window.GazeAllocationAnalyzer = GazeAllocationAnalyzer;
