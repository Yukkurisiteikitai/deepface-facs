/**
 * 感情ラベル付きFACS記録機能
 */

class EmotionRecorder {
    constructor() {
        // FACS 基本6感情 + 軽蔑 の定義
        this.emotions = {
            happiness: {
                name: '喜び',
                emoji: '😊',
                description: '幸福感、楽しさ、満足感',
                typicalAUs: ['AU6', 'AU12'],
                optionalAUs: ['AU25', 'AU26'],
                color: '#FFD700'
            },
            anger: {
                name: '怒り',
                emoji: '😠',
                description: '憤り、苛立ち、敵意',
                typicalAUs: ['AU4', 'AU5', 'AU7', 'AU23'],
                optionalAUs: ['AU17', 'AU24'],
                color: '#FF4444'
            },
            sadness: {
                name: '悲しみ',
                emoji: '😢',
                description: '悲哀、落胆、失望',
                typicalAUs: ['AU1', 'AU4', 'AU15'],
                optionalAUs: ['AU6', 'AU11', 'AU17'],
                color: '#4444FF'
            },
            fear: {
                name: '恐怖',
                emoji: '😨',
                description: '恐れ、不安、脅威への反応',
                typicalAUs: ['AU1', 'AU2', 'AU4', 'AU5', 'AU20', 'AU26'],
                optionalAUs: ['AU25'],
                color: '#9944FF'
            },
            surprise: {
                name: '驚き',
                emoji: '😲',
                description: '予期しない出来事への反応',
                typicalAUs: ['AU1', 'AU2', 'AU5', 'AU26'],
                optionalAUs: ['AU27'],
                color: '#FF9900'
            },
            disgust: {
                name: '嫌悪',
                emoji: '🤢',
                description: '不快感、拒絶反応',
                typicalAUs: ['AU9', 'AU10', 'AU17'],
                optionalAUs: ['AU4', 'AU6', 'AU25', 'AU26'],
                color: '#44AA44'
            },
            contempt: {
                name: '軽蔑',
                emoji: '😏',
                description: '見下し、優越感',
                typicalAUs: ['AU12', 'AU14'],
                optionalAUs: [],
                note: '片側のみ（非対称）',
                color: '#AA44AA'
            },
            neutral: {
                name: '無表情',
                emoji: '😐',
                description: '感情表出なし',
                typicalAUs: [],
                optionalAUs: [],
                color: '#888888'
            }
        };
        
        // 記録データ
        this.recordings = [];
        this.currentSession = null;
        this.isRecording = false;
    }
    
    /**
     * 感情記録セッションを開始
     */
    startSession(emotion) {
        if (!this.emotions[emotion]) {
            throw new Error(`Unknown emotion: ${emotion}`);
        }
        
        this.currentSession = {
            emotion: emotion,
            emotionInfo: this.emotions[emotion],
            startTime: Date.now(),
            frames: [],
            metadata: {
                expectedAUs: this.emotions[emotion].typicalAUs,
                optionalAUs: this.emotions[emotion].optionalAUs
            }
        };
        this.isRecording = true;
        
        return this.currentSession;
    }
    
    /**
     * フレームを記録
     */
    recordFrame(analysisResult) {
        if (!this.isRecording || !this.currentSession) return;
        
        const frame = {
            timestamp: Date.now() - this.currentSession.startTime,
            actionUnits: analysisResult.actionUnits || {},
            facsCode: analysisResult.facsCode || '---',
            blendshapes: analysisResult.blendshapes || {},
            matchScore: this.calculateMatchScore(analysisResult.actionUnits)
        };
        
        this.currentSession.frames.push(frame);
        return frame;
    }
    
    /**
     * 期待されるAUとの一致度を計算
     */
    calculateMatchScore(actionUnits) {
        if (!this.currentSession || !actionUnits) return 0;
        
        const expected = this.currentSession.metadata.expectedAUs;
        if (expected.length === 0) return 1.0; // neutralは常に一致
        
        let matchedCount = 0;
        let totalWeight = 0;
        
        expected.forEach(au => {
            totalWeight += 1;
            if (actionUnits[au] && actionUnits[au] > 0.2) {
                matchedCount += actionUnits[au];
            }
        });
        
        return totalWeight > 0 ? matchedCount / totalWeight : 0;
    }
    
    /**
     * セッションを終了
     */
    stopSession() {
        if (!this.currentSession) return null;
        
        this.isRecording = false;
        
        const session = {
            ...this.currentSession,
            endTime: Date.now(),
            duration: Date.now() - this.currentSession.startTime,
            summary: this.generateSummary()
        };
        
        this.recordings.push(session);
        this.currentSession = null;
        
        return session;
    }
    
    /**
     * セッションのサマリーを生成
     */
    generateSummary() {
        if (!this.currentSession || this.currentSession.frames.length === 0) {
            return null;
        }
        
        const frames = this.currentSession.frames;
        const expected = this.currentSession.metadata.expectedAUs;
        
        // AU別の統計
        const auStats = {};
        frames.forEach(frame => {
            Object.entries(frame.actionUnits).forEach(([au, value]) => {
                if (!auStats[au]) {
                    auStats[au] = { values: [], count: 0 };
                }
                auStats[au].values.push(value);
                if (value > 0.2) auStats[au].count++;
            });
        });
        
        // 各AUの平均・最大値を計算
        const auSummary = {};
        Object.entries(auStats).forEach(([au, stats]) => {
            const values = stats.values;
            auSummary[au] = {
                mean: values.reduce((a, b) => a + b, 0) / values.length,
                max: Math.max(...values),
                min: Math.min(...values),
                activeRatio: stats.count / frames.length
            };
        });
        
        // 平均一致スコア
        const avgMatchScore = frames.reduce((sum, f) => sum + f.matchScore, 0) / frames.length;
        
        // 期待AUの検出率
        const expectedAUDetection = {};
        expected.forEach(au => {
            if (auSummary[au]) {
                expectedAUDetection[au] = {
                    detected: true,
                    avgIntensity: auSummary[au].mean,
                    maxIntensity: auSummary[au].max,
                    activeRatio: auSummary[au].activeRatio
                };
            } else {
                expectedAUDetection[au] = { detected: false };
            }
        });
        
        return {
            totalFrames: frames.length,
            duration: this.currentSession.duration,
            avgMatchScore,
            auSummary,
            expectedAUDetection,
            recommendation: this.generateRecommendation(avgMatchScore, expectedAUDetection)
        };
    }
    
    /**
     * 改善推奨を生成
     */
    generateRecommendation(avgMatchScore, expectedAUDetection) {
        const recommendations = [];
        
        if (avgMatchScore < 0.3) {
            recommendations.push('表情が弱いようです。もう少し大げさに表現してみてください。');
        }
        
        Object.entries(expectedAUDetection).forEach(([au, data]) => {
            if (!data.detected || data.avgIntensity < 0.2) {
                recommendations.push(`${au}（${this.getAUName(au)}）の動きが不足しています。`);
            }
        });
        
        if (recommendations.length === 0) {
            recommendations.push('良い表情です！期待されるAction Unitが適切に検出されています。');
        }
        
        return recommendations;
    }
    
    /**
     * AU名を取得
     */
    getAUName(auCode) {
        const names = {
            'AU1': '眉内側挙上', 'AU2': '眉外側挙上', 'AU4': '眉下制',
            'AU5': '上瞼挙上', 'AU6': '頬挙上', 'AU7': '瞼緊張',
            'AU9': '鼻しわ', 'AU10': '上唇挙上', 'AU11': '鼻唇溝深化',
            'AU12': '口角挙上', 'AU14': 'えくぼ', 'AU15': '口角下制',
            'AU17': '顎挙上', 'AU20': '唇伸展', 'AU23': '唇緊張',
            'AU24': '唇圧迫', 'AU25': '唇分離', 'AU26': '顎下制',
            'AU27': '口大開'
        };
        return names[auCode] || auCode;
    }
    
    /**
     * 記録をJSON形式でエクスポート
     */
    exportJSON() {
        return JSON.stringify({
            exportedAt: new Date().toISOString(),
            emotionDefinitions: this.emotions,
            recordings: this.recordings
        }, null, 2);
    }
    
    /**
     * 記録をダウンロード
     */
    downloadRecordings(filename = 'emotion_recordings.json') {
        const data = this.exportJSON();
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        
        URL.revokeObjectURL(url);
    }
    
    /**
     * 全感情リストを取得
     */
    getEmotionList() {
        return Object.entries(this.emotions).map(([key, value]) => ({
            id: key,
            ...value
        }));
    }
}

// グローバルエクスポート
window.EmotionRecorder = EmotionRecorder;
