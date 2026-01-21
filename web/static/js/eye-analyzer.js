/**
 * 目の分析モジュール - 細い目にも対応した設計
 */

class EyeAnalyzer {
    constructor() {
        // MediaPipe Face Mesh の目関連のランドマークインデックス
        this.landmarks = {
            // 左目（画面から見て右側）
            leftEye: {
                upper: [386, 387, 388, 466],      // 上瞼
                lower: [374, 373, 390, 249],      // 下瞼
                innerCorner: 362,                  // 目頭
                outerCorner: 263,                  // 目尻
                pupil: 468,                        // 瞳（虹彩中心）
                iris: [468, 469, 470, 471, 472],  // 虹彩
                upperLid: 386,                     // 上瞼中央
                lowerLid: 374,                     // 下瞼中央
                // 詳細なポイント
                upperOuter: 388,
                upperInner: 387,
                lowerOuter: 380,
                lowerInner: 373
            },
            // 右目（画面から見て左側）
            rightEye: {
                upper: [159, 158, 157, 246],
                lower: [145, 144, 163, 7],
                innerCorner: 133,
                outerCorner: 33,
                pupil: 473,
                iris: [473, 474, 475, 476, 477],
                upperLid: 159,
                lowerLid: 145,
                upperOuter: 157,
                upperInner: 158,
                lowerOuter: 153,
                lowerInner: 144
            },
            // 眉
            leftBrow: {
                inner: 336,
                middle: 296,
                outer: 334,
                points: [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
            },
            rightBrow: {
                inner: 107,
                middle: 66,
                outer: 105,
                points: [107, 66, 105, 63, 70, 46, 53, 52, 65, 55]
            }
        };
        
        // キャリブレーションデータ（個人差対応）
        this.calibration = {
            isCalibrated: false,
            leftEye: { maxOpen: 0, minOpen: 1, baseline: 0 },
            rightEye: { maxOpen: 0, minOpen: 1, baseline: 0 },
            samples: [],
            sampleCount: 0,
            targetSamples: 30  // キャリブレーションに必要なサンプル数
        };
        
        // 目のタイプ判定
        this.eyeType = {
            isNarrow: false,      // 細い目かどうか
            narrowThreshold: 0.15, // 細い目と判定する閾値
            detected: false
        };
        
        // 履歴（スムージング用）
        this.history = {
            leftEAR: [],
            rightEAR: [],
            maxHistory: 5
        };
    }
    
    /**
     * 目の分析を実行
     */
    analyze(faceLandmarks, blendshapes = {}) {
        if (!faceLandmarks || faceLandmarks.length < 478) {
            return null;
        }
        
        const leftEye = this.analyzeEye(faceLandmarks, 'left');
        const rightEye = this.analyzeEye(faceLandmarks, 'right');
        const eyeContact = this.analyzeGaze(faceLandmarks);
        const brows = this.analyzeBrows(faceLandmarks);
        
        // キャリブレーション更新
        this.updateCalibration(leftEye, rightEye);
        
        // 正規化された値を計算
        const normalizedLeft = this.normalizeOpenness(leftEye.openness, 'left');
        const normalizedRight = this.normalizeOpenness(rightEye.openness, 'right');
        
        // 目関連のAction Unitsを推定
        const actionUnits = this.estimateEyeAUs(
            { ...leftEye, normalizedOpenness: normalizedLeft },
            { ...rightEye, normalizedOpenness: normalizedRight },
            brows,
            blendshapes
        );
        
        return {
            leftEye: {
                ...leftEye,
                normalizedOpenness: normalizedLeft,
                isOpen: normalizedLeft > 0.3,
                isBlinking: normalizedLeft < 0.15
            },
            rightEye: {
                ...rightEye,
                normalizedOpenness: normalizedRight,
                isOpen: normalizedRight > 0.3,
                isBlinking: normalizedRight < 0.15
            },
            gaze: eyeContact,
            brows,
            actionUnits,
            calibration: {
                isCalibrated: this.calibration.isCalibrated,
                progress: Math.min(1, this.calibration.sampleCount / this.calibration.targetSamples),
                eyeType: this.eyeType.isNarrow ? 'narrow' : 'normal'
            }
        };
    }
    
    /**
     * 片目の分析
     */
    analyzeEye(landmarks, side) {
        const eyeIdx = side === 'left' ? this.landmarks.leftEye : this.landmarks.rightEye;
        
        // 目の開き具合（Eye Aspect Ratio - EAR）
        const ear = this.calculateEAR(landmarks, eyeIdx);
        
        // 目の幅と高さ
        const width = this.distance(
            landmarks[eyeIdx.innerCorner],
            landmarks[eyeIdx.outerCorner]
        );
        const height = this.distance(
            landmarks[eyeIdx.upperLid],
            landmarks[eyeIdx.lowerLid]
        );
        
        // アスペクト比
        const aspectRatio = width > 0 ? height / width : 0;
        
        // 瞳の位置（虹彩中心）
        const pupilPos = landmarks[eyeIdx.pupil];
        const eyeCenter = {
            x: (landmarks[eyeIdx.innerCorner].x + landmarks[eyeIdx.outerCorner].x) / 2,
            y: (landmarks[eyeIdx.upperLid].y + landmarks[eyeIdx.lowerLid].y) / 2
        };
        
        // 瞳の相対位置（-1 to 1）
        const pupilOffset = {
            x: width > 0 ? (pupilPos.x - eyeCenter.x) / (width / 2) : 0,
            y: height > 0 ? (pupilPos.y - eyeCenter.y) / (height / 2) : 0
        };
        
        // 上瞼の曲率（細い目の検出に使用）
        const upperLidCurvature = this.calculateLidCurvature(landmarks, eyeIdx, 'upper');
        
        // スムージング適用
        const smoothedEAR = this.smoothValue(ear, side === 'left' ? 'leftEAR' : 'rightEAR');
        
        return {
            ear: smoothedEAR,
            rawEar: ear,
            openness: smoothedEAR,
            width,
            height,
            aspectRatio,
            pupilOffset,
            upperLidCurvature,
            center: eyeCenter
        };
    }
    
    /**
     * Eye Aspect Ratio (EAR) を計算
     * 細い目に対応した改良版
     */
    calculateEAR(landmarks, eyeIdx) {
        // 垂直方向の距離（複数ポイントで計算）
        const v1 = this.distance(
            landmarks[eyeIdx.upperOuter],
            landmarks[eyeIdx.lowerOuter]
        );
        const v2 = this.distance(
            landmarks[eyeIdx.upperInner],
            landmarks[eyeIdx.lowerInner]
        );
        
        // 水平方向の距離
        const h = this.distance(
            landmarks[eyeIdx.innerCorner],
            landmarks[eyeIdx.outerCorner]
        );
        
        if (h === 0) return 0;
        
        // 標準的なEAR計算
        const ear = (v1 + v2) / (2.0 * h);
        
        return ear;
    }
    
    /**
     * 瞼の曲率を計算（細い目の検出用）
     */
    calculateLidCurvature(landmarks, eyeIdx, lid) {
        const points = lid === 'upper' ? eyeIdx.upper : eyeIdx.lower;
        if (points.length < 3) return 0;
        
        // 3点から曲率を計算
        const p1 = landmarks[points[0]];
        const p2 = landmarks[points[Math.floor(points.length / 2)]];
        const p3 = landmarks[points[points.length - 1]];
        
        // 中点が直線からどれだけ離れているか
        const midExpected = {
            x: (p1.x + p3.x) / 2,
            y: (p1.y + p3.y) / 2
        };
        
        const deviation = p2.y - midExpected.y;
        const baseline = this.distance(p1, p3);
        
        return baseline > 0 ? deviation / baseline : 0;
    }
    
    /**
     * 視線方向を分析
     */
    analyzeGaze(landmarks) {
        const leftPupil = landmarks[this.landmarks.leftEye.pupil];
        const rightPupil = landmarks[this.landmarks.rightEye.pupil];
        
        const leftCenter = {
            x: (landmarks[this.landmarks.leftEye.innerCorner].x + 
                landmarks[this.landmarks.leftEye.outerCorner].x) / 2,
            y: (landmarks[this.landmarks.leftEye.upperLid].y + 
                landmarks[this.landmarks.leftEye.lowerLid].y) / 2
        };
        
        const rightCenter = {
            x: (landmarks[this.landmarks.rightEye.innerCorner].x + 
                landmarks[this.landmarks.rightEye.outerCorner].x) / 2,
            y: (landmarks[this.landmarks.rightEye.upperLid].y + 
                landmarks[this.landmarks.rightEye.lowerLid].y) / 2
        };
        
        // 両目の瞳位置の平均
        const avgOffsetX = ((leftPupil.x - leftCenter.x) + (rightPupil.x - rightCenter.x)) / 2;
        const avgOffsetY = ((leftPupil.y - leftCenter.y) + (rightPupil.y - rightCenter.y)) / 2;
        
        // 視線方向を判定
        let direction = 'center';
        if (avgOffsetX < -0.02) direction = 'left';
        else if (avgOffsetX > 0.02) direction = 'right';
        
        let verticalDirection = 'center';
        if (avgOffsetY < -0.015) verticalDirection = 'up';
        else if (avgOffsetY > 0.015) verticalDirection = 'down';
        
        return {
            horizontal: avgOffsetX,
            vertical: avgOffsetY,
            direction,
            verticalDirection,
            isLookingAtCamera: Math.abs(avgOffsetX) < 0.015 && Math.abs(avgOffsetY) < 0.015
        };
    }
    
    /**
     * 眉の分析
     */
    analyzeBrows(landmarks) {
        const analyze = (browIdx, eyeIdx) => {
            const browCenter = landmarks[browIdx.middle];
            const eyeTop = landmarks[eyeIdx.upperLid];
            
            // 眉と目の距離
            const distance = browCenter.y - eyeTop.y;
            
            // 眉の傾き（内側から外側）
            const inner = landmarks[browIdx.inner];
            const outer = landmarks[browIdx.outer];
            const slope = outer.y - inner.y;
            
            // 眉の高さ（顔の高さに対する相対値）
            const height = browCenter.y;
            
            return {
                eyeDistance: Math.abs(distance),
                slope,
                height,
                isRaised: distance > 0.03,
                isLowered: distance < 0.015
            };
        };
        
        return {
            left: analyze(this.landmarks.leftBrow, this.landmarks.leftEye),
            right: analyze(this.landmarks.rightBrow, this.landmarks.rightEye)
        };
    }
    
    /**
     * キャリブレーションを更新
     */
    updateCalibration(leftEye, rightEye) {
        if (this.calibration.isCalibrated) return;
        
        // サンプルを収集
        this.calibration.samples.push({
            left: leftEye.ear,
            right: rightEye.ear
        });
        this.calibration.sampleCount++;
        
        // 十分なサンプルが集まったらキャリブレーション完了
        if (this.calibration.sampleCount >= this.calibration.targetSamples) {
            const leftValues = this.calibration.samples.map(s => s.left);
            const rightValues = this.calibration.samples.map(s => s.right);
            
            // 最大値・最小値・基準値を計算
            this.calibration.leftEye = {
                maxOpen: Math.max(...leftValues),
                minOpen: Math.min(...leftValues),
                baseline: leftValues.reduce((a, b) => a + b, 0) / leftValues.length
            };
            this.calibration.rightEye = {
                maxOpen: Math.max(...rightValues),
                minOpen: Math.min(...rightValues),
                baseline: rightValues.reduce((a, b) => a + b, 0) / rightValues.length
            };
            
            // 細い目かどうかを判定
            const avgBaseline = (this.calibration.leftEye.baseline + this.calibration.rightEye.baseline) / 2;
            this.eyeType.isNarrow = avgBaseline < this.eyeType.narrowThreshold;
            this.eyeType.detected = true;
            
            this.calibration.isCalibrated = true;
            console.log('Eye calibration complete:', {
                leftBaseline: this.calibration.leftEye.baseline.toFixed(3),
                rightBaseline: this.calibration.rightEye.baseline.toFixed(3),
                isNarrowEye: this.eyeType.isNarrow
            });
        }
    }
    
    /**
     * キャリブレーションをリセット
     */
    resetCalibration() {
        this.calibration = {
            isCalibrated: false,
            leftEye: { maxOpen: 0, minOpen: 1, baseline: 0 },
            rightEye: { maxOpen: 0, minOpen: 1, baseline: 0 },
            samples: [],
            sampleCount: 0,
            targetSamples: 30
        };
        this.eyeType.detected = false;
    }
    
    /**
     * 開き具合を正規化（0-1）
     */
    normalizeOpenness(rawValue, side) {
        if (!this.calibration.isCalibrated) {
            // キャリブレーション前は推定値を使用
            // 細い目の人でも機能するよう、低い閾値を使用
            const estimatedMax = 0.3;
            const estimatedMin = 0.05;
            return Math.max(0, Math.min(1, (rawValue - estimatedMin) / (estimatedMax - estimatedMin)));
        }
        
        const cal = side === 'left' ? this.calibration.leftEye : this.calibration.rightEye;
        const range = cal.maxOpen - cal.minOpen;
        
        if (range <= 0) return 0.5;
        
        // 個人の範囲内で正規化
        let normalized = (rawValue - cal.minOpen) / range;
        
        // 細い目の場合は感度を上げる
        if (this.eyeType.isNarrow) {
            normalized = Math.pow(normalized, 0.7); // ガンマ補正で感度UP
        }
        
        return Math.max(0, Math.min(1, normalized));
    }
    
    /**
     * 値をスムージング
     */
    smoothValue(value, historyKey) {
        const history = this.history[historyKey];
        history.push(value);
        
        if (history.length > this.history.maxHistory) {
            history.shift();
        }
        
        // 加重移動平均
        let sum = 0;
        let weightSum = 0;
        history.forEach((v, i) => {
            const weight = i + 1;
            sum += v * weight;
            weightSum += weight;
        });
        
        return sum / weightSum;
    }
    
    /**
     * 目関連のAction Unitsを推定
     */
    estimateEyeAUs(leftEye, rightEye, brows, blendshapes) {
        const aus = {};
        
        // AU1: 眉内側挙上
        if (blendshapes.browInnerUp) {
            aus['AU1'] = blendshapes.browInnerUp;
        } else {
            const innerRaise = Math.max(
                brows.left.isRaised ? 0.5 : 0,
                brows.right.isRaised ? 0.5 : 0
            );
            aus['AU1'] = innerRaise;
        }
        
        // AU2: 眉外側挙上
        const browOuterUp = Math.max(
            blendshapes.browOuterUpLeft || 0,
            blendshapes.browOuterUpRight || 0
        );
        aus['AU2'] = browOuterUp;
        
        // AU4: 眉下制
        const browDown = Math.max(
            blendshapes.browDownLeft || 0,
            blendshapes.browDownRight || 0
        );
        aus['AU4'] = browDown;
        
        // AU5: 上瞼挙上（目を大きく開く）
        // 正規化された開き具合を使用（細い目対応）
        const eyeWide = Math.max(
            blendshapes.eyeWideLeft || 0,
            blendshapes.eyeWideRight || 0
        );
        const openness = (leftEye.normalizedOpenness + rightEye.normalizedOpenness) / 2;
        aus['AU5'] = Math.max(eyeWide, openness > 0.8 ? (openness - 0.8) * 5 : 0);
        
        // AU6: 頬挙上（目を細める/笑顔）
        const cheekSquint = Math.max(
            blendshapes.cheekSquintLeft || 0,
            blendshapes.cheekSquintRight || 0
        );
        aus['AU6'] = cheekSquint;
        
        // AU7: 瞼緊張（目を細める）
        const eyeSquint = Math.max(
            blendshapes.eyeSquintLeft || 0,
            blendshapes.eyeSquintRight || 0
        );
        // 細い目の場合は閾値を調整
        const squintThreshold = this.eyeType.isNarrow ? 0.4 : 0.5;
        const normalizedSquint = openness < squintThreshold ? 
            (squintThreshold - openness) / squintThreshold : 0;
        aus['AU7'] = Math.max(eyeSquint, normalizedSquint);
        
        // AU43: 目閉じ
        const eyeClosed = openness < 0.15 ? (0.15 - openness) / 0.15 : 0;
        aus['AU43'] = eyeClosed;
        
        // AU45: 瞬き
        const blink = Math.max(
            blendshapes.eyeBlinkLeft || 0,
            blendshapes.eyeBlinkRight || 0
        );
        aus['AU45'] = Math.max(blink, eyeClosed);
        
        // AU61-66: 視線方向
        aus['AU61'] = Math.max(blendshapes.eyeLookUpLeft || 0, blendshapes.eyeLookUpRight || 0);
        aus['AU62'] = Math.max(blendshapes.eyeLookDownLeft || 0, blendshapes.eyeLookDownRight || 0);
        aus['AU63'] = Math.max(blendshapes.eyeLookInLeft || 0, blendshapes.eyeLookOutRight || 0);
        aus['AU64'] = Math.max(blendshapes.eyeLookOutLeft || 0, blendshapes.eyeLookInRight || 0);
        
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
     * 2点間の距離を計算
     */
    distance(p1, p2) {
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
}

// グローバルエクスポート
window.EyeAnalyzer = EyeAnalyzer;
