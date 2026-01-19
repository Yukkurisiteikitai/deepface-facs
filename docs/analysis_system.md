# FACS Analysis System - 技術仕様書

## 1. システム概要

本システムは、**Facial Action Coding System (FACS)** に基づいて顔の微表情を解析し、Action Unit (AU) の検出と感情推定を行うライブラリです。

### 1.1 アーキテクチャ図

```
┌─────────────────────────────────────────────────────────────────────┐
│                           入力画像 (BGR)                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LandmarkDetector (顔検出)                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   MediaPipe     │  │      dlib       │  │    OpenCV       │     │
│  │  (478点 → 68点) │  │    (68点)       │  │  (フォールバック) │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ landmarks: np.ndarray (68, 2)
┌─────────────────────────────────────────────────────────────────────┐
│                    FeatureExtractor (特徴量抽出)                     │
│  ┌─────────────────────────┐  ┌─────────────────────────┐          │
│  │   compute_distances()   │  │   compute_angles()      │          │
│  │   - Eye Aspect Ratio    │  │   - Brow angles         │          │
│  │   - Mouth dimensions    │  │   - Mouth angles        │          │
│  │   - Brow distance       │  │                         │          │
│  └─────────────────────────┘  └─────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ distances: Dict, angles: Dict
┌─────────────────────────────────────────────────────────────────────┐
│                      AUDetector (AU検出)                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  VectorizedAUDetector (NumPy最適化版)                        │   │
│  │  - 12種類のAUを同時検出                                       │   │
│  │  - AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU12, AU15, AU25, AU26, AU43│
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ au_results: Dict[int, AUDetectionResult]
┌─────────────────────────────────────────────────────────────────────┐
│                   IntensityEstimator (強度推定)                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  FACS標準強度スケール (A-E)                                   │   │
│  │  ABSENT(0) → TRACE(A) → SLIGHT(B) → MARKED(C) → SEVERE(D) → MAX(E)│
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ intensity_results: Dict[int, IntensityResult]
┌─────────────────────────────────────────────────────────────────────┐
│                    EmotionMapper (感情推定)                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  AU組み合わせ → 感情カテゴリ + Valence/Arousal               │   │
│  │  - Happiness: AU6 + AU12                                     │   │
│  │  - Sadness: AU1 + AU4 + AU15                                 │   │
│  │  - etc.                                                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       AnalysisResult (出力)                          │
│  - facs_code: "AU1C+AU4B+AU12D"                                     │
│  - emotions: [EmotionResult, ...]                                   │
│  - valence: -1.0 ~ +1.0                                             │
│  - arousal: 0.0 ~ +1.0                                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. コンポーネント詳細

### 2.1 LandmarkDetector（ランドマーク検出器）

#### 2.1.1 インターフェース定義

```python
class ILandmarkDetector(ABC):
    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """顔の矩形領域 (x, y, width, height) を検出"""
        pass
    
    @abstractmethod
    def detect_landmarks(self, image: np.ndarray,
                         face_rect: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """68点ランドマークを検出。戻り値: (68, 2) の座標配列"""
        pass
```

#### 2.1.2 68点ランドマークの定義

| インデックス範囲 | 領域 | 説明 |
|-----------------|------|------|
| 0-16 | 顔輪郭 (Jawline) | 右こめかみ → 顎 → 左こめかみ |
| 17-21 | 右眉 | 外側 → 内側 |
| 22-26 | 左眉 | 内側 → 外側 |
| 27-30 | 鼻筋 | 眉間 → 鼻先 |
| 31-35 | 鼻底 | 左 → 右 |
| 36-41 | 右目 | 外側角から時計回り |
| 42-47 | 左目 | 内側角から時計回り |
| 48-59 | 外側唇 | 右角から時計回り |
| 60-67 | 内側唇 | 右角から時計回り |

#### 2.1.3 MediaPipe → 68点マッピング

MediaPipe Face Mesh (478点) から dlib互換の68点形式への変換マッピング:

```python
MEDIAPIPE_TO_68_MAPPING = [
    # 顔の輪郭 (0-16)
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400,
    
    # 右眉 (17-21)
    70, 63, 105, 66, 107,
    
    # 左眉 (22-26)
    336, 296, 334, 293, 300,
    
    # 鼻筋 (27-30)
    168, 6, 197, 195,
    
    # 鼻底 (31-35)
    98, 97, 2, 326, 327,
    
    # 右目 (36-41)
    33, 246, 161, 160, 159, 158,
    
    # 左目 (42-47)
    362, 466, 388, 387, 386, 385,
    
    # 外側唇 (48-59)
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375,
    
    # 内側唇 (60-67)
    78, 191, 80, 81, 82, 13, 312, 311,
]
```

---

### 2.2 FeatureExtractor（特徴量抽出器）

#### 2.2.1 距離特徴量

| 特徴量名 | 計算式 | 説明 |
|----------|--------|------|
| `right_eye_aspect_ratio` | `(‖p37-p41‖ + ‖p38-p40‖) / (2 × ‖p36-p39‖)` | 右目の開き具合 |
| `left_eye_aspect_ratio` | `(‖p43-p47‖ + ‖p44-p46‖) / (2 × ‖p42-p45‖)` | 左目の開き具合 |
| `mouth_width` | `‖p48 - p54‖` | 口の横幅 |
| `mouth_height_outer` | `‖p51 - p57‖` | 外唇の縦幅 |
| `mouth_height_inner` | `‖p62 - p66‖` | 内唇の縦幅 |
| `brow_distance` | `‖p21 - p22‖` | 眉間の距離 |
| `eye_distance` | `‖mean(p36:42) - mean(p42:48)‖` | 両目の中心間距離（正規化基準） |

#### 2.2.2 Eye Aspect Ratio (EAR) の詳細

```
        p37
         *
    p36 *   * p38
         \ /
          X  (瞳孔位置)
         / \
    p41 *   * p40
         *
        p39

EAR = (‖p37-p41‖ + ‖p38-p40‖) / (2 × ‖p36-p39‖)
```

- **EAR > 0.28**: 目が大きく開いている (AU5: Upper Lid Raiser)
- **EAR ≈ 0.25**: 通常の開き
- **EAR < 0.20**: 目が閉じている (AU43: Eyes Closed)

---

### 2.3 AUDetector（Action Unit検出器）

#### 2.3.1 検出対象AU一覧

| AU番号 | 名称 | 筋肉 | 検出方法 |
|--------|------|------|----------|
| AU1 | Inner Brow Raiser | 前頭筋内側部 | 眉内側と鼻筋の距離 |
| AU2 | Outer Brow Raiser | 前頭筋外側部 | 眉外側と目の外角の距離 |
| AU4 | Brow Lowerer | 皺眉筋 | 眉間の距離減少 |
| AU5 | Upper Lid Raiser | 上眼瞼挙筋 | EAR増加 |
| AU6 | Cheek Raiser | 眼輪筋 | 目の下端と口角の距離減少 |
| AU7 | Lid Tightener | 眼輪筋 | EAR減少（軽度） |
| AU9 | Nose Wrinkler | 鼻根筋 | 鼻先と上唇の距離減少 |
| AU12 | Lip Corner Puller | 大頬骨筋 | 口角の上昇 + 口の横幅増加 |
| AU15 | Lip Corner Depressor | 口角下制筋 | 口角の下降 |
| AU25 | Lips Part | 口輪筋 | 内唇の開き |
| AU26 | Jaw Drop | 咀嚼筋群 | 外唇の開き |
| AU43 | Eyes Closed | 眼輪筋 | EAR大幅減少 |

#### 2.3.2 AU検出アルゴリズム

各AUのスコア計算式（`eye_dist` = 目の間の距離で正規化）:

```python
# AU1: Inner Brow Raiser
right_dist = (landmarks[27][1] - landmarks[21][1]) / eye_dist
left_dist = (landmarks[27][1] - landmarks[22][1]) / eye_dist
score_au1 = clip((avg_dist - 0.18) / 0.12, 0.0, 1.0)

# AU4: Brow Lowerer
brow_dist_normalized = brow_distance / eye_dist
score_au4 = clip((0.35 - brow_dist_normalized) / 0.12, 0.0, 1.0)

# AU5: Upper Lid Raiser
avg_ear = (right_ear + left_ear) / 2
score_au5 = clip((avg_ear - 0.28) / 0.12, 0.0, 1.0)

# AU12: Lip Corner Puller
right_elev = (landmarks[51][1] - landmarks[48][1]) / eye_dist
left_elev = (landmarks[51][1] - landmarks[54][1]) / eye_dist
mouth_width_norm = mouth_width / eye_dist
score_au12 = clip((elev - 0.02) / 0.06, 0, 1) * 0.7 + clip((width - 0.48) / 0.1, 0, 1) * 0.3

# AU43: Eyes Closed
score_au43 = clip((0.2 - avg_ear) / 0.15, 0.0, 1.0)
```

#### 2.3.3 閾値設定

| パラメータ | 値 | 説明 |
|------------|-----|------|
| `low_threshold` | 0.15 | 検出閾値（これ以上でdetected=True） |
| `high_threshold` | 0.40 | 高強度閾値 |

---

### 2.4 IntensityEstimator（強度推定器）

#### 2.4.1 FACS標準強度スケール

| 強度 | ラベル | スコア範囲 | 説明 |
|------|--------|-----------|------|
| ABSENT | - | score < 0.075 | 不在 |
| TRACE | A | 0.075 ≤ score < 0.15 | 痕跡 |
| SLIGHT | B | 0.15 ≤ score < 0.264 | 軽度 |
| MARKED | C | 0.264 ≤ score < 0.40 | 顕著 |
| SEVERE | D | 0.40 ≤ score < 0.52 | 重度 |
| MAXIMUM | E | score ≥ 0.52 | 最大 |

#### 2.4.2 FACSコード生成

```python
# 例: AU1が強度C、AU4が強度B、AU12が強度Dの場合
facs_code = "AU1C+AU4B+AU12D"

# アルゴリズム
def format_facs_code(intensity_results: Dict[int, IntensityResult]) -> str:
    active = [r for r in intensity_results.values() 
              if r.intensity != AUIntensity.ABSENT]
    if not active:
        return "Neutral"
    
    sorted_results = sorted(active, key=lambda x: x.au_number)
    return "+".join(f"AU{r.au_number}{r.intensity_label}" for r in sorted_results)
```

---

### 2.5 EmotionMapper（感情推定器）

#### 2.5.1 感情定義

| 感情 | 必須AU | オプションAU | Valence | Arousal |
|------|--------|-------------|---------|---------|
| Happiness | AU6, AU12 | AU25 | +0.8 | +0.6 |
| Sadness | AU1, AU4, AU15 | AU6, AU17 | -0.7 | -0.3 |
| Surprise | AU1, AU2, AU5, AU26 | AU27 | +0.2 | +0.8 |
| Fear | AU1, AU2, AU4, AU5, AU7, AU20, AU26 | - | -0.8 | +0.9 |
| Anger | AU4, AU5, AU7, AU23 | AU17 | -0.9 | +0.7 |
| Disgust | AU9, AU15 | AU16 | -0.6 | +0.3 |
| Contempt | AU12 (片側), AU14 | - | -0.4 | +0.2 |
| Neutral | - | - | 0.0 | 0.0 |

#### 2.5.2 感情スコア計算

```python
def calculate_emotion_score(au_results, emotion_def):
    # 必須AUのマッチング
    required_matched = sum(1 for au in emotion_def.required_aus 
                          if au in au_results and au_results[au].detected)
    required_ratio = required_matched / len(emotion_def.required_aus)
    
    # オプションAUのボーナス
    optional_matched = sum(1 for au in emotion_def.optional_aus 
                          if au in au_results and au_results[au].detected)
    optional_bonus = optional_matched * 0.1
    
    # 必須AUの信頼度平均
    confidence_sum = sum(au_results[au].confidence 
                        for au in emotion_def.required_aus 
                        if au in au_results and au_results[au].detected)
    avg_confidence = confidence_sum / max(required_matched, 1)
    
    # 最終スコア
    score = required_ratio * 0.6 + avg_confidence * 0.3 + optional_bonus
    return min(score, 1.0)
```

#### 2.5.3 Valence-Arousal モデル

```
        Arousal (+1.0)
             │
    Fear     │     Surprise
     ●       │       ●
             │
    Anger ●  │       ● Happiness
─────────────┼─────────────► Valence
   (-1.0)    │      (+1.0)
    Disgust  │
      ●      │
    Sadness  │
      ●      │
             │
        (-1.0)
```

---

## 3. データモデル

### 3.1 AUDetectionResult

```python
@dataclass
class AUDetectionResult:
    au_number: int           # AU番号 (1-46)
    name: str                # AU名称
    detected: bool           # 検出されたか
    confidence: float        # 信頼度 (0.0-1.0)
    intensity: AUIntensity   # 強度レベル
    raw_score: float         # 生スコア
    asymmetry: float         # 左右非対称度 (-1.0 ~ +1.0)
```

### 3.2 AnalysisResult

```python
@dataclass
class AnalysisResult:
    timestamp: float                              # タイムスタンプ
    face_data: Optional[FaceData]                 # 顔データ
    au_results: Dict[int, AUDetectionResult]      # AU検出結果
    intensity_results: Dict[int, IntensityResult] # 強度推定結果
    emotions: List[EmotionResult]                 # 感情推定結果（信頼度降順）
    facs_code: str                                # FACSコード文字列
    valence: float                                # 感情価 (-1.0 ~ +1.0)
    arousal: float                                # 覚醒度 (0.0 ~ +1.0)
    processing_time_ms: float                     # 処理時間
```

---

## 4. 処理フロー

### 4.1 単一画像の分析

```python
def analyze(image: np.ndarray) -> AnalysisResult:
    # 1. ランドマーク検出
    landmarks = landmark_detector.detect_landmarks(image)
    if landmarks is None:
        return AnalysisResult()  # 無効な結果
    
    # 2. 特徴量抽出
    distances = feature_extractor.compute_distances(landmarks)
    angles = feature_extractor.compute_angles(landmarks)
    
    # 3. AU検出
    au_results = au_detector.detect_all(landmarks, distances, angles)
    
    # 4. 強度推定
    intensity_results = intensity_estimator.estimate_all(au_results)
    facs_code = intensity_estimator.format_facs_code(intensity_results)
    
    # 5. 感情推定
    emotions = emotion_mapper.map(au_results, intensity_results)
    valence, arousal = emotion_mapper.get_valence_arousal(au_results, intensity_results)
    
    return AnalysisResult(...)
```

### 4.2 リアルタイム処理（並列版）

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Camera    │────▶│   Input Queue    │────▶│   Worker    │
│   Thread    │     │  (FrameData)     │     │   Process   │
└─────────────┘     └──────────────────┘     └──────┬──────┘
                                                    │
                                                    ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Display   │◀────│   Output Queue   │◀────│   FACS      │
│   Thread    │     │  (ResultData)    │     │   Analysis  │
└─────────────┘     └──────────────────┘     └─────────────┘
```

---

## 5. 設定パラメータ

### 5.1 分析モード

| モード | スケール | 閾値 | 用途 |
|--------|----------|------|------|
| `realtime` | 0.5x | 0.40 | リアルタイム処理（30fps目標） |
| `balanced` | 1.0x | 0.30 | バランス型 |
| `accurate` | 1.0x | 0.20 | 高精度分析 |

### 5.2 パフォーマンス目標

| 環境 | モード | 目標FPS | 処理時間 |
|------|--------|---------|----------|
| M1 Mac | realtime | 30+ fps | < 33ms |
| M1 Mac | balanced | 15-20 fps | < 66ms |
| Intel i7 | realtime | 20-25 fps | < 50ms |

---

## 6. 制限事項

1. **顔の向き**: 正面 ±30° の範囲で最も精度が高い
2. **照明条件**: 均一な照明が推奨（逆光・強い影は精度低下）
3. **解像度**: 顔領域が最低 64x64 ピクセル必要
4. **遮蔽**: マスク・メガネ・髪による遮蔽は精度に影響
5. **複数人**: 同時に1人のみ分析（最も大きい顔を優先）

---

## 7. 参考文献

1. Ekman, P., & Friesen, W. V. (1978). *Facial Action Coding System: A Technique for the Measurement of Facial Movement*. Consulting Psychologists Press.
2. Lucey, P., et al. (2010). *The Extended Cohn-Kanade Dataset (CK+)*. CVPR Workshop.
3. Lugaresi, C., et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines*. arXiv:1906.08172.
