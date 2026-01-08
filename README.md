# FACS (Facial Action Coding System) Analyzer

顔の微表情を解析し、Action Unit (AU) に基づいて感情を推定するPythonライブラリ。

## Abstract

本プロジェクトは、Paul Ekmanが開発したFacial Action Coding System (FACS) に基づき、顔画像から筋肉の動き（Action Units）を検出し、感情を推定するシステムです。

### 主な機能

- **Action Unit検出**: 16種類以上のAUを検出（AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU12, AU15, AU17, AU20, AU23, AU25, AU26, AU43など）
- **強度推定**: FACS標準のA-Eスケール（Trace → Maximum）で強度を評価
- **感情マッピング**: 検出されたAUの組み合わせから7つの基本感情（幸福、悲しみ、驚き、恐怖、怒り、嫌悪、軽蔑）を推定
- **Valence-Arousal推定**: 感情の価値（ポジティブ/ネガティブ）と覚醒度を数値化
- **リアルタイム分析**: カメラ入力によるリアルタイム表情分析
- **日本語対応**: ターミナル出力・可視化の日本語表示

### 技術スタック

- **MediaPipe Face Landmarker**: 478点の顔ランドマーク検出
- **DeepFace** (オプション): 補助的な感情分析・顔の向き推定
- **OpenCV**: 画像処理・可視化
- **Pillow**: 日本語フォント描画

## Installation

### PyPIからインストール（公開後）

```bash
pip install facs-analyzer
```

### ソースからインストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd analysis_face

# 開発モードでインストール
pip install -e .

# DeepFace機能を含める場合
pip install -e ".[deepface]"

# 開発用ツールを含める場合
pip install -e ".[dev]"

# すべての依存関係
pip install -e ".[all]"
```

## Usage

### 基本的な使い方

#### コマンドライン

```bash
# 画像分析
python demo.py image face.jpg

# 動画分析
python demo.py video interview.mp4 --csv

# リアルタイム分析（カメラ）
python demo.py realtime

# 2つの画像を比較
python demo.py compare before.jpg after.jpg

# フォルダ内の画像を一括分析
python demo.py batch ./images/ -o ./results/

# AU一覧を表示
python demo.py list

# 強度スケールの凡例を表示
python demo.py legend
```

#### Pythonコード

```python
from facs import FACSAnalyzer
import cv2

# 分析器を初期化
analyzer = FACSAnalyzer(use_mediapipe=True)

# 画像を読み込んで分析
image = cv2.imread("face.jpg")
result = analyzer.analyze(image)

# 結果を表示
print(f"FACSコード: {result.facs_code}")
print(f"主要感情: {result.dominant_emotion.emotion}")
print(f"Valence: {result.valence:.2f}, Arousal: {result.arousal:.2f}")

# 検出されたAUを表示
for au in result.active_aus:
    intensity = result.intensity_results.get(au.au_number)
    print(f"AU{au.au_number}[{intensity.intensity_label}]: {au.name}")

# 可視化
vis_image = analyzer.visualize(image, result)
cv2.imshow("FACS Analysis", vis_image)
cv2.waitKey(0)

# JSON出力
print(result.to_json())
```

### ライブラリとして使用

```python
from facs import FACSAnalyzer, AnalysisResult
import cv2

# 基本的な使い方
analyzer = FACSAnalyzer()
image = cv2.imread("face.jpg")
result = analyzer.analyze(image)

# 結果にアクセス
print(result.facs_code)                    # "AU6B + AU12C"
print(result.dominant_emotion.emotion)     # "happiness"
print(result.valence)                      # 0.85
print(result.arousal)                      # 0.42

# 検出されたAUをイテレート
for au in result.active_aus:
    print(f"AU{au.au_number}: {au.name} ({au.confidence:.2f})")

# JSON形式で出力
json_data = result.to_json()

# 辞書形式で取得
dict_data = result.to_dict()
```

### カスタマイズ

```python
from facs import FACSAnalyzer, AUDetector, IntensityEstimator

# カスタムコンポーネントを注入
custom_au_detector = AUDetector()
custom_au_detector.register_strategy(MyCustomAUStrategy())

analyzer = FACSAnalyzer(
    au_detector=custom_au_detector,
    use_deepface=False,  # DeepFaceを無効化
)
```

### 動画分析

```python
# 動画を分析
results = analyzer.analyze_video("video.mp4", output_path="output.mp4")

# 各フレームの結果にアクセス
for i, result in enumerate(results):
    if result.is_valid:
        print(f"Frame {i}: {result.dominant_emotion.emotion}")
```

### バッチ処理

```python
from pathlib import Path

analyzer = FACSAnalyzer()
results = {}

for image_path in Path("./images").glob("*.jpg"):
    image = cv2.imread(str(image_path))
    results[image_path.name] = analyzer.analyze(image)
```

### CLIとして使用

```bash
# インストール後
facs analyze face.jpg
facs analyze video.mp4 -o output.mp4
facs realtime
facs list

# または直接
python -m facs.cli analyze face.jpg
```

### オプション

| オプション | 説明 |
|-----------|------|
| `--no-interactive` | マウスホバー機能を無効化 |
| `--no-details` | AU詳細表示を無効化 |
| `-o, --output` | 出力ファイルパス |
| `-j, --json` | JSON形式で結果を保存 |
| `-c, --csv` | CSV形式で結果を保存 |
| `-s, --skip` | 動画のフレームスキップ数 |

## Project Structure

```
analysis_face/
├── facs/                          # メインパッケージ
│   ├── __init__.py
│   ├── analyzer.py                # FACSAnalyzer（Facadeクラス）
│   │
│   ├── core/                      # コアモジュール
│   │   ├── __init__.py
│   │   ├── interfaces.py          # 抽象インターフェース（SOLID: D）
│   │   ├── models.py              # データモデル
│   │   ├── enums.py               # 列挙型
│   │   └── terminal_display.py    # ターミナル表示
│   │
│   ├── config/                    # 設定
│   │   ├── __init__.py
│   │   └── au_definitions.py      # AU・感情定義
│   │
│   ├── detectors/                 # 検出器
│   │   ├── __init__.py
│   │   ├── landmark_detector.py   # ランドマーク検出（MediaPipe/dlib）
│   │   ├── feature_extractor.py   # 特徴量抽出
│   │   ├── face_aligner.py        # 顔の傾き補正
│   │   ├── au_detector.py         # AU検出
│   │   ├── deepface_detector.py   # DeepFace統合
│   │   ├── debug_landmarks.py     # デバッグ用
│   │   └── strategies/            # AU検出戦略（SOLID: O, S）
│   │       ├── __init__.py
│   │       └── au_strategies.py
│   │
│   ├── estimators/                # 推定器
│   │   ├── __init__.py
│   │   ├── intensity_estimator.py # 強度推定
│   │   └── emotion_mapper.py      # 感情マッピング
│   │
│   └── visualization/             # 可視化
│       ├── __init__.py
│       ├── visualizer.py          # 可視化クラス
│       └── font_manager.py        # 日本語フォント管理
│
├── demo.py                        # デモスクリプト
├── test_landmarks.py              # ランドマークテスト
├── check_versions.py              # バージョン確認
├── requirements.txt               # 依存関係
└── README.md
```

## Project Rules

### 設計原則

本プロジェクトは**SOLID原則**に基づいて設計されています。

#### S - Single Responsibility Principle（単一責任原則）
各クラスは単一の責任のみを持つ:
- `LandmarkDetector`: ランドマーク検出のみ
- `AUDetector`: AU検出のみ
- `IntensityEstimator`: 強度推定のみ
- `EmotionMapper`: 感情マッピングのみ

#### O - Open/Closed Principle（開放閉鎖原則）
新しいAU検出ロジックは`IAUDetectionStrategy`を実装するだけで追加可能:
```python
class AU99Strategy(BaseAUStrategy):
    _au_number = 99
    
    def detect(self, landmarks, distances, angles, eye_dist):
        # 新しい検出ロジック
        return score, asymmetry

# 登録
au_detector.register_strategy(AU99Strategy())
```

#### L - Liskov Substitution Principle（リスコフの置換原則）
すべての検出器は基底クラス/インターフェースと置換可能:
```python
detector: ILandmarkDetector = MediaPipeLandmarkDetector()
detector: ILandmarkDetector = DlibLandmarkDetector(path)
```

#### I - Interface Segregation Principle（インターフェース分離原則）
インターフェースは小さく分割:
- `ILandmarkDetector`
- `IAUDetector`
- `IIntensityEstimator`
- `IEmotionMapper`
- `IVisualizer`

#### D - Dependency Inversion Principle（依存性逆転原則）
`FACSAnalyzer`は具象クラスではなくインターフェースに依存:
```python
class FACSAnalyzer:
    def __init__(
        self,
        landmark_detector: Optional[ILandmarkDetector] = None,
        au_detector: Optional[IAUDetector] = None,
        ...
    ):
```

### コーディング規約

- **型ヒント**: すべての関数に型ヒントを付与
- **docstring**: クラス・関数にはdocstringを記述
- **命名規則**: 
  - クラス: PascalCase
  - 関数・変数: snake_case
  - 定数: UPPER_SNAKE_CASE
  - プライベート: `_`プレフィックス

### 依存関係

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| numpy | >=1.24.0 | 数値計算 |
| opencv-python | >=4.8.0 | 画像処理 |
| mediapipe | >=0.10.9 | ランドマーク検出 |
| Pillow | >=10.0.0 | 日本語フォント |
| deepface | >=0.0.96 | 補助分析（オプション） |
| tensorflow | >=2.15.0 | DeepFaceの依存 |
| tf-keras | >=2.15.0 | DeepFaceの依存 |

## Action Units Reference

| AU | 名前 | 説明 | 関連感情 |
|----|------|------|----------|
| 1 | Inner Brow Raiser | 眉の内側を上げる | 悲しみ、恐怖 |
| 2 | Outer Brow Raiser | 眉の外側を上げる | 驚き |
| 4 | Brow Lowerer | 眉を下げる・寄せる | 怒り、悲しみ |
| 5 | Upper Lid Raiser | 上まぶたを上げる | 驚き、恐怖 |
| 6 | Cheek Raiser | 頬を上げる | 幸福 |
| 7 | Lid Tightener | まぶたを締める | 怒り |
| 9 | Nose Wrinkler | 鼻にしわを寄せる | 嫌悪 |
| 12 | Lip Corner Puller | 口角を上げる | 幸福 |
| 15 | Lip Corner Depressor | 口角を下げる | 悲しみ |
| 25 | Lips Part | 唇を開く | 驚き |
| 26 | Jaw Drop | 顎を下げる | 驚き |

## License

MIT License

## References

- Ekman, P., & Friesen, W. V. (1978). Facial Action Coding System
- MediaPipe Face Landmarker: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
- DeepFace: https://github.com/serengil/deepface


