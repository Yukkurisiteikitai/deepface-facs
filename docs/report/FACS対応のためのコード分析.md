# **解剖学的妥当性に基づく感情分析エンジンの再構築：FACS準拠への技術的要件と実装指針**

人間が顔を通じて表出する感情は、単なる特定の点と点の距離の変化ではなく、複雑な顔面筋の収縮と弛緩の相互作用の結果である。現行の EmotionEngine 実装、特に SimpleGeometryEngine と DeepFaceEngine は、計算効率や高レベルな推論においては一定の利便性を提供するものの、科学的・臨床的な妥当性が求められる「顔面行動符号化システム」（Facial Action Coding System: FACS）の基準には到達していない。本報告書では、提示されたコードの致命的な欠陥を詳細に分析し、真にFACSに準拠したシステムへと進化させるために不可欠な技術的要素、数学的正規化、および解剖学的マッピングの要件を網羅的に提示する。

## **現行のEmotionEngine実装における構造的課題の特定**

提示されたコードに含まれる SimpleGeometryEngine は、MediaPipe Face Meshから得られる数点の座標を利用した単純な幾何学計算に基づいている。しかし、このアプローチには解剖学的根拠が欠如しており、FACSが定義する「アクション・ユニット」（Action Unit: AU）の概念が全く反映されていない 1。

### **幾何学的エンジンの理論的限界**

SimpleGeometryEngine における mouth\_open や eyebrow\_dist といった指標は、画素座標または正規化座標（0.0から1.0）の差分を計算しているに過ぎない。この手法の最大の問題は、顔の大きさ、カメラからの距離、および個々人の解剖学的な初期状態（ニュートラル・フェイス）の差異を無視している点にある 3。例えば、特定の人物にとっての「0.08」という口の開きが、別の人にとっては叫び（Surprise）ではなく単なる会話（Speech）の一部である可能性がある。また、Y軸のみの距離計算では、唇の巻き込み（AU23, AU24）や横方向への伸展（AU20）といった、感情の質を決定づける微細な変化を捉えることができない 1。

さらに、判定ロジックが if 文によるハードコードされた閾値に基づいている点は、感情表現の非線形性を無視している 6。実際の笑顔（Happy）の形成過程では、口角の引き上げ（AU12）が先行し、その後に頬の挙上（AU6）が続くといった動的なプロセスが存在するが、現行のコードはこのプロセスを静的な一枚の画像としてしか扱っていない 6。

### **DeepFaceEngineにおける「ブラックボックス」問題**

一方、DeepFaceEngine は深層学習モデル（VGG-Face, ArcFace, 等）を活用しており、幾何学ベースの手法よりも高い精度で「支配的な感情」を分類することが可能である 8。しかし、DeepFaceはトップダウン型のアプローチであり、画像のピクセル情報から直接「Happy」や「Angry」というラベルを導き出す 8。

FACS準拠を謳うシステムにおいて重要なのは、なぜその感情と判断されたのかという「説明可能性」である。FACSは、感情を特定のAUの組み合わせとして定義する 1。例えば、真の笑顔（Duchenne Smile）はAU6とAU12の同時収縮によって定義される 1。DeepFaceのような分類器は、これらの構成要素を明示的に出力しないため、研究者が特定の表情を細分化して分析する（例：その怒りには唇のタイトナーAU23が含まれていたか）といった高度な分析には対応できない 5。

## **FACSの理論的背景とアクション・ユニットの分類体系**

FACS準拠のコードを開発するためには、まずカール＝ヘルマン・ヨルショ、ポール・エクマン、ウォレス・フリーセンらによって確立されたFACSの分類体系をエンジンの核心に据える必要がある 1。

FACSは、顔の動きを解剖学的に可能な最小単位である「アクション・ユニット」（AU）に分解する。これにより、文化や文脈に依存しない客観的な顔面行動の記述が可能となる 1。

| カテゴリ | 主要なアクション・ユニット (AU) | FACS名称 | 関連する表情 |
| :---- | :---- | :---- | :---- |
| **眉・額** | AU 1 | Inner Brow Raiser | Sadness, Surprise, Fear |
|  | AU 2 | Outer Brow Raiser | Surprise, Fear |
|  | AU 4 | Brow Lowerer | Anger, Sadness, Fear |
| **眼・瞼** | AU 5 | Upper Lid Raiser | Surprise, Anger, Fear |
|  | AU 6 | Cheek Raiser | Happiness (Duchenne) |
|  | AU 7 | Lid Tightener | Anger, Fear |
| **鼻** | AU 9 | Nose Wrinkler | Disgust |
| **唇・口** | AU 10 | Upper Lip Raiser | Disgust |
|  | AU 12 | Lip Corner Puller | Happiness, Contempt |
|  | AU 14 | Dimpler | Contempt |
|  | AU 15 | Lip Corner Depressor | Sadness, Disgust |
|  | AU 17 | Chin Raiser | Sadness, Disgust, Anger |
|  | AU 20 | Lip Stretcher | Fear |
|  | AU 23 | Lip Tightener | Anger |
| **下顎** | AU 26 | Jaw Drop | Surprise, Fear |

現行の SimpleGeometryEngine に足りないのは、これらのAUを独立して検知し、それぞれの強さを評価するロジックである。FACSでは各AUの強度をA（微か）からE（最大）までの5段階で評価する 1。

## **解剖学的メカニズムと筋肉の相関**

FACSのAUは、特定の顔面筋の収縮と1対1、あるいは1対多で対応している。コードをFACS対応にするためには、ランドマークの移動がどの筋肉の活動に由来するのかを理解し、それを特徴量として抽出する必要がある 1。

1. **大頬骨筋 (Zygomaticus major) / AU 12**: 口角を耳の方向へ引き上げる。これが SimpleGeometryEngine で扱われている mouth\_width の主な要因であるが、単なる幅ではなく、斜め上方向へのベクトルとして計算されるべきである 1。  
2. **眼輪筋 (Orbicularis oculi) / AU 6 & AU 7**: 目の周囲を囲む筋肉。AU6（外側）は頬を押し上げ、目尻にシワを作る。AU7（内側）は下瞼を引き締める 1。これらは「幸福」と「怒り」を見分けるための決定的な特徴量であるが、現行コードでは完全に無視されている。  
3. **皺眉筋 (Corrugator supercilii) / AU 4**: 眉を中央に寄せ、下方に押し下げる 1。現行コードの eyebrow\_dist はこの活動の一部を捉えている可能性があるが、垂直方向の移動だけでなく、左右の眉の距離（水平方向）の縮小も同時に計算しなければならない 1。  
4. **前頭筋 (Frontalis) / AU 1 & AU 2**: 額の筋肉。内側（AU1）が上がると悲しみや驚きの表情になり、外側（AU2）が上がると驚きの表情が強まる 2。現行コードには額のランドマーク（ID 10, 105等）を活用したAU1/2の検知ロジックが存在しない。

## **MediaPipe Face LandmarkerからFACSへの橋渡し**

MediaPipeの標準的なFace Mesh（468点）を利用する場合、ランドマークの座標からAUを推定する中間層が必要である。しかし、最新のMediaPipe FaceLandmarker タスクは、52種類の「ブレンドシェイプ・スコア」を出力する機能を備えており、これがFACS準拠への最短ルートとなる 14。

これらの52のスコアは、ARKitの規格に準拠しており、それぞれが特定のFACS AUと密接に関連している 15。

| MediaPipeブレンドシェイプ | 対応するFACS AU | 物理的アクション |
| :---- | :---- | :---- |
| browInnerUp | AU 1 | 眉の内側を引き上げる 13 |
| browDownLeft / Right | AU 4 | 眉を下げ、中央に寄せる 13 |
| eyeWideLeft / Right | AU 5 | 上瞼を大きく開く 13 |
| cheekSquintLeft / Right | AU 6 | 頬を押し上げ、目を細める 16 |
| eyeSquintLeft / Right | AU 7 | 下瞼を緊張させる 13 |
| noseSneerLeft / Right | AU 9 | 鼻筋にシワを寄せ、上唇を上げる 13 |
| mouthSmileLeft / Right | AU 12 | 口角を斜め上に引き上げる 16 |
| mouthFrownLeft / Right | AU 15 | 口角を押し下げる 16 |
| jawOpen | AU 26 / 27 | 下顎を下ろす 16 |

コードを改良する際には、landmarks.multi\_face\_landmarks から座標を計算するのではなく、FaceLandmarkerResult.face\_blendshapes からこれらのスコアを直接取得し、それをAU強度として扱うべきである 15。

## **幾何学的正規化と姿勢不変性の数学的実装**

現行の SimpleGeometryEngine に欠けている最も重要な数学的要素は「正規化」である。ピクセル単位の距離は、顔の傾きやカメラとの距離によって大きく変動するため、信頼できる特徴量にはなり得ない 3。

### **瞳孔間距離 (IPD) によるスケール正規化**

すべての幾何学的距離は、瞳孔間距離（Inter-Pupillary Distance: IPD）または眼間距離（Inter-Ocular Distance）で除算されるべきである 20。MediaPipeにおける左目中心（例：ランドマーク468）と右目中心（例：ランドマーク473）の距離 $d\_{io}$ を基準単位とする。

$$d\_{normalized} \= \\frac{\\sqrt{(x\_2 \- x\_1)^2 \+ (y\_2 \- y\_1)^2 \+ (z\_2 \- z\_1)^2}}{d\_{io}}$$  
これにより、顔がカメラに近い場合でも遠い場合でも、一貫したAU強度を算出できる 23。

### **プロクラステス解析による姿勢不変性**

顔の向き（Yaw, Pitch, Roll）の変化は、2D平面上のランドマーク位置を歪める。FACS対応コードには、検出されたランドマークを「標準的な正面顔（Canonical Face Model）」にマッピングする処理が必要である 3。MediaPipeの「Face Transform」モジュールを利用すれば、各ランドマークの3D位置から剛体変換行列を推定し、顔の回転をキャンセルした「正規化座標系」での距離測定が可能となる 24。

現行コードの SimpleGeometryEngine.analyze 内で、以下の処理を追加することが必須となる。

1. Face Meshから基準となる「アンカーポイント」（鼻先、目頭、口角のニュートラル位置）を抽出する 24。  
2. 基準モデルとの最小二乗誤差を最小化する回転・変換行列（Procrustes Analysis）を適用する 24。  
3. 変換後の3D空間において、垂直方向・水平方向の筋肉移動量を測定する 3。

## **時系列ダイナミクス：Onset, Apex, Offsetの検知**

表情は瞬間的な状態ではなく、時間的なプロセスである。真のFACS分析では、AUの強さがどのように変化したかを追跡する 1。

* **Onset（開始）**: 筋肉が収縮を始め、ニュートラルから離れる期間 1。  
* **Apex（頂点）**: 収縮が最大強度に達した期間 1。  
* **Offset（終了）**: 筋肉が弛緩し、ニュートラルに戻る期間 1。

現行のコードは analyze 関数が呼ばれるたびに独立した判定を行っている。FACS準拠のためには、フレーム間の差分を保持する「バッファリング」機能が必要である 11。例えば、驚き（Surprise）と恐怖（Fear）を区別する場合、驚きは立ち上がりが非常に速い（Rapid Onset）のに対し、恐怖は持続時間が長く、他のAU（AU20: Lip Stretcher）との複雑な同期を伴う 1。

研究データによれば、AU14（Dimpler）の強度は学習者の集中状態（Engagement）と相関があるが、これは単一フレームの閾値ではなく、数秒間の平均強度や変動幅によって評価される 28。したがって、EmotionEngine は deque 等を用いて過去数フレームのスコアを蓄積し、移動平均や分散、あるいは隠れマルコフモデル（HMM）やBi-LSTMといった時系列モデルを用いて最終的な判定を下す設計が望ましい 11。

## **発話と表情の分離：Speech Artifactの抑制**

感情分析において、発話（Talking）は「ノイズ」となる。口を開ける動作（AU25/26）は、驚きや恐怖といった感情だけでなく、単に言葉を発している際にも発生する 7。

FACS準拠のコードには、以下の二つのアプローチのいずれか（あるいは両方）を組み込む必要がある。

1. **発話検知による重み調整**: 音声入力または口の動きのリズムから発話状態を検知し、発話中は下顔部（Mouth region）のAUの寄与度を下げ、上顔部（Eyebrow, Eye region）のAUの寄与度を上げる 7。  
2. **個人特化型キャリブレーション**: 使用開始時に数秒間の「無表情」および「音読」を行わせ、各個人のニュートラルな可動範囲を学習させる。これにより、元々口角が上がっている人や、眉が低いといった個体差を吸収できる 29。

提示されたコードの SimpleGeometryEngine では、mouth\_open \> 0.08 という一律の基準が設けられているが、これは発話中の人物をすべて「Surprised」と誤判定するリスクを孕んでいる 7。

## **非対称アクション・ユニットと Contempt (軽蔑) の検知**

FACSの高度な機能の一つに、顔の左右で異なる動き（Unilateral AUs）を符号化できる点がある 1。例えば、「軽蔑（Contempt）」は、口角の一方のみが引き上げられる（Unilateral AU12）ことによって定義される 1。

現行コードの mouth\_width \= abs(left\_corner.x \- right\_corner.x) という計算は、左右の絶対的な距離を求めているため、片側の口角だけが動いた場合と、両側がわずかに動いた場合を区別できない。

FACS対応のためには、鼻の中心線（Landmark 4, 168等）を基準とした左右の対称性を評価するロジックが必要である 31。

* **左AU12強度**: distance(nose\_center, left\_corner)  
* **右AU12強度**: distance(nose\_center, right\_corner)  
* **軽蔑の判定基準**: abs(left\_AU12 \- right\_AU12) \> threshold かつ dimpler\_score \> threshold.1

## **改善された FACS 準拠型 EmotionEngine の設計案**

以下に、調査結果に基づき、提示されたコードに「何が足りないか」を補完した新しいエンジンの設計指針をまとめる。

### **1\. 特徴量抽出の刷新 (AU Detection)**

* MediaPipeのFace Meshから座標を直接使うのではなく、FaceLandmarker のブレンドシェイプを利用する 14。  
* 個々のブレンドシェイプを、解剖学的なAU番号へとマッピングする辞書を定義する 13。

### **2\. 判定ロジックの多次元化**

* 単一の if 文による分類を廃止し、感情ごとの「AUプロファイル」に基づいたスコアリングを導入する 1。

| 感情 | 必須AUの組み合わせ |
| :---- | :---- |
| **Happiness** | AU 6 (Cheek Raiser) \+ AU 12 (Lip Corner Puller) |
| **Sadness** | AU 1 (Inner Brow) \+ AU 4 (Brow Lowerer) \+ AU 15 (Lip Corner Depressor) |
| **Surprise** | AU 1 \+ AU 2 (Outer Brow) \+ AU 5 (Upper Lid) \+ AU 26 (Jaw Drop) |
| **Fear** | AU 1 \+ AU 2 \+ AU 4 \+ AU 5 \+ AU 7 \+ AU 20 (Lip Stretcher) \+ AU 26 |
| **Anger** | AU 4 \+ AU 5 \+ AU 7 \+ AU 23 (Lip Tightener) |
| **Disgust** | AU 9 (Nose Wrinkler) \+ AU 15 \+ AU 16 (Lower Lip Depressor) |
| **Contempt** | Unilateral AU 12 \+ AU 14 (Dimpler) |

### **3\. 微表情（Micro-expressions）への対応**

* 微弱な信号を強調するための増幅アルゴリズムを実装する 32。例えば、Disgustを検知する際には、noseWrinkler (AU9) のスコアに 3.0 程度の係数を掛け、微小な動きでも検出できるように調整する 32。

### **4\. 信頼性スコアの導入**

* MediaPipeが提供する min\_face\_presence\_confidence や min\_tracking\_confidence を活用し、顔の検知精度が低いフレーム（例：顔が大きく横を向いている、手が顔を覆っている）での判定をスキップまたは重み付け解除する処理を追加する 14。

## **結論：次世代 Affective Computing への展望**

提示されたコードを真にFACSに対応させるために足りない要素は、単なるプログラミング上の修正ではなく、顔面解剖学への深い洞察に基づく「中間層（AU Detection Layer）」の設計である。

幾何学ベースの SimpleGeometryEngine は、AUの概念を導入し、瞳孔間距離による正規化と姿勢不変性を備えることで、科学的なツールへと進化できる。一方で、DeepFaceEngine のような深層学習アプローチは、その判断の根拠をAUの強度として出力できる「解読可能（Explainable）」なモデルへと移行することが求められる。

MediaPipeが提供する52のブレンドシェイプは、この進化を実現するための強力な基盤である。それらをFACSのAU定義に正確にマッピングし、時間的な変化を捉えるアルゴリズムを実装することで、単なる感情ラベルの出力機から、人間の微細な心理変化を解剖学的な精度で記述する「デジタル・フェイス・デコーダー」へと昇華させることが可能である。この技術的転換は、教育におけるエンゲージメント測定、臨床における痛み（Pain）の検知、さらにはメタバースにおけるリアルタイムなアバター制御など、広範な応用分野において革新的な価値をもたらすだろう 11。

#### **Works cited**

1. Facial Action Coding System \- Wikipedia, accessed January 7, 2026, [https://en.wikipedia.org/wiki/Facial\_Action\_Coding\_System](https://en.wikipedia.org/wiki/Facial_Action_Coding_System)  
2. Facial Action Units (AUs) Analysis \- Emergent Mind, accessed January 7, 2026, [https://www.emergentmind.com/topics/facial-action-units-aus](https://www.emergentmind.com/topics/facial-action-units-aus)  
3. mediapipe/docs/solutions/face\_mesh.md at master \- GitHub, accessed January 7, 2026, [https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face\_mesh.md](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md)  
4. Facial Landmarks Detection Using Mediapipe Library \- Analytics Vidhya, accessed January 7, 2026, [https://www.analyticsvidhya.com/blog/2022/03/facial-landmarks-detection-using-mediapipe-library/](https://www.analyticsvidhya.com/blog/2022/03/facial-landmarks-detection-using-mediapipe-library/)  
5. Facial Action Coding System (FACS) \- A Visual Guidebook \- iMotions, accessed January 7, 2026, [https://imotions.com/blog/learning/research-fundamentals/facial-action-coding-system/](https://imotions.com/blog/learning/research-fundamentals/facial-action-coding-system/)  
6. Amplifying Emotions \- Zixin Zhao, accessed January 7, 2026, [https://zxnnic.github.io/amplifying-emotion/](https://zxnnic.github.io/amplifying-emotion/)  
7. Action Unit Models of Facial Expression of Emotion in the Presence of Speech \- PMC \- NIH, accessed January 7, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4267560/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4267560/)  
8. serengil/deepface: A Lightweight Face Recognition and Facial Attribute Analysis (Age, Gender, Emotion and Race) Library for Python \- GitHub, accessed January 7, 2026, [https://github.com/serengil/deepface](https://github.com/serengil/deepface)  
9. Understanding DeepFace and Its Powerful Models for Face Recognition \- Medium, accessed January 7, 2026, [https://medium.com/@p4prince2/understanding-deepface-and-its-powerful-models-for-face-recognition-a55413706832](https://medium.com/@p4prince2/understanding-deepface-and-its-powerful-models-for-face-recognition-a55413706832)  
10. Validation-study: Basic emotions and Action Units detection \- Noldus, accessed January 7, 2026, [https://noldus.com/blog/validation-study-facereader](https://noldus.com/blog/validation-study-facereader)  
11. A Non-Invasive Approach for Facial Action Unit Extraction and Its Application in Pain Detection \- MDPI, accessed January 7, 2026, [https://www.mdpi.com/2306-5354/12/2/195](https://www.mdpi.com/2306-5354/12/2/195)  
12. Facial Action Coding System (FACS) Cheat Sheet, accessed January 7, 2026, [https://melindaozel.com/facs-cheat-sheet/](https://melindaozel.com/facs-cheat-sheet/)  
13. The Ultimate Guide to Creating ARKit 52 Facial Blendshapes \- Pooya Deperson, accessed January 7, 2026, [https://pooyadeperson.com/the-ultimate-guide-to-creating-arkits-52-facial-blendshapes/](https://pooyadeperson.com/the-ultimate-guide-to-creating-arkits-52-facial-blendshapes/)  
14. Face landmark detection guide | Google AI Edge, accessed January 7, 2026, [https://ai.google.dev/edge/mediapipe/solutions/vision/face\_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)  
15. MediaPipe: Enhancing Virtual Humans to be more realistic \- Google Developers Blog, accessed January 7, 2026, [https://developers.googleblog.com/mediapipe-enhancing-virtual-humans-to-be-more-realistic/](https://developers.googleblog.com/mediapipe-enhancing-virtual-humans-to-be-more-realistic/)  
16. ARKit to FACS: Blendshape Cheat Sheet, accessed January 7, 2026, [https://melindaozel.com/arkit-to-facs-cheat-sheet/](https://melindaozel.com/arkit-to-facs-cheat-sheet/)  
17. MediaPipe Blendshapes recording and filtering | by Samer Attrah \- Medium, accessed January 7, 2026, [https://medium.com/@samiratra95/mediapipe-blendshapes-recording-and-filtering-29bd6243924e](https://medium.com/@samiratra95/mediapipe-blendshapes-recording-and-filtering-29bd6243924e)  
18. Face landmark detection guide for Python | Google AI Edge, accessed January 7, 2026, [https://ai.google.dev/edge/mediapipe/solutions/vision/face\_landmarker/python](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)  
19. Generalizing to Unseen Head Poses in Facial Expression Recognition and Action Unit Intensity Estimation \- SciSpace, accessed January 7, 2026, [https://scispace.com/pdf/generalizing-to-unseen-head-poses-in-facial-expression-2n0pqamaip.pdf](https://scispace.com/pdf/generalizing-to-unseen-head-poses-in-facial-expression-2n0pqamaip.pdf)  
20. Fast Facial Landmark Detection and Applications: A Survey \- ResearchGate, accessed January 7, 2026, [https://www.researchgate.net/publication/360119534\_Fast\_Facial\_Landmark\_Detection\_and\_Applications\_A\_Survey](https://www.researchgate.net/publication/360119534_Fast_Facial_Landmark_Detection_and_Applications_A_Survey)  
21. How to manipulate dlib landmark points \- c++ \- Stack Overflow, accessed January 7, 2026, [https://stackoverflow.com/questions/38055212/how-to-manipulate-dlib-landmark-points](https://stackoverflow.com/questions/38055212/how-to-manipulate-dlib-landmark-points)  
22. How to calculate inter pupil and inter ocular distance for facial landmarks in python, accessed January 7, 2026, [https://stackoverflow.com/questions/63785689/how-to-calculate-inter-pupil-and-inter-ocular-distance-for-facial-landmarks-in-p](https://stackoverflow.com/questions/63785689/how-to-calculate-inter-pupil-and-inter-ocular-distance-for-facial-landmarks-in-p)  
23. Finding approximate distance between facial landmarks · vladmandic human · Discussion \#310 \- GitHub, accessed January 7, 2026, [https://github.com/vladmandic/human/discussions/310](https://github.com/vladmandic/human/discussions/310)  
24. MediaPipe 3D Face Transform \- Google for Developers Blog, accessed January 7, 2026, [https://developers.googleblog.com/mediapipe-3d-face-transform/](https://developers.googleblog.com/mediapipe-3d-face-transform/)  
25. layout: forward target: https://developers.google.com/mediapipe/solutions/vision/face\_landmarker/ title: Face Mesh parent: MediaPipe Legacy Solutions nav\_order: 2 — MediaPipe v0.7.5 documentation \- Read the Docs, accessed January 7, 2026, [https://mediapipe.readthedocs.io/en/latest/solutions/face\_mesh.html](https://mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html)  
26. MediaPipe Face Mesh \- GitHub, accessed January 7, 2026, [https://github.com/google-ai-edge/mediapipe/wiki/MediaPipe-Face-Mesh](https://github.com/google-ai-edge/mediapipe/wiki/MediaPipe-Face-Mesh)  
27. A Non-Invasive Approach for Facial Action Unit Extraction and Its Application in Pain Detection \- NIH, accessed January 7, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11851526/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11851526/)  
28. Set of action units needed for basic emotions | Download Table \- ResearchGate, accessed January 7, 2026, [https://www.researchgate.net/figure/Set-of-action-units-needed-for-basic-emotions\_tbl1\_303755395](https://www.researchgate.net/figure/Set-of-action-units-needed-for-basic-emotions_tbl1_303755395)  
29. Action unit intensity regression for facial MoCap aimed towards digital humans, accessed January 7, 2026, [https://d-nb.info/1337897655/34](https://d-nb.info/1337897655/34)  
30. Action Units · TadasBaltrusaitis/OpenFace Wiki \- GitHub, accessed January 7, 2026, [https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units)  
31. How to Visualise MediaPipe's Face and Face Landmark Detection in 2D and 3D with Rerun, accessed January 7, 2026, [https://dev.to/rerunio/how-to-visualise-mediapipes-face-and-face-landmark-detection-in-2d-and-3d-with-rerun-94f](https://dev.to/rerunio/how-to-visualise-mediapipes-face-and-face-landmark-detection-in-2d-and-3d-with-rerun-94f)  
32. Seeing the Unseen: Real-Time Micro-Expression Recognition with Action Units and GPT-Based Reasoning \- MDPI, accessed January 7, 2026, [https://www.mdpi.com/2076-3417/15/12/6417](https://www.mdpi.com/2076-3417/15/12/6417)