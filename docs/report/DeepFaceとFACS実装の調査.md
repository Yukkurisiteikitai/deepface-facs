# **DeepFaceライブラリの技術的動作原理とFacial Action Coding System（FACS）の統合に向けた包括的調査報告書**

## **1\. 序論：現代顔認識技術の潮流とDeepFaceの立ち位置**

コンピュータビジョンにおける顔認識技術は、過去十年で深層学習（ディープラーニング）の導入により、人間の認識精度を凌駕するレベルに達した。この進化を支えるのは、膨大なデータセットを用いた畳み込みニューラルネットワーク（CNN）による高次元の特徴量抽出能力である。その中で、DeepFaceライブラリは、Pythonベースの軽量かつ強力なハイブリッド・フレームワークとして、学術研究から実用アプリケーションまで幅広い支持を得ている 1。

DeepFaceの最大の特徴は、単一のライブラリでありながら、VGG-Face、FaceNet、OpenFace、DeepFace（Facebook）、DeepID、ArcFace、Dlib、SFace、GhostFaceNet、Buffalo\_Lといった、世界最高水準の顔認識モデルを統合的にラッピングしている点にある 1。これにより、開発者は複雑なアルゴリズムの深淵に立ち入ることなく、数行のコードで「検証（Verification）」、「認識（Recognition）」、「属性分析（Attribute Analysis）」といった高度なタスクを実行可能となっている 1。

本報告書では、DeepFaceが採用している現代的な顔認識パイプラインの動作原理を、検出、整列、表現、照合の各フェーズにわたって厳密に調査する。また、顔の筋肉運動を客観的に記述するFacial Action Coding System（FACS）の理論的背景を整理し、これをDeepFaceの既存フレームワークにどのように統合・実装しうるか、その技術的手法と将来的な可能性について多角的な検討を行う。

## **2\. DeepFaceライブラリの基本アーキテクチャとパイプライン**

DeepFaceは、TensorFlowおよびKerasを基盤として構築されており、顔認識に必要なプロセスを5つの共通ステージ（検出、整列、正規化、表現、検証）に分解して管理している 1。このモジュール化された設計により、ユーザーは特定のステージで異なるアルゴリズム（例えば、検出にはRetinaFace、表現にはArcFace）を自由に組み合わせて選択することが可能となっている 1。

### **2.1 顔認識の5段階プロセス**

顔認識は、単なる画像の照合ではなく、以下の厳密なステップを経て行われる 1。

1. **検出（Detection）**: 入力画像内から顔の領域（バウンディングボックス）を特定する。  
2. **整列（Alignment）**: 顔のポーズや傾きを補正し、標準的な正面向きの画像へと変換する。  
3. **正規化（Normalization）**: 輝度、コントラスト、および画像サイズを、後続のモデルが期待する形式に調整する 2。  
4. **表現（Representation / Embedding）**: 顔画像を数百から数千次元の数値ベクトル（埋め込み）に変換する。  
5. **検証・照合（Verification / Matching）**: 二つのベクトル間の距離（類似度）を計算し、同一人物か否かを判定する 1。

DeepFaceはこれらのプロセスをバックグラウンドで自動処理するが、精度と速度のトレードオフを最適化するためには、各段階で採用されているアルゴリズムの性質を理解することが不可欠である 1。

## **3\. 顔検出（Face Detection）のアルゴリズムと特性**

顔検出は、認識パイプライン全体の精度の約42%を左右する極めて重要な段階である 1。DeepFaceは複数のバックエンドをサポートしており、それぞれ異なる計算手法を採用している。

### **3.1 主要な検出バックエンドの比較**

| 検出バックエンド | 基本原理 | 主な利点 | 欠点 |
| :---- | :---- | :---- | :---- |
| **OpenCV (Haar)** | Haar-like特徴を用いたカスケード分類器 10。 | 極めて高速、軽量、CPU環境での動作に適する 1。 | 照明変化、回転、オクルージョンに弱い 1。 |
| **SSD (Single Shot)** | CNNベースのシングルショットマルチボックス検出器 12。 | 速度と精度のバランスが良く、リアルタイム性に優れる 1。 | 小さな顔や複雑な角度の検出精度に限界がある 1。 |
| **MTCNN** | 3段階のカスケードCNN（P-Net, R-Net, O-Net）による検出とランドマーク特定 6。 | 5つのランドマークを同時に特定でき、整列に直結する 1。 | 推論速度が比較的遅く、計算リソースを要する 1。 |
| **RetinaFace** | マルチタスク損失関数を用いたSOTAレベルの検出器 6。 | 5つの顔ランドマーク検出において極めて高い精度と堅牢性を誇る 1。 | 推論速度が最も遅く、GPUの利用が推奨される 1。 |
| **MediaPipe** | モバイル環境向けに最適化された軽量CNN 1。 | 非常に高速でリアルタイムビデオ解析に適し、詳細なメッシュ抽出が可能 1。 | 厳密な精度面ではRetinaFaceに及ばない場合がある。 |
| **YOLOv8 Face** | 物体検出モデルYOLOv8を顔検出に特化させたモデル 8。 | 複雑なシーンでも高い検出率を維持しつつ、一定の速度を保つ 6。 | リソース消費がそれなりに大きい。 |

### **3.2 ランドマーク検出の役割**

MTCNNやRetinaFaceのようなモデルは、バウンディングボックスの特定に加えて、左右の目、鼻、口の両端といった「フィデューシャルポイント（基準点）」を検出する 6。これらの座標情報は、次段階のアフィン変換による顔の整列において数学的なアンカーとして機能する 6。特に、低解像度の画像や厳しい角度がついた顔画像において、ランドマークベースの検出器を選択することは、システム全体の堅牢性を担保する上で決定的な要因となる 9。

## **4\. 顔の整列（Face Alignment）と正規化**

顔の整列は、認識精度を約6%向上させる効果がある 1。これは、カメラに対する顔の角度（ヨー、ピッチ、ロール）のバリエーションを吸収し、後続の特徴抽出モデルが常に一貫した構造的配置を持つ画像を受け取れるようにするためである 6。

### **4.1 アフィン変換の幾何学的補正**

DeepFaceにおける整列プロセスは、主に2次元のアフィン変換（Affine Transformation）に基づいている 17。具体的には、検出された左右の目の中心座標を水平線上に配置するための回転行列の算出が行われる。

左右の目の中心をそれぞれ $E\_L(x\_1, y\_1)$ および $E\_R(x\_2, y\_2)$ とすると、両目を結ぶ直線の傾き $\\theta$ は以下の式で求められる。

$$\\Delta y \= y\_2 \- y\_1 \\\\ \\Delta x \= x\_2 \- x\_1 \\\\ \\theta \= \\arctan\\left(\\frac{\\Delta y}{\\Delta x}\\right) \\times \\frac{180}{\\pi}$$  
算出した角度 $\\theta$ に基づき、画像の中心を回転軸とした回転行列 $M$ を構成し、OpenCVの warpAffine 関数等を適用することで、目を水平に補正した画像を生成する 17。さらに、顔のスケール（大きさ）を標準化し、余白部分にパディングを施すことで、アスペクト比を維持したままモデル指定の入力解像度（例：$152 \\times 152$ や $224 \\times 224$）へとリサイズを行う 5。

### **4.2 3D正面化の試み**

一部の高度な実装（オリジナルのFacebook DeepFace論文の手法など）では、67点のランドマークを用いた3Dモデリングによる正面化（Frontalization）が行われる 16。これは、デローネ三角形分割を用いて、2次元画像を3D形状にマッピングし、非平面的な回転（横向きの顔など）を擬似的に正面に引き戻す手法である 16。しかし、計算コストの増大とテクスチャの歪みのリスクがあるため、現在のライブラリレベルのDeepFaceでは、より効率的な2Dアフィン変換がデフォルトとして採用されている 17。

## **5\. 特徴量抽出（Representation / Embedding）のアルゴリズム**

整列された顔画像は、深層畳み込みニューラルネットワークに入力され、最終的には低次元（通常128次元から2622次元）の埋め込みベクトルへと圧縮される 1。このベクトルは、その顔の「デジタル指紋」として機能する 2。

### **5.1 主要モデルのアーキテクチャ詳細**

| モデル | アーキテクチャの根幹 | 埋め込み次元 | 特徴 |
| :---- | :---- | :---- | :---- |
| **VGG-Face** | VGG16 (16層CNN) 14。 | 2622D 8。 | DeepFaceのデフォルトモデル。高い精度だが、パラメータ数が多く（約1.2億）、推論が重い 1。 |
| **FaceNet** | Inception-ResNet 14。 | 128D / 512D 2。 | トリプレットロスを用いた学習により、コンパクトで強力な埋め込みを実現。128バイトで個人の特定が可能 25。 |
| **ArcFace** | ResNet100ベース 11。 | 512D 11。 | Additive Angular Margin Lossを採用し、クラス間の分離性を極限まで高めた現在のSOTAモデル 11。 |
| **OpenFace** | Torch NN4 (軽量CNN) 14。 | 128D 3。 | 処理が非常に高速。モバイルやリアルタイムシステム向け 3。 |
| **DeepID** | マルチスケールCNN 29。 | 160D程度。 | 初めて人間レベルの精度を超えたモデルの一つ。局所的な特徴抽出に優れる 3。 |

### **5.2 損失関数の革新：トリプレットロスとAngular Margin**

モデルが「誰が誰であるか」を学ぶ過程で、損失関数の設計が決定的な役割を果たす。

* **Triplet Loss (FaceNet)**: アンカー（基準顔）、ポジティブ（本人）、ネガティブ（他人）の3つを同時に学習させる。アンカーとポジティブの距離を縮め、アンカーとネガティブの距離をマージン以上に広げることで、識別能力を向上させる 25。  
* **ArcFace (Additive Angular Margin Loss)**: ソフトマックス損失関数に角度マージン $m$ を追加する。特徴量 $x\_i$ とクラス中心 $W\_j$ の間の角度 $\\theta$ に対して、$\\cos(\\theta \+ m)$ を適用することで、同一クラス内の凝集度とクラス間の分離度を幾何学的に強化する 27。

これらの数学的アプローチにより、顔画像はライティングやポーズの変動に不変な（Invariant）特徴空間へと射影される 11。

## **6\. 照合・検証（Verification / Matching）の数学的基盤**

二つの顔画像が同一人物か否かの判定は、それらの埋め込みベクトル間の距離 $\\text{dist}(v\_1, v\_2)$ を計算し、特定のしきい値 $\\tau$ と比較することで行われる 2。

### **6.1 距離メトリックの比較**

DeepFaceは主に以下の3種類の距離指標をサポートしている 1。

1. コサイン類似度 (Cosine Similarity):  
   ベクトルの向きの類似性を測る。DeepFaceのデフォルト設定である 1。

   $$\\text{dist}\_{\\cos} \= 1 \- \\frac{A \\cdot B}{\\|A\\| \\|B\\|}$$

   この指標は、ベクトルの絶対的な大きさ（輝度やコントラストの影響を受けやすい）ではなく、特徴のパターンに重きを置くため、非常に堅牢である 34。  
2. ユークリッド距離 (Euclidean Distance / L2 Norm):  
   ベクトル空間上の直線距離を測る 36。

   $$d(u, v) \= \\sqrt{\\sum\_{i=1}^{n} (u\_i \- v\_i)^2}$$  
3. L2正規化ユークリッド距離 (Euclidean L2):  
   ベクトルを事前に単位長（$\\|v\\| \= 1$）に正規化した上でユークリッド距離を計算する 34。数学的に、正規化後のユークリッド距離の2乗はコサイン距離の2倍に相当するため、提供される情報は同等であるが、スケールが異なる 34。

   $$\\text{dist}\_{L2}^2 \= 2 \\times \\text{dist}\_{\\cos}$$

### **6.2 しきい値の最適化**

認識の精度（Accuracy）は、しきい値 $\\tau$ の設定に強く依存する。DeepFaceでは、モデルと距離メトリックの各組み合わせに対して、LFW（Labeled Faces in the Wild）データセット等を用いて事前にチューニングされたデフォルトのしきい値が設定されている 32。

| モデル | メトリック | しきい値（目安） |
| :---- | :---- | :---- |
| VGG-Face | Cosine | 0.40 33 |
| FaceNet | Cosine | 0.40 33 |
| ArcFace | Cosine | 0.68 33 |
| FaceNet512 | Cosine | 0.30 33 |

これらのしきい値を調整することで、セキュリティレベル（False Acceptance Rate vs False Rejection Rate）をアプリケーションの要件に応じて制御可能である 11。

## **7\. Facial Action Coding System (FACS) の理論的体系**

顔認識が「個人のアイデンティティ」を特定するものであるのに対し、Facial Action Coding System (FACS) は「顔の動きそのもの」を解剖学的な観点から客観的に記述するためのシステムである 37。1978年にポール・エクマンらによって発表されたこの体系は、感情の解釈を排除し、単に「どの筋肉がどのように動いたか」を記述することに特化している 15。

### **7.1 アクションユニット (Action Units, AU)**

FACSの最小構成単位はアクションユニット（AU）と呼ばれる 37。それぞれのAUは、特定の顔筋または筋群の収縮に対応している。

| AU番号 | 名称 | 対応する解剖学的筋肉 | 感情との関連例 |
| :---- | :---- | :---- | :---- |
| **AU 1** | Inner Brow Raiser | 前頭筋（内側） | 悲しみ、驚き 38 |
| **AU 2** | Outer Brow Raiser | 前頭筋（外側） | 驚き 38 |
| **AU 4** | Brow Lowerer | 皺眉筋 | 怒り、集中 38 |
| **AU 6** | Cheek Raiser | 眼輪筋 | 真正の笑顔（デュシェンヌ・スマイル） 38 |
| **AU 9** | Nose Wrinkle | 鼻筋 | 嫌悪、痛み 39 |
| **AU 12** | Lip Corner Puller | 大頬骨筋 | 喜び、笑顔 38 |
| **AU 15** | Lip Corner Depressor | 口角下制筋 | 悲しみ、落胆 38 |
| **AU 17** | Chin Raiser | 頤筋 | 疑念、悲しみ 38 |
| **AU 25** | Lips Part | 口唇下制筋 | 驚き、会話 38 |

### **7.2 強度と時間的ダイナミクス**

FACSでは、AUの発生（Presence）だけでなく、その強度をA（微小）からE（最大）までの5段階で評価する 37。また、表情の発生からピーク（Apex）、消失（Offset）までの時間的推移を記録することで、偽りの笑顔（Micro-expressions）や隠された感情の兆候を分析することが可能となる 38。

## **8\. FACSのDeepFaceへの実装・統合手法の検討**

DeepFaceには現在、カテゴリカルな感情認識（怒り、嫌悪、恐怖、喜び、悲しみ、驚き、無表情）のモジュールが含まれているが、これは画像全体の分類問題として解かれている 1。これをFACSベースのAU検出へと拡張・統合するためには、以下の3つの主要な技術的アプローチが考えられる。

### **8.1 3Dランドマークを用いた回帰分析手法**

DeepFaceの既存の検出バックエンド（MediaPipeやRetinaFace）を拡張し、詳細な顔メッシュ（478点程度のランドマーク）を抽出する 39。

* **手法**: ランドマークの $x, y, z$ 座標を正規化し、軽量なニューラルネットワーク（FCN）に入力する 39。  
* **モデル構成**:  
  * **入力層**: 3D座標の連結ベクトル。  
  * **中間層**: 128個程度のニューロンを持つReLU活性化層 40。  
  * **出力層**: 各AUの強度を予測する回帰層、または発生を予測するシグモイド層 40。  
* **利点**: 画像データを直接扱わないため、計算負荷が低く、個人のプライバシーを保護する「アノニマイザー」として機能しつつ、筋肉の微細な動きを捉えることができる 40。

### **8.2 特徴抽出バックボーンからの転移学習**

VGG-FaceやResNetといったDeepFaceの強力なバックボーンを利用し、AU検出用にファインチューニングを行う手法である 43。

* **手法**: 埋め込みベクトルの手前の層（ボトルネック層）から特徴量を抽出し、複数のバイナリ分類器（Binary Relevance方式）を並列に配置する 46。  
* **データセット**: CK+（Extended Cohn-Kanade）やDISFA（Dataset for Emotional Facial Action Analysis）といったAUアノテーション付きのデータセットを使用して、各AUの活性化パターンを学習させる 43。  
* **可能性**: DeepFaceの analyze 関数のアクションに au を追加し、既存の感情モデルと同様のインターフェースでAUスコアを返却するように統合することが可能である 32。

### **8.3 OpenFace 3.0の統合**

OpenFace 3.0は、ランドマーク検出、視線推定、AU認識（26項目）を統合した強力なツールキットである 50。DeepFaceは既に「OpenFace」という名前の認識モデルをラップしているが、これは顔識別用であり、AU認識用ではない 1。

* **統合案**: OpenFace 3.0のAU検出モジュールをDeepFaceの新しいバックエンドとして正式にラッピングする。  
* **メリット**: OpenFace 3.0は非正面顔や動的な環境においても堅牢なAU検出を実現しており、DeepFaceの「軽量なラッパー」としての有用性を飛躍的に高めることができる 26。

## **9\. FACS統合による実用アプリケーションの展望**

DeepFaceとFACSの統合は、単なる技術的な拡張にとどまらず、多分野でのパラダイムシフトをもたらす可能性がある。

### **9.1 医療分野における痛みと病状の検出**

表情の減少（ハイポミミア）を特徴とするパーキンソン病の診断支援や、発話困難な患者の痛み検出において、AU分析は不可欠なツールとなる 40。

* **痛み検出のメカニズム**: AU4（眉の引き下げ）、AU6/7（眼窩の緊張）、AU9/10（鼻のしわ/唇の引き上げ）の組み合わせをTransformerモデル等で時系列処理することで、高い精度で痛みの有無を判別できる 39。  
* **有効性**: 研究によれば、全AUを用いるよりも、重要な8つのAU（AU5, 6, 8, 9, 10, 12, 14, 18）に絞ることで、計算コストを抑えつつSOTAに匹敵する精度を維持できることが示されている 39。

### **9.2 ヒューマン・コンピュータ・インタラクション（HCI）**

対話型AIやロボットにおいて、ユーザーの微細な表情の変化をFACSレベルで捉えることで、エンゲージメントや混乱、不満の兆候を早期に察知し、応答を動的に調整することが可能となる 51。

### **9.3 感情解析の高度化と教育・マーケティング**

従来の「喜び」という一括りの分類ではなく、AU6（頬の引き上げ）を伴う「本物の喜び」と、AU12（口角の引き上げ）のみの「儀礼的な笑顔」を区別することで、教育現場での学習意欲の測定や、広告に対する真の反応分析といった、より解像度の高いインサイトを得ることができる 15。

## **10\. 結論と提言**

本調査報告を通じて、DeepFaceライブラリは、現代的な顔認識パイプライン（検出・整列・表現・照合）を極めて洗練された形で統合しており、その動作原理は最新の深層学習の成果に基づいていることが明らかとなった。RetinaFaceによる高精度な検出と、ArcFaceによる角度マージンを活用した強力な埋め込み空間の構築は、人間レベルを超える認識精度を支える柱となっている 1。

一方で、FACSの統合は、現在のライブラリが持つ「静的な個人識別」と「大まかな感情分類」という枠組みを、「動的で解剖学的な表情解析」へと拡張する重要なステップである。特に、3Dランドマークをベースとした軽量なAU検出器の統合は、計算効率、プライバシー保護、および解析精度のバランスを保つ上で最も有望な手法と考えられる 39。

開発者および研究者に対する提言として、DeepFaceの柔軟なアーキテクチャを活かし、以下の実装ロードマップを推奨する。

1. **3Dランドマーク抽出の標準化**: extract\_faces 時にMediaPipe等の詳細メッシュをデフォルトで取得・保持する。  
2. **AU解析アクションの追加**: DeepFace.analyze(actions=\['au'\]) のインターフェースを通じて、主要なAction Unitsの強度と存在確率を返却する機能を実装する。  
3. **時系列モジュールの統合**: 動画解析において、AUの推移をLSTMやTransformerで処理するプラグインを導入し、マイクロ表情や痛みの検出といった高付加価値タスクをサポートする。

FACSの統合されたDeepFaceは、単なるセキュリティツールを超え、人間理解のためのバイオメトリック・プラットフォームとしての地位を確立するであろう。

---

*本報告書は、提供された研究資料および現代のコンピュータビジョンにおける学術的知見に基づき、専門家向けに構成されたものである。*

#### **Works cited**

1. deepface-batching \- PyPI, accessed January 7, 2026, [https://pypi.org/project/deepface-batching/](https://pypi.org/project/deepface-batching/)  
2. serengil/deepface: A Lightweight Face Recognition and Facial Attribute Analysis (Age, Gender, Emotion and Race) Library for Python \- GitHub, accessed January 7, 2026, [https://github.com/serengil/deepface](https://github.com/serengil/deepface)  
3. Master Facial Recognition with DeepFace in Python \- Viso Suite, accessed January 7, 2026, [https://viso.ai/computer-vision/deepface/](https://viso.ai/computer-vision/deepface/)  
4. deepface · PyPI, accessed January 7, 2026, [https://pypi.org/project/deepface/0.0.24/](https://pypi.org/project/deepface/0.0.24/)  
5. 开水君/deepface \- Gitee, accessed January 7, 2026, [https://gitee.com/kaishuijun/deepface](https://gitee.com/kaishuijun/deepface)  
6. A Comprehensive Guide to Building a Face Recognition System \- Hailo Community, accessed January 7, 2026, [https://community.hailo.ai/t/a-comprehensive-guide-to-building-a-face-recognition-system/8803](https://community.hailo.ai/t/a-comprehensive-guide-to-building-a-face-recognition-system/8803)  
7. DeepFace: Closing the Gap to Human-Level Performance in Face Verification, accessed January 7, 2026, [https://research.facebook.com/publications/deepface-closing-the-gap-to-human-level-performance-in-face-verification/](https://research.facebook.com/publications/deepface-closing-the-gap-to-human-level-performance-in-face-verification/)  
8. eye-square/wet-deepface: A Lightweight Face Recognition and Facial Attribute Analysis (Age, Gender, Emotion and Race) Library for Python \- GitHub, accessed January 7, 2026, [https://github.com/eye-square/wet-deepface](https://github.com/eye-square/wet-deepface)  
9. DeepFace in Production: Lessons from deploying a 1M+ monthly validation application to production | by DevJ | Medium, accessed January 7, 2026, [https://medium.com/@devenderj2013/deepface-in-production-lessons-from-deploying-a-1m-monthly-validation-application-to-production-a57a9f4674b4](https://medium.com/@devenderj2013/deepface-in-production-lessons-from-deploying-a-1m-monthly-validation-application-to-production-a57a9f4674b4)  
10. Face Alignment for Face Recognition in Python within OpenCV \- Sefik Ilkin Serengil, accessed January 7, 2026, [https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/)  
11. ArcFace: Additive Angular Margin Loss for Deep Face Recognition \- ResearchGate, accessed January 7, 2026, [https://www.researchgate.net/publication/338506499\_ArcFace\_Additive\_Angular\_Margin\_Loss\_for\_Deep\_Face\_Recognition](https://www.researchgate.net/publication/338506499_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition)  
12. DEEPFACE \- Kaggle, accessed January 7, 2026, [https://www.kaggle.com/datasets/sayantankirtaniya/deepface](https://www.kaggle.com/datasets/sayantankirtaniya/deepface)  
13. InsightFace: Open Source Deep Face Analysis Library \- 2D&3D, accessed January 7, 2026, [https://www.insightface.ai/](https://www.insightface.ai/)  
14. Understanding DeepFace and Its Powerful Models for Face ..., accessed January 7, 2026, [https://medium.com/@p4prince2/understanding-deepface-and-its-powerful-models-for-face-recognition-a55413706832](https://medium.com/@p4prince2/understanding-deepface-and-its-powerful-models-for-face-recognition-a55413706832)  
15. FACS: Face Detection, Facial Landmark Detection and Coding \- Marketing Analytics Solutions, accessed January 7, 2026, [https://www.ashokcharan.com/Marketing-Analytics/\~bm-facial-coding-FACS.php/\~ar-marketing-education-fluffy-and-weak.php](https://www.ashokcharan.com/Marketing-Analytics/~bm-facial-coding-FACS.php/~ar-marketing-education-fluffy-and-weak.php)  
16. Deep Face Recognition \- GeeksforGeeks, accessed January 7, 2026, [https://www.geeksforgeeks.org/machine-learning/deep-face-recognition/](https://www.geeksforgeeks.org/machine-learning/deep-face-recognition/)  
17. Face Alignment with OpenCV and Python \- PyImageSearch, accessed January 7, 2026, [https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/](https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)  
18. A Study of Face Alignment Methods in Unmasked and Masked Face Recognition \- uu .diva, accessed January 7, 2026, [https://uu.diva-portal.org/smash/get/diva2:1805927/FULLTEXT01.pdf](https://uu.diva-portal.org/smash/get/diva2:1805927/FULLTEXT01.pdf)  
19. face alignment algorithm on images \- Stack Overflow, accessed January 7, 2026, [https://stackoverflow.com/questions/12046462/face-alignment-algorithm-on-images](https://stackoverflow.com/questions/12046462/face-alignment-algorithm-on-images)  
20. The Math and Code Behind Aligning Faces | by Sabbir Ahmed | Analytics Vidhya \- Medium, accessed January 7, 2026, [https://medium.com/analytics-vidhya/the-math-and-code-behind-aligning-faces-59fcd6664a8c](https://medium.com/analytics-vidhya/the-math-and-code-behind-aligning-faces-59fcd6664a8c)  
21. Affine Align Transformations: A Practical Guide to Image Alignment and Transformation | by Babykrishna Rayaguru | Medium, accessed January 7, 2026, [https://medium.com/@babykrishna/affine-align-transformations-a-practical-guide-to-image-alignment-and-transformation-8844bff2aefd](https://medium.com/@babykrishna/affine-align-transformations-a-practical-guide-to-image-alignment-and-transformation-8844bff2aefd)  
22. deepface-custom \- PyPI, accessed January 7, 2026, [https://pypi.org/project/deepface-custom/](https://pypi.org/project/deepface-custom/)  
23. (PDF) Improving Low-Light Face Recognition using DeepFace Embedding and Multi-Layer Perceptron \- ResearchGate, accessed January 7, 2026, [https://www.researchgate.net/publication/397862463\_Improving\_Low-Light\_Face\_Recognition\_using\_DeepFace\_Embedding\_and\_Multi-Layer\_Perceptron](https://www.researchgate.net/publication/397862463_Improving_Low-Light_Face_Recognition_using_DeepFace_Embedding_and_Multi-Layer_Perceptron)  
24. 8 Different Face Recognition Models in DeepFace \- YouTube, accessed January 7, 2026, [https://www.youtube.com/watch?v=I5oZ3bLChiI](https://www.youtube.com/watch?v=I5oZ3bLChiI)  
25. Triplet loss \- Wikipedia, accessed January 7, 2026, [https://en.wikipedia.org/wiki/Triplet\_loss](https://en.wikipedia.org/wiki/Triplet_loss)  
26. (PDF) Comparison of Two Face Recognition Machine Learning Models (2022) \- SciSpace, accessed January 7, 2026, [https://scispace.com/papers/comparison-of-two-face-recognition-machine-learning-models-3rydxllw](https://scispace.com/papers/comparison-of-two-face-recognition-machine-learning-models-3rydxllw)  
27. \[1801.07698\] ArcFace: Additive Angular Margin Loss for Deep Face Recognition \- arXiv, accessed January 7, 2026, [https://arxiv.org/abs/1801.07698](https://arxiv.org/abs/1801.07698)  
28. ArcFace: Additive Angular Margin Loss for Deep Face Recognition \- Imperial College London, accessed January 7, 2026, [https://ibug.doc.ic.ac.uk/media/uploads/documents/arcface.pdf](https://ibug.doc.ic.ac.uk/media/uploads/documents/arcface.pdf)  
29. khawar-islam/deepface \- GitHub, accessed January 7, 2026, [https://github.com/khawar-islam/deepface](https://github.com/khawar-islam/deepface)  
30. Triplet Loss: Intro, Implementation, Use Cases \- V7 Go, accessed January 7, 2026, [https://www.v7labs.com/blog/triplet-loss](https://www.v7labs.com/blog/triplet-loss)  
31. ExpFace: Exponential Angular Margin Loss for Deep Face Recognition \- arXiv, accessed January 7, 2026, [https://arxiv.org/html/2509.19753v1](https://arxiv.org/html/2509.19753v1)  
32. deepface/deepface/DeepFace.py at master · serengil/deepface \- GitHub, accessed January 7, 2026, [https://github.com/serengil/deepface/blob/master/deepface/DeepFace.py](https://github.com/serengil/deepface/blob/master/deepface/DeepFace.py)  
33. \[FEATURE\]: Add Angular Distance as a Distance Metric · Issue \#1451 · serengil/deepface, accessed January 7, 2026, [https://github.com/serengil/deepface/issues/1451](https://github.com/serengil/deepface/issues/1451)  
34. \[FEATURE\]: euclidean\_l2 and cosine distance have identical ROC curves, so you could drop one of them in benchmarks. · Issue \#1417 · serengil/deepface \- GitHub, accessed January 7, 2026, [https://github.com/serengil/deepface/issues/1417](https://github.com/serengil/deepface/issues/1417)  
35. Euclidean Distance vs Cosine Similarity | Baeldung on Computer Science, accessed January 7, 2026, [https://www.baeldung.com/cs/euclidean-distance-vs-cosine-similarity](https://www.baeldung.com/cs/euclidean-distance-vs-cosine-similarity)  
36. Cosine Distance vs Dot Product vs Euclidean in vector similarity search \- Medium, accessed January 7, 2026, [https://medium.com/data-science-collective/cosine-distance-vs-dot-product-vs-euclidean-in-vector-similarity-search-227a6db32edb](https://medium.com/data-science-collective/cosine-distance-vs-dot-product-vs-euclidean-in-vector-similarity-search-227a6db32edb)  
37. Facial Action Coding System \- Wikipedia, accessed January 7, 2026, [https://en.wikipedia.org/wiki/Facial\_Action\_Coding\_System](https://en.wikipedia.org/wiki/Facial_Action_Coding_System)  
38. Facial Action Units (AUs) Analysis \- Emergent Mind, accessed January 7, 2026, [https://www.emergentmind.com/topics/facial-action-units-aus](https://www.emergentmind.com/topics/facial-action-units-aus)  
39. Facial Action Unit Detection using 3D Face Landmarks for ... \- Arinex, accessed January 7, 2026, [https://microsites.arinex.com.au/EMBC/pdf/full-paper\_268.pdf](https://microsites.arinex.com.au/EMBC/pdf/full-paper_268.pdf)  
40. A Non-Invasive Approach for Facial Action Unit Extraction and Its Application in Pain Detection \- NIH, accessed January 7, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11851526/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11851526/)  
41. Exploring facial expressions and action unit domains for Parkinson detection | PLOS One \- Research journals, accessed January 7, 2026, [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0281248](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0281248)  
42. Facial Emotion Recognition with OpenCV and Deepface: Step-by-Step Tutorial \- AI Mind, accessed January 7, 2026, [https://pub.aimind.so/facial-emotion-recognition-with-opencv-and-deepface-step-by-step-tutorial-6e3ba2c803a3](https://pub.aimind.so/facial-emotion-recognition-with-opencv-and-deepface-step-by-step-tutorial-6e3ba2c803a3)  
43. Transfer Learning for Facial Expression Recognition \- MDPI, accessed January 7, 2026, [https://www.mdpi.com/2078-2489/16/4/320](https://www.mdpi.com/2078-2489/16/4/320)  
44. \[1807.07556\] Transfer Learning for Action Unit Recognition \- arXiv, accessed January 7, 2026, [https://arxiv.org/abs/1807.07556](https://arxiv.org/abs/1807.07556)  
45. Teacher–student training and triplet loss to reduce the effect of drastic face occlusion: Application to emotion recognition, gender identification and age estimation \- PMC \- PubMed Central, accessed January 7, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8693600/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8693600/)  
46. Facial Action Unit Detection With Deep Convolutional Neural Networks \- Digital Commons @ DU, accessed January 7, 2026, [https://digitalcommons.du.edu/cgi/viewcontent.cgi?article=2812\&context=etd](https://digitalcommons.du.edu/cgi/viewcontent.cgi?article=2812&context=etd)  
47. On multi-task learning for facial action unit detection \- IEEE Xplore, accessed January 7, 2026, [http://ieeexplore.ieee.org/document/6727016/](http://ieeexplore.ieee.org/document/6727016/)  
48. Fast Adaptation of Deep Models for Facial Action Unit Detection Using Model-Agnostic Meta-Learning \- Proceedings of Machine Learning Research, accessed January 7, 2026, [http://proceedings.mlr.press/v122/lee20a/lee20a.pdf](http://proceedings.mlr.press/v122/lee20a/lee20a.pdf)  
49. DeepFace.analyze() does not output the region when no action is passed. \#729 \- GitHub, accessed January 7, 2026, [https://github.com/serengil/deepface/issues/729](https://github.com/serengil/deepface/issues/729)  
50. OpenFace 3.0: A Lightweight Multitask System for Comprehensive Facial Behavior Analysis, accessed January 7, 2026, [https://arxiv.org/html/2506.02891v1](https://arxiv.org/html/2506.02891v1)  
51. OpenFace: an open source facial behavior analysis toolkit, accessed January 7, 2026, [https://www.repository.cam.ac.uk/server/api/core/bitstreams/38d96efd-7698-4aea-825f-bda08c664807/content](https://www.repository.cam.ac.uk/server/api/core/bitstreams/38d96efd-7698-4aea-825f-bda08c664807/content)  
52. Attention Based Deep Learning models for Action Unit Recognition \- DiVA portal, accessed January 7, 2026, [http://www.diva-portal.org/smash/get/diva2:1942314/FULLTEXT01.pdf](http://www.diva-portal.org/smash/get/diva2:1942314/FULLTEXT01.pdf)  
53. Hugging Rain Man: A Novel Facial Action Units Dataset for Analyzing Atypical Facial Expressions in Children with Autism Spectrum Disorder \- arXiv, accessed January 7, 2026, [https://arxiv.org/html/2411.13797v1](https://arxiv.org/html/2411.13797v1)