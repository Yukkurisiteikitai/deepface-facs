import numpy as np
import cv2
from typing import Optional, List, Dict
import time

from .core.models import FaceData, AnalysisResult
from .core.enums import DetectorType, AnalysisMode
from .detectors import LandmarkDetectorFactory, FeatureExtractor, AUDetector
from .detectors.vectorized_au_detector import VectorizedAUDetector
from .detectors.optimized_feature_extractor import OptimizedFeatureExtractor
from .estimators import IntensityEstimator, EmotionMapper
from .visualization import FACSVisualizer, InteractiveFACSVisualizer
from .visualization.visualizer import LayoutConfig
from .config import AU_DEFINITIONS


class AnalysisModeConfig:
    """分析モードごとの設定"""
    
    CONFIGS = {
        AnalysisMode.REALTIME: {
            "detection_confidence": 0.3,
            "smoothing": True,
            "smoothing_window": 3,
            "skip_asymmetry": True,
            "skip_micro_expressions": True,
            "au_threshold": 0.4,
            "max_faces": 1,
            "image_scale": 0.5,
            "use_temporal_filter": True,
        },
        AnalysisMode.BALANCED: {
            "detection_confidence": 0.5,
            "smoothing": False,
            "smoothing_window": 1,
            "skip_asymmetry": False,
            "skip_micro_expressions": False,
            "au_threshold": 0.3,
            "max_faces": 5,
            "image_scale": 1.0,
            "use_temporal_filter": False,
        },
        AnalysisMode.ACCURATE: {
            "detection_confidence": 0.7,
            "smoothing": False,
            "smoothing_window": 1,
            "skip_asymmetry": False,
            "skip_micro_expressions": False,
            "au_threshold": 0.2,
            "max_faces": 10,
            "image_scale": 1.0,
            "use_temporal_filter": False,
            # 追加の精密分析オプション
            "multi_scale_detection": True,
            "landmark_refinement": True,
            "bilateral_symmetry_check": True,
            "intensity_calibration": True,
            "emotion_ensemble": True,
        },
    }
    
    @classmethod
    def get(cls, mode: AnalysisMode) -> Dict:
        return cls.CONFIGS.get(mode, cls.CONFIGS[AnalysisMode.BALANCED])


class TemporalFilter:
    """時系列フィルタ（リアルタイム用）"""
    
    def __init__(self, window_size: int = 3):
        self._window_size = window_size
        self._history: List[Dict] = []
    
    def update(self, au_results: Dict) -> Dict:
        """時系列平滑化"""
        self._history.append(au_results)
        if len(self._history) > self._window_size:
            self._history.pop(0)
        
        if len(self._history) < 2:
            return au_results
        
        # 平均化
        smoothed = {}
        for au_num in au_results.keys():
            scores = [h[au_num].raw_score for h in self._history if au_num in h]
            confs = [h[au_num].confidence for h in self._history if au_num in h]
            if scores:
                au = au_results[au_num]
                # 新しいオブジェクトを作成（元のオブジェクトを変更しない）
                from .core.models import AUDetectionResult
                smoothed[au_num] = AUDetectionResult(
                    au_number=au.au_number,
                    name=au.name,
                    detected=au.detected,
                    confidence=sum(confs) / len(confs),
                    intensity=au.intensity,
                    raw_score=sum(scores) / len(scores),
                    asymmetry=au.asymmetry
                )
        
        return smoothed if smoothed else au_results
    
    def reset(self):
        self._history.clear()


class FACSAnalyzer:
    """FACS分析器（モード対応版）"""
    
    def __init__(
        self,
        use_mediapipe: bool = True,
        use_deepface: bool = False,
        predictor_path: Optional[str] = None,
        interactive: bool = False,
        mode: AnalysisMode = AnalysisMode.BALANCED,
        layout_config: Optional[LayoutConfig] = None,
        use_optimized: bool = True,  # 最適化コンポーネントを使用
    ):
        self._mode = mode
        self._config = AnalysisModeConfig.get(mode)
        
        detector_type = DetectorType.MEDIAPIPE if use_mediapipe else DetectorType.DLIB
        self._landmark_detector = LandmarkDetectorFactory.create(detector_type, predictor_path)
        
        # 最適化コンポーネントを選択
        if use_optimized:
            self._feature_extractor = OptimizedFeatureExtractor()
            self._au_detector = VectorizedAUDetector()
        else:
            self._feature_extractor = FeatureExtractor()
            self._au_detector = AUDetector()
        
        self._intensity_estimator = IntensityEstimator()
        self._emotion_mapper = EmotionMapper()
        
        # レイアウト設定
        self._layout_config = layout_config or LayoutConfig()
        if interactive:
            self._visualizer = InteractiveFACSVisualizer(self._layout_config)
        else:
            self._visualizer = FACSVisualizer(self._layout_config)
        
        # リアルタイムモード用のフィルタ
        self._temporal_filter = TemporalFilter(self._config.get("smoothing_window", 3))
        
        # 精密モード用のキャリブレーション
        self._baseline_landmarks: Optional[np.ndarray] = None
        self._calibrated = False
    
    @property
    def mode(self) -> AnalysisMode:
        return self._mode
    
    def set_mode(self, mode: AnalysisMode):
        """分析モードを変更"""
        self._mode = mode
        self._config = AnalysisModeConfig.get(mode)
        self._temporal_filter = TemporalFilter(self._config.get("smoothing_window", 3))
        if mode != AnalysisMode.ACCURATE:
            self._calibrated = False
    
    def calibrate(self, neutral_image: np.ndarray) -> bool:
        """ニュートラル表情でキャリブレーション（精密モード用）"""
        landmarks = self._landmark_detector.detect_landmarks(neutral_image)
        if landmarks is not None:
            self._baseline_landmarks = landmarks.copy()
            self._calibrated = True
            return True
        return False
    
    def analyze(self, image: np.ndarray) -> AnalysisResult:
        """画像を分析"""
        if self._mode == AnalysisMode.REALTIME:
            return self._analyze_realtime_mode(image)
        elif self._mode == AnalysisMode.ACCURATE:
            return self._analyze_accurate_mode(image)
        else:
            return self._analyze_balanced_mode(image)
    
    def _analyze_realtime_mode(self, image: np.ndarray) -> AnalysisResult:
        """軽量・高速分析（リアルタイム向け）"""
        start_time = time.time()
        result = AnalysisResult()
        
        # 画像をスケールダウン
        scale = self._config.get("image_scale", 0.5)
        if scale < 1.0:
            small_image = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            small_image = image
        
        landmarks = self._landmark_detector.detect_landmarks(small_image)
        if landmarks is None:
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result
        
        # スケールを戻す
        if scale < 1.0:
            landmarks = landmarks / scale
        
        distances = self._feature_extractor.compute_distances(landmarks)
        angles = self._feature_extractor.compute_angles(landmarks)
        
        faces = self._landmark_detector.detect_faces(small_image)
        if faces and scale < 1.0:
            faces = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for x, y, w, h in faces]
        
        result.face_data = FaceData(
            rect=faces[0] if faces else None,
            landmarks=landmarks,
            distances=distances,
            angles=angles
        )
        
        # AU検出（簡易版）
        result.au_results = self._au_detector.detect_all(landmarks, distances, angles)
        
        # 時系列フィルタを適用
        if self._config.get("use_temporal_filter", True):
            result.au_results = self._temporal_filter.update(result.au_results)
        
        # 閾値でフィルタ
        threshold = self._config.get("au_threshold", 0.4)
        result.au_results = {
            k: v for k, v in result.au_results.items()
            if v.confidence >= threshold
        }
        
        result.intensity_results = self._intensity_estimator.estimate_all(result.au_results)
        result.facs_code = self._intensity_estimator.format_facs_code(result.intensity_results)
        result.emotions = self._emotion_mapper.map(result.au_results, result.intensity_results)
        result.valence, result.arousal = self._emotion_mapper.get_valence_arousal(
            result.au_results, result.intensity_results
        )
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _analyze_balanced_mode(self, image: np.ndarray) -> AnalysisResult:
        """バランス型分析"""
        start_time = time.time()
        result = AnalysisResult()
        
        landmarks = self._landmark_detector.detect_landmarks(image)
        if landmarks is None:
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result
        
        distances = self._feature_extractor.compute_distances(landmarks)
        angles = self._feature_extractor.compute_angles(landmarks)
        
        faces = self._landmark_detector.detect_faces(image)
        result.face_data = FaceData(
            rect=faces[0] if faces else None,
            landmarks=landmarks,
            distances=distances,
            angles=angles
        )
        
        result.au_results = self._au_detector.detect_all(landmarks, distances, angles)
        result.intensity_results = self._intensity_estimator.estimate_all(result.au_results)
        result.facs_code = self._intensity_estimator.format_facs_code(result.intensity_results)
        result.emotions = self._emotion_mapper.map(result.au_results, result.intensity_results)
        result.valence, result.arousal = self._emotion_mapper.get_valence_arousal(
            result.au_results, result.intensity_results
        )
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _analyze_accurate_mode(self, image: np.ndarray) -> AnalysisResult:
        """高精度分析（詳細分析向け）"""
        start_time = time.time()
        result = AnalysisResult()
        
        # マルチスケール検出
        landmarks_list = []
        if self._config.get("multi_scale_detection", True):
            scales = [1.0, 0.75, 1.25]
            for scale in scales:
                scaled = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image
                lm = self._landmark_detector.detect_landmarks(scaled)
                if lm is not None:
                    if scale != 1.0:
                        lm = lm / scale
                    landmarks_list.append(lm)
        else:
            lm = self._landmark_detector.detect_landmarks(image)
            if lm is not None:
                landmarks_list.append(lm)
        
        if not landmarks_list:
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result
        
        # 複数検出結果の平均化
        landmarks = np.mean(landmarks_list, axis=0)
        
        # ランドマーク精緻化
        if self._config.get("landmark_refinement", True):
            landmarks = self._refine_landmarks(image, landmarks)
        
        # ベースラインからの差分を計算（キャリブレーション済みの場合）
        if self._calibrated and self._baseline_landmarks is not None:
            delta_landmarks = landmarks - self._baseline_landmarks
        else:
            delta_landmarks = None
        
        distances = self._feature_extractor.compute_distances(landmarks)
        angles = self._feature_extractor.compute_angles(landmarks)
        
        # 左右対称性チェック
        symmetry_scores = {}
        if self._config.get("bilateral_symmetry_check", True):
            symmetry_scores = self._compute_symmetry(landmarks)
        
        faces = self._landmark_detector.detect_faces(image)
        result.face_data = FaceData(
            rect=faces[0] if faces else None,
            landmarks=landmarks,
            distances=distances,
            angles=angles
        )
        
        # AU検出（詳細版）
        result.au_results = self._au_detector.detect_all(landmarks, distances, angles)
        
        # デルタベースの補正（キャリブレーション済みの場合）
        if delta_landmarks is not None:
            result.au_results = self._adjust_by_baseline(result.au_results, delta_landmarks)
        
        # 左右非対称性を詳細に記録
        for au_num, au_result in result.au_results.items():
            if au_num in symmetry_scores:
                au_result.asymmetry = symmetry_scores[au_num]
        
        # 強度推定（キャリブレーション付き）
        if self._config.get("intensity_calibration", True) and self._calibrated:
            result.intensity_results = self._estimate_calibrated_intensity(result.au_results)
        else:
            result.intensity_results = self._intensity_estimator.estimate_all(result.au_results)
        
        result.facs_code = self._intensity_estimator.format_facs_code(result.intensity_results)
        
        # 感情推定（アンサンブル）
        if self._config.get("emotion_ensemble", True):
            result.emotions = self._emotion_ensemble(result.au_results, result.intensity_results)
        else:
            result.emotions = self._emotion_mapper.map(result.au_results, result.intensity_results)
        
        result.valence, result.arousal = self._emotion_mapper.get_valence_arousal(
            result.au_results, result.intensity_results
        )
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _refine_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """ランドマーク精緻化"""
        refined = landmarks.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # サブピクセル精度でコーナー検出
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        for i, (x, y) in enumerate(landmarks):
            # 各ランドマーク周辺で精緻化
            window_size = 5
            x_int, y_int = int(x), int(y)
            
            # 境界チェック
            if (x_int - window_size < 0 or x_int + window_size >= gray.shape[1] or
                y_int - window_size < 0 or y_int + window_size >= gray.shape[0]):
                continue
            
            try:
                corners = np.array([[[x, y]]], dtype=np.float32)
                cv2.cornerSubPix(gray, corners, (window_size, window_size), (-1, -1), criteria)
                refined[i] = corners[0, 0]
            except:
                pass
        
        return refined
    
    def _compute_symmetry(self, landmarks: np.ndarray) -> Dict[int, float]:
        """左右対称性を計算"""
        symmetry = {}
        
        # 左右対応するランドマークペア
        pairs = {
            # AU1, AU2: 眉
            1: [(17, 26), (18, 25), (19, 24), (20, 23), (21, 22)],
            2: [(17, 26), (18, 25)],
            # AU6, AU7: 目
            6: [(36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46)],
            7: [(36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46)],
            # AU12, AU15: 口
            12: [(48, 54), (49, 53), (50, 52)],
            15: [(48, 54), (49, 53), (50, 52)],
        }
        
        # 顔の中心線を計算
        center_x = (landmarks[0, 0] + landmarks[16, 0]) / 2
        
        for au_num, pair_list in pairs.items():
            diffs = []
            for left_idx, right_idx in pair_list:
                if left_idx < len(landmarks) and right_idx < len(landmarks):
                    left = landmarks[left_idx]
                    right = landmarks[right_idx]
                    
                    # 中心からの距離の差
                    left_dist = abs(left[0] - center_x)
                    right_dist = abs(right[0] - center_x)
                    
                    # Y座標の差
                    y_diff = left[1] - right[1]
                    
                    # 非対称性スコア
                    diffs.append((left_dist - right_dist, y_diff))
            
            if diffs:
                avg_x_diff = np.mean([d[0] for d in diffs])
                avg_y_diff = np.mean([d[1] for d in diffs])
                symmetry[au_num] = np.sqrt(avg_x_diff**2 + avg_y_diff**2) / 10.0
        
        return symmetry
    
    def _adjust_by_baseline(self, au_results: Dict, delta_landmarks: np.ndarray) -> Dict:
        """ベースラインからの変化で補正"""
        # 変化量の大きさを計算
        magnitude = np.linalg.norm(delta_landmarks, axis=1).mean()
        
        # 変化が小さい場合はスコアを下げる
        adjustment = min(magnitude / 5.0, 1.0)
        
        from .core.models import AUDetectionResult
        adjusted = {}
        for au_num, au in au_results.items():
            adjusted[au_num] = AUDetectionResult(
                au_number=au.au_number,
                name=au.name,
                detected=au.detected,
                confidence=au.confidence * (0.5 + 0.5 * adjustment),
                intensity=au.intensity,
                raw_score=au.raw_score * (0.5 + 0.5 * adjustment),
                asymmetry=au.asymmetry
            )
        
        return adjusted
    
    def _estimate_calibrated_intensity(self, au_results: Dict) -> Dict:
        """キャリブレーション付き強度推定"""
        # 通常の推定を行った後、ベースラインを考慮して調整
        return self._intensity_estimator.estimate_all(au_results)
    
    def _emotion_ensemble(self, au_results: Dict, intensity_results: Dict) -> List:
        """アンサンブル感情推定"""
        # 複数の方法で感情を推定し、統合
        emotions1 = self._emotion_mapper.map(au_results, intensity_results)
        
        # Valence-Arousalモデルからの推定
        v, a = self._emotion_mapper.get_valence_arousal(au_results, intensity_results)
        
        # 統合（重み付け平均）
        for emotion in emotions1:
            # Valence-Arousalとの整合性をチェック
            expected_valence = emotion.valence
            if abs(v - expected_valence) < 0.3:
                emotion.confidence *= 1.1  # 整合性が高い場合はブースト
            else:
                emotion.confidence *= 0.9  # 不整合の場合は減衰
        
        # 再ソート
        emotions1.sort(key=lambda x: x.confidence, reverse=True)
        
        return emotions1
    
    def visualize(self, image: np.ndarray, result: AnalysisResult) -> np.ndarray:
        return self._visualizer.create_analysis_panel(image, result)
    
    def analyze_and_visualize(self, image: np.ndarray) -> tuple:
        result = self.analyze(image)
        return result, self.visualize(image, result)
    
    def show_interactive(self, image: np.ndarray, result: AnalysisResult,
                        window_name: str = "FACS") -> np.ndarray:
        if isinstance(self._visualizer, InteractiveFACSVisualizer):
            return self._visualizer.show_interactive(image, result, window_name)
        return self.visualize(image, result)
    
    def analyze_realtime(self, camera_id: int = 0):
        """リアルタイム分析（自動的にリアルタイムモードを使用）"""
        original_mode = self._mode
        self.set_mode(AnalysisMode.REALTIME)
        
        cap = cv2.VideoCapture(camera_id)
        print("Press 'q' to quit, 'm' to change mode")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            result, vis = self.analyze_and_visualize(frame)
            
            # モード表示
            mode_text = f"Mode: {self._mode.value} | {result.processing_time_ms:.1f}ms"
            cv2.putText(vis, mode_text, (10, vis.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("FACS Realtime", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                # モード切り替え
                modes = list(AnalysisMode)
                current_idx = modes.index(self._mode)
                next_idx = (current_idx + 1) % len(modes)
                self.set_mode(modes[next_idx])
                print(f"Mode changed to: {self._mode.value}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 元のモードに戻す
        self.set_mode(original_mode)
    
    def analyze_video(self, video_path: str, output_path: Optional[str] = None,
                      frame_skip: int = 1) -> List[AnalysisResult]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = None
        if output_path:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                    fps / frame_skip, (w + 350, h))
        
        results, count = [], 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_skip == 0:
                result, vis = self.analyze_and_visualize(frame)
                results.append(result)
                if writer:
                    writer.write(vis)
                print(f"\rProcessing: {count}/{total}", end="")
            count += 1
        
        print()
        cap.release()
        if writer:
            writer.release()
        return results
    
    @staticmethod
    def list_all_aus() -> List[Dict]:
        return [{"number": au.au_number, "name": au.name, "description": au.description,
                 "muscular_basis": au.muscular_basis} for au in AU_DEFINITIONS.values()]
    
    @staticmethod
    def get_au_info(au_number: int) -> Optional[Dict]:
        au = AU_DEFINITIONS.get(au_number)
        if not au:
            return None
        return {"number": au.au_number, "name": au.name, "description": au.description,
                "muscular_basis": au.muscular_basis, "landmarks_involved": au.landmarks_involved}
