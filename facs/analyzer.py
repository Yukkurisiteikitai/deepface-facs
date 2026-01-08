import numpy as np
import cv2
from typing import Optional, List, Dict
import time

from .core.interfaces import ILandmarkDetector, IAUDetector, IIntensityEstimator, IEmotionMapper, IVisualizer
from .core.models import FaceData, AnalysisResult
from .core.enums import DetectorType
from .detectors import (
    LandmarkDetectorFactory, FeatureExtractor, AUDetector,
    FaceAligner, DeepFaceAnalyzer, DeepFaceLandmarkConverter
)
from .estimators import IntensityEstimator, EmotionMapper
from .visualization import FACSVisualizer, InteractiveFACSVisualizer
from .config import AU_DEFINITIONS

class FACSAnalyzer:
    """
    FACS分析器（DeepFace統合版）
    """
    
    def __init__(
        self,
        landmark_detector: Optional[ILandmarkDetector] = None,
        au_detector: Optional[IAUDetector] = None,
        intensity_estimator: Optional[IIntensityEstimator] = None,
        emotion_mapper: Optional[IEmotionMapper] = None,
        visualizer: Optional[IVisualizer] = None,
        use_mediapipe: bool = True,
        use_deepface: bool = True,
        predictor_path: Optional[str] = None,
        interactive: bool = False
    ):
        """
        Args:
            use_mediapipe: MediaPipeを使用するか
            use_deepface: DeepFaceを併用するか（顔の向き推定に使用）
            interactive: インタラクティブモード
        """
        detector_type = DetectorType.MEDIAPIPE if use_mediapipe else DetectorType.DLIB
        
        self._landmark_detector = landmark_detector or LandmarkDetectorFactory.create(
            detector_type, predictor_path
        )
        self._feature_extractor = FeatureExtractor(use_alignment=True)
        self._face_aligner = FaceAligner()
        self._au_detector = au_detector or AUDetector()
        self._intensity_estimator = intensity_estimator or IntensityEstimator()
        self._emotion_mapper = emotion_mapper or EmotionMapper()
        
        # DeepFace（オプション）
        self._use_deepface = use_deepface
        self._deepface: Optional[DeepFaceAnalyzer] = None
        if use_deepface:
            try:
                self._deepface = DeepFaceAnalyzer(detector_backend='retinaface')
            except Exception as e:
                print(f"DeepFace初期化スキップ: {e}")
                self._deepface = None
        
        # 可視化
        if interactive:
            self._visualizer = visualizer or InteractiveFACSVisualizer()
        else:
            self._visualizer = visualizer or FACSVisualizer()
        
        self._interactive = interactive
    
    def analyze(self, image: np.ndarray) -> AnalysisResult:
        """画像を分析"""
        start_time = time.time()
        result = AnalysisResult()
        
        # DeepFaceで補助情報を取得
        deepface_result = None
        if self._deepface and self._deepface.is_available:
            df_results = self._deepface.analyze(image)
            if df_results:
                deepface_result = df_results[0]
        
        # ランドマーク検出
        landmarks = self._landmark_detector.detect_landmarks(image)
        
        # MediaPipeで検出できない場合、DeepFaceのランドマークを使用
        if landmarks is None and deepface_result and deepface_result.landmarks:
            landmarks = DeepFaceLandmarkConverter.convert_5_to_68(
                deepface_result.landmarks,
                deepface_result.face_rect
            )
        
        if landmarks is None:
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result
        
        # 顔のアライメント情報を取得
        alignment = self._face_aligner.compute_alignment(landmarks)
        
        # DeepFaceの向き情報で補完
        if deepface_result:
            # DeepFaceの推定値が有効な場合は使用
            if abs(deepface_result.roll) > 1 or abs(deepface_result.yaw) > 1:
                alignment.roll = deepface_result.roll
                alignment.yaw = deepface_result.yaw
                alignment.pitch = deepface_result.pitch
        
        # 特徴量抽出（回転補正済み）
        distances = self._feature_extractor.compute_distances(landmarks)
        angles = self._feature_extractor.compute_angles(landmarks)
        
        # アライメント情報を角度に追加
        angles['face_roll'] = alignment.roll
        angles['face_yaw'] = alignment.yaw
        angles['face_pitch'] = alignment.pitch
        
        # FaceDataを構築
        faces = self._landmark_detector.detect_faces(image)
        face_rect = faces[0] if faces else None
        result.face_data = FaceData(
            rect=face_rect,
            landmarks=landmarks,
            distances=distances,
            angles=angles
        )
        
        # AU検出
        result.au_results = self._au_detector.detect_all(landmarks, distances, angles)
        
        # 強度推定
        result.intensity_results = self._intensity_estimator.estimate_all(result.au_results)
        
        # FACSコード生成
        result.facs_code = self._intensity_estimator.format_facs_code(result.intensity_results)
        
        # 感情マッピング
        result.emotions = self._emotion_mapper.map(result.au_results, result.intensity_results)
        
        # DeepFaceの感情と比較・補完
        if deepface_result:
            result = self._merge_emotion_results(result, deepface_result)
        
        result.valence, result.arousal = self._emotion_mapper.get_valence_arousal(
            result.au_results, result.intensity_results
        )
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _merge_emotion_results(self, result: AnalysisResult, 
                               deepface_result: 'DeepFaceResult') -> AnalysisResult:
        """FACSとDeepFaceの感情結果をマージ"""
        # DeepFaceの感情スコアを参考情報として追加
        # FACSのAUベースの感情推定を主として使用
        
        df_emotions = deepface_result.emotion
        df_dominant = deepface_result.dominant_emotion
        
        # FACSで検出されなかった感情がDeepFaceで高スコアの場合、信頼度を調整
        for emotion_result in result.emotions:
            emotion_name = emotion_result.emotion.lower()
            if emotion_name in df_emotions:
                df_score = df_emotions[emotion_name] / 100.0
                # DeepFaceのスコアが高く、FACSの信頼度が低い場合は補正
                if df_score > 0.5 and emotion_result.confidence < 0.3:
                    emotion_result.confidence = min(
                        emotion_result.confidence + df_score * 0.2,
                        0.8
                    )
        
        # 再ソート
        result.emotions.sort(key=lambda x: x.confidence, reverse=True)
        
        return result
    
    def visualize(self, image: np.ndarray, result: AnalysisResult) -> np.ndarray:
        """分析結果を可視化"""
        return self._visualizer.create_analysis_panel(image, result)
    
    def analyze_and_visualize(self, image: np.ndarray) -> tuple:
        """分析と可視化を同時実行"""
        result = self.analyze(image)
        vis_image = self.visualize(image, result)
        return result, vis_image
    
    def show_interactive(self, image: np.ndarray, result: AnalysisResult,
                         window_name: str = "FACS Analysis") -> np.ndarray:
        """インタラクティブ表示"""
        if isinstance(self._visualizer, InteractiveFACSVisualizer):
            return self._visualizer.show_interactive(image, result, window_name)
        return self.visualize(image, result)
    
    def analyze_realtime(self, camera_id: int = 0, window_name: str = "FACS Analysis"):
        """リアルタイム分析"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"カメラを開けません: {camera_id}")
        
        print("Press 'q' to quit, 's' to save screenshot")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result, vis_frame = self.analyze_and_visualize(frame)
            
            # 処理時間と顔の向きを表示
            info_text = f"{result.processing_time_ms:.1f}ms"
            if result.face_data and result.face_data.angles:
                roll = result.face_data.angles.get('face_roll', 0)
                yaw = result.face_data.angles.get('face_yaw', 0)
                info_text += f" | R:{roll:+.0f} Y:{yaw:+.0f}"
            
            cv2.putText(vis_frame, info_text, (10, vis_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(window_name, vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"facs_capture_{int(time.time())}.png"
                cv2.imwrite(filename, vis_frame)
                print(f"Saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def analyze_video(self, video_path: str, output_path: Optional[str] = None,
                      frame_skip: int = 1) -> List[AnalysisResult]:
        """動画を分析"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps / frame_skip, (width + 420, height))
        
        results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                result, vis_frame = self.analyze_and_visualize(frame)
                results.append(result)
                
                if writer:
                    writer.write(vis_frame)
                
                print(f"\rProcessing: {frame_count}/{total_frames}", end="")
            
            frame_count += 1
        
        print()
        cap.release()
        if writer:
            writer.release()
        
        return results
    
    @staticmethod
    def list_all_aus() -> List[Dict]:
        return [
            {
                "number": au.au_number,
                "name": au.name,
                "description": au.description,
                "muscular_basis": au.muscular_basis
            }
            for au in AU_DEFINITIONS.values()
        ]
    
    @staticmethod
    def get_au_info(au_number: int) -> Optional[Dict]:
        au = AU_DEFINITIONS.get(au_number)
        if au is None:
            return None
        return {
            "number": au.au_number,
            "name": au.name,
            "description": au.description,
            "muscular_basis": au.muscular_basis,
            "landmarks_involved": au.landmarks_involved
        }
