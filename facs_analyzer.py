import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
import time

from facs import (
    LandmarkDetector, AUDetector, AUDetectionResult,
    IntensityEstimator, IntensityResult,
    EmotionMapper, EmotionResult, FACSVisualizer, AU_DEFINITIONS
)

@dataclass
class FACSAnalysisResult:
    """FACS分析の完全な結果"""
    timestamp: float
    face_rect: Optional[Tuple[int, int, int, int]]
    landmarks: Optional[np.ndarray]
    au_results: Dict[int, AUDetectionResult] = field(default_factory=dict)
    intensity_results: Dict[int, IntensityResult] = field(default_factory=dict)
    emotions: List[EmotionResult] = field(default_factory=list)
    dominant_emotion: Optional[EmotionResult] = None
    valence: float = 0.0
    arousal: float = 0.0
    facs_code: str = "Neutral"
    asymmetry_info: Dict = field(default_factory=dict)
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "facs_code": self.facs_code,
            "active_aus": [{"au": n, "name": r.name, "confidence": r.confidence,
                           "intensity": self.intensity_results[n].intensity_label if n in self.intensity_results else None}
                          for n, r in self.au_results.items() if r.detected],
            "emotions": [{"emotion": e.emotion, "confidence": e.confidence} for e in self.emotions[:5]],
            "dominant_emotion": self.dominant_emotion.emotion if self.dominant_emotion else "neutral",
            "valence": self.valence, "arousal": self.arousal
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

class FACSAnalyzer:
    """Facial Action Coding System 分析器"""
    
    def __init__(self, predictor_path: Optional[str] = None, use_mediapipe: bool = False):
        self.landmark_detector = LandmarkDetector(predictor_path, use_mediapipe)
        self.au_detector = AUDetector()
        self.intensity_estimator = IntensityEstimator()
        self.emotion_mapper = EmotionMapper()
        self.visualizer = FACSVisualizer()
    
    def analyze(self, image: np.ndarray,
                face_rect: Optional[Tuple[int, int, int, int]] = None) -> FACSAnalysisResult:
        start_time = time.time()
        result = FACSAnalysisResult(timestamp=time.time(), face_rect=None, landmarks=None)
        
        if face_rect is None:
            faces = self.landmark_detector.detect_faces(image)
            if faces:
                face_rect = faces[0]
        
        landmarks = self.landmark_detector.detect_landmarks(image, face_rect)
        if landmarks is None:
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result
        
        result.face_rect = face_rect
        result.landmarks = landmarks
        
        distances = self.landmark_detector.compute_distances(landmarks)
        angles = self.landmark_detector.compute_angles(landmarks)
        
        result.au_results = self.au_detector.detect_all_aus(landmarks, distances, angles)
        result.intensity_results = self.intensity_estimator.estimate_all_intensities(result.au_results)
        result.facs_code = self.intensity_estimator.format_facs_code(result.intensity_results)
        result.emotions = self.emotion_mapper.map_emotion(result.au_results, result.intensity_results)
        result.dominant_emotion = result.emotions[0] if result.emotions else None
        result.valence, result.arousal = self.emotion_mapper.get_valence_arousal(result.au_results, result.intensity_results)
        result.asymmetry_info = self.emotion_mapper.detect_asymmetry(result.au_results)
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def visualize(self, image: np.ndarray, result: FACSAnalysisResult) -> np.ndarray:
        if result.landmarks is None:
            return image
        return self.visualizer.create_analysis_panel(
            image, result.landmarks, result.au_results,
            result.intensity_results, result.emotions, result.facs_code
        )
    
    def analyze_realtime(self, camera_id: int = 0):
        cap = cv2.VideoCapture(camera_id)
        print("Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = self.analyze(frame)
            vis = self.visualize(frame, result) if result.landmarks is not None else frame
            cv2.putText(vis, f"{result.processing_time_ms:.1f}ms", (10, vis.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("FACS Analysis", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    
    @staticmethod
    def list_all_aus() -> List[Dict]:
        return [{"number": au.au_number, "name": au.name, "description": au.description}
                for au in AU_DEFINITIONS.values()]
