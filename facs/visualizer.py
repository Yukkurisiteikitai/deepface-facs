import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

from .action_units import AU_DEFINITIONS, AUIntensity
from .au_detector import AUDetectionResult
from .intensity_estimator import IntensityResult
from .emotion_mapper import EmotionResult

class FACSVisualizer:
    """FACS分析結果の可視化"""
    
    INTENSITY_COLORS = {
        AUIntensity.ABSENT: (128, 128, 128), AUIntensity.TRACE: (0, 255, 0),
        AUIntensity.SLIGHT: (0, 255, 128), AUIntensity.MARKED: (0, 255, 255),
        AUIntensity.SEVERE: (0, 165, 255), AUIntensity.MAXIMUM: (0, 0, 255)
    }
    
    def __init__(self, font_scale: float = 0.5):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
    
    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray,
                       show_numbers: bool = False) -> np.ndarray:
        output = image.copy()
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(output, (int(x), int(y)), 2, (0, 255, 0), -1)
            if show_numbers:
                cv2.putText(output, str(i), (int(x) + 3, int(y) - 3), self.font, 0.3, (0, 255, 0), 1)
        return output
    
    def draw_landmark_connections(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        output = image.copy()
        connections = [
            (list(range(0, 17)), False),    # jaw
            (list(range(17, 22)), False),   # right eyebrow
            (list(range(22, 27)), False),   # left eyebrow
            (list(range(27, 31)), False),   # nose bridge
            (list(range(31, 36)), False),   # nose tip
            (list(range(36, 42)), True),    # right eye
            (list(range(42, 48)), True),    # left eye
            (list(range(48, 60)), True),    # outer mouth
            (list(range(60, 68)), True),    # inner mouth
        ]
        for indices, closed in connections:
            pts = landmarks[indices].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(output, [pts], closed, (0, 255, 0), 1)
        return output
    
    def create_analysis_panel(self, image: np.ndarray, landmarks: np.ndarray,
                              au_results: Dict[int, AUDetectionResult],
                              intensity_results: Dict[int, IntensityResult],
                              emotions: List[EmotionResult],
                              facs_code: str) -> np.ndarray:
        h, w = image.shape[:2]
        panel_width = 350
        output = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
        
        # 画像にランドマーク描画
        img_vis = self.draw_landmarks(image, landmarks)
        img_vis = self.draw_landmark_connections(img_vis, landmarks)
        output[:, :w] = img_vis
        
        # 右パネル
        panel = output[:, w:]
        panel[:] = (30, 30, 30)
        y = 25
        
        # FACSコード
        cv2.putText(panel, "FACS Code:", (10, y), self.font, 0.5, (255, 255, 255), 1)
        y += 20
        for i in range(0, len(facs_code), 35):
            cv2.putText(panel, facs_code[i:i+35], (10, y), self.font, 0.4, (0, 255, 255), 1)
            y += 18
        y += 15
        
        # 感情
        cv2.putText(panel, "Emotions:", (10, y), self.font, 0.5, (255, 255, 255), 1)
        y += 20
        for emotion in emotions[:4]:
            if emotion.confidence < 0.1:
                continue
            bar_w = int(emotion.confidence * 150)
            color = (0, 200, 0) if emotion.valence > 0 else (0, 0, 200) if emotion.valence < 0 else (200, 200, 0)
            cv2.rectangle(panel, (10, y), (10 + bar_w, y + 12), color, -1)
            cv2.rectangle(panel, (10, y), (160, y + 12), (80, 80, 80), 1)
            cv2.putText(panel, f"{emotion.emotion} ({emotion.confidence:.2f})", (170, y + 10),
                       self.font, 0.35, (200, 200, 200), 1)
            y += 18
        y += 15
        
        # Active AUs
        cv2.putText(panel, "Active AUs:", (10, y), self.font, 0.5, (255, 255, 255), 1)
        y += 20
        for au_num in sorted(au_results.keys()):
            if y > h - 20:
                break
            result = au_results[au_num]
            if not result.detected:
                continue
            intensity = intensity_results.get(au_num)
            color = self.INTENSITY_COLORS.get(intensity.intensity, (0, 255, 0)) if intensity else (0, 255, 0)
            label = intensity.intensity_label if intensity else ""
            cv2.putText(panel, f"AU{au_num}{label}: {result.name[:22]}", (10, y), self.font, 0.35, color, 1)
            y += 16
        
        return output
