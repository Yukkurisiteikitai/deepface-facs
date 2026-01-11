import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from ..core.interfaces import IVisualizer
from ..core.models import AnalysisResult
from ..core.enums import AUIntensity
from ..config import AU_DEFINITIONS

class FACSVisualizer(IVisualizer):
    """FACS可視化"""
    
    INTENSITY_COLORS = {
        AUIntensity.ABSENT: (128, 128, 128), AUIntensity.TRACE: (0, 255, 0),
        AUIntensity.SLIGHT: (0, 255, 128), AUIntensity.MARKED: (0, 255, 255),
        AUIntensity.SEVERE: (0, 165, 255), AUIntensity.MAXIMUM: (0, 0, 255)
    }
    
    def __init__(self, font_scale: float = 0.5):
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = font_scale
        self._hover_au: Optional[int] = None
        self._au_regions: Dict[int, Tuple[int, int, int, int]] = {}
    
    def create_analysis_panel(self, image: np.ndarray, result: AnalysisResult) -> np.ndarray:
        if not result.is_valid:
            return image
        
        h, w = image.shape[:2]
        panel_width = 350
        output = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
        
        # ランドマーク描画
        img_vis = self._draw_landmarks(image, result.face_data.landmarks)
        output[:, :w] = img_vis
        
        # パネル
        panel = output[:, w:]
        panel[:] = (30, 30, 30)
        y = 25
        
        cv2.putText(panel, f"FACS: {result.facs_code[:35]}", (10, y), self._font, 0.4, (0, 255, 255), 1)
        y += 25
        
        for e in result.emotions[:3]:
            if e.confidence < 0.1: continue
            bar_w = int(e.confidence * 120)
            color = (0, 200, 0) if e.valence > 0 else (0, 0, 200) if e.valence < 0 else (200, 200, 0)
            cv2.rectangle(panel, (10, y), (10 + bar_w, y + 12), color, -1)
            cv2.putText(panel, f"{e.emotion} ({e.confidence:.2f})", (140, y + 10), self._font, 0.35, (200, 200, 200), 1)
            y += 18
        
        y += 10
        for au in result.active_aus[:10]:
            if y > h - 20: break
            intensity = result.intensity_results.get(au.au_number)
            color = self.INTENSITY_COLORS.get(intensity.intensity if intensity else AUIntensity.ABSENT, (0, 255, 0))
            label = intensity.intensity_label if intensity else ""
            cv2.putText(panel, f"AU{au.au_number}{label}: {au.name[:20]}", (10, y), self._font, 0.35, color, 1)
            y += 16
        
        return output
    
    def _draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        output = image.copy()
        for (x, y) in landmarks:
            cv2.circle(output, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        connections = [(range(0, 17), False), (range(17, 22), False), (range(22, 27), False),
                       (range(36, 42), True), (range(42, 48), True), (range(48, 60), True), (range(60, 68), True)]
        for indices, closed in connections:
            pts = landmarks[list(indices)].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(output, [pts], closed, (0, 255, 0), 1)
        
        return output


class InteractiveFACSVisualizer(FACSVisualizer):
    """インタラクティブ可視化"""
    
    def show_interactive(self, image: np.ndarray, result: AnalysisResult, window_name: str = "FACS") -> np.ndarray:
        vis = self.create_analysis_panel(image, result)
        cv2.imshow(window_name, vis)
        return vis
