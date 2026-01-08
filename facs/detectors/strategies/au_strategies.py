import numpy as np
from typing import Dict, Tuple, List

from ...core.interfaces import IAUDetectionStrategy

class BaseAUStrategy(IAUDetectionStrategy):
    """AU検出戦略の基底クラス"""
    
    _au_number: int = 0
    
    @property
    def au_number(self) -> int:
        return self._au_number
    
    def _compute_bilateral_score(self, right_val: float, left_val: float,
                                  baseline: float, scale: float) -> Tuple[float, float]:
        right_score = max(0, (right_val - baseline) / scale)
        left_score = max(0, (left_val - baseline) / scale)
        score = min((right_score + left_score) / 2, 1.0)
        asymmetry = np.clip(left_score - right_score, -1.0, 1.0)
        return score, asymmetry

class AU1Strategy(BaseAUStrategy):
    """AU1: Inner Brow Raiser"""
    _au_number = 1
    
    def detect(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        right_dist = (landmarks[27][1] - landmarks[21][1]) / eye_dist
        left_dist = (landmarks[27][1] - landmarks[22][1]) / eye_dist
        return self._compute_bilateral_score(right_dist, left_dist, 0.18, 0.12)

class AU2Strategy(BaseAUStrategy):
    """AU2: Outer Brow Raiser"""
    _au_number = 2
    
    def detect(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        right_dist = (landmarks[36][1] - landmarks[17][1]) / eye_dist
        left_dist = (landmarks[45][1] - landmarks[26][1]) / eye_dist
        return self._compute_bilateral_score(right_dist, left_dist, 0.15, 0.1)

class AU4Strategy(BaseAUStrategy):
    """AU4: Brow Lowerer"""
    _au_number = 4
    
    def detect(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        brow_dist = distances.get("brow_distance", eye_dist * 0.3) / eye_dist
        dist_score = max(0, (0.35 - brow_dist) / 0.12)
        return min(dist_score, 1.0), 0.0

class AU5Strategy(BaseAUStrategy):
    """AU5: Upper Lid Raiser"""
    _au_number = 5
    
    def detect(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        right_ear = distances.get("right_eye_aspect_ratio", 0.25)
        left_ear = distances.get("left_eye_aspect_ratio", 0.25)
        return self._compute_bilateral_score(right_ear, left_ear, 0.28, 0.12)

class AU6Strategy(BaseAUStrategy):
    """AU6: Cheek Raiser"""
    _au_number = 6
    
    def detect(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        right_bottom = (landmarks[40] + landmarks[41]) / 2
        left_bottom = (landmarks[46] + landmarks[47]) / 2
        right_dist = np.linalg.norm(right_bottom - landmarks[48]) / eye_dist
        left_dist = np.linalg.norm(left_bottom - landmarks[54]) / eye_dist
        right_score = max(0, (0.75 - right_dist) / 0.18)
        left_score = max(0, (0.75 - left_dist) / 0.18)
        return min((right_score + left_score) / 2, 1.0), np.clip(left_score - right_score, -1.0, 1.0)

class AU7Strategy(BaseAUStrategy):
    """AU7: Lid Tightener"""
    _au_number = 7
    
    def detect(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        right_ear = distances.get("right_eye_aspect_ratio", 0.25)
        left_ear = distances.get("left_eye_aspect_ratio", 0.25)
        right_score = max(0, (0.25 - right_ear) / 0.1)
        left_score = max(0, (0.25 - left_ear) / 0.1)
        return min((right_score + left_score) / 2, 1.0), np.clip(left_score - right_score, -1.0, 1.0)

class AU9Strategy(BaseAUStrategy):
    """AU9: Nose Wrinkler"""
    _au_number = 9
    
    def detect(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        dist = np.linalg.norm(landmarks[51] - landmarks[30]) / eye_dist
        return min(max(0, (0.35 - dist) / 0.12), 1.0), 0.0

class AU12Strategy(BaseAUStrategy):
    """AU12: Lip Corner Puller (Smile)"""
    _au_number = 12
    
    def detect(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        right_elev = (landmarks[51][1] - landmarks[48][1]) / eye_dist
        left_elev = (landmarks[51][1] - landmarks[54][1]) / eye_dist
        mouth_width = distances.get("mouth_width", eye_dist * 0.5) / eye_dist
        right_score = max(0, (right_elev - 0.02) / 0.06) * 0.7 + max(0, (mouth_width - 0.48) / 0.1) * 0.3
        left_score = max(0, (left_elev - 0.02) / 0.06) * 0.7 + max(0, (mouth_width - 0.48) / 0.1) * 0.3
        return min((right_score + left_score) / 2, 1.0), np.clip(left_score - right_score, -1.0, 1.0)

class AU15Strategy(BaseAUStrategy):
    """AU15: Lip Corner Depressor"""
    _au_number = 15
    
    def detect(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        right_dep = (landmarks[48][1] - landmarks[57][1]) / eye_dist
        left_dep = (landmarks[54][1] - landmarks[57][1]) / eye_dist
        return self._compute_bilateral_score(right_dep, left_dep, 0.0, 0.05)

class AU25Strategy(BaseAUStrategy):
    """AU25: Lips Part"""
    _au_number = 25
    
    def detect(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        height = distances.get("mouth_height_inner", 0) / eye_dist
        return min(max(0, (height - 0.02) / 0.08), 1.0), 0.0

class AU26Strategy(BaseAUStrategy):
    """AU26: Jaw Drop"""
    _au_number = 26
    
    def detect(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        height = distances.get("mouth_height_outer", 0) / eye_dist
        return min(max(0, (height - 0.05) / 0.12), 1.0), 0.0

class AU43Strategy(BaseAUStrategy):
    """AU43: Eyes Closed"""
    _au_number = 43
    
    def detect(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        right_ear = distances.get("right_eye_aspect_ratio", 0.25)
        left_ear = distances.get("left_eye_aspect_ratio", 0.25)
        right_score = max(0, (0.2 - right_ear) / 0.15)
        left_score = max(0, (0.2 - left_ear) / 0.15)
        return min((right_score + left_score) / 2, 1.0), np.clip(left_score - right_score, -1.0, 1.0)

def get_all_strategies() -> List[IAUDetectionStrategy]:
    """すべてのAU検出戦略を取得"""
    return [
        AU1Strategy(), AU2Strategy(), AU4Strategy(), AU5Strategy(),
        AU6Strategy(), AU7Strategy(), AU9Strategy(), AU12Strategy(),
        AU15Strategy(), AU25Strategy(), AU26Strategy(), AU43Strategy()
    ]
