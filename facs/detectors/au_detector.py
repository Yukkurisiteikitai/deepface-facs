import numpy as np
from typing import Dict, Tuple
from ..core.interfaces import IAUDetector, IAUDetectionStrategy
from ..core.models import AUDetectionResult
from ..core.enums import AUIntensity
from ..config import AU_DEFINITIONS

class AUDetector(IAUDetector):
    """Action Unit検出器"""
    
    def __init__(self):
        self._strategies: Dict[int, IAUDetectionStrategy] = {}
        self._thresholds = {au: {"low": 0.15, "high": 0.4} for au in AU_DEFINITIONS.keys()}
    
    def register_strategy(self, strategy: IAUDetectionStrategy) -> None:
        self._strategies[strategy.au_number] = strategy
    
    def detect_all(self, landmarks: np.ndarray, distances: Dict[str, float],
                   angles: Dict[str, float]) -> Dict[int, AUDetectionResult]:
        results = {}
        eye_dist = max(distances.get("eye_distance", 1.0), 1e-6)
        
        for au_num, au_def in AU_DEFINITIONS.items():
            if au_num in self._strategies:
                raw_score, asymmetry = self._strategies[au_num].detect(landmarks, distances, angles, eye_dist)
            else:
                raw_score, asymmetry = self._detect_builtin(au_num, landmarks, distances, angles, eye_dist)
            
            thresholds = self._thresholds.get(au_num, {"low": 0.15, "high": 0.4})
            detected = raw_score >= thresholds["low"]
            confidence = min(raw_score / thresholds["high"], 1.0) if raw_score > 0 else 0.0
            intensity = self._score_to_intensity(raw_score, thresholds)
            
            results[au_num] = AUDetectionResult(
                au_number=au_num, name=au_def.name, detected=detected,
                confidence=confidence, intensity=intensity, raw_score=raw_score, asymmetry=asymmetry
            )
        
        return results
    
    def _score_to_intensity(self, score: float, thresholds: Dict) -> AUIntensity:
        if score < thresholds["low"] * 0.5: return AUIntensity.ABSENT
        elif score < thresholds["low"]: return AUIntensity.TRACE
        elif score < thresholds["high"] * 0.66: return AUIntensity.SLIGHT
        elif score < thresholds["high"]: return AUIntensity.MARKED
        elif score < thresholds["high"] * 1.3: return AUIntensity.SEVERE
        return AUIntensity.MAXIMUM
    
    def _detect_builtin(self, au_num: int, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        if au_num == 1: return self._detect_au1(landmarks, distances, eye_dist)
        elif au_num == 2: return self._detect_au2(landmarks, distances, eye_dist)
        elif au_num == 4: return self._detect_au4(landmarks, distances, eye_dist)
        elif au_num == 5: return self._detect_au5(distances)
        elif au_num == 6: return self._detect_au6(landmarks, eye_dist)
        elif au_num == 7: return self._detect_au7(distances)
        elif au_num == 12: return self._detect_au12(landmarks, distances, eye_dist)
        elif au_num == 25: return self._detect_au25(distances, eye_dist)
        elif au_num == 26: return self._detect_au26(distances, eye_dist)
        elif au_num == 43: return self._detect_au43(distances)
        return 0.0, 0.0
    
    def _detect_au1(self, landmarks, distances, eye_dist):
        right_dist = (landmarks[27][1] - landmarks[21][1]) / eye_dist
        left_dist = (landmarks[27][1] - landmarks[22][1]) / eye_dist
        score = max(0, ((right_dist + left_dist) / 2 - 0.18) / 0.12)
        return min(score, 1.0), np.clip(left_dist - right_dist, -1.0, 1.0)
    
    def _detect_au2(self, landmarks, distances, eye_dist):
        right_dist = (landmarks[36][1] - landmarks[17][1]) / eye_dist
        left_dist = (landmarks[45][1] - landmarks[26][1]) / eye_dist
        score = max(0, ((right_dist + left_dist) / 2 - 0.15) / 0.1)
        return min(score, 1.0), np.clip(left_dist - right_dist, -1.0, 1.0)
    
    def _detect_au4(self, landmarks, distances, eye_dist):
        brow_dist = distances.get("brow_distance", eye_dist * 0.3) / eye_dist
        return min(max(0, (0.35 - brow_dist) / 0.12), 1.0), 0.0
    
    def _detect_au5(self, distances):
        ear = (distances.get("right_eye_aspect_ratio", 0.25) + distances.get("left_eye_aspect_ratio", 0.25)) / 2
        return min(max(0, (ear - 0.28) / 0.12), 1.0), 0.0
    
    def _detect_au6(self, landmarks, eye_dist):
        right_eye_bottom = (landmarks[40] + landmarks[41]) / 2
        left_eye_bottom = (landmarks[46] + landmarks[47]) / 2
        right_dist = np.linalg.norm(right_eye_bottom - landmarks[48]) / eye_dist
        left_dist = np.linalg.norm(left_eye_bottom - landmarks[54]) / eye_dist
        score = max(0, (0.75 - (right_dist + left_dist) / 2) / 0.18)
        return min(score, 1.0), np.clip(left_dist - right_dist, -1.0, 1.0)
    
    def _detect_au7(self, distances):
        ear = (distances.get("right_eye_aspect_ratio", 0.25) + distances.get("left_eye_aspect_ratio", 0.25)) / 2
        return min(max(0, (0.25 - ear) / 0.1), 1.0), 0.0
    
    def _detect_au12(self, landmarks, distances, eye_dist):
        right_elev = (landmarks[51][1] - landmarks[48][1]) / eye_dist
        left_elev = (landmarks[51][1] - landmarks[54][1]) / eye_dist
        mouth_width = distances.get("mouth_width", eye_dist * 0.5) / eye_dist
        score = max(0, ((right_elev + left_elev) / 2 - 0.02) / 0.06) * 0.7 + max(0, (mouth_width - 0.48) / 0.1) * 0.3
        return min(score, 1.0), np.clip(left_elev - right_elev, -1.0, 1.0)
    
    def _detect_au25(self, distances, eye_dist):
        mouth_height = distances.get("mouth_height_inner", 0) / eye_dist
        return min(max(0, (mouth_height - 0.02) / 0.08), 1.0), 0.0
    
    def _detect_au26(self, distances, eye_dist):
        mouth_height = distances.get("mouth_height_outer", 0) / eye_dist
        return min(max(0, (mouth_height - 0.05) / 0.12), 1.0), 0.0
    
    def _detect_au43(self, distances):
        ear = (distances.get("right_eye_aspect_ratio", 0.25) + distances.get("left_eye_aspect_ratio", 0.25)) / 2
        return min(max(0, (0.2 - ear) / 0.15), 1.0), 0.0
