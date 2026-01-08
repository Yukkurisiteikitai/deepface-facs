import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .action_units import AU_DEFINITIONS, AUIntensity

@dataclass
class AUDetectionResult:
    """AU検出結果"""
    au_number: int
    name: str
    detected: bool
    confidence: float
    intensity: AUIntensity
    raw_score: float
    asymmetry: float = 0.0

class AUDetector:
    """Action Unit検出器"""
    
    def __init__(self):
        self._neutral_baseline: Optional[Dict] = None
        self.thresholds = {
            1: {"low": 0.15, "high": 0.4}, 2: {"low": 0.15, "high": 0.4},
            4: {"low": 0.12, "high": 0.35}, 5: {"low": 0.2, "high": 0.5},
            6: {"low": 0.15, "high": 0.4}, 7: {"low": 0.15, "high": 0.4},
            9: {"low": 0.2, "high": 0.5}, 10: {"low": 0.15, "high": 0.4},
            11: {"low": 0.15, "high": 0.4}, 12: {"low": 0.15, "high": 0.45},
            13: {"low": 0.15, "high": 0.4}, 14: {"low": 0.2, "high": 0.5},
            15: {"low": 0.15, "high": 0.4}, 16: {"low": 0.15, "high": 0.4},
            17: {"low": 0.15, "high": 0.4}, 18: {"low": 0.2, "high": 0.5},
            20: {"low": 0.15, "high": 0.4}, 22: {"low": 0.2, "high": 0.5},
            23: {"low": 0.15, "high": 0.4}, 24: {"low": 0.15, "high": 0.4},
            25: {"low": 0.1, "high": 0.35}, 26: {"low": 0.15, "high": 0.45},
            27: {"low": 0.25, "high": 0.55}, 28: {"low": 0.2, "high": 0.5},
            43: {"low": 0.3, "high": 0.7}, 45: {"low": 0.3, "high": 0.7},
            46: {"low": 0.25, "high": 0.6},
        }
    
    def set_neutral_baseline(self, landmarks: np.ndarray, distances: Dict, angles: Dict):
        self._neutral_baseline = {
            "landmarks": landmarks.copy(),
            "distances": distances.copy(),
            "angles": angles.copy()
        }
    
    def detect_all_aus(self, landmarks: np.ndarray, distances: Dict,
                       angles: Dict) -> Dict[int, AUDetectionResult]:
        results = {}
        eye_dist = max(distances.get("eye_distance", 1.0), 1e-6)
        
        for au_number in AU_DEFINITIONS.keys():
            result = self._detect_single_au(au_number, landmarks, distances, angles, eye_dist)
            results[au_number] = result
        return results
    
    def _detect_single_au(self, au_number: int, landmarks: np.ndarray,
                          distances: Dict, angles: Dict, eye_dist: float) -> AUDetectionResult:
        au_def = AU_DEFINITIONS.get(au_number)
        if au_def is None:
            return AUDetectionResult(au_number=au_number, name="Unknown", detected=False,
                                     confidence=0.0, intensity=AUIntensity.ABSENT, raw_score=0.0)
        
        detector_method = getattr(self, f"_detect_au{au_number}", None)
        if detector_method is None:
            raw_score, asymmetry = 0.0, 0.0
        else:
            result = detector_method(landmarks, distances, angles, eye_dist)
            raw_score, asymmetry = (result if isinstance(result, tuple) else (result, 0.0))
        
        thresholds = self.thresholds.get(au_number, {"low": 0.15, "high": 0.4})
        detected = raw_score >= thresholds["low"]
        
        if raw_score < thresholds["low"]:
            confidence = raw_score / thresholds["low"] * 0.5
        elif raw_score < thresholds["high"]:
            confidence = 0.5 + (raw_score - thresholds["low"]) / (thresholds["high"] - thresholds["low"]) * 0.3
        else:
            confidence = min(0.8 + (raw_score - thresholds["high"]) * 0.4, 1.0)
        
        intensity = self._score_to_intensity(raw_score, thresholds)
        
        return AUDetectionResult(au_number=au_number, name=au_def.name, detected=detected,
                                 confidence=confidence, intensity=intensity,
                                 raw_score=raw_score, asymmetry=asymmetry)
    
    def _score_to_intensity(self, score: float, thresholds: Dict) -> AUIntensity:
        if score < thresholds["low"] * 0.5:
            return AUIntensity.ABSENT
        elif score < thresholds["low"]:
            return AUIntensity.TRACE
        elif score < thresholds["low"] + (thresholds["high"] - thresholds["low"]) * 0.33:
            return AUIntensity.SLIGHT
        elif score < thresholds["high"]:
            return AUIntensity.MARKED
        elif score < thresholds["high"] * 1.3:
            return AUIntensity.SEVERE
        return AUIntensity.MAXIMUM
    
    # === AU検出メソッド ===
    def _detect_au1(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        """AU1: Inner Brow Raiser"""
        right_dist = (landmarks[27][1] - landmarks[21][1]) / eye_dist
        left_dist = (landmarks[27][1] - landmarks[22][1]) / eye_dist
        baseline = 0.18
        right_score = max(0, (right_dist - baseline) / 0.12)
        left_score = max(0, (left_dist - baseline) / 0.12)
        return min((right_score + left_score) / 2, 1.0), np.clip(left_score - right_score, -1.0, 1.0)
    
    def _detect_au2(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        """AU2: Outer Brow Raiser"""
        right_dist = (landmarks[36][1] - landmarks[17][1]) / eye_dist
        left_dist = (landmarks[45][1] - landmarks[26][1]) / eye_dist
        baseline = 0.15
        right_score = max(0, (right_dist - baseline) / 0.1)
        left_score = max(0, (left_dist - baseline) / 0.1)
        return min((right_score + left_score) / 2, 1.0), np.clip(left_score - right_score, -1.0, 1.0)
    
    def _detect_au4(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        """AU4: Brow Lowerer"""
        brow_dist = distances.get("brow_distance", eye_dist * 0.3) / eye_dist
        right_height = (landmarks[39][1] - landmarks[21][1]) / eye_dist
        left_height = (landmarks[42][1] - landmarks[22][1]) / eye_dist
        dist_score = max(0, (0.35 - brow_dist) / 0.12)
        height_score = max(0, (0.12 - (right_height + left_height) / 2) / 0.08)
        return min(dist_score * 0.5 + height_score * 0.5, 1.0), 0.0
    
    def _detect_au5(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        """AU5: Upper Lid Raiser"""
        right_ear = distances.get("right_eye_aspect_ratio", 0.25)
        left_ear = distances.get("left_eye_aspect_ratio", 0.25)
        baseline = 0.28
        right_score = max(0, (right_ear - baseline) / 0.12)
        left_score = max(0, (left_ear - baseline) / 0.12)
        return min((right_score + left_score) / 2, 1.0), np.clip(left_score - right_score, -1.0, 1.0)
    
    def _detect_au6(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        """AU6: Cheek Raiser"""
        right_eye_bottom = (landmarks[40] + landmarks[41]) / 2
        left_eye_bottom = (landmarks[46] + landmarks[47]) / 2
        right_dist = np.linalg.norm(right_eye_bottom - landmarks[48]) / eye_dist
        left_dist = np.linalg.norm(left_eye_bottom - landmarks[54]) / eye_dist
        baseline = 0.75
        right_score = max(0, (baseline - right_dist) / 0.18)
        left_score = max(0, (baseline - left_dist) / 0.18)
        return min((right_score + left_score) / 2, 1.0), np.clip(left_score - right_score, -1.0, 1.0)
    
    def _detect_au7(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        """AU7: Lid Tightener"""
        right_ear = distances.get("right_eye_aspect_ratio", 0.25)
        left_ear = distances.get("left_eye_aspect_ratio", 0.25)
        baseline = 0.25
        right_score = max(0, (baseline - right_ear) / 0.1)
        left_score = max(0, (baseline - left_ear) / 0.1)
        return min((right_score + left_score) / 2, 1.0), np.clip(left_score - right_score, -1.0, 1.0)
    
    def _detect_au9(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> float:
        """AU9: Nose Wrinkler"""
        dist = np.linalg.norm(landmarks[51] - landmarks[30]) / eye_dist
        return min(max(0, (0.35 - dist) / 0.12), 1.0)
    
    def _detect_au10(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> float:
        """AU10: Upper Lip Raiser"""
        dist = np.linalg.norm(landmarks[51] - landmarks[30]) / eye_dist
        return min(max(0, (0.32 - dist) / 0.1), 1.0)
    
    def _detect_au12(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        """AU12: Lip Corner Puller (Smile)"""
        right_elev = (landmarks[51][1] - landmarks[48][1]) / eye_dist
        left_elev = (landmarks[51][1] - landmarks[54][1]) / eye_dist
        mouth_width = distances.get("mouth_width", eye_dist * 0.5) / eye_dist
        baseline_elev, baseline_width = 0.02, 0.48
        right_score = max(0, (right_elev - baseline_elev) / 0.06) * 0.7 + max(0, (mouth_width - baseline_width) / 0.1) * 0.3
        left_score = max(0, (left_elev - baseline_elev) / 0.06) * 0.7 + max(0, (mouth_width - baseline_width) / 0.1) * 0.3
        return min((right_score + left_score) / 2, 1.0), np.clip(left_score - right_score, -1.0, 1.0)
    
    def _detect_au15(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        """AU15: Lip Corner Depressor"""
        right_dep = (landmarks[48][1] - landmarks[57][1]) / eye_dist
        left_dep = (landmarks[54][1] - landmarks[57][1]) / eye_dist
        right_score = max(0, right_dep / 0.05)
        left_score = max(0, left_dep / 0.05)
        return min((right_score + left_score) / 2, 1.0), np.clip(left_score - right_score, -1.0, 1.0)
    
    def _detect_au25(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> float:
        """AU25: Lips Part"""
        mouth_height = distances.get("mouth_height_inner", 0) / eye_dist
        return min(max(0, (mouth_height - 0.02) / 0.08), 1.0)
    
    def _detect_au26(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> float:
        """AU26: Jaw Drop"""
        mouth_height = distances.get("mouth_height_outer", 0) / eye_dist
        return min(max(0, (mouth_height - 0.05) / 0.12), 1.0)
    
    def _detect_au43(self, landmarks: np.ndarray, distances: Dict, angles: Dict, eye_dist: float) -> Tuple[float, float]:
        """AU43: Eyes Closed"""
        right_ear = distances.get("right_eye_aspect_ratio", 0.25)
        left_ear = distances.get("left_eye_aspect_ratio", 0.25)
        baseline = 0.2
        right_score = max(0, (baseline - right_ear) / 0.15)
        left_score = max(0, (baseline - left_ear) / 0.15)
        return min((right_score + left_score) / 2, 1.0), np.clip(left_score - right_score, -1.0, 1.0)
