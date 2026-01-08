import numpy as np
from typing import Dict, Optional

from ..core.interfaces import IAUDetector, IAUDetectionStrategy
from ..core.models import AUDetectionResult
from ..core.enums import AUIntensity
from ..config import AU_DEFINITIONS
from .strategies import get_all_strategies

class AUDetector(IAUDetector):
    """Action Unit検出器"""
    
    def __init__(self):
        self._strategies: Dict[int, IAUDetectionStrategy] = {}
        self._thresholds = self._get_default_thresholds()
        
        # デフォルト戦略を登録
        for strategy in get_all_strategies():
            self.register_strategy(strategy)
    
    def register_strategy(self, strategy: IAUDetectionStrategy) -> None:
        self._strategies[strategy.au_number] = strategy
    
    def detect_all(self, landmarks: np.ndarray, distances: Dict[str, float],
                   angles: Dict[str, float]) -> Dict[int, AUDetectionResult]:
        eye_dist = max(distances.get("eye_distance", 1.0), 1e-6)
        results = {}
        
        for au_number, au_def in AU_DEFINITIONS.items():
            strategy = self._strategies.get(au_number)
            if strategy:
                raw_score, asymmetry = strategy.detect(landmarks, distances, angles, eye_dist)
            else:
                raw_score, asymmetry = 0.0, 0.0
            
            results[au_number] = self._create_result(au_number, au_def.name, raw_score, asymmetry)
        
        return results
    
    def _create_result(self, au_number: int, name: str, raw_score: float, asymmetry: float) -> AUDetectionResult:
        thresholds = self._thresholds.get(au_number, {"low": 0.15, "high": 0.4})
        detected = raw_score >= thresholds["low"]
        confidence = self._calculate_confidence(raw_score, thresholds)
        intensity = self._score_to_intensity(raw_score, thresholds)
        
        return AUDetectionResult(
            au_number=au_number, name=name, detected=detected,
            confidence=confidence, intensity=intensity,
            raw_score=raw_score, asymmetry=asymmetry
        )
    
    def _calculate_confidence(self, score: float, thresholds: Dict) -> float:
        if score < thresholds["low"]:
            return score / thresholds["low"] * 0.5
        elif score < thresholds["high"]:
            return 0.5 + (score - thresholds["low"]) / (thresholds["high"] - thresholds["low"]) * 0.3
        return min(0.8 + (score - thresholds["high"]) * 0.4, 1.0)
    
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
    
    def _get_default_thresholds(self) -> Dict[int, Dict[str, float]]:
        return {
            1: {"low": 0.15, "high": 0.4}, 2: {"low": 0.15, "high": 0.4},
            4: {"low": 0.12, "high": 0.35}, 5: {"low": 0.2, "high": 0.5},
            6: {"low": 0.15, "high": 0.4}, 7: {"low": 0.15, "high": 0.4},
            9: {"low": 0.2, "high": 0.5}, 12: {"low": 0.15, "high": 0.45},
            15: {"low": 0.15, "high": 0.4}, 25: {"low": 0.1, "high": 0.35},
            26: {"low": 0.15, "high": 0.45}, 43: {"low": 0.3, "high": 0.7},
        }
