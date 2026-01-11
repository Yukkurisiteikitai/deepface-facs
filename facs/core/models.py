"""
データモデル定義
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import json
import time

from .enums import AUIntensity


@dataclass
class ActionUnitDefinition:
    """Action Unitの定義"""
    au_number: int
    name: str
    description: str
    muscular_basis: str
    landmarks_involved: Tuple[int, ...]


@dataclass
class EmotionDefinition:
    """感情の定義"""
    name: str
    required_aus: Tuple[int, ...]
    optional_aus: Tuple[int, ...]
    description: str
    valence: float
    arousal: float


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


@dataclass
class IntensityResult:
    """強度推定結果"""
    au_number: int
    intensity: AUIntensity
    intensity_value: float
    intensity_label: str
    confidence: float


@dataclass
class EmotionResult:
    """感情推定結果"""
    emotion: str
    confidence: float
    valence: float
    arousal: float
    matched_aus: List[int]
    missing_aus: List[int]
    description: str


@dataclass
class FaceData:
    """顔データ"""
    rect: Optional[Tuple[int, int, int, int]]
    landmarks: Optional[np.ndarray]
    distances: Dict[str, float] = field(default_factory=dict)
    angles: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        return self.landmarks is not None
    
    @property
    def eye_distance(self) -> float:
        return max(self.distances.get("eye_distance", 1.0), 1e-6)


@dataclass
class AnalysisResult:
    """分析結果"""
    timestamp: float = field(default_factory=time.time)
    face_data: Optional[FaceData] = None
    au_results: Dict[int, AUDetectionResult] = field(default_factory=dict)
    intensity_results: Dict[int, IntensityResult] = field(default_factory=dict)
    emotions: List[EmotionResult] = field(default_factory=list)
    facs_code: str = "Neutral"
    valence: float = 0.0
    arousal: float = 0.0
    processing_time_ms: float = 0.0
    
    @property
    def is_valid(self) -> bool:
        return self.face_data is not None and self.face_data.is_valid
    
    @property
    def dominant_emotion(self) -> Optional[EmotionResult]:
        return self.emotions[0] if self.emotions else None
    
    @property
    def active_aus(self) -> List[AUDetectionResult]:
        return [r for r in self.au_results.values() if r.detected]
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "facs_code": self.facs_code,
            "valence": self.valence,
            "arousal": self.arousal,
            "processing_time_ms": self.processing_time_ms,
            "dominant_emotion": self.dominant_emotion.emotion if self.dominant_emotion else None,
            "emotions": [{"emotion": e.emotion, "confidence": e.confidence} for e in self.emotions[:5]],
            "active_aus": [{"au": r.au_number, "name": r.name, "confidence": r.confidence,
                          "intensity": self.intensity_results[r.au_number].intensity_label 
                          if r.au_number in self.intensity_results else None}
                         for r in self.active_aus],
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
