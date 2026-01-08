from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
import time

from .enums import AUIntensity

@dataclass(frozen=True)
class ActionUnitDefinition:
    """Action Unitの定義（イミュータブル）"""
    au_number: int
    name: str
    description: str
    muscular_basis: str
    landmarks_involved: Tuple[int, ...]

@dataclass(frozen=True)
class EmotionDefinition:
    """感情の定義（イミュータブル）"""
    name: str
    required_aus: Tuple[int, ...]
    optional_aus: Tuple[int, ...]
    description: str
    valence: float
    arousal: float

@dataclass
class Landmark:
    """単一のランドマーク点"""
    x: float
    y: float
    index: int
    name: str = ""

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
    
    @property
    def face_roll(self) -> float:
        """顔のロール角（傾き）"""
        return self.angles.get("face_roll", 0.0)
    
    @property
    def face_yaw(self) -> float:
        """顔のヨー角（左右の向き）"""
        return self.angles.get("face_yaw", 0.0)
    
    @property
    def face_pitch(self) -> float:
        """顔のピッチ角（上下の向き）"""
        return self.angles.get("face_pitch", 0.0)

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
    
    def to_dict(self) -> Dict:
        return {
            "au_number": self.au_number,
            "name": self.name,
            "detected": self.detected,
            "confidence": round(self.confidence, 3),
            "intensity": self.intensity.label,
            "raw_score": round(self.raw_score, 3),
            "asymmetry": round(self.asymmetry, 3)
        }

@dataclass
class IntensityResult:
    """強度推定結果"""
    au_number: int
    intensity: AUIntensity
    intensity_value: float
    confidence: float
    
    @property
    def intensity_label(self) -> str:
        return self.intensity.label

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
    
    def to_dict(self) -> Dict:
        return {
            "emotion": self.emotion,
            "confidence": round(self.confidence, 3),
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "matched_aus": self.matched_aus
        }

@dataclass
class AnalysisResult:
    """FACS分析の完全な結果"""
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
            "active_aus": [r.to_dict() for r in self.active_aus],
            "emotions": [e.to_dict() for e in self.emotions[:5]],
            "dominant_emotion": self.dominant_emotion.emotion if self.dominant_emotion else "neutral",
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "processing_time_ms": round(self.processing_time_ms, 2)
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
