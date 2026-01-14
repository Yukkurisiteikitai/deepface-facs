"""
データモデル定義
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import json
import time

from .enums import AUIntensity


def _to_python_type(value):
    """NumPy型をPython標準型に変換"""
    if isinstance(value, (np.bool_, np.generic)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


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
    
    def to_dict(self) -> Dict:
        """シリアライズ用の辞書に変換"""
        return {
            "rect": list(self.rect) if self.rect else None,
            "landmarks": self.landmarks.tolist() if self.landmarks is not None else None,
            "distances": self.distances,
            "angles": self.angles,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FaceData":
        """辞書からFaceDataを復元"""
        return cls(
            rect=tuple(data["rect"]) if data.get("rect") else None,
            landmarks=np.array(data["landmarks"]) if data.get("landmarks") else None,
            distances=data.get("distances", {}),
            angles=data.get("angles", {}),
        )


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
    frame_number: int = 0  # 追加: フレーム番号
    
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
            "frame_number": self.frame_number,
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
    
    def to_record_dict(self) -> Dict:
        """記録用の完全な辞書（再生時に復元可能）"""
        return {
            "timestamp": _to_python_type(self.timestamp),
            "frame_number": _to_python_type(self.frame_number),
            "face_data": self.face_data.to_dict() if self.face_data else None,
            "au_results": {
                str(k): {
                    "au_number": _to_python_type(v.au_number),
                    "name": v.name,
                    "detected": _to_python_type(v.detected),
                    "confidence": _to_python_type(v.confidence),
                    "intensity": _to_python_type(v.intensity.value),
                    "raw_score": _to_python_type(v.raw_score),
                    "asymmetry": _to_python_type(v.asymmetry),
                }
                for k, v in self.au_results.items()
            },
            "intensity_results": {
                str(k): {
                    "au_number": _to_python_type(v.au_number),
                    "intensity": _to_python_type(v.intensity.value),
                    "intensity_value": _to_python_type(v.intensity_value),
                    "intensity_label": v.intensity_label,
                    "confidence": _to_python_type(v.confidence),
                }
                for k, v in self.intensity_results.items()
            },
            "emotions": [
                {
                    "emotion": e.emotion,
                    "confidence": _to_python_type(e.confidence),
                    "valence": _to_python_type(e.valence),
                    "arousal": _to_python_type(e.arousal),
                    "matched_aus": [_to_python_type(x) for x in e.matched_aus],
                    "missing_aus": [_to_python_type(x) for x in e.missing_aus],
                    "description": e.description,
                }
                for e in self.emotions
            ],
            "facs_code": self.facs_code,
            "valence": _to_python_type(self.valence),
            "arousal": _to_python_type(self.arousal),
            "processing_time_ms": _to_python_type(self.processing_time_ms),
        }
    
    @classmethod
    def from_record_dict(cls, data: Dict) -> "AnalysisResult":
        """記録用辞書からAnalysisResultを復元"""
        face_data = FaceData.from_dict(data["face_data"]) if data.get("face_data") else None
        
        au_results = {
            int(k): AUDetectionResult(
                au_number=v["au_number"],
                name=v["name"],
                detected=v["detected"],
                confidence=v["confidence"],
                intensity=AUIntensity(v["intensity"]),
                raw_score=v["raw_score"],
                asymmetry=v.get("asymmetry", 0.0),
            )
            for k, v in data.get("au_results", {}).items()
        }
        
        intensity_results = {
            int(k): IntensityResult(
                au_number=v["au_number"],
                intensity=AUIntensity(v["intensity"]),
                intensity_value=v["intensity_value"],
                intensity_label=v["intensity_label"],
                confidence=v["confidence"],
            )
            for k, v in data.get("intensity_results", {}).items()
        }
        
        emotions = [
            EmotionResult(
                emotion=e["emotion"],
                confidence=e["confidence"],
                valence=e["valence"],
                arousal=e["arousal"],
                matched_aus=e["matched_aus"],
                missing_aus=e["missing_aus"],
                description=e["description"],
            )
            for e in data.get("emotions", [])
        ]
        
        return cls(
            timestamp=data["timestamp"],
            frame_number=data.get("frame_number", 0),
            face_data=face_data,
            au_results=au_results,
            intensity_results=intensity_results,
            emotions=emotions,
            facs_code=data["facs_code"],
            valence=data["valence"],
            arousal=data["arousal"],
            processing_time_ms=data["processing_time_ms"],
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
