import numpy as np
from typing import Dict, List, Optional, Tuple
from ..core.interfaces import IEmotionMapper
from ..core.models import AUDetectionResult, IntensityResult, EmotionResult
from ..config import EMOTION_DEFINITIONS

class EmotionMapper(IEmotionMapper):
    """感情マッピング"""
    
    def map(self, au_results: Dict[int, AUDetectionResult],
            intensity_results: Optional[Dict[int, IntensityResult]] = None) -> List[EmotionResult]:
        detected_aus = {au_num for au_num, result in au_results.items() if result.detected}
        au_intensities = {au_num: result.raw_score * 5.0 for au_num, result in au_results.items()}
        
        emotions = []
        for name, defn in EMOTION_DEFINITIONS.items():
            result = self._evaluate(name, defn, detected_aus, au_intensities)
            if result:
                emotions.append(result)
        
        emotions.sort(key=lambda x: x.confidence, reverse=True)
        
        if not emotions or emotions[0].confidence < 0.3:
            emotions.insert(0, EmotionResult(
                emotion="neutral", confidence=1.0 - (emotions[0].confidence if emotions else 0.0),
                valence=0.0, arousal=0.0, matched_aus=[], missing_aus=[], description="中立"
            ))
        
        return emotions
    
    def _evaluate(self, name: str, defn, detected_aus: set, au_intensities: Dict) -> Optional[EmotionResult]:
        required = set(defn.required_aus)
        optional = set(defn.optional_aus)
        
        if not required:
            return None
        
        matched_required = required & detected_aus
        missing_required = required - detected_aus
        matched_optional = optional & detected_aus
        
        if len(matched_required) / len(required) < 0.5:
            return None
        
        confidence = (len(matched_required) / len(required)) * 0.7 + (len(matched_optional) / max(len(optional), 1)) * 0.2
        confidence = min(confidence + sum(au_intensities.get(au, 0) for au in matched_required) / 50, 1.0)
        
        return EmotionResult(
            emotion=name, confidence=confidence,
            valence=np.clip(defn.valence * confidence, -1.0, 1.0),
            arousal=np.clip(defn.arousal * confidence, -1.0, 1.0),
            matched_aus=list(matched_required | matched_optional),
            missing_aus=list(missing_required), description=defn.description
        )
    
    def get_valence_arousal(self, au_results: Dict[int, AUDetectionResult],
                            intensity_results: Optional[Dict[int, IntensityResult]] = None) -> Tuple[float, float]:
        emotions = self.map(au_results, intensity_results)
        if not emotions:
            return 0.0, 0.0
        total_weight = sum(e.confidence for e in emotions)
        if total_weight == 0:
            return 0.0, 0.0
        valence = sum(e.valence * e.confidence for e in emotions) / total_weight
        arousal = sum(e.arousal * e.confidence for e in emotions) / total_weight
        return valence, arousal
