import numpy as np
from typing import Dict, List, Optional, Tuple

from ..core.interfaces import IEmotionMapper
from ..core.models import AUDetectionResult, IntensityResult, EmotionResult
from ..config import EMOTION_DEFINITIONS

class EmotionMapper(IEmotionMapper):
    """感情マッピング"""
    
    def map(self, au_results: Dict[int, AUDetectionResult],
            intensity_results: Optional[Dict[int, IntensityResult]] = None) -> List[EmotionResult]:
        detected_aus = {au for au, r in au_results.items() if r.detected}
        au_intensities = self._get_intensities(au_results, intensity_results)
        
        emotions = []
        for name, definition in EMOTION_DEFINITIONS.items():
            result = self._evaluate_emotion(name, definition, detected_aus, au_intensities)
            if result:
                emotions.append(result)
        
        emotions.sort(key=lambda x: x.confidence, reverse=True)
        
        if not emotions or emotions[0].confidence < 0.3:
            emotions.insert(0, EmotionResult(
                emotion="neutral", confidence=1.0 - (emotions[0].confidence if emotions else 0),
                valence=0.0, arousal=0.0, matched_aus=[], missing_aus=[], description="中立"
            ))
        
        return emotions
    
    def get_valence_arousal(self, au_results: Dict[int, AUDetectionResult],
                            intensity_results: Optional[Dict[int, IntensityResult]] = None) -> Tuple[float, float]:
        emotions = self.map(au_results, intensity_results)
        if not emotions:
            return 0.0, 0.0
        
        total = sum(e.confidence for e in emotions)
        if total == 0:
            return 0.0, 0.0
        
        valence = sum(e.valence * e.confidence for e in emotions) / total
        arousal = sum(e.arousal * e.confidence for e in emotions) / total
        return valence, arousal
    
    def _get_intensities(self, au_results: Dict[int, AUDetectionResult],
                         intensity_results: Optional[Dict[int, IntensityResult]]) -> Dict[int, float]:
        if intensity_results:
            return {au: r.intensity_value for au, r in intensity_results.items()}
        return {au: r.raw_score * 5.0 for au, r in au_results.items()}
    
    def _evaluate_emotion(self, name: str, definition, detected_aus: set,
                          au_intensities: Dict[int, float]) -> Optional[EmotionResult]:
        required = set(definition.required_aus)
        optional = set(definition.optional_aus)
        
        if not required:
            return None
        
        matched_required = required & detected_aus
        missing_required = required - detected_aus
        matched_optional = optional & detected_aus
        
        match_ratio = len(matched_required) / len(required)
        if match_ratio < 0.5:
            return None
        
        confidence = match_ratio * 0.7
        if optional:
            confidence += len(matched_optional) / len(optional) * 0.2
        
        intensity_bonus = sum(au_intensities.get(au, 0) / 5.0 * 0.02
                             for au in matched_required | matched_optional)
        confidence = min(confidence + intensity_bonus, 1.0)
        
        return EmotionResult(
            emotion=name, confidence=confidence,
            valence=definition.valence, arousal=definition.arousal,
            matched_aus=list(matched_required | matched_optional),
            missing_aus=list(missing_required), description=definition.description
        )
