from typing import Dict

from ..core.interfaces import IIntensityEstimator
from ..core.models import AUDetectionResult, IntensityResult
from ..core.enums import AUIntensity

class IntensityEstimator(IIntensityEstimator):
    """AU強度推定器"""
    
    def __init__(self):
        self._scaling_factors = {
            1: 1.0, 2: 1.0, 4: 1.1, 5: 0.9, 6: 1.0, 7: 1.0,
            9: 1.1, 12: 1.0, 15: 1.0, 25: 0.9, 26: 1.0, 43: 1.0
        }
    
    def estimate(self, au_result: AUDetectionResult) -> IntensityResult:
        scaling = self._scaling_factors.get(au_result.au_number, 1.0)
        intensity_value = min(au_result.raw_score * 5.0 * scaling, 5.0)
        intensity = self._value_to_intensity(intensity_value)
        
        return IntensityResult(
            au_number=au_result.au_number,
            intensity=intensity,
            intensity_value=intensity_value,
            confidence=au_result.confidence
        )
    
    def estimate_all(self, au_results: Dict[int, AUDetectionResult]) -> Dict[int, IntensityResult]:
        return {au_num: self.estimate(result) for au_num, result in au_results.items()}
    
    def format_facs_code(self, results: Dict[int, IntensityResult]) -> str:
        codes = [f"AU{au_num}{r.intensity_label}"
                 for au_num, r in sorted(results.items())
                 if r.intensity != AUIntensity.ABSENT]
        return " + ".join(codes) if codes else "Neutral"
    
    def _value_to_intensity(self, value: float) -> AUIntensity:
        if value < 0.5:
            return AUIntensity.ABSENT
        elif value < 1.5:
            return AUIntensity.TRACE
        elif value < 2.5:
            return AUIntensity.SLIGHT
        elif value < 3.5:
            return AUIntensity.MARKED
        elif value < 4.5:
            return AUIntensity.SEVERE
        return AUIntensity.MAXIMUM
