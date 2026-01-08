import numpy as np
from typing import Dict
from dataclasses import dataclass

from .action_units import AUIntensity
from .au_detector import AUDetectionResult

@dataclass
class IntensityResult:
    """強度推定結果"""
    au_number: int
    intensity: AUIntensity
    intensity_value: float
    intensity_label: str
    confidence: float

class IntensityEstimator:
    """AU強度推定器（FACS A-Eスケール）"""
    
    INTENSITY_LABELS = {
        AUIntensity.ABSENT: "-", AUIntensity.TRACE: "A", AUIntensity.SLIGHT: "B",
        AUIntensity.MARKED: "C", AUIntensity.SEVERE: "D", AUIntensity.MAXIMUM: "E"
    }
    
    def __init__(self):
        self.scaling_factors = {
            1: 1.0, 2: 1.0, 4: 1.1, 5: 0.9, 6: 1.0, 7: 1.0, 9: 1.1, 10: 1.0,
            11: 1.0, 12: 1.0, 13: 1.0, 14: 1.1, 15: 1.0, 16: 1.0, 17: 1.0,
            18: 1.1, 20: 1.0, 22: 1.1, 23: 1.0, 24: 1.0, 25: 0.9, 26: 1.0,
            27: 1.0, 28: 1.1, 43: 1.0, 45: 1.0, 46: 1.0
        }
    
    def estimate_intensity(self, au_result: AUDetectionResult) -> IntensityResult:
        scaling = self.scaling_factors.get(au_result.au_number, 1.0)
        intensity_value = min(au_result.raw_score * 5.0 * scaling, 5.0)
        
        if intensity_value < 0.5:
            intensity = AUIntensity.ABSENT
        elif intensity_value < 1.5:
            intensity = AUIntensity.TRACE
        elif intensity_value < 2.5:
            intensity = AUIntensity.SLIGHT
        elif intensity_value < 3.5:
            intensity = AUIntensity.MARKED
        elif intensity_value < 4.5:
            intensity = AUIntensity.SEVERE
        else:
            intensity = AUIntensity.MAXIMUM
        
        return IntensityResult(
            au_number=au_result.au_number, intensity=intensity,
            intensity_value=intensity_value,
            intensity_label=self.INTENSITY_LABELS[intensity],
            confidence=au_result.confidence
        )
    
    def estimate_all_intensities(self, au_results: Dict[int, AUDetectionResult]) -> Dict[int, IntensityResult]:
        return {au_num: self.estimate_intensity(result) for au_num, result in au_results.items()}
    
    def format_facs_code(self, intensity_results: Dict[int, IntensityResult]) -> str:
        codes = [f"AU{au_num}{result.intensity_label}"
                 for au_num, result in sorted(intensity_results.items())
                 if result.intensity != AUIntensity.ABSENT]
        return " + ".join(codes) if codes else "Neutral"
    
    def get_intensity_summary(self, intensity_results: Dict[int, IntensityResult]) -> Dict:
        active_aus = [{"au": au_num, "intensity": r.intensity_label, "value": round(r.intensity_value, 2)}
                      for au_num, r in intensity_results.items() if r.intensity != AUIntensity.ABSENT]
        active_aus.sort(key=lambda x: x["value"], reverse=True)
        return {
            "active_au_count": len(active_aus),
            "active_aus": active_aus,
            "average_intensity": np.mean([au["value"] for au in active_aus]) if active_aus else 0.0
        }
