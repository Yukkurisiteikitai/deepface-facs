"""強度推定器"""
from typing import Dict

from ..core.models import AUDetectionResult, IntensityResult
from ..core.enums import AUIntensity


class IntensityEstimator:
    """AU強度推定器"""
    
    # 強度ラベルのマッピング
    INTENSITY_LABELS = {
        AUIntensity.ABSENT: "-",
        AUIntensity.TRACE: "A",
        AUIntensity.SLIGHT: "B",
        AUIntensity.MARKED: "C",
        AUIntensity.SEVERE: "D",
        AUIntensity.MAXIMUM: "E",
    }
    
    def estimate(self, au_result: AUDetectionResult) -> IntensityResult:
        """単一AUの強度を推定"""
        if not au_result.detected:
            return IntensityResult(
                au_number=au_result.au_number,
                intensity=AUIntensity.ABSENT,
                intensity_value=0.0,
                intensity_label="-",
                confidence=au_result.confidence
            )
        
        # raw_scoreから強度を計算（0-1を0-5にスケール）
        intensity_value = au_result.raw_score * 5.0
        
        # 強度レベルを決定
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
            au_number=au_result.au_number,
            intensity=intensity,
            intensity_value=intensity_value,
            intensity_label=self.INTENSITY_LABELS[intensity],
            confidence=au_result.confidence
        )
    
    def estimate_all(self, au_results: Dict[int, AUDetectionResult]) -> Dict[int, IntensityResult]:
        """全AUの強度を推定"""
        return {au_num: self.estimate(au_result) for au_num, au_result in au_results.items()}
    
    def format_facs_code(self, intensity_results: Dict[int, IntensityResult]) -> str:
        """FACSコードを生成"""
        active = [(r.au_number, r.intensity_label) for r in intensity_results.values() 
                  if r.intensity != AUIntensity.ABSENT]
        
        if not active:
            return "Neutral"
        
        # AU番号でソート
        active.sort(key=lambda x: x[0])
        
        return " + ".join(f"AU{au}{label}" for au, label in active)
