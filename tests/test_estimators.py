"""推定器のテスト"""
import pytest

from facs.estimators import IntensityEstimator, EmotionMapper
from facs.core.models import AUDetectionResult
from facs.core.enums import AUIntensity


class TestIntensityEstimator:
    """強度推定器のテスト"""
    
    @pytest.fixture
    def estimator(self):
        return IntensityEstimator()
    
    @pytest.fixture
    def dummy_au_result(self):
        return AUDetectionResult(
            au_number=12,
            name="Lip Corner Puller",
            detected=True,
            confidence=0.8,
            intensity=AUIntensity.MARKED,
            raw_score=0.6
        )
    
    def test_estimate_intensity(self, estimator, dummy_au_result):
        """強度推定"""
        result = estimator.estimate(dummy_au_result)
        assert result is not None
        assert result.au_number == 12
        assert result.intensity_label in ["-", "A", "B", "C", "D", "E"]
    
    def test_estimate_all(self, estimator, dummy_au_result):
        """全AU強度推定"""
        au_results = {12: dummy_au_result}
        results = estimator.estimate_all(au_results)
        assert isinstance(results, dict)
        assert 12 in results
    
    def test_format_facs_code(self, estimator, dummy_au_result):
        """FACSコード生成"""
        au_results = {12: dummy_au_result}
        intensity_results = estimator.estimate_all(au_results)
        code = estimator.format_facs_code(intensity_results)
        assert isinstance(code, str)
    
    def test_estimate_not_detected(self, estimator):
        """未検出AUの強度推定"""
        au_result = AUDetectionResult(
            au_number=1,
            name="Inner Brow Raiser",
            detected=False,
            confidence=0.2,
            intensity=AUIntensity.ABSENT,
            raw_score=0.1
        )
        result = estimator.estimate(au_result)
        assert result.intensity == AUIntensity.ABSENT
        assert result.intensity_label == "-"


class TestEmotionMapper:
    """感情マッピングのテスト"""
    
    @pytest.fixture
    def mapper(self):
        return EmotionMapper()
    
    def test_map_empty(self, mapper):
        """空の入力"""
        emotions = mapper.map({}, {})
        assert isinstance(emotions, list)
    
    def test_get_valence_arousal(self, mapper):
        """Valence-Arousal取得"""
        v, a = mapper.get_valence_arousal({}, {})
        assert isinstance(v, float)
        assert isinstance(a, float)
        assert -1.0 <= v <= 1.0
        assert -1.0 <= a <= 1.0
