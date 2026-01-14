"""FACSAnalyzerのテスト"""
import pytest
import numpy as np

from facs import FACSAnalyzer, AnalysisResult


class TestFACSAnalyzer:
    """FACSAnalyzerのテストクラス"""
    
    @pytest.fixture(scope="class")
    def analyzer(self):
        """Analyzerインスタンスを作成（クラス単位で共有）"""
        try:
            return FACSAnalyzer(use_mediapipe=True, use_deepface=False)
        except Exception as e:
            pytest.skip(f"FACSAnalyzer初期化失敗: {e}")
    
    @pytest.fixture
    def dummy_image(self):
        """ダミー画像を作成"""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_init(self, analyzer):
        """初期化テスト"""
        assert analyzer is not None
        assert analyzer._landmark_detector is not None
        assert analyzer._au_detector is not None
    
    def test_analyze_returns_result(self, analyzer, dummy_image):
        """analyze()がAnalysisResultを返すことを確認"""
        result = analyzer.analyze(dummy_image)
        assert isinstance(result, AnalysisResult)
    
    def test_analyze_empty_image(self, analyzer):
        """空の画像を分析"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = analyzer.analyze(empty_image)
        assert result is not None
        assert result.is_valid == False
    
    def test_list_all_aus(self):
        """AU一覧の取得"""
        aus = FACSAnalyzer.list_all_aus()
        assert isinstance(aus, list)
        assert len(aus) > 0
        assert all("number" in au for au in aus)
        assert all("name" in au for au in aus)
    
    def test_get_au_info(self):
        """AU情報の取得"""
        info = FACSAnalyzer.get_au_info(12)
        assert info is not None
        assert info["number"] == 12
        assert "name" in info
        
        # 存在しないAU
        info = FACSAnalyzer.get_au_info(999)
        assert info is None


class TestAnalysisResult:
    """AnalysisResultのテストクラス"""
    
    def test_default_values(self):
        """デフォルト値の確認"""
        result = AnalysisResult()
        assert result.is_valid == False
        assert result.facs_code == "Neutral"
        assert result.valence == 0.0
        assert result.arousal == 0.0
    
    def test_to_dict(self):
        """辞書変換"""
        result = AnalysisResult()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "facs_code" in d
        assert "valence" in d
        assert "arousal" in d
    
    def test_to_json(self):
        """JSON変換"""
        result = AnalysisResult()
        j = result.to_json()
        assert isinstance(j, str)
        assert "facs_code" in j
