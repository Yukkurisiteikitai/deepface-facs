"""検出器のテスト"""
import pytest
import numpy as np

from facs.detectors import FeatureExtractor, AUDetector


class TestFeatureExtractor:
    """特徴量抽出器のテスト"""
    
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()
    
    @pytest.fixture
    def dummy_landmarks(self):
        """ダミーランドマーク（68点）"""
        np.random.seed(42)
        return np.random.rand(68, 2) * 100 + 100
    
    def test_compute_distances(self, extractor, dummy_landmarks):
        """距離計算"""
        distances = extractor.compute_distances(dummy_landmarks)
        assert isinstance(distances, dict)
        assert "eye_distance" in distances
    
    def test_compute_angles(self, extractor, dummy_landmarks):
        """角度計算"""
        angles = extractor.compute_angles(dummy_landmarks)
        assert isinstance(angles, dict)


class TestAUDetector:
    """AU検出器のテスト"""
    
    @pytest.fixture
    def detector(self):
        return AUDetector()
    
    @pytest.fixture
    def dummy_data(self):
        np.random.seed(42)
        landmarks = np.random.rand(68, 2) * 100 + 100
        distances = {"eye_distance": 50.0, "mouth_width": 40.0, 
                     "right_eye_aspect_ratio": 0.3, "left_eye_aspect_ratio": 0.3,
                     "mouth_height_outer": 10.0, "mouth_height_inner": 5.0,
                     "brow_distance": 30.0}
        angles = {"right_brow_angle": 0.0, "left_brow_angle": 0.0,
                  "right_mouth_angle": 0.0, "left_mouth_angle": 0.0}
        return landmarks, distances, angles
    
    def test_detect_all(self, detector, dummy_data):
        """全AU検出"""
        landmarks, distances, angles = dummy_data
        results = detector.detect_all(landmarks, distances, angles)
        assert isinstance(results, dict)
        assert len(results) > 0


# ランドマーク検出器のテストはMediaPipeの初期化が必要なため別途
class TestLandmarkDetectorFactory:
    """ランドマーク検出器ファクトリのテスト"""
    
    def test_import(self):
        """インポートテスト"""
        from facs.detectors import LandmarkDetectorFactory
        assert LandmarkDetectorFactory is not None
