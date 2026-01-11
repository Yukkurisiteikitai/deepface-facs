"""
抽象インターフェース定義
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np

from .models import AUDetectionResult, IntensityResult, EmotionResult, AnalysisResult


class ILandmarkDetector(ABC):
    """ランドマーク検出器インターフェース"""
    
    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """顔を検出"""
        pass
    
    @abstractmethod
    def detect_landmarks(self, image: np.ndarray,
                         face_rect: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """ランドマークを検出"""
        pass


class IFeatureExtractor(ABC):
    """特徴量抽出器インターフェース"""
    
    @abstractmethod
    def compute_distances(self, landmarks: np.ndarray) -> Dict[str, float]:
        """距離特徴を計算"""
        pass
    
    @abstractmethod
    def compute_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """角度特徴を計算"""
        pass


class IAUDetector(ABC):
    """AU検出器インターフェース"""
    
    @abstractmethod
    def detect_all(self, landmarks: np.ndarray, distances: Dict[str, float],
                   angles: Dict[str, float]) -> Dict[int, AUDetectionResult]:
        """全AUを検出"""
        pass


class IIntensityEstimator(ABC):
    """強度推定器インターフェース"""
    
    @abstractmethod
    def estimate(self, au_result: AUDetectionResult) -> IntensityResult:
        """単一AUの強度を推定"""
        pass
    
    @abstractmethod
    def estimate_all(self, au_results: Dict[int, AUDetectionResult]) -> Dict[int, IntensityResult]:
        """全AUの強度を推定"""
        pass


class IEmotionMapper(ABC):
    """感情マッピングインターフェース"""
    
    @abstractmethod
    def map(self, au_results: Dict[int, AUDetectionResult],
            intensity_results: Dict[int, IntensityResult]) -> List[EmotionResult]:
        """AUから感情を推定"""
        pass


class IVisualizer(ABC):
    """可視化インターフェース"""
    
    @abstractmethod
    def create_analysis_panel(self, image: np.ndarray, result: AnalysisResult) -> np.ndarray:
        """分析パネルを作成"""
        pass


class IAUDetectionStrategy(ABC):
    """AU検出戦略インターフェース"""
    
    @property
    @abstractmethod
    def au_number(self) -> int:
        """対象のAU番号"""
        pass
    
    @abstractmethod
    def detect(self, landmarks: np.ndarray, distances: Dict[str, float],
               angles: Dict[str, float], eye_dist: float) -> Tuple[float, float]:
        """検出を実行し、(raw_score, asymmetry)を返す"""
        pass
