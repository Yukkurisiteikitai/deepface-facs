from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np

class ILandmarkDetector(ABC):
    """ランドマーク検出器のインターフェース"""
    
    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """顔を検出して矩形を返す"""
        pass
    
    @abstractmethod
    def detect_landmarks(self, image: np.ndarray, 
                         face_rect: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """68点ランドマークを検出"""
        pass

class IFeatureExtractor(ABC):
    """特徴量抽出器のインターフェース"""
    
    @abstractmethod
    def compute_distances(self, landmarks: np.ndarray) -> Dict[str, float]:
        """距離特徴を計算"""
        pass
    
    @abstractmethod
    def compute_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """角度特徴を計算"""
        pass

class IAUDetectionStrategy(ABC):
    """AU検出戦略のインターフェース（Strategy Pattern）"""
    
    @property
    @abstractmethod
    def au_number(self) -> int:
        """対象のAU番号"""
        pass
    
    @abstractmethod
    def detect(self, landmarks: np.ndarray, distances: Dict[str, float],
               angles: Dict[str, float], eye_dist: float) -> Tuple[float, float]:
        """AU検出を実行。(スコア, 非対称度)を返す"""
        pass

class IAUDetector(ABC):
    """AU検出器のインターフェース"""
    
    @abstractmethod
    def detect_all(self, landmarks: np.ndarray, distances: Dict[str, float],
                   angles: Dict[str, float]) -> Dict[int, 'AUDetectionResult']:
        """すべてのAUを検出"""
        pass
    
    @abstractmethod
    def register_strategy(self, strategy: IAUDetectionStrategy) -> None:
        """AU検出戦略を登録"""
        pass

class IIntensityEstimator(ABC):
    """強度推定器のインターフェース"""
    
    @abstractmethod
    def estimate(self, au_result: 'AUDetectionResult') -> 'IntensityResult':
        """強度を推定"""
        pass
    
    @abstractmethod
    def format_facs_code(self, results: Dict[int, 'IntensityResult']) -> str:
        """FACSコードを生成"""
        pass

class IEmotionMapper(ABC):
    """感情マッピングのインターフェース"""
    
    @abstractmethod
    def map(self, au_results: Dict[int, 'AUDetectionResult'],
            intensity_results: Optional[Dict[int, 'IntensityResult']] = None) -> List['EmotionResult']:
        """感情を推定"""
        pass
    
    @abstractmethod
    def get_valence_arousal(self, au_results: Dict[int, 'AUDetectionResult'],
                            intensity_results: Optional[Dict[int, 'IntensityResult']] = None) -> Tuple[float, float]:
        """Valence-Arousalを取得"""
        pass

class IVisualizer(ABC):
    """可視化のインターフェース"""
    
    @abstractmethod
    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """ランドマークを描画"""
        pass
    
    @abstractmethod
    def create_analysis_panel(self, image: np.ndarray, result: 'AnalysisResult') -> np.ndarray:
        """分析パネルを作成"""
        pass
