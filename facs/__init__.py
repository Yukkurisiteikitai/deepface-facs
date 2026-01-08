"""
FACS (Facial Action Coding System) Analyzer

顔の微表情を解析し、Action Unit (AU) に基づいて感情を推定するライブラリ。

Example:
    >>> from facs import FACSAnalyzer
    >>> import cv2
    >>> 
    >>> analyzer = FACSAnalyzer()
    >>> image = cv2.imread("face.jpg")
    >>> result = analyzer.analyze(image)
    >>> print(result.facs_code)
    >>> print(result.dominant_emotion.emotion)
"""

__version__ = "0.1.0"
__author__ = "yuuto"

from .core import (
    AUIntensity,
    AUDetectionResult,
    IntensityResult,
    EmotionResult,
    AnalysisResult,
    FaceData,
    TerminalDisplay,
)
from .config import AU_DEFINITIONS, EMOTION_DEFINITIONS, LANDMARK_NAMES
from .detectors import (
    LandmarkDetectorFactory,
    FeatureExtractor,
    AUDetector,
    FaceAligner,
    FaceAlignment,
)
from .estimators import IntensityEstimator, EmotionMapper
from .visualization import FACSVisualizer, InteractiveFACSVisualizer
from .analyzer import FACSAnalyzer

__all__ = [
    # Version
    "__version__",
    # Main class
    "FACSAnalyzer",
    # Data models
    "AUIntensity",
    "AUDetectionResult",
    "IntensityResult",
    "EmotionResult",
    "AnalysisResult",
    "FaceData",
    # Config
    "AU_DEFINITIONS",
    "EMOTION_DEFINITIONS",
    "LANDMARK_NAMES",
    # Detectors
    "LandmarkDetectorFactory",
    "FeatureExtractor",
    "AUDetector",
    "FaceAligner",
    "FaceAlignment",
    # Estimators
    "IntensityEstimator",
    "EmotionMapper",
    # Visualization
    "FACSVisualizer",
    "InteractiveFACSVisualizer",
    "TerminalDisplay",
]
