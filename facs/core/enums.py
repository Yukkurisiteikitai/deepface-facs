"""列挙型定義"""
from enum import Enum


class AUIntensity(Enum):
    """AU強度スケール (FACS標準)"""
    ABSENT = 0
    TRACE = 1      # A
    SLIGHT = 2     # B
    MARKED = 3     # C
    SEVERE = 4     # D
    MAXIMUM = 5    # E


class DetectorType(Enum):
    """検出器タイプ"""
    MEDIAPIPE = "mediapipe"
    DLIB = "dlib"
    OPENCV = "opencv"


class AnalysisMode(Enum):
    """分析モード"""
    REALTIME = "realtime"      # 軽量・高速（リアルタイム向け）
    BALANCED = "balanced"      # バランス型（デフォルト）
    ACCURATE = "accurate"      # 高精度・低速（詳細分析向け）
