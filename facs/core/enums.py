from enum import Enum, auto

class AUIntensity(Enum):
    """AU強度スケール (FACS標準 A-E)"""
    ABSENT = 0
    TRACE = 1       # A
    SLIGHT = 2      # B
    MARKED = 3      # C
    SEVERE = 4      # D
    MAXIMUM = 5     # E
    
    @property
    def label(self) -> str:
        labels = {0: "-", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
        return labels.get(self.value, "-")

class DetectorType(Enum):
    """ランドマーク検出器の種類"""
    DLIB = auto()
    MEDIAPIPE = auto()
    OPENCV = auto()
