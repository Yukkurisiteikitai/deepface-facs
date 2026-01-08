from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional

class AUIntensity(Enum):
    """AU強度スケール (FACS標準)"""
    ABSENT = 0      # 不在
    TRACE = 1       # A - 微弱
    SLIGHT = 2      # B - 軽度
    MARKED = 3      # C - 中程度
    SEVERE = 4      # D - 顕著
    MAXIMUM = 5     # E - 最大

@dataclass
class ActionUnit:
    """Action Unitの定義"""
    au_number: int
    name: str
    description: str
    muscular_basis: str
    landmarks_involved: List[int]  # 68点ランドマークのインデックス
    detection_method: str

# FACS Action Unit 定義 (主要なAU)
AU_DEFINITIONS: Dict[int, ActionUnit] = {
    # 上顔面 (Upper Face)
    1: ActionUnit(
        au_number=1,
        name="Inner Brow Raiser",
        description="眉の内側を上げる",
        muscular_basis="Frontalis (pars medialis)",
        landmarks_involved=[17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        detection_method="eyebrow_inner_raise"
    ),
    2: ActionUnit(
        au_number=2,
        name="Outer Brow Raiser",
        description="眉の外側を上げる",
        muscular_basis="Frontalis (pars lateralis)",
        landmarks_involved=[17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        detection_method="eyebrow_outer_raise"
    ),
    4: ActionUnit(
        au_number=4,
        name="Brow Lowerer",
        description="眉を下げる・寄せる",
        muscular_basis="Corrugator supercilii, Depressor supercilii",
        landmarks_involved=[17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
        detection_method="brow_lowerer"
    ),
    5: ActionUnit(
        au_number=5,
        name="Upper Lid Raiser",
        description="上まぶたを上げる",
        muscular_basis="Levator palpebrae superioris",
        landmarks_involved=[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        detection_method="upper_lid_raise"
    ),
    6: ActionUnit(
        au_number=6,
        name="Cheek Raiser",
        description="頬を上げる",
        muscular_basis="Orbicularis oculi (pars orbitalis)",
        landmarks_involved=[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 54],
        detection_method="cheek_raise"
    ),
    7: ActionUnit(
        au_number=7,
        name="Lid Tightener",
        description="まぶたを締める",
        muscular_basis="Orbicularis oculi (pars palpebralis)",
        landmarks_involved=[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        detection_method="lid_tighten"
    ),
    
    # 下顔面 (Lower Face)
    9: ActionUnit(
        au_number=9,
        name="Nose Wrinkler",
        description="鼻にしわを寄せる",
        muscular_basis="Levator labii superioris alaeque nasi",
        landmarks_involved=[27, 28, 29, 30, 31, 32, 33, 34, 35],
        detection_method="nose_wrinkle"
    ),
    10: ActionUnit(
        au_number=10,
        name="Upper Lip Raiser",
        description="上唇を上げる",
        muscular_basis="Levator labii superioris",
        landmarks_involved=[31, 32, 33, 34, 35, 48, 49, 50, 51, 52, 61, 62, 63],
        detection_method="upper_lip_raise"
    ),
    11: ActionUnit(
        au_number=11,
        name="Nasolabial Deepener",
        description="鼻唇溝を深くする",
        muscular_basis="Zygomaticus minor",
        landmarks_involved=[31, 35, 48, 54],
        detection_method="nasolabial_deepen"
    ),
    12: ActionUnit(
        au_number=12,
        name="Lip Corner Puller",
        description="口角を引き上げる（笑顔）",
        muscular_basis="Zygomaticus major",
        landmarks_involved=[48, 54, 60, 64],
        detection_method="lip_corner_pull"
    ),
    13: ActionUnit(
        au_number=13,
        name="Sharp Lip Puller",
        description="口角を鋭く引く",
        muscular_basis="Levator anguli oris (Caninus)",
        landmarks_involved=[48, 49, 53, 54],
        detection_method="sharp_lip_pull"
    ),
    14: ActionUnit(
        au_number=14,
        name="Dimpler",
        description="えくぼを作る",
        muscular_basis="Buccinator",
        landmarks_involved=[48, 54],
        detection_method="dimpler"
    ),
    15: ActionUnit(
        au_number=15,
        name="Lip Corner Depressor",
        description="口角を下げる",
        muscular_basis="Depressor anguli oris (Triangularis)",
        landmarks_involved=[48, 54, 55, 56, 57, 58, 59],
        detection_method="lip_corner_depress"
    ),
    16: ActionUnit(
        au_number=16,
        name="Lower Lip Depressor",
        description="下唇を下げる",
        muscular_basis="Depressor labii inferioris",
        landmarks_involved=[55, 56, 57, 58, 59, 65, 66, 67],
        detection_method="lower_lip_depress"
    ),
    17: ActionUnit(
        au_number=17,
        name="Chin Raiser",
        description="あごを上げる",
        muscular_basis="Mentalis",
        landmarks_involved=[7, 8, 9, 55, 56, 57, 58, 59],
        detection_method="chin_raise"
    ),
    18: ActionUnit(
        au_number=18,
        name="Lip Puckerer",
        description="唇をすぼめる",
        muscular_basis="Incisivii labii superioris and inferioris",
        landmarks_involved=[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        detection_method="lip_pucker"
    ),
    20: ActionUnit(
        au_number=20,
        name="Lip Stretcher",
        description="唇を横に伸ばす",
        muscular_basis="Risorius",
        landmarks_involved=[48, 54, 60, 64],
        detection_method="lip_stretch"
    ),
    22: ActionUnit(
        au_number=22,
        name="Lip Funneler",
        description="唇を漏斗状にする",
        muscular_basis="Orbicularis oris",
        landmarks_involved=[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
        detection_method="lip_funnel"
    ),
    23: ActionUnit(
        au_number=23,
        name="Lip Tightener",
        description="唇を締める",
        muscular_basis="Orbicularis oris",
        landmarks_involved=[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        detection_method="lip_tighten"
    ),
    24: ActionUnit(
        au_number=24,
        name="Lip Presser",
        description="唇を押し付ける",
        muscular_basis="Orbicularis oris",
        landmarks_involved=[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
        detection_method="lip_press"
    ),
    25: ActionUnit(
        au_number=25,
        name="Lips Part",
        description="唇を開く",
        muscular_basis="Depressor labii inferioris, Relaxation of Mentalis/Orbicularis oris",
        landmarks_involved=[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
        detection_method="lips_part"
    ),
    26: ActionUnit(
        au_number=26,
        name="Jaw Drop",
        description="顎を下げる",
        muscular_basis="Masseter, Relaxation of Temporalis and internal pterygoid",
        landmarks_involved=[5, 6, 7, 8, 9, 10, 11, 48, 54, 55, 56, 57, 58, 59],
        detection_method="jaw_drop"
    ),
    27: ActionUnit(
        au_number=27,
        name="Mouth Stretch",
        description="口を大きく開く",
        muscular_basis="Pterygoids, Digastric",
        landmarks_involved=[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
        detection_method="mouth_stretch"
    ),
    28: ActionUnit(
        au_number=28,
        name="Lip Suck",
        description="唇を吸い込む",
        muscular_basis="Orbicularis oris",
        landmarks_involved=[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        detection_method="lip_suck"
    ),
    
    # 頭部動作 (Head Movements)
    43: ActionUnit(
        au_number=43,
        name="Eyes Closed",
        description="目を閉じる",
        muscular_basis="Relaxation of Levator palpebrae superioris",
        landmarks_involved=[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        detection_method="eyes_closed"
    ),
    45: ActionUnit(
        au_number=45,
        name="Blink",
        description="まばたき",
        muscular_basis="Relaxation of Levator palpebrae superioris; Orbicularis oculi",
        landmarks_involved=[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        detection_method="blink"
    ),
    46: ActionUnit(
        au_number=46,
        name="Wink",
        description="ウィンク",
        muscular_basis="Orbicularis oculi",
        landmarks_involved=[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        detection_method="wink"
    ),
}

# 感情とAUの組み合わせ (Ekmanの基本感情)
AU_COMBINATIONS: Dict[str, Dict[str, any]] = {
    "happiness": {
        "required_aus": [6, 12],
        "optional_aus": [25, 26],
        "description": "幸福・喜び - 頬を上げ、口角を引き上げる",
        "valence": 1.0,
        "arousal": 0.5
    },
    "sadness": {
        "required_aus": [1, 4, 15],
        "optional_aus": [6, 11, 17],
        "description": "悲しみ - 眉の内側を上げ、眉を寄せ、口角を下げる",
        "valence": -0.7,
        "arousal": -0.3
    },
    "surprise": {
        "required_aus": [1, 2, 5, 26],
        "optional_aus": [27],
        "description": "驚き - 眉を上げ、目を見開き、口を開く",
        "valence": 0.0,
        "arousal": 0.8
    },
    "fear": {
        "required_aus": [1, 2, 4, 5, 20, 26],
        "optional_aus": [7, 25, 27],
        "description": "恐怖 - 眉を上げて寄せ、目を見開き、唇を伸ばす",
        "valence": -0.8,
        "arousal": 0.9
    },
    "anger": {
        "required_aus": [4, 5, 7, 23],
        "optional_aus": [10, 17, 24, 25, 26],
        "description": "怒り - 眉を寄せ下げ、目を見開き、唇を締める",
        "valence": -0.8,
        "arousal": 0.7
    },
    "disgust": {
        "required_aus": [9, 15, 16],
        "optional_aus": [4, 6, 10, 17, 25, 26],
        "description": "嫌悪 - 鼻にしわを寄せ、口角と下唇を下げる",
        "valence": -0.6,
        "arousal": 0.3
    },
    "contempt": {
        "required_aus": [12, 14],  # 片側のみ
        "optional_aus": [],
        "description": "軽蔑 - 片側の口角を上げる（非対称）",
        "valence": -0.4,
        "arousal": 0.1
    },
    "neutral": {
        "required_aus": [],
        "optional_aus": [],
        "description": "中立 - 特定のAUが検出されない",
        "valence": 0.0,
        "arousal": 0.0
    }
}

# 68点ランドマークの名称マッピング
LANDMARK_NAMES: Dict[int, str] = {
    # 顔の輪郭 (0-16)
    0: "jaw_right_1", 1: "jaw_right_2", 2: "jaw_right_3", 3: "jaw_right_4",
    4: "jaw_right_5", 5: "jaw_right_6", 6: "jaw_right_7", 7: "jaw_right_8",
    8: "chin", 9: "jaw_left_8", 10: "jaw_left_7", 11: "jaw_left_6",
    12: "jaw_left_5", 13: "jaw_left_4", 14: "jaw_left_3", 15: "jaw_left_2",
    16: "jaw_left_1",
    
    # 右眉 (17-21)
    17: "right_eyebrow_outer", 18: "right_eyebrow_2", 19: "right_eyebrow_3",
    20: "right_eyebrow_4", 21: "right_eyebrow_inner",
    
    # 左眉 (22-26)
    22: "left_eyebrow_inner", 23: "left_eyebrow_2", 24: "left_eyebrow_3",
    25: "left_eyebrow_4", 26: "left_eyebrow_outer",
    
    # 鼻 (27-35)
    27: "nose_bridge_1", 28: "nose_bridge_2", 29: "nose_bridge_3", 30: "nose_tip",
    31: "nose_right_2", 32: "nose_right_1", 33: "nose_bottom",
    34: "nose_left_1", 35: "nose_left_2",
    
    # 右目 (36-41)
    36: "right_eye_outer", 37: "right_eye_top_outer", 38: "right_eye_top_inner",
    39: "right_eye_inner", 40: "right_eye_bottom_inner", 41: "right_eye_bottom_outer",
    
    # 左目 (42-47)
    42: "left_eye_inner", 43: "left_eye_top_inner", 44: "left_eye_top_outer",
    45: "left_eye_outer", 46: "left_eye_bottom_outer", 47: "left_eye_bottom_inner",
    
    # 外側唇 (48-59)
    48: "mouth_right", 49: "mouth_top_right_2", 50: "mouth_top_right_1",
    51: "mouth_top_center", 52: "mouth_top_left_1", 53: "mouth_top_left_2",
    54: "mouth_left", 55: "mouth_bottom_left_2", 56: "mouth_bottom_left_1",
    57: "mouth_bottom_center", 58: "mouth_bottom_right_1", 59: "mouth_bottom_right_2",
    
    # 内側唇 (60-67)
    60: "inner_mouth_right", 61: "inner_mouth_top_right", 62: "inner_mouth_top_center",
    63: "inner_mouth_top_left", 64: "inner_mouth_left", 65: "inner_mouth_bottom_left",
    66: "inner_mouth_bottom_center", 67: "inner_mouth_bottom_right"
}
