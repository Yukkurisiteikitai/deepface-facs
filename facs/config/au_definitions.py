from typing import Dict
from ..core.models import ActionUnitDefinition, EmotionDefinition

# FACS Action Unit 定義
AU_DEFINITIONS: Dict[int, ActionUnitDefinition] = {
    1: ActionUnitDefinition(1, "Inner Brow Raiser", "眉の内側を上げる",
                            "Frontalis (pars medialis)", (17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)),
    2: ActionUnitDefinition(2, "Outer Brow Raiser", "眉の外側を上げる",
                            "Frontalis (pars lateralis)", (17, 18, 19, 20, 21, 22, 23, 24, 25, 26)),
    4: ActionUnitDefinition(4, "Brow Lowerer", "眉を下げる・寄せる",
                            "Corrugator supercilii", (17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)),
    5: ActionUnitDefinition(5, "Upper Lid Raiser", "上まぶたを上げる",
                            "Levator palpebrae superioris", (36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47)),
    6: ActionUnitDefinition(6, "Cheek Raiser", "頬を上げる",
                            "Orbicularis oculi (pars orbitalis)", (36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 54)),
    7: ActionUnitDefinition(7, "Lid Tightener", "まぶたを締める",
                            "Orbicularis oculi (pars palpebralis)", (36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47)),
    9: ActionUnitDefinition(9, "Nose Wrinkler", "鼻にしわを寄せる",
                            "Levator labii superioris alaeque nasi", (27, 28, 29, 30, 31, 32, 33, 34, 35)),
    10: ActionUnitDefinition(10, "Upper Lip Raiser", "上唇を上げる",
                             "Levator labii superioris", (31, 32, 33, 34, 35, 48, 49, 50, 51, 52)),
    12: ActionUnitDefinition(12, "Lip Corner Puller", "口角を引き上げる（笑顔）",
                             "Zygomaticus major", (48, 54, 60, 64)),
    15: ActionUnitDefinition(15, "Lip Corner Depressor", "口角を下げる",
                             "Depressor anguli oris", (48, 54, 55, 56, 57, 58, 59)),
    17: ActionUnitDefinition(17, "Chin Raiser", "あごを上げる",
                             "Mentalis", (7, 8, 9, 55, 56, 57, 58, 59)),
    20: ActionUnitDefinition(20, "Lip Stretcher", "唇を横に伸ばす",
                             "Risorius", (48, 54, 60, 64)),
    23: ActionUnitDefinition(23, "Lip Tightener", "唇を締める",
                             "Orbicularis oris", (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59)),
    25: ActionUnitDefinition(25, "Lips Part", "唇を開く",
                             "Depressor labii inferioris", (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59)),
    26: ActionUnitDefinition(26, "Jaw Drop", "顎を下げる",
                             "Masseter", (5, 6, 7, 8, 9, 10, 11, 48, 54)),
    43: ActionUnitDefinition(43, "Eyes Closed", "目を閉じる",
                             "Relaxation of Levator palpebrae superioris", (36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47)),
}

# 感情定義
EMOTION_DEFINITIONS: Dict[str, EmotionDefinition] = {
    "happiness": EmotionDefinition("happiness", (6, 12), (25, 26), "幸福・喜び", 1.0, 0.5),
    "sadness": EmotionDefinition("sadness", (1, 4, 15), (6, 17), "悲しみ", -0.7, -0.3),
    "surprise": EmotionDefinition("surprise", (1, 2, 5, 26), (27,), "驚き", 0.0, 0.8),
    "fear": EmotionDefinition("fear", (1, 2, 4, 5, 20, 26), (7, 25), "恐怖", -0.8, 0.9),
    "anger": EmotionDefinition("anger", (4, 5, 7, 23), (10, 17, 25, 26), "怒り", -0.8, 0.7),
    "disgust": EmotionDefinition("disgust", (9, 15), (4, 6, 10, 17), "嫌悪", -0.6, 0.3),
    "contempt": EmotionDefinition("contempt", (12,), (), "軽蔑（非対称）", -0.4, 0.1),
}

# ランドマーク名（互換性のため維持）
LANDMARK_NAMES: Dict[int, str] = {
    0: "jaw_right_1", 8: "chin", 16: "jaw_left_1",
    17: "right_eyebrow_outer", 21: "right_eyebrow_inner",
    22: "left_eyebrow_inner", 26: "left_eyebrow_outer",
    27: "nose_bridge_1", 30: "nose_tip", 33: "nose_bottom",
    36: "right_eye_outer", 39: "right_eye_inner",
    42: "left_eye_inner", 45: "left_eye_outer",
    48: "mouth_right", 51: "mouth_top_center", 54: "mouth_left", 57: "mouth_bottom_center",
    62: "inner_mouth_top_center", 66: "inner_mouth_bottom_center"
}
