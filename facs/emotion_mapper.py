import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .action_units import AU_COMBINATIONS, AUIntensity
from .au_detector import AUDetectionResult
from .intensity_estimator import IntensityResult

@dataclass
class EmotionResult:
    """感情推定結果"""
    emotion: str
    confidence: float
    valence: float      # -1.0 (negative) to 1.0 (positive)
    arousal: float      # -1.0 (low) to 1.0 (high)
    matched_aus: List[int]
    missing_aus: List[int]
    description: str

class EmotionMapper:
    """AUの組み合わせから感情を推定"""
    
    def __init__(self):
        """初期化"""
        self.emotion_definitions = AU_COMBINATIONS
        
        # 追加の感情定義（複合感情）
        self.complex_emotions = {
            "amusement": {
                "required_aus": [6, 12],
                "optional_aus": [25, 26],
                "modifiers": {"12": "high_intensity"},
                "description": "楽しさ・面白さ",
                "valence": 0.9,
                "arousal": 0.6
            },
            "embarrassment": {
                "required_aus": [6, 12, 15],
                "optional_aus": [7, 17],
                "description": "恥ずかしさ",
                "valence": -0.2,
                "arousal": 0.4
            },
            "pain": {
                "required_aus": [4, 6, 7, 9, 10],
                "optional_aus": [25, 26, 43],
                "description": "痛み",
                "valence": -0.9,
                "arousal": 0.7
            },
            "concentration": {
                "required_aus": [4],
                "optional_aus": [7, 23, 24],
                "description": "集中",
                "valence": 0.0,
                "arousal": 0.3
            },
            "confusion": {
                "required_aus": [1, 2, 4],
                "optional_aus": [7, 15],
                "description": "困惑",
                "valence": -0.3,
                "arousal": 0.4
            },
            "interest": {
                "required_aus": [1, 2],
                "optional_aus": [5, 26],
                "description": "興味・関心",
                "valence": 0.4,
                "arousal": 0.5
            },
            "boredom": {
                "required_aus": [43],
                "optional_aus": [15, 17],
                "description": "退屈",
                "valence": -0.3,
                "arousal": -0.6
            },
            "relief": {
                "required_aus": [12],
                "optional_aus": [6, 26],
                "description": "安堵",
                "valence": 0.6,
                "arousal": -0.3
            }
        }
    
    def map_emotion(self, au_results: Dict[int, AUDetectionResult],
                    intensity_results: Optional[Dict[int, IntensityResult]] = None) -> List[EmotionResult]:
        """AUから感情を推定"""
        detected_aus = {au_num for au_num, result in au_results.items() if result.detected}
        
        # AU強度を取得
        au_intensities = {}
        if intensity_results:
            au_intensities = {au_num: result.intensity_value 
                            for au_num, result in intensity_results.items()}
        else:
            au_intensities = {au_num: result.raw_score * 5.0 
                            for au_num, result in au_results.items()}
        
        emotions = []
        
        # 基本感情を評価
        for emotion_name, definition in self.emotion_definitions.items():
            result = self._evaluate_emotion(emotion_name, definition, detected_aus, au_intensities)
            if result:
                emotions.append(result)
        
        # 複合感情を評価
        for emotion_name, definition in self.complex_emotions.items():
            result = self._evaluate_emotion(emotion_name, definition, detected_aus, au_intensities)
            if result:
                emotions.append(result)
        
        # 信頼度でソート
        emotions.sort(key=lambda x: x.confidence, reverse=True)
        
        # 感情が検出されない場合はneutralを追加
        if not emotions or emotions[0].confidence < 0.3:
            neutral_result = EmotionResult(
                emotion="neutral",
                confidence=1.0 - (emotions[0].confidence if emotions else 0.0),
                valence=0.0,
                arousal=0.0,
                matched_aus=[],
                missing_aus=[],
                description="中立 - 特定の感情表現なし"
            )
            emotions.insert(0, neutral_result)
        
        return emotions
    
    def _evaluate_emotion(self, emotion_name: str, definition: Dict,
                          detected_aus: set, au_intensities: Dict[int, float]) -> Optional[EmotionResult]:
        """単一の感情を評価"""
        required = set(definition.get("required_aus", []))
        optional = set(definition.get("optional_aus", []))
        
        if len(required) == 0:
            return None
        
        # 必須AUの一致
        matched_required = required & detected_aus
        missing_required = required - detected_aus
        matched_optional = optional & detected_aus
        
        # 必須AUが半分以上一致していない場合はスキップ
        required_match_ratio = len(matched_required) / len(required) if required else 1.0
        if required_match_ratio < 0.5:
            return None
        
        # 信頼度を計算
        required_score = required_match_ratio
        optional_score = len(matched_optional) / len(optional) if optional else 0.0
        
        # マッチしたAUの強度を考慮
        intensity_bonus = 0.0
        for au_num in matched_required | matched_optional:
            intensity = au_intensities.get(au_num, 0.0)
            intensity_bonus += intensity / 5.0 * 0.1
        
        confidence = required_score * 0.7 + optional_score * 0.2 + min(intensity_bonus, 0.1)
        confidence = min(confidence, 1.0)
        
        # valenceとarousalを取得
        valence = definition.get("valence", 0.0)
        arousal = definition.get("arousal", 0.0)
        
        # 強度に基づいてvalence/arousalを調整
        avg_intensity = np.mean([au_intensities.get(au, 0) for au in matched_required | matched_optional]) if (matched_required | matched_optional) else 0
        intensity_factor = avg_intensity / 3.0  # 中程度の強度で1.0
        
        valence *= min(intensity_factor, 1.5)
        arousal *= min(intensity_factor, 1.5)
        
        return EmotionResult(
            emotion=emotion_name,
            confidence=confidence,
            valence=np.clip(valence, -1.0, 1.0),
            arousal=np.clip(arousal, -1.0, 1.0),
            matched_aus=list(matched_required | matched_optional),
            missing_aus=list(missing_required),
            description=definition.get("description", "")
        )
    
    def get_dominant_emotion(self, au_results: Dict[int, AUDetectionResult],
                             intensity_results: Optional[Dict[int, IntensityResult]] = None) -> EmotionResult:
        """最も可能性の高い感情を取得"""
        emotions = self.map_emotion(au_results, intensity_results)
        return emotions[0] if emotions else EmotionResult(
            emotion="neutral", confidence=1.0, valence=0.0, arousal=0.0,
            matched_aus=[], missing_aus=[], description="中立"
        )
    
    def get_emotion_blend(self, au_results: Dict[int, AUDetectionResult],
                          intensity_results: Optional[Dict[int, IntensityResult]] = None,
                          threshold: float = 0.3) -> Dict[str, float]:
        """感情のブレンド（複数感情の混合）を取得"""
        emotions = self.map_emotion(au_results, intensity_results)
        
        blend = {}
        total_confidence = 0.0
        
        for emotion in emotions:
            if emotion.confidence >= threshold:
                blend[emotion.emotion] = emotion.confidence
                total_confidence += emotion.confidence
        
        # 正規化
        if total_confidence > 0:
            blend = {k: v / total_confidence for k, v in blend.items()}
        
        return blend
    
    def get_valence_arousal(self, au_results: Dict[int, AUDetectionResult],
                            intensity_results: Optional[Dict[int, IntensityResult]] = None) -> Tuple[float, float]:
        """Valence-Arousal空間での位置を取得"""
        emotions = self.map_emotion(au_results, intensity_results)
        
        if not emotions:
            return 0.0, 0.0
        
        # 重み付き平均
        total_weight = sum(e.confidence for e in emotions)
        if total_weight == 0:
            return 0.0, 0.0
        
        valence = sum(e.valence * e.confidence for e in emotions) / total_weight
        arousal = sum(e.arousal * e.confidence for e in emotions) / total_weight
        
        return valence, arousal
    
    def detect_asymmetry(self, au_results: Dict[int, AUDetectionResult]) -> Dict[str, any]:
        """表情の非対称性を検出（軽蔑の検出に重要）"""
        asymmetric_aus = {}
        
        for au_num, result in au_results.items():
            if abs(result.asymmetry) > 0.3:
                asymmetric_aus[au_num] = {
                    "asymmetry": result.asymmetry,
                    "side": "left" if result.asymmetry > 0 else "right",
                    "description": f"AU{au_num}が{'左' if result.asymmetry > 0 else '右'}側で強い"
                }
        
        # 軽蔑の可能性を評価
        contempt_likelihood = 0.0
        if 12 in asymmetric_aus or 14 in asymmetric_aus:
            for au in [12, 14]:
                if au in asymmetric_aus:
                    contempt_likelihood = max(contempt_likelihood, abs(asymmetric_aus[au]["asymmetry"]))
        
        return {
            "asymmetric_aus": asymmetric_aus,
            "is_asymmetric": len(asymmetric_aus) > 0,
            "contempt_likelihood": contempt_likelihood,
            "possible_contempt": contempt_likelihood > 0.4
        }