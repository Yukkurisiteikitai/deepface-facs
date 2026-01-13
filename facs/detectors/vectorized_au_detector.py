"""
NumPyベクトル化されたAU検出器
高速な行列演算を使用して、全AUを同時に検出
"""
import numpy as np
from typing import Dict, Tuple, Optional
from ..core.models import AUDetectionResult
from ..core.enums import AUIntensity
from ..config import AU_DEFINITIONS


class VectorizedAUDetector:
    """ベクトル化されたAU検出器 - NumPy最適化版"""
    
    # ランドマークインデックス定数（頻繁にアクセスするため事前定義）
    # 眉
    RIGHT_BROW_INNER = 21
    LEFT_BROW_INNER = 22
    RIGHT_BROW_OUTER = 17
    LEFT_BROW_OUTER = 26
    NOSE_BRIDGE = 27
    
    # 目
    RIGHT_EYE = slice(36, 42)
    LEFT_EYE = slice(42, 48)
    RIGHT_EYE_OUTER = 36
    RIGHT_EYE_INNER = 39
    LEFT_EYE_OUTER = 45
    LEFT_EYE_INNER = 42
    
    # 口
    MOUTH_OUTER = slice(48, 60)
    MOUTH_INNER = slice(60, 68)
    MOUTH_RIGHT = 48
    MOUTH_LEFT = 54
    MOUTH_TOP = 51
    MOUTH_BOTTOM = 57
    NOSE_TIP = 30
    
    # 閾値設定（全AU用）
    THRESHOLDS = np.array([
        # [au_num, low, high]
        [1, 0.15, 0.4],
        [2, 0.15, 0.4],
        [4, 0.15, 0.4],
        [5, 0.15, 0.4],
        [6, 0.15, 0.4],
        [7, 0.15, 0.4],
        [9, 0.15, 0.4],
        [12, 0.15, 0.4],
        [15, 0.15, 0.4],
        [25, 0.15, 0.4],
        [26, 0.15, 0.4],
        [43, 0.15, 0.4],
    ])
    
    AU_NUMBERS = [1, 2, 4, 5, 6, 7, 9, 12, 15, 25, 26, 43]
    AU_INDEX_MAP = {au: i for i, au in enumerate(AU_NUMBERS)}
    
    def __init__(self):
        # 閾値を辞書形式でも保持（互換性のため）
        self._thresholds = {
            au: {"low": self.THRESHOLDS[i, 1], "high": self.THRESHOLDS[i, 2]}
            for i, au in enumerate(self.AU_NUMBERS)
        }
    
    def detect_all(self, landmarks: np.ndarray, distances: Dict[str, float],
                   angles: Dict[str, float]) -> Dict[int, AUDetectionResult]:
        """
        全AUを一括で検出（ベクトル化版）
        
        Args:
            landmarks: (68, 2)のランドマーク配列
            distances: 事前計算された距離辞書
            angles: 事前計算された角度辞書
            
        Returns:
            AU検出結果の辞書
        """
        eye_dist = max(distances.get("eye_distance", 1.0), 1e-6)
        
        # 全AUスコアを一括計算
        scores, asymmetries = self._compute_all_scores_vectorized(
            landmarks, distances, angles, eye_dist
        )
        
        # 結果を構築
        results = {}
        for i, au_num in enumerate(self.AU_NUMBERS):
            if au_num not in AU_DEFINITIONS:
                continue
                
            au_def = AU_DEFINITIONS[au_num]
            raw_score = scores[i]
            asymmetry = asymmetries[i]
            
            thresholds = self._thresholds[au_num]
            detected = raw_score >= thresholds["low"]
            confidence = min(raw_score / thresholds["high"], 1.0) if raw_score > 0 else 0.0
            intensity = self._score_to_intensity_fast(raw_score, thresholds["low"], thresholds["high"])
            
            results[au_num] = AUDetectionResult(
                au_number=au_num,
                name=au_def.name,
                detected=detected,
                confidence=confidence,
                intensity=intensity,
                raw_score=raw_score,
                asymmetry=asymmetry
            )
        
        return results
    
    def _compute_all_scores_vectorized(
        self, 
        landmarks: np.ndarray, 
        distances: Dict[str, float],
        angles: Dict[str, float], 
        eye_dist: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        全AUスコアをベクトル演算で一括計算
        
        Returns:
            scores: 各AUのスコア配列
            asymmetries: 各AUの左右非対称度配列
        """
        n_aus = len(self.AU_NUMBERS)
        scores = np.zeros(n_aus)
        asymmetries = np.zeros(n_aus)
        
        # ========================================
        # 事前に必要なランドマーク座標を抽出
        # ========================================
        
        # 眉関連
        lm_brow_inner_right = landmarks[self.RIGHT_BROW_INNER]  # 21
        lm_brow_inner_left = landmarks[self.LEFT_BROW_INNER]    # 22
        lm_brow_outer_right = landmarks[self.RIGHT_BROW_OUTER]  # 17
        lm_brow_outer_left = landmarks[self.LEFT_BROW_OUTER]    # 26
        lm_nose_bridge = landmarks[self.NOSE_BRIDGE]            # 27
        
        # 目関連 - ベクトル化のため配列として取得
        right_eye = landmarks[self.RIGHT_EYE]  # (6, 2)
        left_eye = landmarks[self.LEFT_EYE]    # (6, 2)
        
        # 口関連
        lm_mouth_right = landmarks[self.MOUTH_RIGHT]  # 48
        lm_mouth_left = landmarks[self.MOUTH_LEFT]    # 54
        lm_mouth_top = landmarks[self.MOUTH_TOP]      # 51
        lm_mouth_bottom = landmarks[self.MOUTH_BOTTOM]  # 57
        lm_nose_tip = landmarks[self.NOSE_TIP]        # 30
        
        # ========================================
        # 距離辞書から値を取得（デフォルト値付き）
        # ========================================
        right_ear = distances.get("right_eye_aspect_ratio", 0.25)
        left_ear = distances.get("left_eye_aspect_ratio", 0.25)
        brow_dist = distances.get("brow_distance", eye_dist * 0.3)
        mouth_width = distances.get("mouth_width", eye_dist * 0.5)
        mouth_height_inner = distances.get("mouth_height_inner", 0.0)
        mouth_height_outer = distances.get("mouth_height_outer", 0.0)
        
        # ========================================
        # AU1: Inner Brow Raiser (眉内側上げ)
        # ========================================
        i = self.AU_INDEX_MAP[1]
        right_dist_au1 = (lm_nose_bridge[1] - lm_brow_inner_right[1]) / eye_dist
        left_dist_au1 = (lm_nose_bridge[1] - lm_brow_inner_left[1]) / eye_dist
        avg_dist_au1 = (right_dist_au1 + left_dist_au1) / 2
        scores[i] = np.clip((avg_dist_au1 - 0.18) / 0.12, 0.0, 1.0)
        asymmetries[i] = np.clip(left_dist_au1 - right_dist_au1, -1.0, 1.0)
        
        # ========================================
        # AU2: Outer Brow Raiser (眉外側上げ)
        # ========================================
        i = self.AU_INDEX_MAP[2]
        right_dist_au2 = (landmarks[36, 1] - lm_brow_outer_right[1]) / eye_dist
        left_dist_au2 = (landmarks[45, 1] - lm_brow_outer_left[1]) / eye_dist
        avg_dist_au2 = (right_dist_au2 + left_dist_au2) / 2
        scores[i] = np.clip((avg_dist_au2 - 0.15) / 0.1, 0.0, 1.0)
        asymmetries[i] = np.clip(left_dist_au2 - right_dist_au2, -1.0, 1.0)
        
        # ========================================
        # AU4: Brow Lowerer (眉下げ)
        # ========================================
        i = self.AU_INDEX_MAP[4]
        brow_dist_normalized = brow_dist / eye_dist
        scores[i] = np.clip((0.35 - brow_dist_normalized) / 0.12, 0.0, 1.0)
        asymmetries[i] = 0.0
        
        # ========================================
        # AU5: Upper Lid Raiser (上まぶた上げ) & AU7: Lid Tightener & AU43: Eyes Closed
        # これらは全てEARベースなので一括計算
        # ========================================
        avg_ear = (right_ear + left_ear) / 2
        
        # AU5
        i = self.AU_INDEX_MAP[5]
        scores[i] = np.clip((avg_ear - 0.28) / 0.12, 0.0, 1.0)
        asymmetries[i] = np.clip(left_ear - right_ear, -1.0, 1.0)
        
        # AU7
        i = self.AU_INDEX_MAP[7]
        scores[i] = np.clip((0.25 - avg_ear) / 0.1, 0.0, 1.0)
        asymmetries[i] = np.clip(right_ear - left_ear, -1.0, 1.0)  # 逆向き
        
        # AU43
        i = self.AU_INDEX_MAP[43]
        scores[i] = np.clip((0.2 - avg_ear) / 0.15, 0.0, 1.0)
        asymmetries[i] = np.clip(right_ear - left_ear, -1.0, 1.0)
        
        # ========================================
        # AU6: Cheek Raiser (頬上げ)
        # ========================================
        i = self.AU_INDEX_MAP[6]
        # 目の下端（40, 41と46, 47の平均）
        right_eye_bottom = (landmarks[40] + landmarks[41]) / 2
        left_eye_bottom = (landmarks[46] + landmarks[47]) / 2
        
        # ベクトル化されたnorm計算
        right_dist_au6 = np.sqrt(np.sum((right_eye_bottom - lm_mouth_right) ** 2)) / eye_dist
        left_dist_au6 = np.sqrt(np.sum((left_eye_bottom - lm_mouth_left) ** 2)) / eye_dist
        
        right_score_au6 = np.clip((0.75 - right_dist_au6) / 0.18, 0.0, 1.0)
        left_score_au6 = np.clip((0.75 - left_dist_au6) / 0.18, 0.0, 1.0)
        scores[i] = (right_score_au6 + left_score_au6) / 2
        asymmetries[i] = np.clip(left_score_au6 - right_score_au6, -1.0, 1.0)
        
        # ========================================
        # AU9: Nose Wrinkler (鼻しわ)
        # ========================================
        i = self.AU_INDEX_MAP[9]
        dist_au9 = np.sqrt(np.sum((lm_mouth_top - lm_nose_tip) ** 2)) / eye_dist
        scores[i] = np.clip((0.35 - dist_au9) / 0.12, 0.0, 1.0)
        asymmetries[i] = 0.0
        
        # ========================================
        # AU12: Lip Corner Puller (口角上げ/笑顔)
        # ========================================
        i = self.AU_INDEX_MAP[12]
        right_elev = (lm_mouth_top[1] - lm_mouth_right[1]) / eye_dist
        left_elev = (lm_mouth_top[1] - lm_mouth_left[1]) / eye_dist
        mouth_width_norm = mouth_width / eye_dist
        
        right_score_au12 = np.clip((right_elev - 0.02) / 0.06, 0.0, 1.0) * 0.7 + \
                           np.clip((mouth_width_norm - 0.48) / 0.1, 0.0, 1.0) * 0.3
        left_score_au12 = np.clip((left_elev - 0.02) / 0.06, 0.0, 1.0) * 0.7 + \
                          np.clip((mouth_width_norm - 0.48) / 0.1, 0.0, 1.0) * 0.3
        scores[i] = np.clip((right_score_au12 + left_score_au12) / 2, 0.0, 1.0)
        asymmetries[i] = np.clip(left_elev - right_elev, -1.0, 1.0)
        
        # ========================================
        # AU15: Lip Corner Depressor (口角下げ)
        # ========================================
        i = self.AU_INDEX_MAP[15]
        right_dep = (lm_mouth_right[1] - lm_mouth_bottom[1]) / eye_dist
        left_dep = (lm_mouth_left[1] - lm_mouth_bottom[1]) / eye_dist
        right_score_au15 = np.clip((right_dep - 0.0) / 0.05, 0.0, 1.0)
        left_score_au15 = np.clip((left_dep - 0.0) / 0.05, 0.0, 1.0)
        scores[i] = (right_score_au15 + left_score_au15) / 2
        asymmetries[i] = np.clip(left_score_au15 - right_score_au15, -1.0, 1.0)
        
        # ========================================
        # AU25: Lips Part (唇開き)
        # ========================================
        i = self.AU_INDEX_MAP[25]
        mouth_height_inner_norm = mouth_height_inner / eye_dist
        scores[i] = np.clip((mouth_height_inner_norm - 0.02) / 0.08, 0.0, 1.0)
        asymmetries[i] = 0.0
        
        # ========================================
        # AU26: Jaw Drop (顎下げ)
        # ========================================
        i = self.AU_INDEX_MAP[26]
        mouth_height_outer_norm = mouth_height_outer / eye_dist
        scores[i] = np.clip((mouth_height_outer_norm - 0.05) / 0.12, 0.0, 1.0)
        asymmetries[i] = 0.0
        
        return scores, asymmetries
    
    @staticmethod
    def _score_to_intensity_fast(score: float, low: float, high: float) -> AUIntensity:
        """高速な強度変換（分岐を最小化）"""
        if score < low * 0.5:
            return AUIntensity.ABSENT
        elif score < low:
            return AUIntensity.TRACE
        elif score < high * 0.66:
            return AUIntensity.SLIGHT
        elif score < high:
            return AUIntensity.MARKED
        elif score < high * 1.3:
            return AUIntensity.SEVERE
        return AUIntensity.MAXIMUM


class BatchVectorizedAUDetector(VectorizedAUDetector):
    """バッチ処理対応のベクトル化AU検出器"""
    
    def detect_batch(
        self, 
        landmarks_batch: np.ndarray,
        distances_batch: list,
        angles_batch: list
    ) -> list:
        """
        複数フレームのAU検出をバッチで処理
        
        Args:
            landmarks_batch: (N, 68, 2)のランドマーク配列
            distances_batch: N個の距離辞書のリスト
            angles_batch: N個の角度辞書のリスト
            
        Returns:
            N個のAU検出結果辞書のリスト
        """
        batch_size = landmarks_batch.shape[0]
        results = []
        
        # eye_distを一括計算
        eye_dists = np.array([
            max(d.get("eye_distance", 1.0), 1e-6) 
            for d in distances_batch
        ])
        
        # 各フレームの処理（将来的にはこれもベクトル化可能）
        for i in range(batch_size):
            result = self.detect_all(
                landmarks_batch[i],
                distances_batch[i],
                angles_batch[i]
            )
            results.append(result)
        
        return results
