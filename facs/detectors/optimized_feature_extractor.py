"""
最適化された特徴量抽出器
NumPyのブロードキャスト演算を活用して高速に特徴量を計算
"""
import numpy as np
from typing import Dict, Tuple
from ..core.interfaces import IFeatureExtractor


class OptimizedFeatureExtractor(IFeatureExtractor):
    """最適化された顔特徴量抽出器"""
    
    # ランドマークインデックス定数
    RIGHT_EYE = slice(36, 42)
    LEFT_EYE = slice(42, 48)
    OUTER_MOUTH = slice(48, 60)
    INNER_MOUTH = slice(60, 68)
    RIGHT_EYEBROW = slice(17, 22)
    LEFT_EYEBROW = slice(22, 27)
    
    # 目のアスペクト比計算用インデックス
    # 右目: 上1-下5, 上2-下4 / 幅0-3
    RIGHT_EYE_V1 = (37, 41)  # 上下ペア1
    RIGHT_EYE_V2 = (38, 40)  # 上下ペア2
    RIGHT_EYE_H = (36, 39)   # 横幅
    
    # 左目: 同様
    LEFT_EYE_V1 = (43, 47)
    LEFT_EYE_V2 = (44, 46)
    LEFT_EYE_H = (42, 45)
    
    # 口
    MOUTH_WIDTH = (48, 54)
    MOUTH_HEIGHT_OUTER = (51, 57)
    MOUTH_HEIGHT_INNER = (62, 66)
    
    # 眉
    BROW_DIST = (21, 22)
    
    def __init__(self):
        # 事前計算用のインデックス配列
        self._setup_index_arrays()
    
    def _setup_index_arrays(self):
        """計算用のインデックス配列を事前設定"""
        # 目のアスペクト比計算用
        self._right_eye_v_pairs = np.array([[37, 41], [38, 40]])
        self._left_eye_v_pairs = np.array([[43, 47], [44, 46]])
    
    def compute_distances(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        距離特徴量を計算（ベクトル化版）
        
        Args:
            landmarks: (68, 2)のランドマーク配列
            
        Returns:
            距離特徴量の辞書
        """
        # ========================================
        # 目のアスペクト比（EAR）を一括計算
        # ========================================
        # 右目
        right_eye_v1 = np.linalg.norm(landmarks[37] - landmarks[41])
        right_eye_v2 = np.linalg.norm(landmarks[38] - landmarks[40])
        right_eye_height = (right_eye_v1 + right_eye_v2) / 2
        right_eye_width = np.linalg.norm(landmarks[36] - landmarks[39])
        
        # 左目
        left_eye_v1 = np.linalg.norm(landmarks[43] - landmarks[47])
        left_eye_v2 = np.linalg.norm(landmarks[44] - landmarks[46])
        left_eye_height = (left_eye_v1 + left_eye_v2) / 2
        left_eye_width = np.linalg.norm(landmarks[42] - landmarks[45])
        
        # ========================================
        # 口の距離
        # ========================================
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
        mouth_height_outer = np.linalg.norm(landmarks[51] - landmarks[57])
        
        # 内唇（存在確認）
        if landmarks.shape[0] > 66:
            mouth_height_inner = np.linalg.norm(landmarks[62] - landmarks[66])
        else:
            mouth_height_inner = 0.0
        
        # ========================================
        # 眉と目の距離
        # ========================================
        brow_distance = np.linalg.norm(landmarks[21] - landmarks[22])
        
        # 目の中心間距離（ベクトル演算）
        right_eye_center = np.mean(landmarks[36:42], axis=0)
        left_eye_center = np.mean(landmarks[42:48], axis=0)
        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        
        # ========================================
        # 安全な除算でアスペクト比計算
        # ========================================
        eps = 1e-6
        right_ear = right_eye_height / max(right_eye_width, eps)
        left_ear = left_eye_height / max(left_eye_width, eps)
        mar = mouth_height_outer / max(mouth_width, eps)
        
        return {
            "right_eye_aspect_ratio": float(right_ear),
            "left_eye_aspect_ratio": float(left_ear),
            "right_eye_height": float(right_eye_height),
            "left_eye_height": float(left_eye_height),
            "right_eye_width": float(right_eye_width),
            "left_eye_width": float(left_eye_width),
            "mouth_width": float(mouth_width),
            "mouth_height_outer": float(mouth_height_outer),
            "mouth_height_inner": float(mouth_height_inner),
            "mouth_aspect_ratio": float(mar),
            "brow_distance": float(brow_distance),
            "eye_distance": float(max(eye_distance, eps)),
        }
    
    def compute_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        角度特徴量を計算（ベクトル化版）
        
        Args:
            landmarks: (68, 2)のランドマーク配列
            
        Returns:
            角度特徴量の辞書
        """
        # 眉の角度をベクトル演算で計算
        right_brow_vec = landmarks[17] - landmarks[21]
        left_brow_vec = landmarks[26] - landmarks[22]
        
        # arctan2をベクトル化
        right_brow_angle = np.degrees(np.arctan2(right_brow_vec[1], right_brow_vec[0]))
        left_brow_angle = np.degrees(np.arctan2(left_brow_vec[1], left_brow_vec[0]))
        
        return {
            'right_brow_angle': float(right_brow_angle),
            'left_brow_angle': float(left_brow_angle)
        }
    
    def compute_all(self, landmarks: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        全特徴量を一括計算
        
        Args:
            landmarks: (68, 2)のランドマーク配列
            
        Returns:
            (distances, angles) のタプル
        """
        return self.compute_distances(landmarks), self.compute_angles(landmarks)


class BatchFeatureExtractor(OptimizedFeatureExtractor):
    """バッチ処理対応の特徴量抽出器"""
    
    def compute_distances_batch(self, landmarks_batch: np.ndarray) -> list:
        """
        複数フレームの距離特徴量をバッチで計算
        
        Args:
            landmarks_batch: (N, 68, 2)のランドマーク配列
            
        Returns:
            N個の距離特徴量辞書のリスト
        """
        batch_size = landmarks_batch.shape[0]
        
        # ========================================
        # ベクトル化されたバッチ計算
        # ========================================
        
        # 目の垂直距離（バッチ）
        right_eye_v1 = np.linalg.norm(
            landmarks_batch[:, 37] - landmarks_batch[:, 41], axis=1
        )
        right_eye_v2 = np.linalg.norm(
            landmarks_batch[:, 38] - landmarks_batch[:, 40], axis=1
        )
        right_eye_heights = (right_eye_v1 + right_eye_v2) / 2
        right_eye_widths = np.linalg.norm(
            landmarks_batch[:, 36] - landmarks_batch[:, 39], axis=1
        )
        
        left_eye_v1 = np.linalg.norm(
            landmarks_batch[:, 43] - landmarks_batch[:, 47], axis=1
        )
        left_eye_v2 = np.linalg.norm(
            landmarks_batch[:, 44] - landmarks_batch[:, 46], axis=1
        )
        left_eye_heights = (left_eye_v1 + left_eye_v2) / 2
        left_eye_widths = np.linalg.norm(
            landmarks_batch[:, 42] - landmarks_batch[:, 45], axis=1
        )
        
        # 口の距離（バッチ）
        mouth_widths = np.linalg.norm(
            landmarks_batch[:, 48] - landmarks_batch[:, 54], axis=1
        )
        mouth_heights_outer = np.linalg.norm(
            landmarks_batch[:, 51] - landmarks_batch[:, 57], axis=1
        )
        
        # 内唇
        if landmarks_batch.shape[1] > 66:
            mouth_heights_inner = np.linalg.norm(
                landmarks_batch[:, 62] - landmarks_batch[:, 66], axis=1
            )
        else:
            mouth_heights_inner = np.zeros(batch_size)
        
        # 眉間距離（バッチ）
        brow_distances = np.linalg.norm(
            landmarks_batch[:, 21] - landmarks_batch[:, 22], axis=1
        )
        
        # 目の中心間距離（バッチ）
        right_eye_centers = np.mean(landmarks_batch[:, 36:42], axis=1)
        left_eye_centers = np.mean(landmarks_batch[:, 42:48], axis=1)
        eye_distances = np.linalg.norm(right_eye_centers - left_eye_centers, axis=1)
        
        # アスペクト比（バッチ）
        eps = 1e-6
        right_ears = right_eye_heights / np.maximum(right_eye_widths, eps)
        left_ears = left_eye_heights / np.maximum(left_eye_widths, eps)
        mars = mouth_heights_outer / np.maximum(mouth_widths, eps)
        
        # 結果をリストに変換
        results = []
        for i in range(batch_size):
            results.append({
                "right_eye_aspect_ratio": float(right_ears[i]),
                "left_eye_aspect_ratio": float(left_ears[i]),
                "right_eye_height": float(right_eye_heights[i]),
                "left_eye_height": float(left_eye_heights[i]),
                "right_eye_width": float(right_eye_widths[i]),
                "left_eye_width": float(left_eye_widths[i]),
                "mouth_width": float(mouth_widths[i]),
                "mouth_height_outer": float(mouth_heights_outer[i]),
                "mouth_height_inner": float(mouth_heights_inner[i]),
                "mouth_aspect_ratio": float(mars[i]),
                "brow_distance": float(brow_distances[i]),
                "eye_distance": float(max(eye_distances[i], eps)),
            })
        
        return results
    
    def compute_angles_batch(self, landmarks_batch: np.ndarray) -> list:
        """
        複数フレームの角度特徴量をバッチで計算
        
        Args:
            landmarks_batch: (N, 68, 2)のランドマーク配列
            
        Returns:
            N個の角度特徴量辞書のリスト
        """
        # 眉のベクトル（バッチ）
        right_brow_vecs = landmarks_batch[:, 17] - landmarks_batch[:, 21]
        left_brow_vecs = landmarks_batch[:, 26] - landmarks_batch[:, 22]
        
        # 角度（バッチ）
        right_brow_angles = np.degrees(
            np.arctan2(right_brow_vecs[:, 1], right_brow_vecs[:, 0])
        )
        left_brow_angles = np.degrees(
            np.arctan2(left_brow_vecs[:, 1], left_brow_vecs[:, 0])
        )
        
        return [
            {
                'right_brow_angle': float(right_brow_angles[i]),
                'left_brow_angle': float(left_brow_angles[i])
            }
            for i in range(landmarks_batch.shape[0])
        ]
