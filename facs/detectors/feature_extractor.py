import numpy as np
from typing import Dict

from ..core.interfaces import IFeatureExtractor
from .face_aligner import FaceAligner, FaceAlignment

class FeatureExtractor(IFeatureExtractor):
    """顔特徴量抽出器（回転補正対応）"""
    
    def __init__(self, use_alignment: bool = True):
        """
        Args:
            use_alignment: 顔の傾きを補正するかどうか
        """
        self._use_alignment = use_alignment
        self._aligner = FaceAligner() if use_alignment else None
        self._last_alignment: FaceAlignment = None
    
    def compute_distances(self, landmarks: np.ndarray) -> Dict[str, float]:
        """距離特徴を計算（回転補正済み）"""
        if self._use_alignment:
            self._last_alignment = self._aligner.compute_alignment(landmarks)
            normalized = self._aligner.normalize_landmarks(landmarks, self._last_alignment)
            # 正規化されたランドマークで計算
            return self._compute_distances_internal(normalized, self._last_alignment.eye_distance)
        else:
            # 補正なしで計算
            eye_dist = self._compute_eye_distance(landmarks)
            return self._compute_distances_internal(landmarks / eye_dist, eye_dist)
    
    def _compute_eye_distance(self, landmarks: np.ndarray) -> float:
        """目の間の距離を計算"""
        right_eye_center = np.mean(landmarks[36:42], axis=0)
        left_eye_center = np.mean(landmarks[42:48], axis=0)
        return max(np.linalg.norm(left_eye_center - right_eye_center), 1e-6)
    
    def _compute_distances_internal(self, landmarks: np.ndarray, scale: float) -> Dict[str, float]:
        """正規化されたランドマークから距離を計算"""
        right_eye = landmarks[36:42]
        left_eye = landmarks[42:48]
        outer_mouth = landmarks[48:60]
        inner_mouth = landmarks[60:68]
        right_eyebrow = landmarks[17:22]
        left_eyebrow = landmarks[22:27]
        
        # 目のアスペクト比
        right_eye_height = (np.linalg.norm(right_eye[1] - right_eye[5]) +
                           np.linalg.norm(right_eye[2] - right_eye[4])) / 2
        right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3])
        
        left_eye_height = (np.linalg.norm(left_eye[1] - left_eye[5]) +
                          np.linalg.norm(left_eye[2] - left_eye[4])) / 2
        left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3])
        
        # 口
        mouth_width = np.linalg.norm(outer_mouth[0] - outer_mouth[6])
        mouth_height_outer = np.linalg.norm(outer_mouth[3] - outer_mouth[9])
        mouth_height_inner = np.linalg.norm(inner_mouth[2] - inner_mouth[6])
        
        # 眉間の距離
        brow_distance = np.linalg.norm(right_eyebrow[4] - left_eyebrow[0])
        
        # 目の中心
        right_eye_center = np.mean(right_eye, axis=0)
        left_eye_center = np.mean(left_eye, axis=0)
        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        
        return {
            "right_eye_aspect_ratio": right_eye_height / max(right_eye_width, 1e-6),
            "left_eye_aspect_ratio": left_eye_height / max(left_eye_width, 1e-6),
            "right_eye_height": right_eye_height * scale,
            "left_eye_height": left_eye_height * scale,
            "right_eye_width": right_eye_width * scale,
            "left_eye_width": left_eye_width * scale,
            "mouth_width": mouth_width * scale,
            "mouth_height_outer": mouth_height_outer * scale,
            "mouth_height_inner": mouth_height_inner * scale,
            "mouth_aspect_ratio": mouth_height_outer / max(mouth_width, 1e-6),
            "brow_distance": brow_distance * scale,
            "eye_distance": scale,  # 元のスケールを保持
        }
    
    def compute_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """角度特徴を計算（回転補正済み）"""
        if self._use_alignment:
            if self._last_alignment is None:
                self._last_alignment = self._aligner.compute_alignment(landmarks)
            normalized = self._aligner.normalize_landmarks(landmarks, self._last_alignment)
            angles = self._compute_angles_internal(normalized)
            # 顔の向き情報を追加
            angles['face_roll'] = self._last_alignment.roll
            angles['face_yaw'] = self._last_alignment.yaw
            angles['face_pitch'] = self._last_alignment.pitch
            return angles
        else:
            return self._compute_angles_internal(landmarks)
    
    def _compute_angles_internal(self, landmarks: np.ndarray) -> Dict[str, float]:
        """正規化されたランドマークから角度を計算"""
        right_eyebrow = landmarks[17:22]
        left_eyebrow = landmarks[22:27]
        outer_mouth = landmarks[48:60]
        
        right_brow_angle = np.degrees(np.arctan2(
            right_eyebrow[0][1] - right_eyebrow[4][1],
            right_eyebrow[0][0] - right_eyebrow[4][0]
        ))
        left_brow_angle = np.degrees(np.arctan2(
            left_eyebrow[4][1] - left_eyebrow[0][1],
            left_eyebrow[4][0] - left_eyebrow[0][0]
        ))
        
        mouth_center_y = (outer_mouth[3][1] + outer_mouth[9][1]) / 2
        right_mouth_angle = np.degrees(np.arctan2(
            outer_mouth[0][1] - mouth_center_y,
            outer_mouth[0][0] - outer_mouth[3][0]
        ))
        left_mouth_angle = np.degrees(np.arctan2(
            outer_mouth[6][1] - mouth_center_y,
            outer_mouth[6][0] - outer_mouth[3][0]
        ))
        
        return {
            'right_brow_angle': right_brow_angle,
            'left_brow_angle': left_brow_angle,
            'right_mouth_angle': right_mouth_angle,
            'left_mouth_angle': left_mouth_angle,
        }
    
    def get_last_alignment(self) -> FaceAlignment:
        """最後に計算したアライメント情報を取得"""
        return self._last_alignment
