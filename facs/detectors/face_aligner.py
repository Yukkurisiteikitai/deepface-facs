"""
顔の傾き（回転・スケール）を正規化するモジュール
"""
import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class FaceAlignment:
    """顔のアライメント情報"""
    rotation_angle: float      # 回転角度（度）
    scale: float               # スケール
    center: Tuple[float, float]  # 顔の中心
    eye_distance: float        # 目の間の距離
    roll: float                # ロール（頭の傾き）
    yaw: float                 # ヨー（左右の向き）- 推定値
    pitch: float               # ピッチ（上下の向き）- 推定値

class FaceAligner:
    """顔の傾きを正規化するクラス"""
    
    def __init__(self):
        pass
    
    def compute_alignment(self, landmarks: np.ndarray) -> FaceAlignment:
        """
        ランドマークから顔のアライメント情報を計算
        
        Args:
            landmarks: 68点ランドマーク
            
        Returns:
            FaceAlignment: アライメント情報
        """
        # 目の中心を計算
        left_eye_center = np.mean(landmarks[42:48], axis=0)
        right_eye_center = np.mean(landmarks[36:42], axis=0)
        
        # 目の間の距離
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
        
        # 顔の中心（両目の中点）
        face_center = (left_eye_center + right_eye_center) / 2
        
        # ロール角（頭の傾き）を計算
        # 両目を結ぶ線の角度
        dy = left_eye_center[1] - right_eye_center[1]
        dx = left_eye_center[0] - right_eye_center[0]
        roll_angle = np.degrees(np.arctan2(dy, dx))
        
        # ヨー角（左右の向き）を推定
        # 鼻の位置と顔の中心のずれから推定
        nose_tip = landmarks[30]
        nose_to_center_x = nose_tip[0] - face_center[0]
        yaw_estimate = np.clip(nose_to_center_x / (eye_distance * 0.3) * 30, -45, 45)
        
        # ピッチ角（上下の向き）を推定
        # 鼻と口の位置関係から推定
        mouth_center = (landmarks[48] + landmarks[54]) / 2
        nose_to_mouth = mouth_center[1] - nose_tip[1]
        expected_distance = eye_distance * 0.6
        pitch_estimate = np.clip((nose_to_mouth - expected_distance) / expected_distance * 30, -30, 30)
        
        return FaceAlignment(
            rotation_angle=roll_angle,
            scale=eye_distance,
            center=(face_center[0], face_center[1]),
            eye_distance=eye_distance,
            roll=roll_angle,
            yaw=yaw_estimate,
            pitch=pitch_estimate
        )
    
    def normalize_landmarks(self, landmarks: np.ndarray, 
                           alignment: Optional[FaceAlignment] = None) -> np.ndarray:
        """
        ランドマークを正規化（回転・スケール・位置を補正）
        
        Args:
            landmarks: 元のランドマーク
            alignment: アライメント情報（Noneの場合は自動計算）
            
        Returns:
            正規化されたランドマーク
        """
        if alignment is None:
            alignment = self.compute_alignment(landmarks)
        
        # 1. 中心を原点に移動
        centered = landmarks - np.array(alignment.center)
        
        # 2. 回転を補正（ロール角を0に）
        angle_rad = -np.radians(alignment.roll)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        rotated = centered @ rotation_matrix.T
        
        # 3. スケールを正規化（目の間の距離を1.0に）
        if alignment.eye_distance > 0:
            normalized = rotated / alignment.eye_distance
        else:
            normalized = rotated
        
        return normalized
    
    def denormalize_landmarks(self, normalized_landmarks: np.ndarray,
                              alignment: FaceAlignment) -> np.ndarray:
        """
        正規化されたランドマークを元の座標系に戻す
        """
        # 逆順に処理
        # 1. スケールを戻す
        scaled = normalized_landmarks * alignment.eye_distance
        
        # 2. 回転を戻す
        angle_rad = np.radians(alignment.roll)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        rotated = scaled @ rotation_matrix.T
        
        # 3. 中心を戻す
        denormalized = rotated + np.array(alignment.center)
        
        return denormalized
    
    def align_image(self, image: np.ndarray, landmarks: np.ndarray,
                    output_size: Tuple[int, int] = (256, 256),
                    eye_position: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
        """
        画像とランドマークを正面向きに整列
        
        Args:
            image: 入力画像
            landmarks: 68点ランドマーク
            output_size: 出力画像サイズ
            eye_position: 目の縦位置（0-1、上からの割合）
            
        Returns:
            (整列された画像, 整列されたランドマーク)
        """
        alignment = self.compute_alignment(landmarks)
        
        # 目の位置を基準にアフィン変換を計算
        left_eye = np.mean(landmarks[42:48], axis=0)
        right_eye = np.mean(landmarks[36:42], axis=0)
        
        # 出力画像での目の位置
        desired_eye_distance = output_size[0] * 0.4
        desired_left_eye = np.array([
            output_size[0] * 0.5 + desired_eye_distance * 0.5,
            output_size[1] * eye_position
        ])
        desired_right_eye = np.array([
            output_size[0] * 0.5 - desired_eye_distance * 0.5,
            output_size[1] * eye_position
        ])
        
        # アフィン変換行列を計算
        src_pts = np.array([right_eye, left_eye, landmarks[30]], dtype=np.float32)  # 右目、左目、鼻
        
        # 鼻の位置も計算
        nose_offset = landmarks[30] - (left_eye + right_eye) / 2
        nose_offset_normalized = nose_offset / alignment.eye_distance * desired_eye_distance
        desired_nose = (desired_left_eye + desired_right_eye) / 2 + nose_offset_normalized
        
        dst_pts = np.array([desired_right_eye, desired_left_eye, desired_nose], dtype=np.float32)
        
        # アフィン変換
        M = cv2.getAffineTransform(src_pts, dst_pts)
        
        aligned_image = cv2.warpAffine(image, M, output_size,
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(128, 128, 128))
        
        # ランドマークも変換
        ones = np.ones((landmarks.shape[0], 1))
        landmarks_h = np.hstack([landmarks, ones])
        aligned_landmarks = landmarks_h @ M.T
        
        return aligned_image, aligned_landmarks
    
    def compute_rotation_invariant_features(self, landmarks: np.ndarray) -> dict:
        """
        回転に不変な特徴量を計算
        
        顔が傾いていても一貫した値を返す特徴量
        """
        alignment = self.compute_alignment(landmarks)
        normalized = self.normalize_landmarks(landmarks, alignment)
        
        # 正規化されたランドマークから特徴量を計算
        features = {}
        
        # 目のアスペクト比（回転補正済み）
        right_eye = normalized[36:42]
        left_eye = normalized[42:48]
        
        right_eye_height = (np.linalg.norm(right_eye[1] - right_eye[5]) +
                          np.linalg.norm(right_eye[2] - right_eye[4])) / 2
        right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3])
        
        left_eye_height = (np.linalg.norm(left_eye[1] - left_eye[5]) +
                         np.linalg.norm(left_eye[2] - left_eye[4])) / 2
        left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3])
        
        features['right_eye_aspect_ratio'] = right_eye_height / max(right_eye_width, 1e-6)
        features['left_eye_aspect_ratio'] = left_eye_height / max(left_eye_width, 1e-6)
        
        # 口のアスペクト比
        outer_mouth = normalized[48:60]
        inner_mouth = normalized[60:68]
        
        mouth_width = np.linalg.norm(outer_mouth[0] - outer_mouth[6])
        mouth_height_outer = np.linalg.norm(outer_mouth[3] - outer_mouth[9])
        mouth_height_inner = np.linalg.norm(inner_mouth[2] - inner_mouth[6])
        
        features['mouth_width'] = mouth_width
        features['mouth_height_outer'] = mouth_height_outer
        features['mouth_height_inner'] = mouth_height_inner
        features['mouth_aspect_ratio'] = mouth_height_outer / max(mouth_width, 1e-6)
        
        # 眉の位置（目からの相対距離）
        right_eyebrow = normalized[17:22]
        left_eyebrow = normalized[22:27]
        
        right_eye_center = np.mean(right_eye, axis=0)
        left_eye_center = np.mean(left_eye, axis=0)
        
        features['right_brow_height'] = np.mean(right_eyebrow[:, 1]) - right_eye_center[1]
        features['left_brow_height'] = np.mean(left_eyebrow[:, 1]) - left_eye_center[1]
        
        # 眉間の距離（正規化済み）
        features['brow_distance'] = np.linalg.norm(right_eyebrow[4] - left_eyebrow[0])
        
        # 目の距離（正規化後は常に約1.0）
        features['eye_distance'] = np.linalg.norm(right_eye_center - left_eye_center)
        
        # アライメント情報も含める
        features['roll'] = alignment.roll
        features['yaw'] = alignment.yaw
        features['pitch'] = alignment.pitch
        
        return features


class FeatureExtractorWithAlignment:
    """
    顔の傾きを考慮した特徴量抽出器
    """
    
    def __init__(self):
        self._aligner = FaceAligner()
    
    def compute_distances(self, landmarks: np.ndarray) -> dict:
        """
        回転に不変な距離特徴を計算
        """
        return self._aligner.compute_rotation_invariant_features(landmarks)
    
    def compute_angles(self, landmarks: np.ndarray) -> dict:
        """
        回転補正後の角度特徴を計算
        """
        alignment = self._aligner.compute_alignment(landmarks)
        normalized = self._aligner.normalize_landmarks(landmarks, alignment)
        
        # 正規化後のランドマークで角度を計算
        right_eyebrow = normalized[17:22]
        left_eyebrow = normalized[22:27]
        outer_mouth = normalized[48:60]
        
        # 眉の角度（正規化後）
        right_brow_angle = np.degrees(np.arctan2(
            right_eyebrow[0][1] - right_eyebrow[4][1],
            right_eyebrow[0][0] - right_eyebrow[4][0]
        ))
        left_brow_angle = np.degrees(np.arctan2(
            left_eyebrow[4][1] - left_eyebrow[0][1],
            left_eyebrow[4][0] - left_eyebrow[0][0]
        ))
        
        # 口角の角度
        mouth_center = (outer_mouth[3] + outer_mouth[9]) / 2
        right_mouth_angle = np.degrees(np.arctan2(
            outer_mouth[0][1] - mouth_center[1],
            outer_mouth[0][0] - mouth_center[0]
        ))
        left_mouth_angle = np.degrees(np.arctan2(
            outer_mouth[6][1] - mouth_center[1],
            outer_mouth[6][0] - mouth_center[0]
        ))
        
        return {
            'right_brow_angle': right_brow_angle,
            'left_brow_angle': left_brow_angle,
            'right_mouth_angle': right_mouth_angle,
            'left_mouth_angle': left_mouth_angle,
            'face_roll': alignment.roll,
            'face_yaw': alignment.yaw,
            'face_pitch': alignment.pitch,
        }
    
    def get_alignment(self, landmarks: np.ndarray) -> FaceAlignment:
        """アライメント情報を取得"""
        return self._aligner.compute_alignment(landmarks)
