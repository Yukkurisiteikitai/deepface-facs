import numpy as np
from typing import Dict
from ..core.interfaces import IFeatureExtractor

class FeatureExtractor(IFeatureExtractor):
    """顔特徴量抽出器"""
    
    def compute_distances(self, landmarks: np.ndarray) -> Dict[str, float]:
        right_eye = landmarks[36:42]
        left_eye = landmarks[42:48]
        outer_mouth = landmarks[48:60]
        inner_mouth = landmarks[60:68]
        right_eyebrow = landmarks[17:22]
        left_eyebrow = landmarks[22:27]
        
        right_eye_height = (np.linalg.norm(right_eye[1] - right_eye[5]) + np.linalg.norm(right_eye[2] - right_eye[4])) / 2
        right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3])
        left_eye_height = (np.linalg.norm(left_eye[1] - left_eye[5]) + np.linalg.norm(left_eye[2] - left_eye[4])) / 2
        left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3])
        
        mouth_width = np.linalg.norm(outer_mouth[0] - outer_mouth[6])
        mouth_height_outer = np.linalg.norm(outer_mouth[3] - outer_mouth[9])
        mouth_height_inner = np.linalg.norm(inner_mouth[2] - inner_mouth[6]) if len(inner_mouth) > 6 else 0
        
        brow_distance = np.linalg.norm(right_eyebrow[4] - left_eyebrow[0])
        right_eye_center = np.mean(right_eye, axis=0)
        left_eye_center = np.mean(left_eye, axis=0)
        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        
        return {
            "right_eye_aspect_ratio": right_eye_height / max(right_eye_width, 1e-6),
            "left_eye_aspect_ratio": left_eye_height / max(left_eye_width, 1e-6),
            "right_eye_height": right_eye_height, "left_eye_height": left_eye_height,
            "right_eye_width": right_eye_width, "left_eye_width": left_eye_width,
            "mouth_width": mouth_width, "mouth_height_outer": mouth_height_outer,
            "mouth_height_inner": mouth_height_inner,
            "mouth_aspect_ratio": mouth_height_outer / max(mouth_width, 1e-6),
            "brow_distance": brow_distance, "eye_distance": max(eye_distance, 1e-6),
        }
    
    def compute_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        right_eyebrow = landmarks[17:22]
        left_eyebrow = landmarks[22:27]
        outer_mouth = landmarks[48:60]
        
        right_brow_angle = np.degrees(np.arctan2(right_eyebrow[0][1] - right_eyebrow[4][1], right_eyebrow[0][0] - right_eyebrow[4][0]))
        left_brow_angle = np.degrees(np.arctan2(left_eyebrow[4][1] - left_eyebrow[0][1], left_eyebrow[4][0] - left_eyebrow[0][0]))
        
        return {'right_brow_angle': right_brow_angle, 'left_brow_angle': left_brow_angle}
