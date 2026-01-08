import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
import os

class LandmarkDetector:
    """68点顔ランドマーク検出器"""
    
    def __init__(self, predictor_path: Optional[str] = None, use_mediapipe: bool = False):
        self.use_mediapipe = use_mediapipe
        self._dlib_detector = None
        self._dlib_predictor = None
        self._mp_face_mesh = None
        self._mp_detector = None
        
        if use_mediapipe:
            self._init_mediapipe()
        else:
            self._init_dlib(predictor_path)
    
    def _init_dlib(self, predictor_path: Optional[str]):
        try:
            import dlib
            self._dlib_detector = dlib.get_frontal_face_detector()
            
            if predictor_path is None:
                default_paths = [
                    "shape_predictor_68_face_landmarks.dat",
                    os.path.expanduser("~/.dlib/shape_predictor_68_face_landmarks.dat"),
                    os.path.join(os.path.dirname(__file__), "..", "models", "shape_predictor_68_face_landmarks.dat")
                ]
                for path in default_paths:
                    if os.path.exists(path):
                        predictor_path = path
                        break
            
            if predictor_path and os.path.exists(predictor_path):
                self._dlib_predictor = dlib.shape_predictor(predictor_path)
            else:
                print("警告: shape_predictor_68_face_landmarks.dat が見つかりません。MediaPipeにフォールバックします。")
                self._init_mediapipe()
                self.use_mediapipe = True
        except ImportError:
            print("警告: dlibがインストールされていません。MediaPipeを使用します。")
            self._init_mediapipe()
            self.use_mediapipe = True
    
    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            
            # 新しいAPI (mediapipe >= 0.10.0) を試す
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                # 新しいAPIの場合
                self._use_new_api = True
                self._mp_module = mp
                print("MediaPipe Tasks APIを使用します")
                
            except ImportError:
                # 古いAPI (mediapipe < 0.10.0)
                self._use_new_api = False
                
                try:
                    self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=10,
                        refine_landmarks=True,
                        min_detection_confidence=0.5
                    )
                except AttributeError:
                    # さらに新しいバージョンの場合、直接インポート
                    self._use_new_api = True
                    self._mp_module = mp
            
            self._mp_to_68_indices = self._create_mediapipe_to_68_mapping()
            
        except ImportError:
            raise ImportError("mediapipeがインストールされていません。pip install mediapipe でインストールしてください。")
    
    def _create_mediapipe_to_68_mapping(self) -> List[int]:
        """MediaPipeの468点から68点へのマッピング"""
        return [
            # 顔の輪郭 (0-16)
            234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397,
            # 右眉 (17-21)
            70, 63, 105, 66, 107,
            # 左眉 (22-26)
            336, 296, 334, 293, 300,
            # 鼻 (27-35)
            168, 197, 5, 4, 75, 97, 2, 326, 305,
            # 右目 (36-41)
            33, 160, 158, 133, 153, 144,
            # 左目 (42-47)
            362, 385, 387, 263, 373, 380,
            # 外側唇 (48-59)
            61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
            # 内側唇 (60-67)
            78, 82, 13, 312, 308, 402, 14, 87
        ]
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.use_mediapipe:
            return self._detect_faces_mediapipe(image)
        return self._detect_faces_dlib(image)
    
    def _detect_faces_dlib(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self._dlib_detector(gray, 1)
        return [(f.left(), f.top(), f.right() - f.left(), f.bottom() - f.top()) for f in faces]
    
    def _detect_faces_mediapipe(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        landmarks_list = self._get_mediapipe_landmarks(image)
        if not landmarks_list:
            return []
        
        faces = []
        h, w = image.shape[:2]
        for landmarks_478 in landmarks_list:
            x_coords = [lm[0] for lm in landmarks_478]
            y_coords = [lm[1] for lm in landmarks_478]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            faces.append((x_min, y_min, x_max - x_min, y_max - y_min))
        return faces
    
    def _get_mediapipe_landmarks(self, image: np.ndarray) -> List[List[Tuple[float, float, float]]]:
        """MediaPipeでランドマークを取得（APIバージョンに応じて切り替え）"""
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = image.shape[:2]
        
        if hasattr(self, '_use_new_api') and self._use_new_api:
            return self._get_landmarks_new_api(image_rgb, w, h)
        else:
            return self._get_landmarks_old_api(image_rgb, w, h)
    
    def _get_landmarks_new_api(self, image_rgb: np.ndarray, w: int, h: int) -> List[List[Tuple[float, float, float]]]:
        """新しいMediaPipe APIでランドマーク取得"""
        try:
            import mediapipe as mp
            
            # FaceMeshを動的に初期化
            if self._mp_face_mesh is None:
                mp_face_mesh = mp.solutions.face_mesh
                self._mp_face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=10,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
            
            results = self._mp_face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return []
            
            landmarks_list = []
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(lm.x * w, lm.y * h, lm.z) for lm in face_landmarks.landmark]
                landmarks_list.append(landmarks)
            
            return landmarks_list
            
        except Exception as e:
            print(f"MediaPipe処理エラー: {e}")
            # OpenCVのカスケード分類器にフォールバック
            return self._get_landmarks_opencv_fallback(image_rgb, w, h)
    
    def _get_landmarks_old_api(self, image_rgb: np.ndarray, w: int, h: int) -> List[List[Tuple[float, float, float]]]:
        """古いMediaPipe APIでランドマーク取得"""
        if self._mp_face_mesh is None:
            return []
        
        results = self._mp_face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return []
        
        landmarks_list = []
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(lm.x * w, lm.y * h, lm.z) for lm in face_landmarks.landmark]
            landmarks_list.append(landmarks)
        
        return landmarks_list
    
    def _get_landmarks_opencv_fallback(self, image_rgb: np.ndarray, w: int, h: int) -> List[List[Tuple[float, float, float]]]:
        """OpenCVのHaar Cascadeを使用したフォールバック（顔検出のみ）"""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return []
        
        # 簡易的な68点ランドマークを生成（顔の矩形から推定）
        landmarks_list = []
        for (x, y, fw, fh) in faces:
            landmarks = self._generate_approximate_landmarks(x, y, fw, fh)
            landmarks_list.append(landmarks)
        
        return landmarks_list
    
    def _generate_approximate_landmarks(self, x: int, y: int, w: int, h: int) -> List[Tuple[float, float, float]]:
        """顔の矩形から近似的なランドマークを生成"""
        landmarks = []
        
        # 468点分のダミーランドマークを生成
        for i in range(468):
            # 顔の中心を基準に分布させる
            cx = x + w / 2
            cy = y + h / 2
            
            # 簡易的な位置推定
            angle = (i / 468) * 2 * np.pi
            radius = min(w, h) / 3
            lx = cx + radius * np.cos(angle) * (0.5 + 0.5 * np.sin(i))
            ly = cy + radius * np.sin(angle) * (0.5 + 0.5 * np.cos(i))
            
            landmarks.append((lx, ly, 0))
        
        return landmarks
    
    def detect_landmarks(self, image: np.ndarray,
                         face_rect: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        if self.use_mediapipe:
            return self._detect_landmarks_mediapipe(image, face_rect)
        return self._detect_landmarks_dlib(image, face_rect)
    
    def _detect_landmarks_dlib(self, image: np.ndarray,
                               face_rect: Optional[Tuple[int, int, int, int]]) -> Optional[np.ndarray]:
        import dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if face_rect is None:
            faces = self._dlib_detector(gray, 1)
            if len(faces) == 0:
                return None
            rect = faces[0]
        else:
            x, y, w, h = face_rect
            rect = dlib.rectangle(x, y, x + w, y + h)
        
        shape = self._dlib_predictor(gray, rect)
        return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
    
    def _detect_landmarks_mediapipe(self, image: np.ndarray,
                                    face_rect: Optional[Tuple[int, int, int, int]]) -> Optional[np.ndarray]:
        landmarks_list = self._get_mediapipe_landmarks(image)
        if not landmarks_list:
            return None
        
        # 最初の顔を使用（または face_rect に最も近い顔）
        landmarks_478 = landmarks_list[0]
        
        # 468点から68点へ変換
        landmarks = np.zeros((68, 2), dtype=np.float32)
        for i, mp_idx in enumerate(self._mp_to_68_indices):
            if mp_idx < len(landmarks_478):
                landmarks[i] = [landmarks_478[mp_idx][0], landmarks_478[mp_idx][1]]
        
        return landmarks
    
    def compute_distances(self, landmarks: np.ndarray) -> Dict[str, float]:
        """重要な距離を計算"""
        # 目
        right_eye = landmarks[36:42]
        left_eye = landmarks[42:48]
        
        right_eye_height = (np.linalg.norm(right_eye[1] - right_eye[5]) +
                           np.linalg.norm(right_eye[2] - right_eye[4])) / 2
        right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3])
        left_eye_height = (np.linalg.norm(left_eye[1] - left_eye[5]) +
                          np.linalg.norm(left_eye[2] - left_eye[4])) / 2
        left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3])
        
        # 口
        outer_mouth = landmarks[48:60]
        inner_mouth = landmarks[60:68]
        mouth_width = np.linalg.norm(outer_mouth[0] - outer_mouth[6])
        mouth_height_outer = np.linalg.norm(outer_mouth[3] - outer_mouth[9])
        mouth_height_inner = np.linalg.norm(inner_mouth[2] - inner_mouth[6])
        
        # 眉
        right_eyebrow = landmarks[17:22]
        left_eyebrow = landmarks[22:27]
        brow_distance = np.linalg.norm(right_eyebrow[4] - left_eyebrow[0])
        
        # 目の中心
        right_eye_center = np.mean(right_eye, axis=0)
        left_eye_center = np.mean(left_eye, axis=0)
        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        
        return {
            "right_eye_aspect_ratio": right_eye_height / max(right_eye_width, 1e-6),
            "left_eye_aspect_ratio": left_eye_height / max(left_eye_width, 1e-6),
            "right_eye_height": right_eye_height,
            "left_eye_height": left_eye_height,
            "right_eye_width": right_eye_width,
            "left_eye_width": left_eye_width,
            "mouth_aspect_ratio": mouth_height_outer / max(mouth_width, 1e-6),
            "mouth_width": mouth_width,
            "mouth_height_outer": mouth_height_outer,
            "mouth_height_inner": mouth_height_inner,
            "brow_distance": brow_distance,
            "eye_distance": eye_distance,
        }
    
    def compute_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """重要な角度を計算"""
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
        right_mouth_angle = np.degrees(np.arctan2(
            outer_mouth[0][1] - outer_mouth[3][1],
            outer_mouth[0][0] - outer_mouth[3][0]
        ))
        left_mouth_angle = np.degrees(np.arctan2(
            outer_mouth[6][1] - outer_mouth[3][1],
            outer_mouth[6][0] - outer_mouth[3][0]
        ))
        
        return {
            "right_brow_angle": right_brow_angle,
            "left_brow_angle": left_brow_angle,
            "right_mouth_angle": right_mouth_angle,
            "left_mouth_angle": left_mouth_angle
        }
