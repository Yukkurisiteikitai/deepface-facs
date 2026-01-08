"""
DeepFaceライブラリを使用した顔分析
DeepFace v0.0.89+ 対応
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DeepFaceResult:
    """DeepFace分析結果"""
    face_rect: Tuple[int, int, int, int]
    emotion: Dict[str, float]
    dominant_emotion: str
    age: float
    gender: str
    dominant_race: str
    landmarks: Optional[Dict[str, Tuple[float, float]]] = None
    roll: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0

class DeepFaceAnalyzer:
    """DeepFaceを使用した顔分析（v0.0.89+対応）"""
    
    # 利用可能なバックエンド
    BACKENDS = ['retinaface', 'mtcnn', 'opencv', 'ssd', 'mediapipe', 'yolov8', 'yunet', 'fastmtcnn']
    
    def __init__(self, 
                 detector_backend: str = 'retinaface',
                 enforce_detection: bool = False):
        self._detector_backend = detector_backend
        self._enforce_detection = enforce_detection
        self._deepface = None
        self._version = None
        self._init_deepface()
    
    def _init_deepface(self):
        """DeepFaceを初期化"""
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            
            # バージョン確認
            try:
                import deepface
                self._version = getattr(deepface, '__version__', 'unknown')
                print(f"DeepFace v{self._version} (backend: {self._detector_backend})")
            except:
                self._version = 'unknown'
                
        except ImportError:
            print("警告: DeepFaceがインストールされていません")
            print("pip install deepface でインストールしてください")
            self._deepface = None
    
    @property
    def is_available(self) -> bool:
        return self._deepface is not None
    
    def analyze(self, image: np.ndarray) -> List[DeepFaceResult]:
        """画像を分析"""
        if not self.is_available:
            return []
        
        try:
            # DeepFace v0.0.89+ API
            results = self._deepface.analyze(
                img_path=image,
                actions=['emotion', 'age', 'gender', 'race'],
                detector_backend=self._detector_backend,
                enforce_detection=self._enforce_detection,
                silent=True,
                align=True  # 顔のアライメントを有効化
            )
            
            if isinstance(results, dict):
                results = [results]
            
            parsed_results = []
            for result in results:
                parsed = self._parse_result(result, image)
                if parsed:
                    parsed_results.append(parsed)
            
            return parsed_results
            
        except Exception as e:
            if "Face could not be detected" not in str(e):
                print(f"DeepFace分析エラー: {e}")
            return []
    
    def _parse_result(self, result: Dict, image: np.ndarray) -> Optional[DeepFaceResult]:
        """結果をパース（v0.0.89+ API対応）"""
        try:
            # 顔の矩形 - 新APIでは 'region' キー
            region = result.get('region', {})
            face_rect = (
                region.get('x', 0),
                region.get('y', 0),
                region.get('w', 0),
                region.get('h', 0)
            )
            
            # 感情
            emotion = result.get('emotion', {})
            dominant_emotion = result.get('dominant_emotion', 'neutral')
            
            # 年齢
            age = result.get('age', 0)
            
            # 性別 - 新APIでは辞書形式
            gender_data = result.get('gender', {})
            if isinstance(gender_data, dict):
                gender = result.get('dominant_gender', 'Unknown')
            else:
                gender = str(gender_data)
            
            # 人種
            race_data = result.get('race', {})
            dominant_race = result.get('dominant_race', 'Unknown')
            
            # ランドマーク（RetinaFace/MTCNN使用時）
            landmarks = None
            roll, yaw, pitch = 0.0, 0.0, 0.0
            
            # facial_area にランドマークが含まれる場合
            facial_area = result.get('facial_area', region)
            if isinstance(facial_area, dict):
                # RetinaFaceの場合、ランドマークは別途取得が必要
                pass
            
            # 顔の向きを推定（顔の矩形から）
            if face_rect[2] > 0 and face_rect[3] > 0:
                # アスペクト比から簡易的にヨーを推定
                aspect = face_rect[2] / face_rect[3]
                if aspect < 0.8:
                    yaw = (0.8 - aspect) * 50  # 顔が細い = 横向き
            
            return DeepFaceResult(
                face_rect=face_rect,
                emotion=emotion,
                dominant_emotion=dominant_emotion,
                age=age,
                gender=gender,
                dominant_race=dominant_race,
                landmarks=landmarks,
                roll=roll,
                yaw=yaw,
                pitch=pitch
            )
            
        except Exception as e:
            print(f"結果パースエラー: {e}")
            return None
    
    def extract_faces(self, image: np.ndarray) -> List[Dict]:
        """顔を抽出（ランドマーク付き）"""
        if not self.is_available:
            return []
        
        try:
            # DeepFace v0.0.89+ の extract_faces API
            faces = self._deepface.extract_faces(
                img_path=image,
                detector_backend=self._detector_backend,
                enforce_detection=self._enforce_detection,
                align=False  # アライメントなしで生の位置を取得
            )
            return faces
        except Exception as e:
            if "Face could not be detected" not in str(e):
                print(f"顔抽出エラー: {e}")
            return []
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """顔を検出"""
        faces = self.extract_faces(image)
        results = []
        for face in faces:
            area = face.get('facial_area', {})
            if isinstance(area, dict):
                rect = (area.get('x', 0), area.get('y', 0),
                       area.get('w', 0), area.get('h', 0))
                results.append(rect)
        return results
    
    def represent(self, image: np.ndarray, model_name: str = 'Facenet512') -> List[np.ndarray]:
        """顔の埋め込みベクトルを取得"""
        if not self.is_available:
            return []
        
        try:
            embeddings = self._deepface.represent(
                img_path=image,
                model_name=model_name,
                detector_backend=self._detector_backend,
                enforce_detection=self._enforce_detection
            )
            return [np.array(e['embedding']) for e in embeddings]
        except Exception:
            return []


class DeepFaceLandmarkConverter:
    """DeepFaceのランドマークを68点形式に変換"""
    
    @staticmethod
    def convert_5_to_68(landmarks_5: Dict[str, Tuple[float, float]], 
                        face_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """
        5点ランドマーク（RetinaFace）を68点に拡張
        
        DeepFace/RetinaFaceの5点:
        - left_eye
        - right_eye
        - nose
        - mouth_left
        - mouth_right
        """
        x, y, w, h = face_rect
        
        left_eye = np.array(landmarks_5.get('left_eye', (x + w * 0.65, y + h * 0.35)))
        right_eye = np.array(landmarks_5.get('right_eye', (x + w * 0.35, y + h * 0.35)))
        nose = np.array(landmarks_5.get('nose', (x + w * 0.5, y + h * 0.55)))
        mouth_left = np.array(landmarks_5.get('mouth_left', (x + w * 0.65, y + h * 0.75)))
        mouth_right = np.array(landmarks_5.get('mouth_right', (x + w * 0.35, y + h * 0.75)))
        
        # 基準値を計算
        eye_center = (left_eye + right_eye) / 2
        eye_distance = np.linalg.norm(left_eye - right_eye)
        mouth_center = (mouth_left + mouth_right) / 2
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        
        # 68点を生成
        landmarks_68 = np.zeros((68, 2), dtype=np.float32)
        
        # 顔の輪郭 (0-16)
        for i in range(17):
            t = i / 16.0
            angle = np.pi * (0.1 + t * 0.8)
            r = h * 0.48
            cx, cy = x + w * 0.5, y + h * 0.45
            landmarks_68[i] = [cx - r * np.cos(angle), cy + r * np.sin(angle) * 0.9]
        
        # 右眉 (17-21)
        for i in range(5):
            t = i / 4.0
            landmarks_68[17 + i] = [
                right_eye[0] - eye_distance * 0.3 + t * eye_distance * 0.4,
                right_eye[1] - eye_distance * 0.25
            ]
        
        # 左眉 (22-26)
        for i in range(5):
            t = i / 4.0
            landmarks_68[22 + i] = [
                left_eye[0] - eye_distance * 0.1 + t * eye_distance * 0.4,
                left_eye[1] - eye_distance * 0.25
            ]
        
        # 鼻筋 (27-30)
        for i in range(4):
            t = i / 3.0
            landmarks_68[27 + i] = [nose[0], eye_center[1] + t * (nose[1] - eye_center[1])]
        
        # 鼻先 (31-35)
        for i in range(5):
            t = (i - 2) / 2.0
            landmarks_68[31 + i] = [nose[0] + t * eye_distance * 0.2, nose[1] + eye_distance * 0.1]
        
        # 右目 (36-41)
        eye_w, eye_h = eye_distance * 0.25, eye_distance * 0.1
        for i, angle in enumerate([0, 0.5, 0.8, 1.0, 1.5, 1.8]):
            landmarks_68[36 + i] = [
                right_eye[0] + eye_w * np.cos(np.pi * angle),
                right_eye[1] + eye_h * np.sin(np.pi * angle)
            ]
        
        # 左目 (42-47)
        for i, angle in enumerate([0, 0.5, 0.8, 1.0, 1.5, 1.8]):
            landmarks_68[42 + i] = [
                left_eye[0] + eye_w * np.cos(np.pi * angle),
                left_eye[1] + eye_h * np.sin(np.pi * angle)
            ]
        
        # 外側の唇 (48-59)
        for i in range(12):
            angle = 2 * np.pi * i / 12
            landmarks_68[48 + i] = [
                mouth_center[0] + mouth_width * 0.5 * np.cos(angle),
                mouth_center[1] + mouth_width * 0.25 * np.sin(angle)
            ]
        
        # 内側の唇 (60-67)
        for i in range(8):
            angle = 2 * np.pi * i / 8
            landmarks_68[60 + i] = [
                mouth_center[0] + mouth_width * 0.3 * np.cos(angle),
                mouth_center[1] + mouth_width * 0.12 * np.sin(angle)
            ]
        
        return landmarks_68
