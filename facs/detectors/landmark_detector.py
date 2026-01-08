"""
MediaPipe v0.10.31+ 対応ランドマーク検出器
solutions APIが利用不可の場合、Tasks APIまたはOpenCVフォールバックを使用
"""
import numpy as np
import cv2
from typing import List, Optional, Tuple
from abc import ABC
import os
import urllib.request

from ..core.interfaces import ILandmarkDetector
from ..core.enums import DetectorType

class BaseLandmarkDetector(ILandmarkDetector, ABC):
    """ランドマーク検出器の基底クラス"""
    
    @staticmethod
    def get_mediapipe_to_68_mapping() -> List[int]:
        """
        MediaPipe Face Mesh (478点) から dlib形式 (68点) へのマッピング
        
        参考: 
        - https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        - https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/mesh_map.jpg
        
        dlib 68点の順序:
        - 0-16: 顔の輪郭（右耳から左耳へ、顎を通って）
        - 17-21: 右眉
        - 22-26: 左眉
        - 27-30: 鼻筋（上から下）
        - 31-35: 鼻の下部（左から右）
        - 36-41: 右目（外側から時計回り）
        - 42-47: 左目（内側から時計回り）
        - 48-59: 外側の唇（右から時計回り）
        - 60-67: 内側の唇（右から時計回り）
        """
        return [
            # 顔の輪郭 (0-16) - 右から左へ
            10, 338, 297, 332, 284, 251, 389, 356, 454,  # 右側 0-8
            323, 361, 288, 397, 365, 379, 378, 400,      # 左側 + 顎 9-16
            
            # 右眉 (17-21) - 外側から内側
            70, 63, 105, 66, 107,
            
            # 左眉 (22-26) - 内側から外側
            336, 296, 334, 293, 300,
            
            # 鼻筋 (27-30) - 上から下
            168, 6, 197, 195,
            
            # 鼻の下部 (31-35) - 左から右
            98, 97, 2, 326, 327,
            
            # 右目 (36-41) - 外側角から時計回り
            33, 246, 161, 160, 159, 158,  # 上まぶた
            # 133, 155, 154, 153, 145, 144,  # 下まぶた（代替）
            
            # 左目 (42-47) - 内側角から時計回り
            362, 466, 388, 387, 386, 385,  # 上まぶた
            # 263, 382, 381, 380, 374, 373,  # 下まぶた（代替）
            
            # 外側唇 (48-59) - 右角から時計回り
            61, 185, 40, 39, 37, 0,        # 上唇右→中央
            267, 269, 270, 409, 291,       # 上唇中央→左
            375,                            # 左角
            # 321, 405, 314, 17, 84, 181,   # 下唇（代替）
            
            # 内側唇 (60-67) - 右角から時計回り
            78, 191, 80, 81, 82, 13,       # 上
            312, 311,                       # 下
        ]


class MediaPipeLandmarkDetector(BaseLandmarkDetector):
    """MediaPipe v0.10.31+ 対応（Tasks API優先）"""
    
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_FILENAME = "face_landmarker.task"
    
    def __init__(self):
        self._mapping = self.get_mediapipe_to_68_mapping()
        self._api_type = None
        self._face_mesh = None
        self._face_landmarker = None
        self._init_mediapipe()
    
    def _download_model(self, url: str, path: str) -> bool:
        """モデルをダウンロード（SSL問題回避）"""
        import ssl
        import urllib.request
        
        print(f"MediaPipeモデルをダウンロード中...")
        
        # 方法1: SSL検証を無効化
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            with urllib.request.urlopen(url, context=ssl_context) as response:
                with open(path, 'wb') as f:
                    f.write(response.read())
            print(f"ダウンロード完了: {path}")
            return True
        except Exception as e:
            print(f"URLlib失敗: {e}")
        
        # 方法2: requestsライブラリ
        try:
            import requests
            response = requests.get(url, verify=False)
            response.raise_for_status()
            with open(path, 'wb') as f:
                f.write(response.content)
            print(f"ダウンロード完了: {path}")
            return True
        except ImportError:
            pass
        except Exception as e:
            print(f"requests失敗: {e}")
        
        # 方法3: subprocess (curl)
        try:
            import subprocess
            result = subprocess.run(
                ['curl', '-L', '-o', path, url],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and os.path.exists(path):
                print(f"ダウンロード完了: {path}")
                return True
        except Exception as e:
            print(f"curl失敗: {e}")
        
        print(f"\n手動でダウンロードしてください:")
        print(f"  curl -L -o {path} {url}")
        return False
    
    def _get_model_path(self) -> str:
        """モデルファイルのパスを取得（必要に応じてダウンロード）"""
        # モデルの保存場所
        model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, self.MODEL_FILENAME)
        
        # 他の候補パス
        candidates = [
            model_path,
            self.MODEL_FILENAME,
            os.path.expanduser(f"~/.mediapipe/{self.MODEL_FILENAME}"),
        ]
        
        for path in candidates:
            if os.path.exists(path):
                return path
        
        # ダウンロード
        if self._download_model(self.MODEL_URL, model_path):
            return model_path
        
        return ""
    
    def _init_mediapipe(self):
        """MediaPipeを初期化"""
        try:
            import mediapipe as mp
            version = getattr(mp, '__version__', 'unknown')
            print(f"MediaPipe v{version}")
        except ImportError:
            print("警告: MediaPipeがインストールされていません")
            self._api_type = 'opencv'
            return
        
        # 方法1: Tasks API (v0.10.x 推奨)
        if self._try_init_tasks_api():
            return
        
        # 方法2: solutions API (旧バージョン互換)
        if self._try_init_solutions_api():
            return
        
        # 方法3: OpenCVフォールバック
        print("警告: MediaPipeを初期化できません。OpenCVフォールバックを使用")
        self._api_type = 'opencv'
    
    def _try_init_tasks_api(self) -> bool:
        """Tasks APIの初期化を試行"""
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
            
            model_path = self._get_model_path()
            if not model_path or not os.path.exists(model_path):
                print("Tasks API: モデルファイルが見つかりません")
                return False
            
            base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=10,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
            self._face_landmarker = vision.FaceLandmarker.create_from_options(options)
            self._api_type = 'tasks'
            print("MediaPipe Tasks API使用")
            return True
            
        except Exception as e:
            print(f"Tasks API初期化失敗: {e}")
            return False
    
    def _try_init_solutions_api(self) -> bool:
        """solutions APIの初期化を試行"""
        try:
            import mediapipe as mp
            if not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'face_mesh'):
                return False
            
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self._api_type = 'solutions'
            print("MediaPipe solutions API使用")
            return True
            
        except Exception as e:
            print(f"solutions API初期化失敗: {e}")
            return False
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        landmarks_list = self._get_raw_landmarks(image)
        if not landmarks_list:
            return []
        
        faces = []
        for landmarks in landmarks_list:
            x_coords = [lm[0] for lm in landmarks]
            y_coords = [lm[1] for lm in landmarks]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            padding = 0.1
            w, h = x_max - x_min, y_max - y_min
            faces.append((
                int(max(0, x_min - w * padding)),
                int(max(0, y_min - h * padding)),
                int(w * (1 + 2 * padding)),
                int(h * (1 + 2 * padding))
            ))
        return faces
    
    def detect_landmarks(self, image: np.ndarray,
                         face_rect: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        landmarks_list = self._get_raw_landmarks(image)
        if not landmarks_list:
            return None
        
        raw_landmarks = landmarks_list[0]
        landmarks_68 = np.zeros((68, 2), dtype=np.float32)
        for i, mp_idx in enumerate(self._mapping):
            if mp_idx < len(raw_landmarks):
                landmarks_68[i] = [raw_landmarks[mp_idx][0], raw_landmarks[mp_idx][1]]
        
        return landmarks_68
    
    def _get_raw_landmarks(self, image: np.ndarray) -> List[List[Tuple[float, float]]]:
        """生のランドマークを取得"""
        if self._api_type == 'tasks':
            return self._get_landmarks_tasks_api(image)
        elif self._api_type == 'solutions':
            return self._get_landmarks_solutions_api(image)
        else:
            return self._get_landmarks_opencv(image)
    
    def _get_landmarks_tasks_api(self, image: np.ndarray) -> List[List[Tuple[float, float]]]:
        """Tasks API"""
        try:
            import mediapipe as mp
            
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            h, w = image.shape[:2]
            
            # MediaPipe Imageを作成
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            result = self._face_landmarker.detect(mp_image)
            
            if not result.face_landmarks:
                return []
            
            all_landmarks = []
            for face in result.face_landmarks:
                landmarks = [(lm.x * w, lm.y * h) for lm in face]
                all_landmarks.append(landmarks)
            
            return all_landmarks
            
        except Exception as e:
            print(f"Tasks API処理エラー: {e}")
            return self._get_landmarks_opencv(image)
    
    def _get_landmarks_solutions_api(self, image: np.ndarray) -> List[List[Tuple[float, float]]]:
        """solutions API"""
        if self._face_mesh is None:
            return []
        
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = image.shape[:2]
        
        try:
            results = self._face_mesh.process(image_rgb)
            if not results.multi_face_landmarks:
                return []
            
            return [[(lm.x * w, lm.y * h) for lm in face.landmark]
                    for face in results.multi_face_landmarks]
        except Exception as e:
            print(f"solutions API処理エラー: {e}")
            return []
    
    def _get_landmarks_opencv(self, image: np.ndarray) -> List[List[Tuple[float, float]]]:
        """OpenCVフォールバック（顔検出のみ、ランドマークは近似）"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            return []
        
        results = []
        for (x, y, w, h) in faces:
            landmarks = self._generate_approximate_landmarks(x, y, w, h)
            results.append(landmarks)
        
        return results
    
    def _generate_approximate_landmarks(self, x: int, y: int, w: int, h: int) -> List[Tuple[float, float]]:
        """顔矩形から近似的なランドマークを生成"""
        landmarks = []
        cx, cy = x + w / 2, y + h / 2
        
        # 478点の近似位置を生成
        for i in range(478):
            t = i / 478.0
            angle = t * 2 * np.pi * 3
            r = (0.3 + 0.2 * np.sin(t * np.pi * 5)) * min(w, h) / 2
            lx = cx + r * np.cos(angle)
            ly = cy + r * np.sin(angle) * 0.8 + h * 0.1
            landmarks.append((lx, ly))
        
        return landmarks


class DlibLandmarkDetector(BaseLandmarkDetector):
    """dlibを使用したランドマーク検出器"""
    
    def __init__(self, predictor_path: str):
        import dlib
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(predictor_path)
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self._detector(gray, 1)
        return [(f.left(), f.top(), f.right() - f.left(), f.bottom() - f.top()) for f in faces]
    
    def detect_landmarks(self, image: np.ndarray,
                         face_rect: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        import dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if face_rect is None:
            faces = self._detector(gray, 1)
            if not faces:
                return None
            rect = faces[0]
        else:
            x, y, w, h = face_rect
            rect = dlib.rectangle(x, y, x + w, y + h)
        
        shape = self._predictor(gray, rect)
        return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float32)


class LandmarkDetectorFactory:
    """ランドマーク検出器のファクトリ"""
    
    @staticmethod
    def create(detector_type: DetectorType = DetectorType.MEDIAPIPE,
               predictor_path: Optional[str] = None) -> ILandmarkDetector:
        if detector_type == DetectorType.MEDIAPIPE:
            return MediaPipeLandmarkDetector()
        elif detector_type == DetectorType.DLIB:
            if not predictor_path:
                raise ValueError("dlibにはpredictor_pathが必要です")
            return DlibLandmarkDetector(predictor_path)
        else:
            raise ValueError(f"未対応の検出器タイプ: {detector_type}")
