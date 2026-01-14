"""
マルチプロセス並列処理モジュール
推論と描画を分離して並列実行するためのクラス群
"""
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import numpy as np
import cv2
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from queue import Empty, Full
import threading


@dataclass
class FrameData:
    """フレームデータの転送用構造体"""
    frame_id: int
    image: np.ndarray
    timestamp: float


@dataclass  
class ResultData:
    """分析結果の転送用構造体"""
    frame_id: int
    result_dict: Dict[str, Any]
    processing_time_ms: float


class InferenceWorker:
    """
    推論ワーカープロセス
    カメラからのフレームを受け取り、FACS分析を実行
    """
    
    def __init__(
        self,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        stop_event: mp.Event,
        use_mediapipe: bool = True,
        mode: str = 'realtime'
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.use_mediapipe = use_mediapipe
        self.mode = mode
        self._analyzer = None
    
    def _init_analyzer(self):
        """プロセス内でアナライザーを初期化（pickling回避）"""
        from ..core.enums import AnalysisMode, DetectorType
        from ..detectors.vectorized_au_detector import VectorizedAUDetector
        from ..detectors.optimized_feature_extractor import OptimizedFeatureExtractor
        from ..detectors import LandmarkDetectorFactory
        from ..estimators import IntensityEstimator, EmotionMapper
        from ..core.models import FaceData, AnalysisResult
        
        # 最適化されたコンポーネントを使用
        detector_type = DetectorType.MEDIAPIPE if self.use_mediapipe else DetectorType.DLIB
        self._landmark_detector = LandmarkDetectorFactory.create(detector_type)
        self._feature_extractor = OptimizedFeatureExtractor()
        self._au_detector = VectorizedAUDetector()
        self._intensity_estimator = IntensityEstimator()
        self._emotion_mapper = EmotionMapper()
        
        # モード設定
        mode_map = {
            'realtime': {'scale': 0.5, 'threshold': 0.4},
            'balanced': {'scale': 1.0, 'threshold': 0.3},
            'accurate': {'scale': 1.0, 'threshold': 0.2}
        }
        self._config = mode_map.get(self.mode, mode_map['realtime'])
    
    def run(self):
        """ワーカーのメインループ"""
        self._init_analyzer()
        
        from ..core.models import FaceData, AnalysisResult
        
        while not self.stop_event.is_set():
            try:
                # タイムアウト付きで入力を取得
                frame_data = self.input_queue.get(timeout=0.1)
            except Empty:
                continue
            
            start_time = time.time()
            
            # 画像スケーリング
            scale = self._config['scale']
            if scale < 1.0:
                small_image = cv2.resize(
                    frame_data.image, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                small_image = frame_data.image
            
            # ランドマーク検出
            landmarks = self._landmark_detector.detect_landmarks(small_image)
            
            result_dict = {'is_valid': False, 'frame_id': frame_data.frame_id}
            
            if landmarks is not None:
                # スケールを戻す
                if scale < 1.0:
                    landmarks = landmarks / scale
                
                # 特徴量抽出
                distances = self._feature_extractor.compute_distances(landmarks)
                angles = self._feature_extractor.compute_angles(landmarks)
                
                # 顔矩形検出
                faces = self._landmark_detector.detect_faces(small_image)
                if faces and scale < 1.0:
                    faces = [
                        (int(x/scale), int(y/scale), int(w/scale), int(h/scale))
                        for x, y, w, h in faces
                    ]
                
                # AU検出
                au_results = self._au_detector.detect_all(landmarks, distances, angles)
                
                # 閾値フィルタ
                threshold = self._config['threshold']
                au_results = {
                    k: v for k, v in au_results.items()
                    if v.confidence >= threshold
                }
                
                # 強度推定
                intensity_results = self._intensity_estimator.estimate_all(au_results)
                facs_code = self._intensity_estimator.format_facs_code(intensity_results)
                
                # 感情推定
                emotions = self._emotion_mapper.map(au_results, intensity_results)
                valence, arousal = self._emotion_mapper.get_valence_arousal(
                    au_results, intensity_results
                )
                
                # 結果をシリアライズ可能な形式に変換
                result_dict = {
                    'is_valid': True,
                    'frame_id': frame_data.frame_id,
                    'face_rect': faces[0] if faces else None,
                    'landmarks': landmarks.tolist(),
                    'distances': distances,
                    'angles': angles,
                    'au_results': {
                        k: {
                            'au_number': v.au_number,
                            'name': v.name,
                            'detected': v.detected,
                            'confidence': v.confidence,
                            'intensity': v.intensity.value,
                            'raw_score': v.raw_score,
                            'asymmetry': v.asymmetry
                        }
                        for k, v in au_results.items()
                    },
                    'intensity_results': {
                        k: {
                            'au_number': v.au_number,
                            'intensity': v.intensity.value,
                            'confidence': v.confidence
                        }
                        for k, v in intensity_results.items()
                    },
                    'facs_code': facs_code,
                    'emotions': [
                        {
                            'emotion': e.emotion,
                            'confidence': e.confidence,
                            'valence': e.valence,
                            'arousal': e.arousal
                        }
                        for e in emotions
                    ],
                    'valence': valence,
                    'arousal': arousal
                }
            
            processing_time = (time.time() - start_time) * 1000
            
            # 結果を出力キューに送信
            try:
                result_data = ResultData(
                    frame_id=frame_data.frame_id,
                    result_dict=result_dict,
                    processing_time_ms=processing_time
                )
                self.output_queue.put(result_data, timeout=0.05)
            except Full:
                # キューがいっぱいの場合は古い結果を捨てる
                try:
                    self.output_queue.get_nowait()
                    self.output_queue.put(result_data, timeout=0.05)
                except:
                    pass


def inference_worker_process(
    input_queue: mp.Queue,
    output_queue: mp.Queue, 
    stop_event: mp.Event,
    use_mediapipe: bool,
    mode: str
):
    """推論ワーカープロセスのエントリーポイント"""
    worker = InferenceWorker(
        input_queue, output_queue, stop_event,
        use_mediapipe, mode
    )
    worker.run()


class ParallelFACSProcessor:
    """
    並列FACS処理マネージャー
    推論と描画を別プロセスで実行
    """
    
    def __init__(
        self,
        use_mediapipe: bool = True,
        mode: str = 'realtime',
        num_workers: int = 1,
        queue_size: int = 4
    ):
        self.use_mediapipe = use_mediapipe
        self.mode = mode
        self.num_workers = num_workers
        self.queue_size = queue_size
        
        # プロセス間通信用
        self._input_queue: Optional[mp.Queue] = None
        self._output_queue: Optional[mp.Queue] = None
        self._stop_event: Optional[mp.Event] = None
        self._workers: list = []
        
        # 状態管理
        self._frame_counter = 0
        self._latest_result: Optional[Dict] = None
        self._result_lock = threading.Lock()
        
        # パフォーマンス計測
        self._fps_counter = FPSCounter()
        
        # ビジュアライザー（メインプロセスで使用）
        self._visualizer = None
    
    def start(self):
        """並列処理を開始"""
        # キューとイベントを作成
        self._input_queue = mp.Queue(maxsize=self.queue_size)
        self._output_queue = mp.Queue(maxsize=self.queue_size)
        self._stop_event = mp.Event()
        
        # ワーカープロセスを起動
        for _ in range(self.num_workers):
            p = Process(
                target=inference_worker_process,
                args=(
                    self._input_queue,
                    self._output_queue,
                    self._stop_event,
                    self.use_mediapipe,
                    self.mode
                )
            )
            p.daemon = True
            p.start()
            self._workers.append(p)
        
        # ビジュアライザーを初期化（メインプロセス）
        from ..visualization import FACSVisualizer
        from ..visualization.visualizer import LayoutConfig
        self._visualizer = FACSVisualizer(LayoutConfig())
        
        print(f"[ParallelFACS] Started {self.num_workers} worker(s)")
    
    def stop(self):
        """並列処理を停止"""
        if self._stop_event:
            self._stop_event.set()
        
        # ワーカーの終了を待機
        for p in self._workers:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        
        self._workers.clear()
        
        # キューをクリーンアップ
        if self._input_queue:
            self._input_queue.close()
        if self._output_queue:
            self._output_queue.close()
        
        print("[ParallelFACS] Stopped")
    
    def submit_frame(self, image: np.ndarray) -> int:
        """
        フレームを推論キューに送信
        
        Args:
            image: BGR画像
            
        Returns:
            フレームID
        """
        self._frame_counter += 1
        frame_data = FrameData(
            frame_id=self._frame_counter,
            image=image.copy(),
            timestamp=time.time()
        )
        
        try:
            self._input_queue.put(frame_data, timeout=0.01)
        except Full:
            # キューがいっぱいの場合は古いフレームを捨てる
            try:
                self._input_queue.get_nowait()
                self._input_queue.put(frame_data, timeout=0.01)
            except:
                pass
        
        return self._frame_counter
    
    def get_latest_result(self) -> Optional[Dict]:
        """最新の分析結果を取得（ノンブロッキング）"""
        # 新しい結果があれば取得
        while True:
            try:
                result_data = self._output_queue.get_nowait()
                with self._result_lock:
                    self._latest_result = result_data.result_dict
                    self._latest_result['processing_time_ms'] = result_data.processing_time_ms
            except Empty:
                break
        
        with self._result_lock:
            return self._latest_result
    
    def process_and_visualize(self, image: np.ndarray) -> Tuple[Optional[Dict], np.ndarray]:
        """
        フレームを送信し、最新結果で可視化
        
        Args:
            image: BGR画像
            
        Returns:
            (result_dict, visualized_image) のタプル
        """
        # フレームを送信
        self.submit_frame(image)
        
        # 最新結果を取得
        result = self.get_latest_result()
        
        # 可視化
        if result and result.get('is_valid', False) and self._visualizer:
            vis = self._visualize_from_dict(image, result)
        else:
            vis = image.copy()
        
        # FPS計算
        self._fps_counter.update()
        fps = self._fps_counter.get_fps()
        
        # FPSとモード情報を表示
        info_text = f"FPS: {fps:.1f} | Mode: {self.mode}"
        if result:
            info_text += f" | Inference: {result.get('processing_time_ms', 0):.1f}ms"
        cv2.putText(
            vis, info_text, (10, vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        
        return result, vis
    
    def _visualize_from_dict(self, image: np.ndarray, result_dict: Dict) -> np.ndarray:
        """辞書形式の結果からAnalysisResultを再構築して可視化"""
        from ..core.models import (
            AnalysisResult, FaceData, AUDetectionResult, 
            IntensityResult, EmotionResult
        )
        from ..core.enums import AUIntensity
        
        # ランドマークを復元
        landmarks = np.array(result_dict['landmarks'])
        
        # FaceDataを再構築
        face_data = FaceData(
            rect=result_dict.get('face_rect'),
            landmarks=landmarks,
            distances=result_dict.get('distances', {}),
            angles=result_dict.get('angles', {})
        )
        
        # AU結果を再構築
        au_results = {}
        for k, v in result_dict.get('au_results', {}).items():
            au_results[int(k)] = AUDetectionResult(
                au_number=v['au_number'],
                name=v['name'],
                detected=v['detected'],
                confidence=v['confidence'],
                intensity=AUIntensity(v['intensity']),
                raw_score=v['raw_score'],
                asymmetry=v['asymmetry']
            )
        
        # 強度結果を再構築
        intensity_results = {}
        for k, v in result_dict.get('intensity_results', {}).items():
            intensity_results[int(k)] = IntensityResult(
                au_number=v['au_number'],
                intensity=AUIntensity(v['intensity']),
                confidence=v['confidence']
            )
        
        # 感情結果を再構築
        emotions = [
            EmotionResult(
                emotion=e['emotion'],
                confidence=e['confidence'],
                valence=e['valence'],
                arousal=e['arousal']
            )
            for e in result_dict.get('emotions', [])
        ]
        
        # AnalysisResultを構築
        analysis_result = AnalysisResult()
        analysis_result.face_data = face_data
        analysis_result.au_results = au_results
        analysis_result.intensity_results = intensity_results
        analysis_result.facs_code = result_dict.get('facs_code', '')
        analysis_result.emotions = emotions
        analysis_result.valence = result_dict.get('valence', 0.0)
        analysis_result.arousal = result_dict.get('arousal', 0.0)
        analysis_result.processing_time_ms = result_dict.get('processing_time_ms', 0.0)
        
        return self._visualizer.create_analysis_panel(image, analysis_result)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class FPSCounter:
    """FPS計測用ユーティリティ"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps: list = []
    
    def update(self):
        """タイムスタンプを記録"""
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
    
    def get_fps(self) -> float:
        """現在のFPSを計算"""
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


def run_parallel_realtime(
    camera_id: int = 0,
    use_mediapipe: bool = True,
    mode: str = 'realtime',
    num_workers: int = 1
):
    """
    並列処理でリアルタイムFACS分析を実行
    
    Args:
        camera_id: カメラID
        use_mediapipe: MediaPipeを使用するか
        mode: 分析モード ('realtime', 'balanced', 'accurate')
        num_workers: ワーカープロセス数
    """
    with ParallelFACSProcessor(
        use_mediapipe=use_mediapipe,
        mode=mode,
        num_workers=num_workers
    ) as processor:
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result, vis = processor.process_and_visualize(frame)
            
            cv2.imshow("FACS Parallel Realtime", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
