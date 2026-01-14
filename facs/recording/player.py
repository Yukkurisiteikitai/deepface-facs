"""
FACS分析結果の再生
"""
import json
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Iterator, Callable
import threading

from ..core.models import AnalysisResult
from .recorder import RecordingMetadata


class PlaybackState(Enum):
    """再生状態"""
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()


@dataclass
class PlaybackInfo:
    """再生情報"""
    state: PlaybackState
    current_frame: int
    total_frames: int
    current_time: float
    total_time: float
    speed: float
    
    @property
    def progress(self) -> float:
        """進捗率 (0.0-1.0)"""
        if self.total_frames == 0:
            return 0.0
        return self.current_frame / self.total_frames


class FACSPlayer:
    """FACS分析結果のプレイヤー"""
    
    def __init__(self, recording_path: str):
        """
        Args:
            recording_path: 記録ファイルのパス（.jsonl または ディレクトリ/名前）
        """
        self._resolve_paths(recording_path)
        self._load_metadata()
        self._load_frames()
        
        self._state = PlaybackState.STOPPED
        self._current_frame = 0
        self._speed = 1.0
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_callback: Optional[Callable[[AnalysisResult, int], None]] = None
    
    def _resolve_paths(self, recording_path: str) -> None:
        """パスを解決"""
        path = Path(recording_path)
        
        if path.suffix == ".jsonl":
            self.data_path = path
            self.meta_path = path.with_name(path.stem + "_meta.json")
        else:
            # ディレクトリ + 名前として解釈
            self.data_path = path.with_suffix(".jsonl")
            self.meta_path = path.with_name(path.name + "_meta.json")
    
    def _load_metadata(self) -> None:
        """メタデータを読み込み"""
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = RecordingMetadata.from_json(f.read())
        else:
            self.metadata = RecordingMetadata()
    
    def _load_frames(self) -> None:
        """全フレームを読み込み（メモリに保持）"""
        self._frames: List[AnalysisResult] = []
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    result = AnalysisResult.from_record_dict(data)
                    self._frames.append(result)
        
        # メタデータが不完全な場合は更新
        if self.metadata.total_frames == 0:
            self.metadata.total_frames = len(self._frames)
    
    @property
    def state(self) -> PlaybackState:
        return self._state
    
    @property
    def total_frames(self) -> int:
        return len(self._frames)
    
    @property
    def current_frame(self) -> int:
        return self._current_frame
    
    @property
    def fps(self) -> float:
        return self.metadata.fps
    
    @property
    def duration(self) -> float:
        return self.metadata.duration_sec
    
    @property
    def current_time(self) -> float:
        return self._current_frame / self.fps if self.fps > 0 else 0.0
    
    @property
    def playback_info(self) -> PlaybackInfo:
        return PlaybackInfo(
            state=self._state,
            current_frame=self._current_frame,
            total_frames=self.total_frames,
            current_time=self.current_time,
            total_time=self.duration,
            speed=self._speed,
        )
    
    def get_frame(self, frame_number: int) -> Optional[AnalysisResult]:
        """指定フレームの結果を取得"""
        if 0 <= frame_number < len(self._frames):
            return self._frames[frame_number]
        return None
    
    def seek(self, frame_number: int) -> None:
        """指定フレームにシーク"""
        self._current_frame = max(0, min(frame_number, len(self._frames) - 1))
    
    def seek_time(self, time_sec: float) -> None:
        """指定時間にシーク"""
        frame = int(time_sec * self.fps)
        self.seek(frame)
    
    def seek_progress(self, progress: float) -> None:
        """進捗率でシーク (0.0-1.0)"""
        frame = int(progress * self.total_frames)
        self.seek(frame)
    
    def set_speed(self, speed: float) -> None:
        """再生速度を設定 (0.25-4.0)"""
        self._speed = max(0.25, min(4.0, speed))
    
    def set_frame_callback(self, callback: Callable[[AnalysisResult, int], None]) -> None:
        """フレームコールバックを設定"""
        self._frame_callback = callback
    
    def play(self) -> None:
        """再生開始"""
        if self._state == PlaybackState.PLAYING:
            return
        
        self._state = PlaybackState.PLAYING
        self._stop_event.clear()
        
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()
    
    def pause(self) -> None:
        """一時停止"""
        if self._state == PlaybackState.PLAYING:
            self._state = PlaybackState.PAUSED
    
    def resume(self) -> None:
        """再開"""
        if self._state == PlaybackState.PAUSED:
            self.play()
    
    def stop(self) -> None:
        """停止"""
        self._stop_event.set()
        self._state = PlaybackState.STOPPED
        self._current_frame = 0
        
        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)
            self._playback_thread = None
    
    def toggle_play_pause(self) -> None:
        """再生/一時停止をトグル"""
        if self._state == PlaybackState.PLAYING:
            self.pause()
        else:
            self.play()
    
    def step_forward(self, frames: int = 1) -> Optional[AnalysisResult]:
        """指定フレーム数進める"""
        self.seek(self._current_frame + frames)
        return self.get_frame(self._current_frame)
    
    def step_backward(self, frames: int = 1) -> Optional[AnalysisResult]:
        """指定フレーム数戻る"""
        self.seek(self._current_frame - frames)
        return self.get_frame(self._current_frame)
    
    def _playback_loop(self) -> None:
        """再生ループ（別スレッドで実行）"""
        frame_interval = 1.0 / (self.fps * self._speed)
        
        while not self._stop_event.is_set():
            if self._state != PlaybackState.PLAYING:
                time.sleep(0.01)
                continue
            
            if self._current_frame >= len(self._frames):
                self._state = PlaybackState.STOPPED
                break
            
            start_time = time.perf_counter()
            
            # フレーム取得とコールバック
            result = self._frames[self._current_frame]
            if self._frame_callback:
                self._frame_callback(result, self._current_frame)
            
            self._current_frame += 1
            
            # フレームレート制御
            elapsed = time.perf_counter() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def iterate_frames(self) -> Iterator[AnalysisResult]:
        """全フレームをイテレート"""
        for frame in self._frames:
            yield frame
    
    def __len__(self) -> int:
        return len(self._frames)
    
    def __getitem__(self, index: int) -> AnalysisResult:
        return self._frames[index]
