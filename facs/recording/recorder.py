"""
FACS分析結果の記録
"""
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, TextIO
from datetime import datetime

from ..core.models import AnalysisResult


@dataclass
class RecordingMetadata:
    """記録のメタデータ"""
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    fps: float = 30.0
    width: int = 640
    height: int = 480
    total_frames: int = 0
    duration_sec: float = 0.0
    source: str = "camera"  # "camera" or "video"
    source_path: Optional[str] = None
    description: str = ""
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "RecordingMetadata":
        return cls(**json.loads(json_str))


class FACSRecorder:
    """FACS分析結果のレコーダー"""
    
    def __init__(self, output_dir: str, name: Optional[str] = None):
        """
        Args:
            output_dir: 出力ディレクトリ
            name: 記録名（省略時はタイムスタンプ）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_path = self.output_dir / f"{self.name}.jsonl"
        self.meta_path = self.output_dir / f"{self.name}_meta.json"
        
        self.metadata = RecordingMetadata()
        self._file: Optional[TextIO] = None
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._is_recording = False
    
    @property
    def is_recording(self) -> bool:
        return self._is_recording
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    @property
    def elapsed_time(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    def start(
        self,
        fps: float = 30.0,
        width: int = 640,
        height: int = 480,
        source: str = "camera",
        source_path: Optional[str] = None,
        description: str = "",
    ) -> None:
        """記録を開始"""
        if self._is_recording:
            raise RuntimeError("Already recording")
        
        self.metadata = RecordingMetadata(
            fps=fps,
            width=width,
            height=height,
            source=source,
            source_path=source_path,
            description=description,
        )
        
        self._file = open(self.data_path, "w", encoding="utf-8")
        self._frame_count = 0
        self._start_time = time.time()
        self._is_recording = True
        
        print(f"Recording started: {self.data_path}")
    
    def record_frame(self, result: AnalysisResult) -> None:
        """1フレームの分析結果を記録"""
        if not self._is_recording or self._file is None:
            return
        
        result.frame_number = self._frame_count
        record = result.to_record_dict()
        
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._frame_count += 1
    
    def stop(self) -> RecordingMetadata:
        """記録を停止してメタデータを返す"""
        if not self._is_recording:
            raise RuntimeError("Not recording")
        
        if self._file:
            self._file.close()
            self._file = None
        
        # メタデータを更新
        self.metadata.total_frames = self._frame_count
        self.metadata.duration_sec = self.elapsed_time
        
        # メタデータを保存
        with open(self.meta_path, "w", encoding="utf-8") as f:
            f.write(self.metadata.to_json())
        
        self._is_recording = False
        print(f"Recording stopped: {self._frame_count} frames, {self.metadata.duration_sec:.1f}s")
        
        return self.metadata
    
    def __enter__(self) -> "FACSRecorder":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._is_recording:
            self.stop()
