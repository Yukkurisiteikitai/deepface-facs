"""
FACS記録をMP4動画としてエクスポート
"""
import json
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from ..core.models import AnalysisResult
from ..core.enums import AUIntensity
from .recorder import RecordingMetadata


class FACSVideoExporter:
    """FACS記録を動画としてエクスポート"""
    
    # 強度に応じた色
    INTENSITY_COLORS = {
        AUIntensity.ABSENT: (128, 128, 128),
        AUIntensity.TRACE: (0, 255, 0),
        AUIntensity.SLIGHT: (0, 255, 128),
        AUIntensity.MARKED: (0, 255, 255),
        AUIntensity.SEVERE: (0, 165, 255),
        AUIntensity.MAXIMUM: (0, 0, 255)
    }
    
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.face_area_width = int(width * 0.5)  # 左半分: 顔
        self.panel_width = width - self.face_area_width  # 右半分: 情報パネル
    
    def export(
        self,
        recording_path: str,
        output_path: Optional[str] = None,
        fps: Optional[float] = None,
        show_progress: bool = True
    ) -> str:
        """
        記録をMP4動画としてエクスポート
        
        Args:
            recording_path: 記録ファイルのパス (.jsonl)
            output_path: 出力ファイルのパス (省略時は自動生成)
            fps: 出力FPS (省略時はメタデータから)
            show_progress: 進捗表示
        
        Returns:
            出力ファイルのパス
        """
        # パス解決
        data_path = Path(recording_path)
        if data_path.suffix != ".jsonl":
            data_path = data_path.with_suffix(".jsonl")
        
        meta_path = data_path.with_name(data_path.stem + "_meta.json")
        
        # メタデータ読み込み
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = RecordingMetadata.from_json(f.read())
        else:
            metadata = RecordingMetadata()
        
        # FPS設定
        output_fps = fps or metadata.fps or 30.0
        
        # 出力パス
        if output_path is None:
            output_path = str(data_path.with_suffix(".mp4"))
        
        # フレームデータ読み込み
        frames_data = []
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    result = AnalysisResult.from_record_dict(data)
                    frames_data.append(result)
        
        total_frames = len(frames_data)
        if total_frames == 0:
            raise ValueError("記録にフレームがありません")
        
        # VideoWriter設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, output_fps, (self.width, self.height))
        
        if not writer.isOpened():
            raise RuntimeError(f"動画ファイルを作成できません: {output_path}")
        
        try:
            for i, result in enumerate(frames_data):
                # フレーム描画
                frame = self._render_frame(result, i, total_frames, metadata)
                writer.write(frame)
                
                if show_progress and i % 30 == 0:
                    progress = (i + 1) / total_frames * 100
                    print(f"\r書き出し中: {progress:.1f}% ({i+1}/{total_frames})", end="")
            
            if show_progress:
                print(f"\r書き出し完了: {output_path}")
        
        finally:
            writer.release()
        
        return output_path
    
    def _render_frame(
        self,
        result: AnalysisResult,
        frame_idx: int,
        total_frames: int,
        metadata: RecordingMetadata
    ) -> np.ndarray:
        """1フレームを描画"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)  # ダークグレー背景
        
        # 左側: 顔のランドマーク描画
        self._draw_face_area(frame, result)
        
        # 右側: 情報パネル
        self._draw_info_panel(frame, result, frame_idx, total_frames, metadata)
        
        # 下部: プログレスバー
        self._draw_progress_bar(frame, frame_idx, total_frames, metadata)
        
        return frame
    
    def _draw_face_area(self, frame: np.ndarray, result: AnalysisResult):
        """顔エリアを描画"""
        area_x = 0
        area_w = self.face_area_width
        area_h = self.height - 60  # プログレスバー分を引く
        
        # 背景
        cv2.rectangle(frame, (area_x, 0), (area_x + area_w, area_h), (25, 25, 35), -1)
        
        if not result.face_data or result.face_data.landmarks is None:
            # 顔なし
            cv2.putText(frame, "No Face Detected", (area_x + 50, area_h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            return
        
        landmarks = result.face_data.landmarks
        
        # ランドマークをエリアに収めるようにスケーリング
        min_x, min_y = landmarks.min(axis=0)
        max_x, max_y = landmarks.max(axis=0)
        face_w = max_x - min_x
        face_h = max_y - min_y
        
        # スケールとオフセット計算
        margin = 50
        scale = min((area_w - margin * 2) / face_w, (area_h - margin * 2) / face_h)
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        offset_x = area_x + area_w / 2 - center_x * scale
        offset_y = area_h / 2 - center_y * scale
        
        # ランドマークを変換
        scaled_landmarks = landmarks * scale + np.array([offset_x, offset_y])
        
        # 接続線を描画
        connections = [
            (range(0, 17), False, (80, 80, 80)),      # 顔輪郭
            (range(17, 22), False, (100, 100, 150)),  # 左眉
            (range(22, 27), False, (100, 100, 150)),  # 右眉
            (range(27, 31), False, (100, 100, 100)),  # 鼻筋
            (range(31, 36), False, (100, 100, 100)),  # 鼻底
            (range(36, 42), True, (150, 150, 50)),    # 左目
            (range(42, 48), True, (150, 150, 50)),    # 右目
            (range(48, 60), True, (50, 50, 150)),     # 外唇
            (range(60, 68), True, (100, 50, 150)),    # 内唇
        ]
        
        for indices, closed, color in connections:
            idx_list = list(indices)
            if max(idx_list) < len(scaled_landmarks):
                pts = scaled_landmarks[idx_list].astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], closed, color, 2)
        
        # ランドマーク点を描画
        for i, (x, y) in enumerate(scaled_landmarks):
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # 顔の矩形（元の座標をスケーリング）
        if result.face_data.rect:
            rx, ry, rw, rh = result.face_data.rect
            rx = int(rx * scale + offset_x)
            ry = int(ry * scale + offset_y)
            rw = int(rw * scale)
            rh = int(rh * scale)
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 2)
    
    def _draw_info_panel(
        self,
        frame: np.ndarray,
        result: AnalysisResult,
        frame_idx: int,
        total_frames: int,
        metadata: RecordingMetadata
    ):
        """情報パネルを描画"""
        panel_x = self.face_area_width
        panel_w = self.panel_width
        panel_h = self.height - 60
        
        # パネル背景
        cv2.rectangle(frame, (panel_x, 0), (panel_x + panel_w, panel_h), (20, 20, 30), -1)
        
        y = 30
        margin = 20
        
        # タイトル
        cv2.putText(frame, "FACS Analysis", (panel_x + margin, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y += 40
        
        # 区切り線
        cv2.line(frame, (panel_x + margin, y), (panel_x + panel_w - margin, y), (60, 60, 80), 1)
        y += 20
        
        # 感情セクション
        cv2.putText(frame, "Emotions", (panel_x + margin, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 200), 1)
        y += 25
        
        if result.emotions:
            for emotion in result.emotions[:5]:
                # 感情名
                cv2.putText(frame, emotion.emotion, (panel_x + margin, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # バー背景
                bar_x = panel_x + margin + 100
                bar_w = panel_w - 180
                bar_h = 15
                cv2.rectangle(frame, (bar_x, y - 12), (bar_x + bar_w, y + 3), (50, 50, 60), -1)
                
                # バー
                fill_w = int(bar_w * emotion.confidence)
                color = (0, 200, 100) if emotion.valence > 0 else (0, 100, 200) if emotion.valence < 0 else (200, 200, 0)
                cv2.rectangle(frame, (bar_x, y - 12), (bar_x + fill_w, y + 3), color, -1)
                
                # パーセント
                cv2.putText(frame, f"{emotion.confidence:.0%}", (bar_x + bar_w + 5, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                y += 25
        else:
            cv2.putText(frame, "No emotions detected", (panel_x + margin, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            y += 25
        
        y += 10
        
        # Valence / Arousal
        cv2.putText(frame, "Valence / Arousal", (panel_x + margin, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 200), 1)
        y += 30
        
        # Valence
        v_color = (0, 200, 0) if result.valence > 0 else (0, 0, 200) if result.valence < 0 else (200, 200, 0)
        cv2.putText(frame, f"V: {result.valence:+.2f}", (panel_x + margin, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, v_color, 2)
        
        # Arousal
        cv2.putText(frame, f"A: {result.arousal:+.2f}", (panel_x + margin + 120, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y += 35
        
        # FACSコード
        cv2.putText(frame, "FACS Code", (panel_x + margin, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 200), 1)
        y += 25
        
        facs_code = result.facs_code[:40] + "..." if len(result.facs_code) > 40 else result.facs_code
        cv2.putText(frame, facs_code, (panel_x + margin, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 35
        
        # Active AUs
        cv2.putText(frame, "Active AUs", (panel_x + margin, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 200), 1)
        y += 25
        
        if result.active_aus:
            for au in result.active_aus[:8]:
                intensity = result.intensity_results.get(au.au_number)
                color = self.INTENSITY_COLORS.get(
                    intensity.intensity if intensity else AUIntensity.ABSENT,
                    (200, 200, 200)
                )
                label = intensity.intensity_label if intensity else ""
                
                text = f"AU{au.au_number}: {au.name[:15]}"
                cv2.putText(frame, text, (panel_x + margin, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                
                cv2.putText(frame, f"[{label}]", (panel_x + margin + 180, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                conf_text = f"{au.confidence:.0%}"
                cv2.putText(frame, conf_text, (panel_x + panel_w - 60, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                y += 22
        else:
            cv2.putText(frame, "No AUs detected", (panel_x + margin, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    def _draw_progress_bar(
        self,
        frame: np.ndarray,
        frame_idx: int,
        total_frames: int,
        metadata: RecordingMetadata
    ):
        """プログレスバーを描画"""
        bar_y = self.height - 50
        bar_height = 40
        margin = 20
        
        # 背景
        cv2.rectangle(frame, (0, bar_y), (self.width, self.height), (15, 15, 20), -1)
        
        # プログレスバー
        bar_x = 100
        bar_w = self.width - 200
        bar_h = 8
        bar_center_y = bar_y + bar_height // 2
        
        # バー背景
        cv2.rectangle(frame, (bar_x, bar_center_y - bar_h // 2),
                     (bar_x + bar_w, bar_center_y + bar_h // 2), (60, 60, 80), -1)
        
        # プログレス
        progress = frame_idx / max(total_frames - 1, 1)
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_center_y - bar_h // 2),
                     (bar_x + fill_w, bar_center_y + bar_h // 2), (0, 200, 255), -1)
        
        # 再生位置マーカー
        marker_x = bar_x + fill_w
        cv2.circle(frame, (marker_x, bar_center_y), 8, (255, 255, 255), -1)
        
        # 時間表示
        current_time = frame_idx / metadata.fps if metadata.fps > 0 else 0
        total_time = metadata.duration_sec
        
        time_text = f"{current_time:.1f}s / {total_time:.1f}s"
        cv2.putText(frame, time_text, (margin, bar_center_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # フレーム番号
        frame_text = f"Frame {frame_idx + 1}/{total_frames}"
        text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, frame_text, (self.width - margin - text_size[0], bar_center_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    """コマンドライン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FACS記録をMP4動画としてエクスポート")
    parser.add_argument("input", help="記録ファイルのパス (.jsonl)")
    parser.add_argument("-o", "--output", help="出力ファイルのパス")
    parser.add_argument("--width", type=int, default=1280, help="出力動画の幅")
    parser.add_argument("--height", type=int, default=720, help="出力動画の高さ")
    parser.add_argument("--fps", type=float, help="出力FPS")
    
    args = parser.parse_args()
    
    exporter = FACSVideoExporter(width=args.width, height=args.height)
    output = exporter.export(args.input, args.output, args.fps)
    print(f"出力: {output}")


if __name__ == "__main__":
    main()
