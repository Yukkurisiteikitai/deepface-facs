"""
FACS記録・再生デモ
"""
import cv2
import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from facs import FACSAnalyzer
from facs.recording import FACSRecorder, FACSPlayer, PlaybackState
from facs.visualization import FACSVisualizer


def record_session(output_dir: str, duration: int = 30):
    """カメラから記録"""
    analyzer = FACSAnalyzer()
    visualizer = FACSVisualizer()
    recorder = FACSRecorder(output_dir)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラを開けません")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    recorder.start(fps=fps, width=width, height=height, description="Demo recording")
    
    print(f"Recording... (Press 'q' to stop, max {duration}s)")
    
    try:
        while recorder.elapsed_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = analyzer.analyze(frame)
            recorder.record_frame(result)
            
            # 描画
            vis_frame = visualizer.create_analysis_panel(frame, result)
            
            # 記録インジケータ
            cv2.circle(vis_frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(vis_frame, f"REC {recorder.frame_count}f / {recorder.elapsed_time:.1f}s",
                       (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow("Recording", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        metadata = recorder.stop()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Saved: {recorder.data_path}")
        print(f"Frames: {metadata.total_frames}, Duration: {metadata.duration_sec:.1f}s")


def playback_session(recording_path: str):
    """記録を再生"""
    player = FACSPlayer(recording_path)
    visualizer = FACSVisualizer()
    
    print(f"Playing: {recording_path}")
    print(f"Frames: {player.total_frames}, Duration: {player.duration:.1f}s, FPS: {player.fps}")
    print("\nControls:")
    print("  Space: Play/Pause")
    print("  Left/Right: -/+ 1 frame")
    print("  Up/Down: Speed up/down")
    print("  0-9: Seek to 0%-90%")
    print("  q: Quit")
    
    # フレームサイズ
    frame_width = player.metadata.width
    frame_height = player.metadata.height
    
    current_result = player.get_frame(0)
    
    while True:
        info = player.playback_info
        
        # 再生中ならフレームを進める
        if info.state == PlaybackState.PLAYING:
            current_result = player.step_forward()
            if player.current_frame >= player.total_frames - 1:
                player.pause()
        
        # 記録されたランドマークから描画用フレームを作成
        frame = create_landmark_frame(frame_width, frame_height, current_result)
        
        # Visualizerでパネル付き表示を作成
        vis_frame = visualizer.create_analysis_panel(frame, current_result)
        
        # 再生情報オーバーレイ
        draw_playback_overlay(vis_frame, info)
        
        cv2.imshow("Playback", vis_frame)
        
        # キー処理（FPS考慮）
        wait_time = max(1, int(1000 / player.fps / info.speed))
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            player.toggle_play_pause()
        elif key == 81 or key == 2 or key == ord('a'):  # Left
            player.pause()
            player.step_backward()
            current_result = player.get_frame(player.current_frame)
        elif key == 83 or key == 3 or key == ord('d'):  # Right
            player.pause()
            player.step_forward()
            current_result = player.get_frame(player.current_frame)
        elif key == 82 or key == 0 or key == ord('w'):  # Up
            player.set_speed(info.speed * 1.5)
            print(f"Speed: {player._speed:.2f}x")
        elif key == 84 or key == 1 or key == ord('s'):  # Down
            player.set_speed(info.speed / 1.5)
            print(f"Speed: {player._speed:.2f}x")
        elif ord('0') <= key <= ord('9'):
            progress = (key - ord('0')) / 10
            player.seek_progress(progress)
            current_result = player.get_frame(player.current_frame)
    
    player.stop()
    cv2.destroyAllWindows()


def create_landmark_frame(width: int, height: int, result) -> np.ndarray:
    """記録されたランドマークから描画用フレームを作成"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)  # ダークグレー背景
    
    if result and result.face_data and result.face_data.landmarks is not None:
        landmarks = result.face_data.landmarks
        
        # ランドマークの接続線を描画
        connections = [
            (range(0, 17), False, (100, 100, 100)),      # 顔輪郭
            (range(17, 22), False, (150, 150, 150)),     # 左眉
            (range(22, 27), False, (150, 150, 150)),     # 右眉
            (range(27, 31), False, (150, 150, 150)),     # 鼻筋
            (range(31, 36), False, (150, 150, 150)),     # 鼻底
            (range(36, 42), True, (200, 200, 100)),      # 左目
            (range(42, 48), True, (200, 200, 100)),      # 右目
            (range(48, 60), True, (100, 100, 200)),      # 外唇
            (range(60, 68), True, (150, 100, 200)),      # 内唇
        ]
        
        for indices, closed, color in connections:
            idx_list = list(indices)
            if max(idx_list) < len(landmarks):
                pts = landmarks[idx_list].astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], closed, color, 2)
        
        # ランドマーク点を描画
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # 顔の矩形
        if result.face_data.rect:
            x, y, w, h = result.face_data.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
    return frame


def draw_playback_overlay(frame: np.ndarray, info):
    """再生情報オーバーレイを描画"""
    h, w = frame.shape[:2]
    
    # 下部にオーバーレイ領域を作成
    overlay_height = 60
    overlay_y = h - overlay_height
    
    # 半透明背景
    overlay = frame[overlay_y:h, :].copy()
    cv2.rectangle(overlay, (0, 0), (w, overlay_height), (0, 0, 0), -1)
    frame[overlay_y:h, :] = cv2.addWeighted(frame[overlay_y:h, :], 0.3, overlay, 0.7, 0)
    
    # プログレスバー
    bar_height = 6
    bar_y = overlay_y + 10
    bar_margin = 20
    bar_width = w - bar_margin * 2
    
    # バー背景
    cv2.rectangle(frame, (bar_margin, bar_y), (bar_margin + bar_width, bar_y + bar_height), 
                  (80, 80, 80), -1)
    
    # プログレス
    progress_width = int(bar_width * info.progress)
    cv2.rectangle(frame, (bar_margin, bar_y), (bar_margin + progress_width, bar_y + bar_height), 
                  (0, 200, 255), -1)
    
    # 再生状態アイコン
    state_icons = {
        PlaybackState.PLAYING: ">",
        PlaybackState.PAUSED: "||",
        PlaybackState.STOPPED: "[]",
    }
    state_text = state_icons.get(info.state, "?")
    
    # テキスト情報
    text_y = overlay_y + 40
    cv2.putText(frame, state_text, (bar_margin, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    time_text = f"{info.current_time:.1f}s / {info.total_time:.1f}s"
    cv2.putText(frame, time_text, (bar_margin + 50, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    frame_text = f"Frame: {info.current_frame}/{info.total_frames}"
    cv2.putText(frame, frame_text, (bar_margin + 200, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    speed_text = f"x{info.speed:.1f}"
    cv2.putText(frame, speed_text, (w - bar_margin - 60, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    parser = argparse.ArgumentParser(description="FACS Recording Demo")
    parser.add_argument("mode", choices=["record", "play"], help="Mode: record or play")
    parser.add_argument("--output", "-o", default="./recordings", help="Output directory for recording")
    parser.add_argument("--input", "-i", help="Recording file to play")
    parser.add_argument("--duration", "-d", type=int, default=30, help="Max recording duration (sec)")
    
    args = parser.parse_args()
    
    if args.mode == "record":
        record_session(args.output, args.duration)
    elif args.mode == "play":
        if not args.input:
            # 最新の記録を探す
            recordings = list(Path(args.output).glob("*.jsonl"))
            if not recordings:
                print(f"No recordings found in {args.output}")
                return
            args.input = str(max(recordings, key=lambda p: p.stat().st_mtime))
            print(f"Playing latest recording: {args.input}")
        playback_session(args.input)


if __name__ == "__main__":
    main()
