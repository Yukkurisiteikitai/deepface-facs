#!/usr/bin/env python3
# filepath: /Users/yuuto/learn_lab/analysis_face/facs_cli.py
"""
FACS Analyzer CLI - インタラクティブコマンドラインツール
"""
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime


class TerminalMenu:
    """ターミナルメニュー（矢印キー対応）"""
    
    def __init__(self, title: str, options: list, show_back: bool = False):
        self.title = title
        self.options = options
        self.show_back = show_back
        self.selected = 0
    
    def _clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _get_key(self):
        """キー入力を取得"""
        if os.name == 'nt':
            import msvcrt
            key = msvcrt.getch()
            if key == b'\xe0':
                key = msvcrt.getch()
                if key == b'H': return 'up'
                if key == b'P': return 'down'
            if key == b'\r': return 'enter'
            if key == b'q': return 'quit'
            return key.decode('utf-8', errors='ignore')
        else:
            import tty
            import termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == '\x1b':
                    ch2 = sys.stdin.read(2)
                    if ch2 == '[A': return 'up'
                    if ch2 == '[B': return 'down'
                    if ch2 == '[C': return 'right'
                    if ch2 == '[D': return 'left'
                if ch == '\r' or ch == '\n': return 'enter'
                if ch == 'q' or ch == '\x03': return 'quit'
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
    
    def _render(self):
        """メニューを描画"""
        self._clear_screen()
        
        print("\n" + "=" * 50)
        print(f"  🎭 {self.title}")
        print("=" * 50 + "\n")
        
        all_options = list(self.options)
        if self.show_back:
            all_options.append(("← 戻る", "back"))
        
        for i, (label, _) in enumerate(all_options):
            if i == self.selected:
                print(f"  \033[7m > {label} \033[0m")
            else:
                print(f"    {label}")
        
        print("\n" + "-" * 50)
        print("  ↑/↓: 選択  Enter: 決定  q: 終了")
        print("-" * 50)
    
    def show(self):
        """メニューを表示して選択結果を返す"""
        all_options = list(self.options)
        if self.show_back:
            all_options.append(("← 戻る", "back"))
        
        while True:
            self._render()
            key = self._get_key()
            
            if key == 'up':
                self.selected = (self.selected - 1) % len(all_options)
            elif key == 'down':
                self.selected = (self.selected + 1) % len(all_options)
            elif key == 'enter':
                return all_options[self.selected][1]
            elif key == 'quit':
                return 'quit'


class DirectoryBrowser:
    """ディレクトリブラウザ"""
    
    def __init__(self, start_path: str = ".", title: str = "ディレクトリ選択"):
        self.current_path = Path(start_path).resolve()
        self.title = title
        self.selected = 0
    
    def _clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _get_key(self):
        """キー入力を取得"""
        if os.name == 'nt':
            import msvcrt
            key = msvcrt.getch()
            if key == b'\xe0':
                key = msvcrt.getch()
                if key == b'H': return 'up'
                if key == b'P': return 'down'
            if key == b'\r': return 'enter'
            if key == b'q': return 'quit'
            if key == b's': return 'select'
            return key.decode('utf-8', errors='ignore')
        else:
            import tty
            import termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == '\x1b':
                    ch2 = sys.stdin.read(2)
                    if ch2 == '[A': return 'up'
                    if ch2 == '[B': return 'down'
                if ch == '\r' or ch == '\n': return 'enter'
                if ch == 'q' or ch == '\x03': return 'quit'
                if ch == 's': return 'select'
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
    
    def _get_entries(self):
        """ディレクトリ内のエントリを取得"""
        entries = []
        
        # 親ディレクトリ
        if self.current_path.parent != self.current_path:
            entries.append(("📁 ..", "parent"))
        
        # ディレクトリ
        try:
            for p in sorted(self.current_path.iterdir()):
                if p.is_dir() and not p.name.startswith('.'):
                    entries.append((f"📁 {p.name}", ("dir", p)))
        except PermissionError:
            pass
        
        return entries
    
    def _render(self, entries):
        """ブラウザを描画"""
        self._clear_screen()
        
        print("\n" + "=" * 50)
        print(f"  📂 {self.title}")
        print("=" * 50)
        print(f"  現在: {self.current_path}\n")
        
        if not entries:
            print("    (空のディレクトリ)")
        else:
            start = max(0, self.selected - 10)
            end = min(len(entries), start + 20)
            
            for i in range(start, end):
                label, _ = entries[i]
                if i == self.selected:
                    print(f"  \033[7m > {label} \033[0m")
                else:
                    print(f"    {label}")
        
        print("\n" + "-" * 50)
        print("  ↑/↓: 選択  Enter: 開く  s: このフォルダを選択  q: キャンセル")
        print("-" * 50)
    
    def browse(self):
        """ブラウズして選択されたパスを返す"""
        while True:
            entries = self._get_entries()
            self.selected = min(self.selected, max(0, len(entries) - 1))
            
            self._render(entries)
            key = self._get_key()
            
            if key == 'up' and entries:
                self.selected = (self.selected - 1) % len(entries)
            elif key == 'down' and entries:
                self.selected = (self.selected + 1) % len(entries)
            elif key == 'enter' and entries:
                _, value = entries[self.selected]
                if value == "parent":
                    self.current_path = self.current_path.parent
                    self.selected = 0
                elif isinstance(value, tuple) and value[0] == "dir":
                    self.current_path = value[1]
                    self.selected = 0
            elif key == 'select':
                return str(self.current_path)
            elif key == 'quit':
                return None


class RecordingBrowser:
    """記録ファイルブラウザ"""
    
    def __init__(self, recordings_dir: str = "./recordings"):
        self.recordings_dir = Path(recordings_dir)
        self.selected = 0
    
    def _clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _get_key(self):
        """キー入力を取得"""
        if os.name == 'nt':
            import msvcrt
            key = msvcrt.getch()
            if key == b'\xe0':
                key = msvcrt.getch()
                if key == b'H': return 'up'
                if key == b'P': return 'down'
            if key == b'\r': return 'enter'
            if key == b'q': return 'quit'
            if key == b'd': return 'delete'
            return key.decode('utf-8', errors='ignore')
        else:
            import tty
            import termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == '\x1b':
                    ch2 = sys.stdin.read(2)
                    if ch2 == '[A': return 'up'
                    if ch2 == '[B': return 'down'
                if ch == '\r' or ch == '\n': return 'enter'
                if ch == 'q' or ch == '\x03': return 'quit'
                if ch == 'd': return 'delete'
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
    
    def _load_recordings(self):
        """記録一覧を読み込み"""
        import json
        
        recordings = []
        if not self.recordings_dir.exists():
            return recordings
        
        for meta_file in sorted(self.recordings_dir.glob("*_meta.json"), reverse=True):
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    meta["id"] = meta_file.stem.replace("_meta", "")
                    meta["path"] = str(meta_file.with_name(meta["id"] + ".jsonl"))
                    recordings.append(meta)
            except:
                pass
        
        return recordings
    
    def _render(self, recordings):
        """ブラウザを描画"""
        self._clear_screen()
        
        print("\n" + "=" * 60)
        print(f"  📹 記録一覧 ({self.recordings_dir})")
        print("=" * 60 + "\n")
        
        if not recordings:
            print("    記録がありません\n")
        else:
            start = max(0, self.selected - 8)
            end = min(len(recordings), start + 15)
            
            for i in range(start, end):
                r = recordings[i]
                created = r.get("created_at", "")[:16].replace("T", " ")
                duration = f"{r.get('duration_sec', 0):.1f}s"
                frames = r.get("total_frames", 0)
                source = r.get("source", "camera")[:8]
                
                info = f"{r['id'][:18]:<18} {created}  {duration:>7}  {frames:>5}f  {source}"
                
                if i == self.selected:
                    print(f"  \033[7m > {info} \033[0m")
                else:
                    print(f"    {info}")
        
        print("\n" + "-" * 60)
        print("  ↑/↓: 選択  Enter: 決定  d: 削除  q: 戻る")
        print("-" * 60)
    
    def browse(self):
        """ブラウズして選択された記録のパスを返す"""
        while True:
            recordings = self._load_recordings()
            
            if not recordings:
                self._render(recordings)
                key = self._get_key()
                if key in ('quit', 'enter'):
                    return None
                continue
            
            self.selected = min(self.selected, len(recordings) - 1)
            self._render(recordings)
            key = self._get_key()
            
            if key == 'up':
                self.selected = (self.selected - 1) % len(recordings)
            elif key == 'down':
                self.selected = (self.selected + 1) % len(recordings)
            elif key == 'enter':
                return recordings[self.selected]["path"]
            elif key == 'delete':
                self._delete_recording(recordings[self.selected])
            elif key == 'quit':
                return None
    
    def _delete_recording(self, recording):
        """記録を削除"""
        import os
        
        data_file = Path(recording["path"])
        meta_file = data_file.with_name(data_file.stem + "_meta.json")
        
        try:
            if data_file.exists():
                os.remove(data_file)
            if meta_file.exists():
                os.remove(meta_file)
        except:
            pass


class FACSInteractiveCLI:
    """インタラクティブCLI"""
    
    def __init__(self):
        self.recordings_dir = "./recordings"
    
    def run(self):
        """メインループ"""
        while True:
            menu = TerminalMenu("FACS Analyzer", [
                ("🎥 リアルタイム分析", "realtime"),
                ("📹 記録モード", "record"),
                ("▶️  記録を再生", "play"),
                ("🎬 MP4にエクスポート", "export"),
                ("📁 記録一覧", "list"),
                ("🌐 Webサーバー起動", "server"),
                ("🖼️  画像/動画を分析", "analyze"),
                ("⚙️  設定", "settings"),
                ("❌ 終了", "exit"),
            ])
            
            choice = menu.show()
            
            if choice == 'quit' or choice == 'exit':
                self._clear_screen()
                print("\n👋 終了します\n")
                break
            elif choice == 'realtime':
                self._realtime_menu()
            elif choice == 'record':
                self._record_menu()
            elif choice == 'play':
                self._play_menu()
            elif choice == 'export':
                self._export_menu()
            elif choice == 'list':
                self._list_recordings()
            elif choice == 'server':
                self._server_menu()
            elif choice == 'analyze':
                self._analyze_menu()
            elif choice == 'settings':
                self._settings_menu()
    
    def _clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _wait_key(self):
        """キー入力を待つ"""
        print("\n[Enter]で続行...")
        input()
    
    def _realtime_menu(self):
        """リアルタイム分析メニュー"""
        menu = TerminalMenu("リアルタイム分析 - モード選択", [
            ("⚡ 高速モード (fast)", "fast"),
            ("⚖️  バランスモード (balanced)", "balanced"),
            ("🎯 高精度モード (accurate)", "accurate"),
        ], show_back=True)
        
        mode = menu.show()
        if mode == 'back' or mode == 'quit':
            return
        
        # カメラ選択
        camera_menu = TerminalMenu("カメラ選択", [
            ("カメラ 0 (デフォルト)", 0),
            ("カメラ 1", 1),
            ("カメラ 2", 2),
        ], show_back=True)
        
        camera = camera_menu.show()
        if camera == 'back' or camera == 'quit':
            return
        
        self._clear_screen()
        print(f"\n🎥 リアルタイム分析を開始します...")
        print(f"   モード: {mode}")
        print(f"   カメラ: {camera}")
        print("\n   'q'で終了\n")
        
        self._run_realtime(camera, mode)
    
    def _run_realtime(self, camera: int, mode: str):
        """リアルタイム分析を実行"""
        from facs import FACSAnalyzer
        from facs.core.enums import AnalysisMode
        
        mode_map = {
            'fast': AnalysisMode.REALTIME,
            'balanced': AnalysisMode.BALANCED,
            'accurate': AnalysisMode.ACCURATE,
        }
        
        analyzer = FACSAnalyzer(mode=mode_map.get(mode, AnalysisMode.REALTIME))
        analyzer.analyze_realtime(camera)
    
    def _record_menu(self):
        """記録モードメニュー"""
        # 出力ディレクトリ選択
        menu = TerminalMenu("記録モード - 保存先", [
            (f"📁 デフォルト ({self.recordings_dir})", "default"),
            ("📂 フォルダを選択...", "browse"),
        ], show_back=True)
        
        choice = menu.show()
        if choice == 'back' or choice == 'quit':
            return
        
        if choice == 'browse':
            browser = DirectoryBrowser(".", "保存先フォルダを選択")
            output_dir = browser.browse()
            if not output_dir:
                return
        else:
            output_dir = self.recordings_dir
        
        # 記録時間選択
        duration_menu = TerminalMenu("最大記録時間", [
            ("1分", 60),
            ("5分", 300),
            ("10分", 600),
            ("30分", 1800),
            ("無制限 (手動停止)", 99999),
        ], show_back=True)
        
        duration = duration_menu.show()
        if duration == 'back' or duration == 'quit':
            return
        
        self._clear_screen()
        print(f"\n📹 記録を開始します...")
        print(f"   保存先: {output_dir}")
        print(f"   最大時間: {duration}秒")
        print("\n   'q'で停止\n")
        
        self._run_record(output_dir, duration)
    
    def _run_record(self, output_dir: str, duration: int):
        """記録を実行"""
        import cv2
        from facs import FACSAnalyzer
        from facs.recording import FACSRecorder
        from facs.visualization import FACSVisualizer
        from facs.core.enums import AnalysisMode
        
        analyzer = FACSAnalyzer(mode=AnalysisMode.REALTIME)
        visualizer = FACSVisualizer()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        recorder = FACSRecorder(output_dir)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("エラー: カメラを開けません")
            self._wait_key()
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        recorder.start(fps=fps, width=width, height=height)
        
        try:
            while recorder.elapsed_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result = analyzer.analyze(frame)
                recorder.record_frame(result)
                
                vis_frame = visualizer.create_analysis_panel(frame, result)
                
                cv2.circle(vis_frame, (30, 30), 10, (0, 0, 255), -1)
                info = f"REC {recorder.frame_count}f / {recorder.elapsed_time:.1f}s"
                cv2.putText(vis_frame, info, (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imshow("FACS Recording", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            metadata = recorder.stop()
            cap.release()
            cv2.destroyAllWindows()
            
            self._clear_screen()
            print(f"\n✅ 記録完了!")
            print(f"   ファイル: {recorder.data_path}")
            print(f"   フレーム数: {metadata.total_frames}")
            print(f"   時間: {metadata.duration_sec:.1f}秒")
            self._wait_key()
    
    def _play_menu(self):
        """再生メニュー"""
        browser = RecordingBrowser(self.recordings_dir)
        path = browser.browse()
        
        if not path:
            return
        
        self._clear_screen()
        print(f"\n▶️  再生: {path}")
        print("\n   Space: 再生/停止")
        print("   ←/→: フレーム移動")
        print("   ↑/↓: 速度変更")
        print("   q: 終了\n")
        
        self._run_play(path)
    
    def _run_play(self, path: str):
        """再生を実行"""
        import cv2
        import numpy as np
        from facs.recording import FACSPlayer, PlaybackState
        from facs.visualization import FACSVisualizer
        
        player = FACSPlayer(path)
        visualizer = FACSVisualizer()
        
        current_result = player.get_frame(0)
        
        while True:
            info = player.playback_info
            
            if info.state == PlaybackState.PLAYING:
                current_result = player.step_forward()
                if player.current_frame >= player.total_frames - 1:
                    player.pause()
            
            frame = self._create_playback_frame(player, current_result)
            vis_frame = visualizer.create_analysis_panel(frame, current_result)
            self._draw_playback_overlay(vis_frame, info)
            
            cv2.imshow("FACS Playback", vis_frame)
            
            wait_time = max(1, int(1000 / player.fps / info.speed))
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                player.toggle_play_pause()
            elif key in (81, 2, ord('a')):
                player.pause()
                player.step_backward()
                current_result = player.get_frame(player.current_frame)
            elif key in (83, 3, ord('d')):
                player.pause()
                player.step_forward()
                current_result = player.get_frame(player.current_frame)
            elif key in (82, 0, ord('w')):
                player.set_speed(info.speed * 1.5)
            elif key in (84, 1, ord('s')):
                player.set_speed(info.speed / 1.5)
            elif ord('0') <= key <= ord('9'):
                player.seek_progress((key - ord('0')) / 10)
                current_result = player.get_frame(player.current_frame)
        
        player.stop()
        cv2.destroyAllWindows()
    
    def _create_playback_frame(self, player, result):
        """再生フレームを作成"""
        import numpy as np
        import cv2
        
        w, h = player.metadata.width, player.metadata.height
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)
        
        if result and result.face_data and result.face_data.landmarks is not None:
            landmarks = result.face_data.landmarks
            for x, y in landmarks:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        return frame
    
    def _draw_playback_overlay(self, frame, info):
        """再生オーバーレイを描画"""
        import cv2
        from facs.recording import PlaybackState
        
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
        
        bar_x, bar_w = 100, w - 200
        bar_y = h - 25
        
        cv2.rectangle(frame, (bar_x, bar_y - 3), (bar_x + bar_w, bar_y + 3), (60, 60, 80), -1)
        progress_w = int(bar_w * info.progress)
        cv2.rectangle(frame, (bar_x, bar_y - 3), (bar_x + progress_w, bar_y + 3), (0, 200, 255), -1)
        
        state = ">" if info.state == PlaybackState.PLAYING else "||"
        cv2.putText(frame, state, (20, bar_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        time_text = f"{info.current_time:.1f}s / {info.total_time:.1f}s  x{info.speed:.1f}"
        cv2.putText(frame, time_text, (50, bar_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def _export_menu(self):
        """エクスポートメニュー"""
        browser = RecordingBrowser(self.recordings_dir)
        path = browser.browse()
        
        if not path:
            return
        
        # 解像度選択
        res_menu = TerminalMenu("出力解像度", [
            ("720p (1280x720)", (1280, 720)),
            ("1080p (1920x1080)", (1920, 1080)),
            ("480p (854x480)", (854, 480)),
        ], show_back=True)
        
        resolution = res_menu.show()
        if resolution == 'back' or resolution == 'quit':
            return
        
        self._clear_screen()
        print(f"\n🎬 エクスポート中...")
        print(f"   入力: {path}")
        print(f"   解像度: {resolution[0]}x{resolution[1]}")
        
        self._run_export(path, resolution)
    
    def _run_export(self, path: str, resolution: tuple):
        """エクスポートを実行"""
        from facs.recording import FACSVideoExporter
        
        exporter = FACSVideoExporter(width=resolution[0], height=resolution[1])
        output = exporter.export(path)
        
        print(f"\n✅ 完了: {output}")
        self._wait_key()
    
    def _list_recordings(self):
        """記録一覧を表示"""
        browser = RecordingBrowser(self.recordings_dir)
        browser.browse()
    
    def _server_menu(self):
        """サーバーメニュー"""
        menu = TerminalMenu("Webサーバー設定", [
            ("🔓 HTTP (ポート8000)", ("http", 8000)),
            ("🔒 HTTPS (ポート8443)", ("https", 8443)),
        ], show_back=True)
        
        choice = menu.show()
        if choice == 'back' or choice == 'quit':
            return
        
        protocol, port = choice
        
        self._clear_screen()
        print(f"\n🌐 Webサーバーを起動します...")
        print(f"   プロトコル: {protocol.upper()}")
        print(f"   ポート: {port}")
        print("\n   Ctrl+Cで停止\n")
        
        from web.server import FACSWebServer
        server = FACSWebServer(recordings_dir=self.recordings_dir)
        server.run(port=port, use_https=(protocol == "https"))
    
    def _analyze_menu(self):
        """分析メニュー"""
        self._clear_screen()
        print("\n🖼️  画像/動画を分析")
        print("-" * 40)
        
        path = input("\nファイルパスを入力: ").strip()
        if not path:
            return
        
        if not Path(path).exists():
            print(f"\nエラー: ファイルが見つかりません: {path}")
            self._wait_key()
            return
        
        self._run_analyze(path)
    
    def _run_analyze(self, path: str):
        """分析を実行"""
        import cv2
        from facs import FACSAnalyzer
        
        p = Path(path)
        analyzer = FACSAnalyzer()
        
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        
        if p.suffix.lower() in image_exts:
            image = cv2.imread(str(p))
            if image is None:
                print(f"\nエラー: 画像を読み込めません")
                self._wait_key()
                return
            
            result = analyzer.analyze(image)
            
            self._clear_screen()
            print(f"\n🎭 分析結果: {p.name}")
            print("-" * 40)
            print(f"FACSコード: {result.facs_code}")
            if result.dominant_emotion:
                print(f"感情: {result.dominant_emotion.emotion} ({result.dominant_emotion.confidence:.0%})")
            print(f"Valence: {result.valence:+.2f}")
            print(f"Arousal: {result.arousal:+.2f}")
            
            vis = analyzer.visualize(image, result)
            cv2.imshow("FACS Analysis", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("\n動画分析は時間がかかります...")
            output = str(p.stem) + "_analyzed.mp4"
            results = analyzer.analyze_video(str(p), output, frame_skip=2)
            print(f"\n✅ 完了: {len(results)}フレーム")
            print(f"   出力: {output}")
            self._wait_key()
    
    def _settings_menu(self):
        """設定メニュー"""
        while True:
            menu = TerminalMenu("設定", [
                (f"📁 記録ディレクトリ: {self.recordings_dir}", "recordings_dir"),
            ], show_back=True)
            
            choice = menu.show()
            if choice == 'back' or choice == 'quit':
                break
            elif choice == 'recordings_dir':
                browser = DirectoryBrowser(self.recordings_dir, "記録ディレクトリを選択")
                new_dir = browser.browse()
                if new_dir:
                    self.recordings_dir = new_dir


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(description="FACS Analyzer CLI")
    parser.add_argument("command", nargs="?", help="コマンド (省略時: インタラクティブモード)")
    
    args, remaining = parser.parse_known_args()
    
    if args.command is None:
        # インタラクティブモード
        cli = FACSInteractiveCLI()
        cli.run()
    else:
        # コマンドモード（後方互換性）
        print(f"コマンド '{args.command}' はインタラクティブモードで実行してください")
        print("python facs_cli.py")


if __name__ == "__main__":
    main()