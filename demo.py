"""
FACS (Facial Action Coding System) デモスクリプト

使用例:
    python demo.py image face.jpg              # 画像分析
    python demo.py video video.mp4             # 動画分析
    python demo.py realtime                    # リアルタイム分析
    python demo.py realtime --parallel         # 並列処理リアルタイム分析
    python demo.py list                        # AU一覧表示
    python demo.py compare img1.jpg img2.jpg   # 2つの表情を比較
    python demo.py batch ./images/             # フォルダ内の画像を一括分析
"""

import argparse
import sys
import cv2
import json
import csv
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from facs import FACSAnalyzer, AnalysisResult, TerminalDisplay, InteractiveFACSVisualizer, AnalysisMode
from facs import ParallelFACSProcessor, run_parallel_realtime
from facs.visualization.visualizer import LayoutConfig

class FACSDemo:
    """FACSデモアプリケーション"""
    
    def __init__(self, use_mediapipe: bool = True, interactive: bool = True,
                 mode: AnalysisMode = AnalysisMode.BALANCED,
                 layout_config: Optional[LayoutConfig] = None):
        self.analyzer = FACSAnalyzer(
            use_mediapipe=use_mediapipe,
            interactive=interactive,
            mode=mode,
            layout_config=layout_config
        )
        self.terminal = TerminalDisplay(use_colors=True)
        self._interactive = interactive
        self._colors = {
            'positive': '\033[92m',
            'negative': '\033[91m',
            'neutral': '\033[93m',
            'reset': '\033[0m',
            'bold': '\033[1m',
            'dim': '\033[2m'
        }
    
    def _color(self, text: str, color: str) -> str:
        """テキストに色を付ける"""
        return f"{self._colors.get(color, '')}{text}{self._colors['reset']}"
    
    def _print_header(self, title: str):
        """ヘッダーを表示"""
        width = 60
        print("\n" + "=" * width)
        print(f"{title:^{width}}")
        print("=" * width)
    
    def _print_result(self, result, verbose: bool = True):
        """分析結果を表示"""
        self.terminal.print_full_analysis(result, show_au_details=verbose)
    
    def analyze_image(self, image_path: str, output_path: Optional[str] = None, 
                      save_json: bool = False, verbose: bool = True,
                      show_au_details: bool = True):
        """画像を分析"""
        self.terminal.print_header(f"画像分析: {Path(image_path).name}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(self.terminal._c(f"エラー: 画像を読み込めません: {image_path}", 'red'))
            sys.exit(1)
        
        result = self.analyzer.analyze(image)
        
        # ターミナルに詳細表示
        self.terminal.print_full_analysis(result, show_au_details=show_au_details)
        
        # 保存
        if output_path:
            vis_image = self.analyzer.visualize(image, result)
            cv2.imwrite(output_path, vis_image)
            print(f"\n{self.terminal._c('保存:', 'dim')} {output_path}")
        
        if save_json:
            json_path = Path(image_path).stem + "_facs.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write(result.to_json())
            print(f"{self.terminal._c('JSON保存:', 'dim')} {json_path}")
        
        # インタラクティブ表示
        if self._interactive:
            print(f"\n{self.terminal._c('マウスをAUにホバーすると詳細が表示されます', 'cyan')}")
            print(f"{self.terminal._c('Enterで終了, Sで保存', 'dim')}")
            
            vis_image = self.analyzer.show_interactive(image, result)
        else:
            vis_image = self.analyzer.visualize(image, result)
            cv2.imshow("FACS Analysis", vis_image)
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == 13 or key == 27:  # Enter or ESC
                break
            elif key == ord('s'):
                save_path = f"facs_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                # 現在の表示を保存
                current_vis = self.analyzer.visualize(image, result)
                cv2.imwrite(save_path, current_vis)
                print(f"保存しました: {save_path}")
            elif key == ord('l'):
                # 凡例を表示
                self.terminal.print_au_legend()
        
        cv2.destroyAllWindows()
        return result
    
    def analyze_video(self, video_path: str, output_path: Optional[str] = None,
                      frame_skip: int = 2, save_csv: bool = False):
        """動画を分析"""
        self._print_header(f"動画分析: {Path(video_path).name}")
        
        if output_path is None:
            output_path = str(Path(video_path).stem) + "_analyzed.mp4"
        
        results = self.analyzer.analyze_video(video_path, output_path, frame_skip)
        
        print(f"\n分析完了: {len(results)} フレーム")
        print(f"出力: {output_path}")
        
        # 統計
        self._print_video_statistics(results)
        
        # CSV保存
        if save_csv:
            csv_path = Path(video_path).stem + "_facs.csv"
            self._save_results_csv(results, csv_path)
            print(f"\nCSV保存: {csv_path}")
        
        return results
    
    def _print_video_statistics(self, results: List[AnalysisResult]):
        """動画分析の統計を表示"""
        emotion_counts = {}
        valence_sum, arousal_sum, valid_count = 0.0, 0.0, 0
        
        for r in results:
            if r.is_valid:
                valid_count += 1
                valence_sum += r.valence
                arousal_sum += r.arousal
                if r.dominant_emotion:
                    e = r.dominant_emotion.emotion
                    emotion_counts[e] = emotion_counts.get(e, 0) + 1
        
        print(f"\n{self._color('【統計】', 'bold')}")
        print(f"  有効フレーム: {valid_count}/{len(results)}")
        
        if valid_count > 0:
            print(f"  平均 Valence: {valence_sum/valid_count:+.2f}")
            print(f"  平均 Arousal: {arousal_sum/valid_count:+.2f}")
        
        print(f"\n{self._color('【感情分布】', 'bold')}")
        total = sum(emotion_counts.values())
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"  {emotion:12s} {bar:20s} {count:4d} ({pct:5.1f}%)")
    
    def _save_results_csv(self, results: List[AnalysisResult], csv_path: str):
        """結果をCSVに保存"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'timestamp', 'facs_code', 'emotion', 'confidence', 'valence', 'arousal', 'active_aus'])
            
            for i, r in enumerate(results):
                writer.writerow([
                    i,
                    r.timestamp,
                    r.facs_code,
                    r.dominant_emotion.emotion if r.dominant_emotion else 'N/A',
                    r.dominant_emotion.confidence if r.dominant_emotion else 0,
                    r.valence,
                    r.arousal,
                    ','.join(f"AU{au.au_number}" for au in r.active_aus)
                ])
    
    def realtime_analysis(self, camera_id: int = 0, use_parallel: bool = False, 
                          num_workers: int = 1):
        """リアルタイム分析"""
        self._print_header("リアルタイム分析")
        print("操作方法:")
        print("  q: 終了")
        print("  s: スクリーンショット保存")
        print("  r: 結果をJSON保存")
        if use_parallel:
            print(f"  並列処理モード（ワーカー数: {num_workers}）")
        print()
        
        if use_parallel:
            # 並列処理モード
            self._realtime_parallel(camera_id, num_workers)
        else:
            # 通常モード
            self.analyzer.analyze_realtime(camera_id)
    
    def _realtime_parallel(self, camera_id: int = 0, num_workers: int = 1):
        """並列処理でリアルタイム分析を実行"""
        mode_str = 'realtime'  # デフォルトはリアルタイムモード
        
        with ParallelFACSProcessor(
            use_mediapipe=True,
            mode=mode_str,
            num_workers=num_workers
        ) as processor:
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                print(self._color("エラー: カメラを開けません", 'negative'))
                return
            
            print("Press 'q' to quit, 's' to save screenshot")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result, vis = processor.process_and_visualize(frame)
                
                cv2.imshow("FACS Parallel Realtime", vis)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    save_path = f"facs_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(save_path, vis)
                    print(f"保存しました: {save_path}")
            
            cap.release()
            cv2.destroyAllWindows()
    
    def compare_images(self, image_path1: str, image_path2: str):
        """2つの画像を比較"""
        self._print_header("表情比較")
        
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)
        
        if img1 is None or img2 is None:
            print(self._color("エラー: 画像を読み込めません", 'negative'))
            sys.exit(1)
        
        result1 = self.analyzer.analyze(img1)
        result2 = self.analyzer.analyze(img2)
        
        print(f"\n{self._color('【画像1】', 'bold')} {Path(image_path1).name}")
        self._print_result(result1, verbose=False)
        
        print(f"\n{self._color('【画像2】', 'bold')} {Path(image_path2).name}")
        self._print_result(result2, verbose=False)
        
        # 比較
        if result1.is_valid and result2.is_valid:
            print(f"\n{self._color('【変化】', 'bold')}")
            
            aus1 = {au.au_number for au in result1.active_aus}
            aus2 = {au.au_number for au in result2.active_aus}
            
            added = aus2 - aus1
            removed = aus1 - aus2
            
            if added:
                print(f"  追加されたAU: {', '.join(f'AU{au}' for au in sorted(added))}")
            if removed:
                print(f"  消えたAU: {', '.join(f'AU{au}' for au in sorted(removed))}")
            
            v_diff = result2.valence - result1.valence
            a_diff = result2.arousal - result1.arousal
            print(f"  Valence変化: {v_diff:+.2f}")
            print(f"  Arousal変化: {a_diff:+.2f}")
            
            if result1.dominant_emotion and result2.dominant_emotion:
                if result1.dominant_emotion.emotion != result2.dominant_emotion.emotion:
                    print(f"  感情変化: {result1.dominant_emotion.emotion} → {result2.dominant_emotion.emotion}")
        
        # 並べて表示
        vis1 = self.analyzer.visualize(img1, result1)
        vis2 = self.analyzer.visualize(img2, result2)
        
        # 高さを揃える
        h1, h2 = vis1.shape[0], vis2.shape[0]
        if h1 != h2:
            target_h = max(h1, h2)
            if h1 < target_h:
                vis1 = cv2.copyMakeBorder(vis1, 0, target_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            if h2 < target_h:
                vis2 = cv2.copyMakeBorder(vis2, 0, target_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        
        combined = cv2.hconcat([vis1, vis2])
        cv2.imshow("FACS Comparison", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def batch_analyze(self, folder_path: str, output_folder: Optional[str] = None):
        """フォルダ内の画像を一括分析"""
        self._print_header(f"一括分析: {folder_path}")
        
        folder = Path(folder_path)
        if not folder.is_dir():
            print(self._color(f"エラー: フォルダが見つかりません: {folder_path}", 'negative'))
            sys.exit(1)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
        
        if not images:
            print("画像が見つかりません")
            return
        
        print(f"画像数: {len(images)}")
        
        if output_folder:
            out_path = Path(output_folder)
            out_path.mkdir(parents=True, exist_ok=True)
        
        results_summary = []
        
        for i, img_path in enumerate(images, 1):
            print(f"\r処理中: {i}/{len(images)} - {img_path.name}", end="")
            
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            result = self.analyzer.analyze(image)
            
            results_summary.append({
                'file': img_path.name,
                'valid': result.is_valid,
                'facs_code': result.facs_code,
                'emotion': result.dominant_emotion.emotion if result.dominant_emotion else 'N/A',
                'valence': result.valence,
                'arousal': result.arousal
            })
            
            if output_folder and result.is_valid:
                vis = self.analyzer.visualize(image, result)
                cv2.imwrite(str(out_path / f"analyzed_{img_path.name}"), vis)
        
        print("\n")
        
        # サマリー
        valid_count = sum(1 for r in results_summary if r['valid'])
        print(f"\n{self._color('【サマリー】', 'bold')}")
        print(f"  処理: {len(results_summary)}, 成功: {valid_count}")
        
        # CSVに保存
        csv_path = folder / "batch_results.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'valid', 'facs_code', 'emotion', 'valence', 'arousal'])
            writer.writeheader()
            writer.writerows(results_summary)
        print(f"  結果保存: {csv_path}")
    
    def list_aus(self):
        """AU一覧を表示"""
        self._print_header("Action Units 一覧")
        
        aus = FACSAnalyzer.list_all_aus()
        
        for au in aus:
            au_num = au["number"]
            au_name = au['name']
            print(f"\n{self._color(f'AU{au_num:2d}', 'bold')}: {au_name}")
            print(f"      {self._color(au['description'], 'dim')}")
            print(f"      筋肉: {au['muscular_basis']}")


def main():
    parser = argparse.ArgumentParser(
        description="FACS (Facial Action Coding System) 分析ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='コマンド')
    
    # 共通オプションを各サブパーサーに追加する関数
    def add_common_options(subparser):
        subparser.add_argument('--no-interactive', action='store_true', help='インタラクティブモードを無効化')
        subparser.add_argument('--no-details', action='store_true', help='AU詳細表示を無効化')
        subparser.add_argument('--mode', '-m', type=str, choices=['realtime', 'balanced', 'accurate'],
                              default='balanced', help='分析モード')
        subparser.add_argument('--face-height', type=int, default=400,
                              help='顔画像の目標高さ (デフォルト: 400)')
        subparser.add_argument('--panel-width', type=int, default=420,
                              help='パネル幅 (デフォルト: 420)')

    # image コマンド
    img_parser = subparsers.add_parser('image', aliases=['i'], help='画像を分析')
    img_parser.add_argument('path', type=str, help='画像パス')
    img_parser.add_argument('-o', '--output', type=str, help='出力画像パス')
    img_parser.add_argument('-j', '--json', action='store_true', help='JSONを保存')
    img_parser.add_argument('-q', '--quiet', action='store_true', help='簡易出力')
    add_common_options(img_parser)
    
    # video コマンド
    vid_parser = subparsers.add_parser('video', aliases=['v'], help='動画を分析')
    vid_parser.add_argument('path', type=str, help='動画パス')
    vid_parser.add_argument('-o', '--output', type=str, help='出力動画パス')
    vid_parser.add_argument('-s', '--skip', type=int, default=2, help='フレームスキップ数')
    vid_parser.add_argument('-c', '--csv', action='store_true', help='CSVを保存')
    add_common_options(vid_parser)
    
    # realtime コマンド
    rt_parser = subparsers.add_parser('realtime', aliases=['r'], help='リアルタイム分析')
    rt_parser.add_argument('-c', '--camera', type=int, default=0, help='カメラID')
    rt_parser.add_argument('--parallel', '-p', action='store_true', 
                          help='並列処理モードを有効化（推論と描画を分離）')
    rt_parser.add_argument('--workers', '-w', type=int, default=1,
                          help='ワーカープロセス数 (デフォルト: 1)')
    add_common_options(rt_parser)
    
    # compare コマンド
    cmp_parser = subparsers.add_parser('compare', aliases=['c'], help='2つの画像を比較')
    cmp_parser.add_argument('image1', type=str, help='画像1')
    cmp_parser.add_argument('image2', type=str, help='画像2')
    add_common_options(cmp_parser)
    
    # batch コマンド
    batch_parser = subparsers.add_parser('batch', aliases=['b'], help='フォルダ内を一括分析')
    batch_parser.add_argument('folder', type=str, help='フォルダパス')
    batch_parser.add_argument('-o', '--output', type=str, help='出力フォルダ')
    add_common_options(batch_parser)
    
    # list コマンド
    subparsers.add_parser('list', aliases=['l'], help='AU一覧を表示')
    
    # legend コマンド
    subparsers.add_parser('legend', help='AU強度の凡例を表示')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        print("\n使用例:")
        print("  python demo.py image face.jpg")
        print("  python demo.py image face.jpg --mode accurate  # 高精度モード")
        print("  python demo.py image face.jpg --mode realtime  # 高速モード")
        print("  python demo.py video interview.mp4 --csv")
        print("  python demo.py realtime")
        print("  python demo.py compare happy.jpg sad.jpg")
        print("  python demo.py realtime --parallel        # 並列処理モード")
        print("  python demo.py realtime --parallel -w 2   # ワーカー2つで並列処理")
        print("  python demo.py legend")
        print("  python demo.py list")
        return
    
    interactive = not getattr(args, 'no_interactive', False)
    show_details = not getattr(args, 'no_details', False)
    
    # モードを解析
    mode_map = {
        'realtime': AnalysisMode.REALTIME,
        'balanced': AnalysisMode.BALANCED,
        'accurate': AnalysisMode.ACCURATE,
    }
    mode = mode_map.get(getattr(args, 'mode', 'balanced'), AnalysisMode.BALANCED)
    
    # レイアウト設定
    layout_config = LayoutConfig(
        target_face_height=getattr(args, 'face_height', 400),
        panel_width=getattr(args, 'panel_width', 420),
    )
    
    demo = FACSDemo(interactive=interactive, mode=mode, layout_config=layout_config)
    
    if args.command in ('image', 'i'):
        demo.analyze_image(args.path, args.output, args.json, 
                          not args.quiet, show_au_details=show_details)
    elif args.command in ('video', 'v'):
        demo.analyze_video(args.path, args.output, args.skip, args.csv)
    elif args.command in ('realtime', 'r'):
        use_parallel = getattr(args, 'parallel', False)
        num_workers = getattr(args, 'workers', 1)
        demo.realtime_analysis(args.camera, use_parallel=use_parallel, num_workers=num_workers)
    elif args.command in ('compare', 'c'):
        demo.compare_images(args.image1, args.image2)
    elif args.command in ('batch', 'b'):
        demo.batch_analyze(args.folder, args.output)
    elif args.command in ('list', 'l'):
        demo.list_aus()
    elif args.command == 'legend':
        demo.terminal.print_au_legend()


if __name__ == "__main__":
    main()
