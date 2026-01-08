"""
FACS Analyzer CLI
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        prog="facs",
        description="FACS (Facial Action Coding System) Analyzer",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="コマンド")
    
    # analyze コマンド
    analyze_parser = subparsers.add_parser("analyze", aliases=["a"], help="画像/動画を分析")
    analyze_parser.add_argument("input", type=str, help="入力ファイルパス")
    analyze_parser.add_argument("-o", "--output", type=str, help="出力ファイルパス")
    analyze_parser.add_argument("-j", "--json", action="store_true", help="JSON出力")
    analyze_parser.add_argument("--no-display", action="store_true", help="表示しない")
    
    # realtime コマンド
    rt_parser = subparsers.add_parser("realtime", aliases=["r"], help="リアルタイム分析")
    rt_parser.add_argument("-c", "--camera", type=int, default=0, help="カメラID")
    
    # list コマンド
    subparsers.add_parser("list", aliases=["l"], help="AU一覧")
    
    # version コマンド
    subparsers.add_parser("version", aliases=["v"], help="バージョン表示")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command in ("version", "v"):
        from facs import __version__
        print(f"facs-analyzer {__version__}")
        return 0
    
    if args.command in ("list", "l"):
        from facs import FACSAnalyzer
        for au in FACSAnalyzer.list_all_aus():
            print(f"AU{au['number']:2d}: {au['name']} - {au['description']}")
        return 0
    
    if args.command in ("realtime", "r"):
        from facs import FACSAnalyzer
        analyzer = FACSAnalyzer()
        analyzer.analyze_realtime(args.camera)
        return 0
    
    if args.command in ("analyze", "a"):
        return _analyze(args)
    
    return 0

def _analyze(args) -> int:
    """分析を実行"""
    import cv2
    from facs import FACSAnalyzer
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ファイルが見つかりません: {args.input}", file=sys.stderr)
        return 1
    
    analyzer = FACSAnalyzer()
    
    # 画像か動画か判定
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    
    if input_path.suffix.lower() in image_exts:
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"画像を読み込めません: {args.input}", file=sys.stderr)
            return 1
        
        result = analyzer.analyze(image)
        
        if args.json:
            print(result.to_json())
        else:
            print(f"FACSコード: {result.facs_code}")
            if result.dominant_emotion:
                print(f"感情: {result.dominant_emotion.emotion} ({result.dominant_emotion.confidence:.2f})")
            print(f"Valence: {result.valence:.2f}, Arousal: {result.arousal:.2f}")
        
        if args.output:
            vis = analyzer.visualize(image, result)
            cv2.imwrite(args.output, vis)
            print(f"保存: {args.output}")
        
        if not args.no_display:
            vis = analyzer.visualize(image, result)
            cv2.imshow("FACS Analysis", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif input_path.suffix.lower() in video_exts:
        output = args.output or str(input_path.stem) + "_analyzed.mp4"
        results = analyzer.analyze_video(str(input_path), output)
        print(f"分析完了: {len(results)} フレーム")
        print(f"出力: {output}")
    
    else:
        print(f"未対応のファイル形式: {input_path.suffix}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
