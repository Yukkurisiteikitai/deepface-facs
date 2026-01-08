"""
FACS (Facial Action Coding System) デモスクリプト

使用方法:
    python demo.py --image <画像パス>        # 画像分析
    python demo.py --video <動画パス>        # 動画分析
    python demo.py --realtime                 # リアルタイム分析
    python demo.py --list-aus                 # AU一覧表示
"""

import argparse
import cv2
import sys
import os

# 親ディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from facs_analyzer import FACSAnalyzer, FACSAnalysisResult

def analyze_image(image_path: str, output_path: str = None):
    """画像を分析"""
    print(f"画像を分析中: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"エラー: 画像を読み込めません: {image_path}")
        return
    
    analyzer = FACSAnalyzer(use_mediapipe=True)
    result = analyzer.analyze(image)
    
    if result.landmarks is None:
        print("顔が検出されませんでした")
        return
    
    # 結果を表示
    print("\n" + "=" * 50)
    print("FACS分析結果")
    print("=" * 50)
    print(f"\nFACSコード: {result.facs_code}")
    print(f"\n処理時間: {result.processing_time_ms:.1f}ms")
    
    print("\n--- 検出されたAction Units ---")
    for au_num, au_result in sorted(result.au_results.items()):
        if au_result.detected:
            intensity = result.intensity_results.get(au_num)
            intensity_str = f"[{intensity.intensity_label}]" if intensity else ""
            print(f"  AU{au_num}{intensity_str}: {au_result.name} "
                  f"(confidence: {au_result.confidence:.2f})")
    
    print("\n--- 感情推定 ---")
    for emotion in result.emotions[:5]:
        if emotion.confidence > 0.1:
            print(f"  {emotion.emotion}: {emotion.confidence:.2f} "
                  f"(V: {emotion.valence:+.2f}, A: {emotion.arousal:+.2f})")
    
    if result.dominant_emotion:
        print(f"\n主要感情: {result.dominant_emotion.emotion}")
        print(f"Valence: {result.valence:+.2f}, Arousal: {result.arousal:+.2f}")
    
    # 非対称性
    if result.asymmetry_info.get("is_asymmetric"):
        print("\n--- 非対称性検出 ---")
        for au, info in result.asymmetry_info.get("asymmetric_aus", {}).items():
            print(f"  {info['description']}")
        if result.asymmetry_info.get("possible_contempt"):
            print("  ⚠ 軽蔑表情の可能性あり")
    
    # 可視化
    vis_image = analyzer.visualize(image, result)
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"\n結果を保存しました: {output_path}")
    
    # 表示
    cv2.imshow("FACS Analysis", vis_image)
    print("\n何かキーを押すと終了します...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # JSON出力
    print("\n--- JSON出力 ---")
    print(result.to_json())

def analyze_video(video_path: str, output_path: str = None):
    """動画を分析"""
    print(f"動画を分析中: {video_path}")
    
    analyzer = FACSAnalyzer(use_mediapipe=True)
    
    def progress_callback(frame, total, result):
        emotion = result.dominant_emotion.emotion if result.dominant_emotion else "N/A"
        print(f"\rFrame {frame}/{total} - Emotion: {emotion:15s}", end="")
    
    results = analyzer.analyze_video(
        video_path,
        output_path=output_path,
        frame_skip=2,
        callback=progress_callback
    )
    
    print(f"\n\n分析完了: {len(results)} フレーム")
    
    if output_path:
        print(f"結果を保存しました: {output_path}")
    
    # 統計
    emotion_counts = {}
    for r in results:
        if r.dominant_emotion:
            e = r.dominant_emotion.emotion
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
    
    print("\n--- 感情の分布 ---")
    total = len(results)
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {emotion}: {count} ({pct:.1f}%)")

def realtime_analysis():
    """リアルタイム分析"""
    print("リアルタイム分析を開始します...")
    print("操作方法:")
    print("  'q': 終了")
    print("  'c': 中立表情をキャリブレーション")
    print()
    
    analyzer = FACSAnalyzer(use_mediapipe=True)
    analyzer.analyze_realtime()

def list_aus():
    """AU一覧を表示"""
    aus = FACSAnalyzer.list_all_aus()
    
    print("\n" + "=" * 60)
    print("Facial Action Coding System - Action Units")
    print("=" * 60)
    
    for au in aus:
        print(f"\nAU{au['number']:2d}: {au['name']}")
        print(f"      {au['description']}")

def main():
    parser = argparse.ArgumentParser(
        description="FACS (Facial Action Coding System) 分析ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--image", "-i", type=str, help="分析する画像のパス")
    parser.add_argument("--video", "-v", type=str, help="分析する動画のパス")
    parser.add_argument("--output", "-o", type=str, help="出力ファイルのパス")
    parser.add_argument("--realtime", "-r", action="store_true", help="リアルタイム分析")
    parser.add_argument("--list-aus", "-l", action="store_true", help="AU一覧を表示")
    parser.add_argument("--use-dlib", action="store_true", help="dlibを使用（デフォルトはMediaPipe）")
    
    args = parser.parse_args()
    
    if args.list_aus:
        list_aus()
    elif args.image:
        analyze_image(args.image, args.output)
    elif args.video:
        output = args.output or str(Path(args.video).stem) + "_analyzed.mp4"
        analyze_video(args.video, output)
    elif args.realtime:
        realtime_analysis()
    else:
        parser.print_help()
        print("\n例:")
        print("  python demo.py --image face.jpg")
        print("  python demo.py --video interview.mp4 --output result.mp4")
        print("  python demo.py --realtime")
        print("  python demo.py --list-aus")

if __name__ == "__main__":
    main()
