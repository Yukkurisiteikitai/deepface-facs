"""
DeepFace統合のテストスクリプト
"""
import cv2
import sys
from facs.detectors import DeepFaceAnalyzer, DeepFaceLandmarkConverter
from facs.detectors.debug_landmarks import visualize_landmarks_debug

def main():
    if len(sys.argv) < 2:
        print("使用方法: python test_deepface.py <画像パス>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"画像を読み込めません: {image_path}")
        sys.exit(1)
    
    print("=== DeepFace分析テスト ===\n")
    
    # DeepFaceで分析
    analyzer = DeepFaceAnalyzer(detector_backend='retinaface')
    
    if not analyzer.is_available:
        print("DeepFaceが利用できません")
        sys.exit(1)
    
    results = analyzer.analyze(image)
    
    if not results:
        print("顔が検出されませんでした")
        sys.exit(1)
    
    result = results[0]
    
    print(f"顔の矩形: {result.face_rect}")
    print(f"顔の向き: Roll={result.roll:.1f}° Yaw={result.yaw:.1f}° Pitch={result.pitch:.1f}°")
    print(f"\n感情スコア:")
    for emotion, score in sorted(result.emotion.items(), key=lambda x: -x[1]):
        bar = "█" * int(score / 5)
        print(f"  {emotion:12s} {bar:20s} {score:.1f}%")
    print(f"\n主要感情: {result.dominant_emotion}")
    print(f"推定年齢: {result.age:.0f}歳")
    
    # ランドマークがあれば68点に変換してテスト
    if result.landmarks:
        print(f"\nランドマーク: {list(result.landmarks.keys())}")
        
        # 68点に変換
        landmarks_68 = DeepFaceLandmarkConverter.convert_5_to_68(
            result.landmarks, result.face_rect
        )
        
        # 可視化
        vis = visualize_landmarks_debug(image, landmarks_68, show_numbers=True)
        
        # 顔の向きを表示
        cv2.putText(vis, f"Roll: {result.roll:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis, f"Yaw: {result.yaw:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis, f"Pitch: {result.pitch:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("DeepFace Landmarks (68-point)", vis)
        print("\nキーを押すと終了...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
