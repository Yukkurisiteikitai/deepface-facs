"""
ランドマーク検出のテストスクリプト
詳細なデバッグ情報を表示
"""
import cv2
import sys
import numpy as np
from facs.detectors import (
    LandmarkDetectorFactory,
    visualize_landmarks_debug,
    test_landmark_mapping
)
from facs.core.enums import DetectorType

def visualize_mediapipe_raw(image, raw_landmarks):
    """MediaPipeの生のランドマーク（478点）を可視化"""
    output = image.copy()
    h, w = image.shape[:2]
    
    # 重要なランドマークのインデックス（MediaPipe Face Mesh）
    important_points = {
        # 顔の輪郭
        'silhouette': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162],
        # 唇外側
        'lips_outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
        # 唇内側
        'lips_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],
        # 左目
        'left_eye': [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398],
        # 右目
        'right_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173],
        # 左眉
        'left_eyebrow': [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
        # 右眉
        'right_eyebrow': [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
        # 鼻
        'nose': [168, 6, 197, 195, 5, 4, 1, 19, 94, 2],
    }
    
    colors = {
        'silhouette': (255, 255, 0),
        'lips_outer': (0, 0, 255),
        'lips_inner': (255, 0, 255),
        'left_eye': (255, 0, 0),
        'right_eye': (255, 0, 0),
        'left_eyebrow': (0, 255, 0),
        'right_eyebrow': (0, 255, 0),
        'nose': (0, 255, 255),
    }
    
    # 各領域を描画
    for region_name, indices in important_points.items():
        color = colors.get(region_name, (255, 255, 255))
        pts = []
        for idx in indices:
            if idx < len(raw_landmarks):
                x, y = raw_landmarks[idx]
                pts.append((int(x), int(y)))
                cv2.circle(output, (int(x), int(y)), 2, color, -1)
        
        # ポリライン
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                cv2.line(output, pts[i], pts[i+1], color, 1)
    
    return output


def test_mapping():
    """マッピングの正確性をテスト"""
    # 正確なMediaPipe → 68点マッピング
    # 参考: https://github.com/google/mediapipe/issues/1615
    
    correct_mapping = {
        # 顔の輪郭 (0-16) - 顎のライン、右から左
        0: 162, 1: 234, 2: 93, 3: 132, 4: 58, 5: 172, 6: 136, 7: 150, 8: 149,
        9: 176, 10: 148, 11: 152, 12: 377, 13: 400, 14: 378, 15: 379, 16: 365,
        
        # 右眉 (17-21)
        17: 70, 18: 63, 19: 105, 20: 66, 21: 107,
        
        # 左眉 (22-26)
        22: 336, 23: 296, 24: 334, 25: 293, 26: 300,
        
        # 鼻筋 (27-30)
        27: 168, 28: 6, 29: 197, 30: 195,
        
        # 鼻先 (31-35)
        31: 5, 32: 4, 33: 1, 34: 19, 35: 94,
        
        # 右目 (36-41)
        36: 33, 37: 160, 38: 158, 39: 133, 40: 153, 41: 144,
        
        # 左目 (42-47)
        42: 362, 43: 385, 44: 387, 45: 263, 46: 373, 47: 380,
        
        # 外側唇 (48-59)
        48: 61, 49: 40, 50: 37, 51: 0, 52: 267, 53: 270,
        54: 291, 55: 321, 56: 314, 57: 17, 58: 84, 59: 181,
        
        # 内側唇 (60-67)
        60: 78, 61: 82, 62: 13, 63: 312, 64: 308, 65: 317, 66: 14, 67: 87,
    }
    
    print("=== 68点マッピング ===")
    for i in range(68):
        mp_idx = correct_mapping.get(i, -1)
        region = ""
        if i <= 16:
            region = "顎"
        elif i <= 21:
            region = "右眉"
        elif i <= 26:
            region = "左眉"
        elif i <= 30:
            region = "鼻筋"
        elif i <= 35:
            region = "鼻先"
        elif i <= 41:
            region = "右目"
        elif i <= 47:
            region = "左目"
        elif i <= 59:
            region = "外唇"
        else:
            region = "内唇"
        print(f"  68pt[{i:2d}] ({region:4s}) -> MP[{mp_idx:3d}]")
    
    return [correct_mapping[i] for i in range(68)]


def download_model(url: str, path: str) -> bool:
    """モデルをダウンロード（SSL問題回避）"""
    import ssl
    import urllib.request
    
    # 方法1: SSL検証を無効化（開発用）
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        print(f"ダウンロード中: {url}")
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(path, 'wb') as f:
                f.write(response.read())
        print(f"ダウンロード完了: {path}")
        return True
    except Exception as e:
        print(f"方法1失敗: {e}")
    
    # 方法2: requestsライブラリを使用
    try:
        import requests
        print(f"requestsでダウンロード中...")
        response = requests.get(url, verify=False)
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f"ダウンロード完了: {path}")
        return True
    except ImportError:
        print("requestsがインストールされていません")
    except Exception as e:
        print(f"方法2失敗: {e}")
    
    # 方法3: curlコマンドを使用
    try:
        import subprocess
        print(f"curlでダウンロード中...")
        subprocess.run(['curl', '-L', '-o', path, url], check=True)
        print(f"ダウンロード完了: {path}")
        return True
    except Exception as e:
        print(f"方法3失敗: {e}")
    
    return False


def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python test_landmarks.py <画像パス>")
        print("  python test_landmarks.py --test-mapping")
        print("  python test_landmarks.py --raw <画像パス>  # MediaPipe生データを表示")
        sys.exit(1)
    
    if sys.argv[1] == '--test-mapping':
        test_mapping()
        return
    
    show_raw = False
    image_path = sys.argv[1]
    
    if sys.argv[1] == '--raw':
        show_raw = True
        if len(sys.argv) < 3:
            print("画像パスを指定してください")
            sys.exit(1)
        image_path = sys.argv[2]
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"画像を読み込めません: {image_path}")
        sys.exit(1)
    
    print(f"画像サイズ: {image.shape}")
    
    # MediaPipeで直接検出
    try:
        import mediapipe as mp
        print(f"MediaPipe version: {mp.__version__}")
        
        # Tasks APIを使用
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        
        # モデルをダウンロード/取得
        import os
        
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            if not download_model(url, model_path):
                print("\nモデルを手動でダウンロードしてください:")
                print(f"  curl -L -o {model_path} {url}")
                print("または:")
                print(f"  wget -O {model_path} {url}")
                sys.exit(1)
        
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # 検出
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = landmarker.detect(mp_image)
        
        if not result.face_landmarks:
            print("顔が検出されませんでした")
            sys.exit(1)
        
        h, w = image.shape[:2]
        raw_landmarks = [(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]]
        print(f"検出されたランドマーク数: {len(raw_landmarks)}")
        
        if show_raw:
            # 生データを表示
            vis = visualize_mediapipe_raw(image, raw_landmarks)
            cv2.imshow("MediaPipe Raw Landmarks", vis)
            print("\nキーを押すと終了...")
            cv2.waitKey(0)
        else:
            # 68点に変換して表示
            from facs.detectors.debug_landmarks import visualize_landmarks_debug
            from facs.detectors.landmark_detector import BaseLandmarkDetector
            
            mapping = BaseLandmarkDetector.get_mediapipe_to_68_mapping()
            landmarks_68 = np.zeros((68, 2), dtype=np.float32)
            for i, mp_idx in enumerate(mapping):
                if mp_idx < len(raw_landmarks):
                    landmarks_68[i] = raw_landmarks[mp_idx]
            
            vis = visualize_landmarks_debug(image, landmarks_68, show_numbers=True, show_regions=True)
            cv2.imshow("68-point Landmarks", vis)
            print("\nキーを押すと終了...")
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
