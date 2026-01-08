"""
ランドマーク検出のデバッグツール
68点のランドマークが正しく配置されているか確認する
"""
import numpy as np
import cv2
from typing import Optional

def visualize_landmarks_debug(image: np.ndarray, landmarks: np.ndarray, 
                               show_numbers: bool = True,
                               show_regions: bool = True) -> np.ndarray:
    """
    ランドマークを詳細に可視化（デバッグ用）
    
    Args:
        image: 入力画像
        landmarks: 68点ランドマーク
        show_numbers: ランドマーク番号を表示するか
        show_regions: 領域ごとに色分けするか
    """
    output = image.copy()
    
    # 領域ごとの色定義
    region_colors = {
        'jaw': (255, 255, 0),        # シアン - 顎ライン
        'right_eyebrow': (0, 255, 0), # 緑 - 右眉
        'left_eyebrow': (0, 255, 0),  # 緑 - 左眉
        'nose_bridge': (255, 0, 255), # マゼンタ - 鼻筋
        'nose_tip': (255, 0, 255),    # マゼンタ - 鼻先
        'right_eye': (255, 0, 0),     # 青 - 右目
        'left_eye': (255, 0, 0),      # 青 - 左目
        'outer_lip': (0, 0, 255),     # 赤 - 外側唇
        'inner_lip': (0, 165, 255),   # オレンジ - 内側唇
    }
    
    # 領域定義
    regions = {
        'jaw': (0, 17, False),
        'right_eyebrow': (17, 22, False),
        'left_eyebrow': (22, 27, False),
        'nose_bridge': (27, 31, False),
        'nose_tip': (31, 36, False),
        'right_eye': (36, 42, True),
        'left_eye': (42, 48, True),
        'outer_lip': (48, 60, True),
        'inner_lip': (60, 68, True),
    }
    
    # 各領域を描画
    for region_name, (start, end, closed) in regions.items():
        color = region_colors.get(region_name, (255, 255, 255))
        pts = landmarks[start:end].astype(np.int32)
        
        if show_regions:
            # ポリラインを描画
            cv2.polylines(output, [pts.reshape((-1, 1, 2))], closed, color, 2)
        
        # 各点を描画
        for i, (x, y) in enumerate(pts):
            global_idx = start + i
            cv2.circle(output, (int(x), int(y)), 3, color, -1)
            cv2.circle(output, (int(x), int(y)), 4, (255, 255, 255), 1)
            
            if show_numbers:
                # 番号を表示
                cv2.putText(output, str(global_idx), (int(x) + 5, int(y) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # 凡例を追加
    legend_y = 30
    for region_name, color in region_colors.items():
        cv2.rectangle(output, (10, legend_y - 12), (25, legend_y), color, -1)
        cv2.putText(output, region_name, (30, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y += 18
    
    return output


def compare_mediapipe_dlib(image: np.ndarray, mp_landmarks: np.ndarray, 
                           dlib_landmarks: Optional[np.ndarray] = None) -> np.ndarray:
    """
    MediaPipeとdlibのランドマークを比較
    """
    output = image.copy()
    
    # MediaPipeのランドマーク（緑）
    for i, (x, y) in enumerate(mp_landmarks):
        cv2.circle(output, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    # dlibのランドマーク（赤）
    if dlib_landmarks is not None:
        for i, (x, y) in enumerate(dlib_landmarks):
            cv2.circle(output, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        # 対応点を線で結ぶ
        for i in range(68):
            mp_pt = tuple(mp_landmarks[i].astype(int))
            dlib_pt = tuple(dlib_landmarks[i].astype(int))
            cv2.line(output, mp_pt, dlib_pt, (255, 255, 0), 1)
    
    # 凡例
    cv2.putText(output, "Green: MediaPipe", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(output, "Red: dlib", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return output


def test_landmark_mapping():
    """マッピングのテスト用関数"""
    from .landmark_detector import BaseLandmarkDetector
    
    mapping = BaseLandmarkDetector.get_mediapipe_to_68_mapping()
    
    print("=== MediaPipe to 68-point Mapping ===")
    print(f"Total points: {len(mapping)}")
    
    # 重複チェック
    if len(mapping) != len(set(mapping)):
        duplicates = [x for x in mapping if mapping.count(x) > 1]
        print(f"WARNING: Duplicate indices found: {set(duplicates)}")
    
    # 領域ごとに表示
    regions = [
        ("Jaw (0-16)", 0, 17),
        ("Right Eyebrow (17-21)", 17, 22),
        ("Left Eyebrow (22-26)", 22, 27),
        ("Nose Bridge (27-30)", 27, 31),
        ("Nose Tip (31-35)", 31, 36),
        ("Right Eye (36-41)", 36, 42),
        ("Left Eye (42-47)", 42, 48),
        ("Outer Lip (48-59)", 48, 60),
        ("Inner Lip (60-67)", 60, 68),
    ]
    
    for name, start, end in regions:
        indices = mapping[start:end]
        print(f"\n{name}:")
        print(f"  MediaPipe indices: {indices}")
    
    return mapping
