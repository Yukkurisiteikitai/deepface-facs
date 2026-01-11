"""pytest共通設定とフィクスチャ"""
import pytest
import numpy as np
import cv2
from pathlib import Path


@pytest.fixture
def sample_image_path():
    """サンプル画像パス"""
    return Path(__file__).parent.parent / "sample_image"


@pytest.fixture
def dummy_image():
    """ダミー画像（黒画像）"""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_face_image():
    """ダミー顔画像（簡易的な楕円）"""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # 顔の楕円
    cv2.ellipse(image, (320, 240), (100, 130), 0, 0, 360, (200, 180, 160), -1)
    # 目
    cv2.circle(image, (280, 200), 15, (50, 50, 50), -1)
    cv2.circle(image, (360, 200), 15, (50, 50, 50), -1)
    # 口
    cv2.ellipse(image, (320, 300), (40, 15), 0, 0, 180, (100, 80, 80), -1)
    return image


@pytest.fixture
def dummy_landmarks():
    """ダミー68点ランドマーク"""
    # 基本的な顔の形状を模したランドマーク
    landmarks = np.zeros((68, 2), dtype=np.float32)
    
    # 顔の輪郭 (0-16)
    for i in range(17):
        angle = np.pi * (i / 16)
        landmarks[i] = [320 + 100 * np.cos(angle), 240 + 130 * np.sin(angle)]
    
    # 右眉 (17-21)
    for i in range(5):
        landmarks[17 + i] = [250 + i * 15, 180]
    
    # 左眉 (22-26)
    for i in range(5):
        landmarks[22 + i] = [340 + i * 15, 180]
    
    # 鼻 (27-35)
    for i in range(9):
        landmarks[27 + i] = [320, 200 + i * 10]
    
    # 右目 (36-41)
    for i in range(6):
        angle = 2 * np.pi * i / 6
        landmarks[36 + i] = [280 + 15 * np.cos(angle), 200 + 8 * np.sin(angle)]
    
    # 左目 (42-47)
    for i in range(6):
        angle = 2 * np.pi * i / 6
        landmarks[42 + i] = [360 + 15 * np.cos(angle), 200 + 8 * np.sin(angle)]
    
    # 外側唇 (48-59)
    for i in range(12):
        angle = 2 * np.pi * i / 12
        landmarks[48 + i] = [320 + 40 * np.cos(angle), 300 + 15 * np.sin(angle)]
    
    # 内側唇 (60-67)
    for i in range(8):
        angle = 2 * np.pi * i / 8
        landmarks[60 + i] = [320 + 25 * np.cos(angle), 300 + 8 * np.sin(angle)]
    
    return landmarks
