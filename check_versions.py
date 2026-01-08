"""
インストール済みライブラリのバージョン確認
"""
import sys

def check_versions():
    print("=== ライブラリバージョン確認 ===\n")
    
    # NumPy
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        print("✗ NumPy: 未インストール")
    
    # OpenCV
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV: 未インストール")
    
    # TensorFlow
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow: {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow: 未インストール")
    
    # tf-keras
    try:
        import tf_keras
        version = getattr(tf_keras, '__version__', 'installed')
        print(f"✓ tf-keras: {version}")
    except ImportError:
        print("✗ tf-keras: 未インストール")
        print("  → pip install tf-keras")
    
    # MediaPipe
    try:
        import mediapipe as mp
        version = getattr(mp, '__version__', 'unknown')
        print(f"✓ MediaPipe: {version}")
        
        # solutions API確認
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
            print("  └─ solutions.face_mesh: 利用可能")
        else:
            print("  └─ solutions.face_mesh: 利用不可")
        
        # Tasks API確認
        try:
            from mediapipe.tasks import python
            print("  └─ tasks API: 利用可能")
        except ImportError:
            print("  └─ tasks API: 利用不可")
            
    except ImportError:
        print("✗ MediaPipe: 未インストール")
    
    # DeepFace
    try:
        # tf-kerasがないとインポートエラーになるので先にチェック
        try:
            import tf_keras
        except ImportError:
            print("✗ DeepFace: tf-kerasが必要です")
            print("  → pip install tf-keras")
        else:
            from deepface import DeepFace
            import deepface
            version = getattr(deepface, '__version__', 'unknown')
            print(f"✓ DeepFace: {version}")
            
    except ImportError as e:
        print(f"✗ DeepFace: インポートエラー - {e}")
    except ValueError as e:
        print(f"✗ DeepFace: {e}")
        print("  → pip install tf-keras")
    
    # Pillow
    try:
        from PIL import Image
        import PIL
        print(f"✓ Pillow: {PIL.__version__}")
    except ImportError:
        print("✗ Pillow: 未インストール")
    
    print("\n=== 推奨インストールコマンド ===")
    print("pip install numpy opencv-python mediapipe Pillow tensorflow tf-keras deepface")
    print("\n# tf-kerasは必須（DeepFace/RetinaFaceが依存）:")
    print("pip install tf-keras")

if __name__ == "__main__":
    check_versions()
