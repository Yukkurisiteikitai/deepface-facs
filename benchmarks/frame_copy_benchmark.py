"""
フレームコピーのパフォーマンスベンチマーク
"""
import numpy as np
import time
from typing import Callable, List, Tuple

def benchmark_copy_method(
    frame: np.ndarray, 
    copy_func: Callable[[np.ndarray], np.ndarray],
    iterations: int = 1000
) -> Tuple[float, float]:
    """コピー方法のベンチマーク"""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = copy_func(frame)
        times.append((time.perf_counter() - start) * 1000)
    return np.mean(times), np.std(times)

def run_benchmark():
    """ベンチマーク実行"""
    resolutions = [
        ("480p", (640, 480)),
        ("720p", (1280, 720)),
        ("1080p", (1920, 1080)),
    ]
    
    copy_methods = [
        ("np.copy()", lambda f: np.copy(f)),
        ("frame.copy()", lambda f: f.copy()),
        ("np.array()", lambda f: np.array(f)),
        ("np.empty + [:] 代入", lambda f: _copy_with_empty(f)),
        ("コピーなし(参照)", lambda f: f),  # 比較用
    ]
    
    print("=" * 70)
    print("フレームコピー パフォーマンスベンチマーク")
    print("=" * 70)
    
    for res_name, (w, h) in resolutions:
        # BGR画像を想定 (3チャンネル)
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        size_mb = frame.nbytes / (1024 * 1024)
        
        print(f"\n【{res_name}】 {w}x{h} ({size_mb:.2f} MB)")
        print("-" * 50)
        
        for method_name, copy_func in copy_methods:
            mean_ms, std_ms = benchmark_copy_method(frame, copy_func)
            fps_impact = mean_ms / (1000 / 30) * 100  # 30FPS想定での影響
            print(f"  {method_name:25s}: {mean_ms:.3f} ms (±{std_ms:.3f}) "
                  f"| 30FPS時 {fps_impact:.1f}%")
    
    # 実際の処理時間との比較
    print("\n" + "=" * 70)
    print("処理時間に対するコピーコストの割合")
    print("=" * 70)
    
    frame_1080p = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    copy_time, _ = benchmark_copy_method(frame_1080p, lambda f: f.copy(), 100)
    
    processing_times = [10, 20, 30, 50, 100]  # 想定される処理時間(ms)
    print(f"\n1080pコピー時間: {copy_time:.3f} ms")
    print("-" * 50)
    for proc_time in processing_times:
        ratio = copy_time / proc_time * 100
        print(f"  処理時間 {proc_time:3d}ms の場合: コピーは {ratio:.1f}%")

def _copy_with_empty(frame: np.ndarray) -> np.ndarray:
    """np.emptyを使ったコピー"""
    new_frame = np.empty_like(frame)
    new_frame[:] = frame
    return new_frame

if __name__ == "__main__":
    run_benchmark()
