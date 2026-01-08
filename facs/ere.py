import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import colorsys

from .action_units import AU_DEFINITIONS, AUIntensity, LANDMARK_NAMES
from .au_detector import AUDetectionResult
from .intensity_estimator import IntensityResult
from .emotion_mapper import EmotionResult

class FACSVisualizer:
    """FACS分析結果の可視化"""
    
    # カラーパレット
    COLORS = {
        "landmark": (0, 255, 0),
        "landmark_highlight": (0, 255, 255),
        "face_rect": (255, 0, 0),
        "au_active": (0, 255, 0),
        "au_inactive": (128, 128, 128),
        "text": (255, 255, 255),
        "background": (0, 0, 0),
        "positive": (0, 255, 0),
        "negative": (0, 0, 255),
        "neutral": (255, 255, 0)
    }
    
    # 強度に対応する色（緑→黄→赤）
    INTENSITY_COLORS = {
        AUIntensity.ABSENT: (128, 128, 128),
        AUIntensity.TRACE: (0, 255, 0),
        AUIntensity.SLIGHT: (0, 255, 128),
        AUIntensity.MARKED: (0, 255, 255),
        AUIntensity.SEVERE: (0, 165, 255),
        AUIntensity.MAXIMUM: (0, 0, 255)
    }
    
    def __init__(self, font_scale: float = 0.5, line_thickness: int = 1):
        """初期化"""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.line_thickness = line_thickness
    
    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray,
                       highlight_indices: Optional[List[int]] = None,
                       show_numbers: bool = False) -> np.ndarray:
        """ランドマークを描画"""
        output = image.copy()
        
        for i, (x, y) in enumerate(landmarks):
            x, y = int(x), int(y)
            
            if highlight_indices and i in highlight_indices:
                color = self.COLORS["landmark_highlight"]
                radius = 3
            else:
                color = self.COLORS["landmark"]
                radius = 2
            
            cv2.circle(output, (x, y), radius, color, -1)
            
            if show_numbers:
                cv2.putText(output, str(i), (x + 3, y - 3),
                           self.font, 0.3, color, 1)
        
        return output
    
    def draw_landmark_connections(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """ランドマーク間の接続線を描画"""
        output = image.copy()
        
        # 顔の輪郭
        self._draw_polyline(output, landmarks[0:17], False)
        
        # 眉
        self._draw_polyline(output, landmarks[17:22], False)
        self._draw_polyline(output, landmarks[22:27], False)
        
        # 鼻
        self._draw_polyline(output, landmarks[27:31], False)
        self._draw_polyline(output, landmarks[31:36], False)
        
        # 目
        self._draw_polyline(output, landmarks[36:42], True)
        self._draw_polyline(output, landmarks[42:48], True)
        
        # 唇
        self._draw_polyline(output, landmarks[48:60], True)
        self._draw_polyline(output, landmarks[60:68], True)
        
        return output
    
    def _draw_polyline(self, image: np.ndarray, points: np.ndarray, 
                       closed: bool = False, color: Tuple[int, int, int] = None):
        """ポリラインを描画"""
        if color is None:
            color = self.COLORS["landmark"]
        
        pts = points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], closed, color, self.line_thickness)
    
    def draw_au_results(self, image: np.ndarray, au_results: Dict[int, AUDetectionResult],
                        intensity_results: Optional[Dict[int, IntensityResult]] = None,
                        position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """AU検出結果を描画"""
        output = image.copy()
        x, y = position
        line_height = 20
        
        # 背景パネル
        active_aus = [r for r in au_results.values() if r.detected]
        panel_height = len(active_aus) * line_height + 40
        panel_width = 250
        
        overlay = output.copy()
        cv2.rectangle(overlay, (x - 5, y - 25), (x + panel_width, y + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        # タイトル
        cv2.putText(output, "Active Action Units:", (x, y),
                   self.font, self.font_scale, self.COLORS["text"], 1)
        y += line_height + 5
        
        # 各AU
        for au_num in sorted(au_results.keys()):
            result = au_results[au_num]
            if not result.detected:
                continue
            
            # 強度情報
            if intensity_results and au_num in intensity_results:
                intensity = intensity_results[au_num]
                intensity_label = intensity.intensity_label
                color = self.INTENSITY_COLORS.get(intensity.intensity, self.COLORS["au_active"])
            else:
                intensity_label = ""
                color = self.COLORS["au_active"]
            
            # テキスト
            text = f"AU{au_num}{intensity_label}: {result.name}"
            conf_text = f"({result.confidence:.2f})"
            
            cv2.putText(output, text, (x, y), self.font, self.font_scale * 0.9, color, 1)
            cv2.putText(output, conf_text, (x + 180, y), self.font, self.font_scale * 0.8,
                       (180, 180, 180), 1)
            
            y += line_height
        
        return output
    
    def draw_emotion_results(self, image: np.ndarray, emotions: List[EmotionResult],
                             position: Tuple[int, int] = None) -> np.ndarray:
        """感情推定結果を描画"""
        output = image.copy()
        h, w = image.shape[:2]
        
        if position is None:
            position = (w - 260, 30)
        
        x, y = position
        line_height = 25
        
        # 上位3つの感情を表示
        top_emotions = emotions[:3]
        
        # 背景パネル
        panel_height = len(top_emotions) * line_height + 80
        panel_width = 250
        
        overlay = output.copy()
        cv2.rectangle(overlay, (x - 5, y - 25), (x + panel_width, y + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        # タイトル
        cv2.putText(output, "Detected Emotions:", (x, y),
                   self.font, self.font_scale, self.COLORS["text"], 1)
        y += line_height + 5
        
        for emotion in top_emotions:
            # 感情に応じた色
            if emotion.valence > 0.2:
                color = self.COLORS["positive"]
            elif emotion.valence < -0.2:
                color = self.COLORS["negative"]
            else:
                color = self.COLORS["neutral"]
            
            # バーを描画
            bar_width = int(emotion.confidence * 150)
            cv2.rectangle(output, (x, y), (x + bar_width, y + 15), color, -1)
            cv2.rectangle(output, (x, y), (x + 150, y + 15), (100, 100, 100), 1)
            
            # テキスト
            text = f"{emotion.emotion}: {emotion.confidence:.2f}"
            cv2.putText(output, text, (x + 155, y + 12),
                       self.font, self.font_scale * 0.9, self.COLORS["text"], 1)
            
            y += line_height
        
        # Valence-Arousal
        y += 10
        dominant = emotions[0] if emotions else None
        if dominant:
            va_text = f"V: {dominant.valence:+.2f} | A: {dominant.arousal:+.2f}"
            cv2.putText(output, va_text, (x, y), self.font, self.font_scale * 0.8,
                       (180, 180, 180), 1)
        
        return output
    
    def draw_valence_arousal_plot(self, image: np.ndarray, valence: float, arousal: float,
                                  position: Tuple[int, int] = None, size: int = 150) -> np.ndarray:
        """Valence-Arousal空間のプロットを描画"""
        output = image.copy()
        h, w = image.shape[:2]
        
        if position is None:
            position = (w - size - 20, h - size - 20)
        
        x, y = position
        center_x = x + size // 2
        center_y = y + size // 2
        
        # 背景
        overlay = output.copy()
        cv2.rectangle(overlay, (x, y), (x + size, y + size), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, output, 0.2, 0, output)
        
        # 軸
        cv2.line(output, (x, center_y), (x + size, center_y), (100, 100, 100), 1)
        cv2.line(output, (center_x, y), (center_x, y + size), (100, 100, 100), 1)
        
        # ラベル
        cv2.putText(output, "+V", (x + size - 20, center_y - 5), self.font, 0.3, (150, 150, 150), 1)
        cv2.putText(output, "-V", (x + 2, center_y - 5), self.font, 0.3, (150, 150, 150), 1)
        cv2.putText(output, "+A", (center_x + 3, y + 12), self.font, 0.3, (150, 150, 150), 1)
        cv2.putText(output, "-A", (center_x + 3, y + size - 3), self.font, 0.3, (150, 150, 150), 1)
        
        # 感情の位置をプロット
        plot_x = int(center_x + valence * (size // 2 - 10))
        plot_y = int(center_y - arousal * (size // 2 - 10))
        
        # 色を決定（valenceに基づく）
        if valence > 0:
            color = (0, int(255 * min(valence + 0.5, 1)), 0)
        else:
            color = (0, 0, int(255 * min(-valence + 0.5, 1)))
        
        cv2.circle(output, (plot_x, plot_y), 8, color, -1)
        cv2.circle(output, (plot_x, plot_y), 8, (255, 255, 255), 1)
        
        return output
    
    def draw_au_heatmap(self, image: np.ndarray, landmarks: np.ndarray,
                        au_results: Dict[int, AUDetectionResult]) -> np.ndarray:
        """AU活性化のヒートマップを描画"""
        output = image.copy()
        h, w = image.shape[:2]
        
        # 各AUに関連するランドマークの活性化を計算
        landmark_activation = np.zeros(68)
        
        for au_num, result in au_results.items():
            if not result.detected:
                continue
            
            au_def = AU_DEFINITIONS.get(au_num)
            if au_def:
                for lm_idx in au_def.landmarks_involved:
                    if 0 <= lm_idx < 68:
                        landmark_activation[lm_idx] = max(
                            landmark_activation[lm_idx],
                            result.raw_score
                        )
        
        # ヒートマップを描画
        for i, (x, y) in enumerate(landmarks):
            x, y = int(x), int(y)
            activation = landmark_activation[i]
            
            if activation > 0.1:
                # 色を計算（青→緑→黄→赤）
                hue = (1.0 - min(activation, 1.0)) * 0.4  # 0.4 (緑) から 0 (赤)
                rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
                
                radius = int(5 + activation * 10)
                
                # 半透明の円を描画
                overlay = output.copy()
                cv2.circle(overlay, (x, y), radius, color, -1)
                cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
        
        return output
    
    def draw_face_regions(self, image: np.ndarray, landmarks: np.ndarray,
                          highlight_regions: Optional[List[str]] = None) -> np.ndarray:
        """顔の領域を描画"""
        output = image.copy()
        
        regions = {
            "right_eyebrow": landmarks[17:22],
            "left_eyebrow": landmarks[22:27],
            "right_eye": landmarks[36:42],
            "left_eye": landmarks[42:48],
            "nose": np.vstack([landmarks[27:36]]),
            "outer_mouth": landmarks[48:60],
            "jaw": landmarks[0:17]
        }
        
        for region_name, points in regions.items():
            if highlight_regions and region_name in highlight_regions:
                color = (0, 255, 255)
                thickness = 2
            else:
                color = (0, 200, 0)
                thickness = 1
            
            hull = cv2.convexHull(points.astype(np.int32))
            cv2.polylines(output, [hull], True, color, thickness)
        
        return output
    
    def create_analysis_panel(self, image: np.ndarray, landmarks: np.ndarray,
                              au_results: Dict[int, AUDetectionResult],
                              intensity_results: Dict[int, IntensityResult],
                              emotions: List[EmotionResult],
                              facs_code: str) -> np.ndarray:
        """分析結果の総合パネルを作成"""
        h, w = image.shape[:2]
        
        # パネルサイズ
        panel_width = 400
        total_width = w + panel_width
        
        # 出力画像を作成
        output = np.zeros((h, total_width, 3), dtype=np.uint8)
        
        # 元画像にランドマークを描画
        img_with_landmarks = self.draw_landmarks(image, landmarks)
        img_with_landmarks = self.draw_landmark_connections(img_with_landmarks, landmarks)
        output[:, :w] = img_with_landmarks
        
        # 右パネル
        panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        
        y = 30
        
        # FACSコード
        cv2.putText(panel, "FACS Code:", (10, y), self.font, 0.6, (255, 255, 255), 1)
        y += 25
        
        # コードを複数行に分割
        code_parts = facs_code.split(" + ")
        for i in range(0, len(code_parts), 4):
            line = " + ".join(code_parts[i:i+4])
            cv2.putText(panel, line, (10, y), self.font, 0.5, (0, 255, 255), 1)
            y += 20
        
        y += 20
        
        # 感情結果
        cv2.putText(panel, "Emotions:", (10, y), self.font, 0.6, (255, 255, 255), 1)
        y += 25
        
        for emotion in emotions[:5]:
            if emotion.confidence < 0.1:
                continue
            
            # バー
            bar_width = int(emotion.confidence * 200)
            if emotion.valence > 0:
                color = (0, 200, 0)
            elif emotion.valence < 0:
                color = (0, 0, 200)
            else:
                color = (200, 200, 0)
            
            cv2.rectangle(panel, (10, y), (10 + bar_width, y + 12), color, -1)
            cv2.rectangle(panel, (10, y), (210, y + 12), (80, 80, 80), 1)
            
            text = f"{emotion.emotion} ({emotion.confidence:.2f})"
            cv2.putText(panel, text, (220, y + 10), self.font, 0.4, (200, 200, 200), 1)
            y += 20
        
        y += 20
        
        # Active AUs
        cv2.putText(panel, "Active AUs:", (10, y), self.font, 0.6, (255, 255, 255), 1)
        y += 25
        
        for au_num in sorted(au_results.keys()):
            if y > h - 30:
                break
            
            result = au_results[au_num]
            if not result.detected:
                continue
            
            intensity = intensity_results.get(au_num)
            if intensity:
                color = self.INTENSITY_COLORS.get(intensity.intensity, (0, 255, 0))
                label = intensity.intensity_label
            else:
                color = (0, 255, 0)
                label = ""
            
            text = f"AU{au_num}{label}: {result.name[:20]}"
            cv2.putText(panel, text, (10, y), self.font, 0.4, color, 1)
            y += 18
        
        output[:, w:] = panel
        
        return output
    
    def save_visualization(self, output: np.ndarray, filepath: str):
        """可視化結果を保存"""
        cv2.imwrite(filepath, output)
