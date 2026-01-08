import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

from ..core.interfaces import IVisualizer
from ..core.models import AnalysisResult, AUDetectionResult
from ..core.enums import AUIntensity
from ..config import AU_DEFINITIONS
from .font_manager import get_text_renderer

class FACSVisualizer(IVisualizer):
    """FACS可視化（インタラクティブ機能付き・日本語対応）"""
    
    INTENSITY_COLORS = {
        AUIntensity.ABSENT: (128, 128, 128), AUIntensity.TRACE: (0, 255, 0),
        AUIntensity.SLIGHT: (0, 255, 128), AUIntensity.MARKED: (0, 255, 255),
        AUIntensity.SEVERE: (0, 165, 255), AUIntensity.MAXIMUM: (0, 0, 255)
    }
    
    AU_HIGHLIGHT_COLOR = (255, 0, 255)
    AU_LANDMARK_COLOR = (0, 255, 255)
    
    def __init__(self, font_scale: float = 0.5):
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = font_scale
        self._hover_au: Optional[int] = None
        self._au_regions: Dict[int, Tuple[int, int, int, int]] = {}
        self._text_renderer = get_text_renderer()
    
    def _put_text(self, image: np.ndarray, text: str, position: Tuple[int, int],
                  font_size: int = 14, color: Tuple[int, int, int] = (255, 255, 255),
                  bg_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """テキスト描画（日本語対応）"""
        return self._text_renderer.put_text(image, text, position, font_size, color, bg_color)
    
    def set_hover_au(self, au_number: Optional[int]):
        self._hover_au = au_number
    
    def get_au_at_position(self, x: int, y: int, panel_x_offset: int) -> Optional[int]:
        for au_num, (rx, ry, rw, rh) in self._au_regions.items():
            if rx + panel_x_offset <= x <= rx + rw + panel_x_offset and ry <= y <= ry + rh:
                return au_num
        return None
    
    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray,
                       highlight_indices: Optional[List[int]] = None) -> np.ndarray:
        output = image.copy()
        
        for i, (x, y) in enumerate(landmarks):
            if highlight_indices and i in highlight_indices:
                color = self.AU_HIGHLIGHT_COLOR
                radius = 4
            else:
                color = (0, 255, 0)
                radius = 2
            cv2.circle(output, (int(x), int(y)), radius, color, -1)
        
        connections = [
            (range(0, 17), False), (range(17, 22), False), (range(22, 27), False),
            (range(27, 31), False), (range(31, 36), False),
            (range(36, 42), True), (range(42, 48), True),
            (range(48, 60), True), (range(60, 68), True)
        ]
        
        for indices, closed in connections:
            idx_list = list(indices)
            is_highlighted = highlight_indices and any(i in highlight_indices for i in idx_list)
            color = self.AU_HIGHLIGHT_COLOR if is_highlighted else (0, 255, 0)
            thickness = 2 if is_highlighted else 1
            pts = landmarks[idx_list].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(output, [pts], closed, color, thickness)
        
        return output
    
    def draw_au_landmarks_highlight(self, image: np.ndarray, landmarks: np.ndarray,
                                    au_number: int) -> np.ndarray:
        au_def = AU_DEFINITIONS.get(au_number)
        if au_def is None:
            return image
        highlight_indices = list(au_def.landmarks_involved)
        return self.draw_landmarks(image, landmarks, highlight_indices)
    
    def create_analysis_panel(self, image: np.ndarray, result: AnalysisResult) -> np.ndarray:
        if not result.is_valid:
            return image
        
        h, w = image.shape[:2]
        panel_width = 420
        output = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
        
        # 画像にランドマーク描画
        if self._hover_au and self._hover_au in AU_DEFINITIONS:
            img_vis = self.draw_au_landmarks_highlight(image, result.face_data.landmarks, self._hover_au)
        else:
            img_vis = self.draw_landmarks(image, result.face_data.landmarks)
        
        # 顔の傾きを表示（画像上）
        img_vis = self._draw_face_orientation(img_vis, result)
        
        output[:, :w] = img_vis
        
        # 右パネル
        panel = output[:, w:]
        panel[:] = (30, 30, 30)
        
        self._au_regions.clear()
        
        y = 25
        panel = self._put_text(panel, "FACS分析結果", (10, y), 18, (255, 255, 255))
        y += 30
        
        # 顔の向き情報を追加
        angles = result.face_data.angles
        if 'face_roll' in angles:
            roll = angles['face_roll']
            yaw = angles.get('face_yaw', 0)
            pitch = angles.get('face_pitch', 0)
            
            orientation_text = f"顔の向き: R{roll:+.1f}° Y{yaw:+.1f}° P{pitch:+.1f}°"
            panel = self._put_text(panel, orientation_text, (10, y), 11, (180, 180, 180))
            y += 20
        
        # FACSコード
        panel = self._put_text(panel, f"コード: {result.facs_code[:40]}", (10, y), 13, (0, 255, 255))
        y += 25
        
        # 感情
        panel = self._put_text(panel, "【感情推定】", (10, y), 14, (255, 255, 255))
        y += 22
        
        for e in result.emotions[:4]:
            if e.confidence < 0.1:
                continue
            bar_w = int(e.confidence * 150)
            color = (0, 200, 0) if e.valence > 0 else (0, 0, 200) if e.valence < 0 else (200, 200, 0)
            cv2.rectangle(panel, (10, y), (10 + bar_w, y + 14), color, -1)
            cv2.rectangle(panel, (10, y), (160, y + 14), (80, 80, 80), 1)
            
            emotion_text = f"{e.emotion} ({e.confidence:.2f})"
            panel = self._put_text(panel, emotion_text, (170, y + 2), 12, (200, 200, 200))
            y += 20
        
        y += 15
        panel = self._put_text(panel, "【検出AU】マウスホバーで詳細", (10, y), 13, (255, 255, 255))
        y += 22
        
        # Active AUs
        for au in result.active_aus[:12]:
            if y > h - 100:
                break
            
            intensity = result.intensity_results.get(au.au_number)
            is_hovered = self._hover_au == au.au_number
            
            if is_hovered:
                cv2.rectangle(panel, (5, y - 14), (panel_width - 10, y + 6), (60, 60, 60), -1)
            
            color = self.AU_HIGHLIGHT_COLOR if is_hovered else \
                    self.INTENSITY_COLORS.get(intensity.intensity if intensity else AUIntensity.ABSENT, (0, 255, 0))
            label = intensity.intensity_label if intensity else ""
            
            text = f"AU{au.au_number:2d}[{label}] {au.name}"
            panel = self._put_text(panel, text, (10, y - 10), 12, color)
            
            # 信頼度バー
            bar_x = 250
            bar_w = int(au.confidence * 80)
            cv2.rectangle(panel, (bar_x, y - 10), (bar_x + bar_w, y), color, -1)
            cv2.rectangle(panel, (bar_x, y - 10), (bar_x + 80, y), (80, 80, 80), 1)
            
            conf_text = f"{au.confidence:.2f}"
            panel = self._put_text(panel, conf_text, (bar_x + 85, y - 10), 10, (150, 150, 150))
            
            self._au_regions[au.au_number] = (5, y - 14, panel_width - 15, 22)
            y += 22
        
        # ホバー中のAU詳細
        if self._hover_au:
            panel = self._draw_au_detail_popup(panel, result, y)
        
        output[:, w:] = panel
        return output
    
    def _draw_face_orientation(self, image: np.ndarray, result: AnalysisResult) -> np.ndarray:
        """顔の向きを画像上に可視化"""
        if not result.face_data or result.face_data.landmarks is None:
            return image
        
        output = image.copy()
        landmarks = result.face_data.landmarks
        angles = result.face_data.angles
        
        # 顔の中心を計算
        left_eye = np.mean(landmarks[42:48], axis=0)
        right_eye = np.mean(landmarks[36:42], axis=0)
        face_center = ((left_eye + right_eye) / 2).astype(int)
        
        # 軸の長さ
        eye_dist = np.linalg.norm(left_eye - right_eye)
        axis_length = int(eye_dist * 0.5)
        
        if 'face_roll' in angles:
            roll = np.radians(angles['face_roll'])
            yaw = np.radians(angles.get('face_yaw', 0))
            pitch = np.radians(angles.get('face_pitch', 0))
            
            # 各軸の方向ベクトルを計算
            # X軸（赤）- 右方向
            x_end = (
                int(face_center[0] + axis_length * np.cos(roll) * np.cos(yaw)),
                int(face_center[1] + axis_length * np.sin(roll))
            )
            cv2.arrowedLine(output, tuple(face_center), x_end, (0, 0, 255), 2, tipLength=0.3)
            
            # Y軸（緑）- 下方向
            y_end = (
                int(face_center[0] - axis_length * np.sin(roll)),
                int(face_center[1] + axis_length * np.cos(roll) * np.cos(pitch))
            )
            cv2.arrowedLine(output, tuple(face_center), y_end, (0, 255, 0), 2, tipLength=0.3)
            
            # Z軸（青）- 手前方向（簡略化）
            z_length = int(axis_length * 0.5 * (1 - abs(np.sin(yaw))))
            z_end = (
                int(face_center[0] - axis_length * 0.3 * np.sin(yaw)),
                int(face_center[1] - axis_length * 0.3 * np.sin(pitch))
            )
            cv2.arrowedLine(output, tuple(face_center), z_end, (255, 0, 0), 2, tipLength=0.3)
        
        return output

    def _draw_au_detail_popup(self, panel: np.ndarray, result: AnalysisResult, start_y: int) -> np.ndarray:
        """AU詳細ポップアップを描画（日本語対応）"""
        au_def = AU_DEFINITIONS.get(self._hover_au)
        au_result = result.au_results.get(self._hover_au)
        intensity = result.intensity_results.get(self._hover_au)
        
        if not au_def or not au_result:
            return panel
        
        panel_h, panel_w = panel.shape[:2]
        
        # ポップアップ背景
        popup_y = min(start_y + 10, panel_h - 220)
        cv2.rectangle(panel, (5, popup_y), (panel_w - 10, popup_y + 210), (45, 45, 45), -1)
        cv2.rectangle(panel, (5, popup_y), (panel_w - 10, popup_y + 210), self.AU_HIGHLIGHT_COLOR, 2)
        
        y = popup_y + 20
        
        # タイトル
        title = f"AU{self._hover_au}: {au_def.name}"
        panel = self._put_text(panel, title, (15, y), 15, self.AU_HIGHLIGHT_COLOR)
        y += 25
        
        # 日本語説明
        panel = self._put_text(panel, au_def.description, (15, y), 14, (255, 255, 255))
        y += 22
        
        # 筋肉（長い場合は折り返し）
        muscle_text = f"筋肉: {au_def.muscular_basis}"
        if len(muscle_text) > 40:
            panel = self._put_text(panel, muscle_text[:40], (15, y), 11, (180, 180, 180))
            y += 16
            panel = self._put_text(panel, muscle_text[40:], (15, y), 11, (180, 180, 180))
        else:
            panel = self._put_text(panel, muscle_text, (15, y), 11, (180, 180, 180))
        y += 20
        
        # 区切り線
        cv2.line(panel, (15, y), (panel_w - 25, y), (80, 80, 80), 1)
        y += 12
        
        # 検出値
        panel = self._put_text(panel, "【検出値】", (15, y), 12, (200, 200, 200))
        y += 18
        
        # Raw Score
        raw_bar_w = int(min(au_result.raw_score, 1.0) * 120)
        cv2.rectangle(panel, (100, y), (100 + raw_bar_w, y + 10), (100, 255, 100), -1)
        cv2.rectangle(panel, (100, y), (220, y + 10), (80, 80, 80), 1)
        panel = self._put_text(panel, "Raw:", (15, y - 2), 11, (150, 150, 150))
        panel = self._put_text(panel, f"{au_result.raw_score:.3f}", (225, y - 2), 11, (150, 255, 150))
        y += 16
        
        # Confidence
        conf_bar_w = int(au_result.confidence * 120)
        cv2.rectangle(panel, (100, y), (100 + conf_bar_w, y + 10), (100, 200, 255), -1)
        cv2.rectangle(panel, (100, y), (220, y + 10), (80, 80, 80), 1)
        panel = self._put_text(panel, "信頼度:", (15, y - 2), 11, (150, 150, 150))
        panel = self._put_text(panel, f"{au_result.confidence:.3f}", (225, y - 2), 11, (100, 200, 255))
        y += 16
        
        # Intensity
        if intensity:
            int_bar_w = int(intensity.intensity_value / 5.0 * 120)
            int_color = self.INTENSITY_COLORS.get(intensity.intensity, (200, 200, 200))
            cv2.rectangle(panel, (100, y), (100 + int_bar_w, y + 10), int_color, -1)
            cv2.rectangle(panel, (100, y), (220, y + 10), (80, 80, 80), 1)
            panel = self._put_text(panel, "強度:", (15, y - 2), 11, (150, 150, 150))
            panel = self._put_text(panel, f"{intensity.intensity_label} ({intensity.intensity_value:.2f})", 
                                   (225, y - 2), 11, int_color)
            y += 16
        
        # 非対称性
        if abs(au_result.asymmetry) > 0.1:
            side = "左が強い" if au_result.asymmetry > 0 else "右が強い"
            panel = self._put_text(panel, f"非対称: {au_result.asymmetry:+.3f} ({side})", 
                                   (15, y), 11, (255, 200, 100))
            y += 16
        
        # 関連ランドマーク
        panel = self._put_text(panel, f"関連点: {len(au_def.landmarks_involved)}箇所（ハイライト中）", 
                               (15, y), 10, (150, 150, 150))
        
        return panel


class InteractiveFACSVisualizer(FACSVisualizer):
    """インタラクティブなFACS可視化（マウスホバー対応）"""
    
    def __init__(self, font_scale: float = 0.5):
        super().__init__(font_scale)
        self._window_name = "FACS Analysis"
        self._panel_x_offset = 0
        self._current_result: Optional[AnalysisResult] = None
        self._current_image: Optional[np.ndarray] = None
    
    def _mouse_callback(self, event, x, y, flags, param):
        """マウスイベントコールバック"""
        if event == cv2.EVENT_MOUSEMOVE:
            # パネル領域内かチェック
            if x >= self._panel_x_offset:
                au = self.get_au_at_position(x, y, self._panel_x_offset)
                if au != self._hover_au:
                    self._hover_au = au
                    self._update_display()
            else:
                if self._hover_au is not None:
                    self._hover_au = None
                    self._update_display()
    
    def _update_display(self):
        """表示を更新"""
        if self._current_image is not None and self._current_result is not None:
            vis = self.create_analysis_panel(self._current_image, self._current_result)
            cv2.imshow(self._window_name, vis)
    
    def show_interactive(self, image: np.ndarray, result: AnalysisResult, 
                         window_name: str = "FACS Analysis"):
        """インタラクティブ表示"""
        self._window_name = window_name
        self._current_image = image
        self._current_result = result
        self._panel_x_offset = image.shape[1]
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        vis = self.create_analysis_panel(image, result)
        cv2.imshow(window_name, vis)
        
        return vis
