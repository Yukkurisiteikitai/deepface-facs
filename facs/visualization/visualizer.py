import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..core.interfaces import IVisualizer
from ..core.models import AnalysisResult, AUDetectionResult
from ..core.enums import AUIntensity
from ..config import AU_DEFINITIONS
from .components import PanelBuilder, ProgressBar


@dataclass
class LayoutConfig:
    """レイアウト設定"""
    # 画像サイズ
    target_face_height: int = 400  # 顔画像の目標高さ
    min_face_height: int = 200
    max_face_height: int = 600
    
    # パネルサイズ
    panel_width: int = 420
    min_panel_height: int = 400
    
    # マージン
    margin: int = 10


class ImageScaler:
    """画像スケーリングユーティリティ"""
    
    @staticmethod
    def compute_scale(image_shape: Tuple[int, int], target_height: int,
                      min_height: int, max_height: int) -> float:
        """適切なスケールを計算"""
        h = image_shape[0]
        scale = target_height / h
        
        # 結果の高さが範囲内になるように調整
        result_h = h * scale
        if result_h < min_height:
            scale = min_height / h
        elif result_h > max_height:
            scale = max_height / h
        
        return scale
    
    @staticmethod
    def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
        """画像をリサイズ"""
        if abs(scale - 1.0) < 0.01:
            return image
        
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 高品質リサイズ
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        return cv2.resize(image, (new_w, new_h), interpolation=interp)
    
    @staticmethod
    def scale_landmarks(landmarks: np.ndarray, scale: float) -> np.ndarray:
        """ランドマークをスケーリング"""
        return landmarks * scale


class FACSVisualizer(IVisualizer):
    """FACS可視化"""
    
    INTENSITY_COLORS = {
        AUIntensity.ABSENT: (128, 128, 128),
        AUIntensity.TRACE: (0, 255, 0),
        AUIntensity.SLIGHT: (0, 255, 128),
        AUIntensity.MARKED: (0, 255, 255),
        AUIntensity.SEVERE: (0, 165, 255),
        AUIntensity.MAXIMUM: (0, 0, 255)
    }
    AU_HIGHLIGHT_COLOR = (255, 0, 255)
    
    def __init__(self, layout_config: Optional[LayoutConfig] = None):
        self._layout = layout_config or LayoutConfig()
        self._hover_au: Optional[int] = None
        self._au_regions: Dict[int, Tuple[int, int, int, int]] = {}
        self._current_scale: float = 1.0
    
    def set_layout(self, **kwargs):
        """レイアウト設定を変更"""
        for key, value in kwargs.items():
            if hasattr(self._layout, key):
                setattr(self._layout, key, value)
    
    def set_hover_au(self, au_number: Optional[int]):
        self._hover_au = au_number
    
    def get_au_at_position(self, x: int, y: int, panel_x_offset: int) -> Optional[int]:
        for au_num, (rx, ry, rw, rh) in self._au_regions.items():
            if rx + panel_x_offset <= x <= rx + rw + panel_x_offset and ry <= y <= ry + rh:
                return au_num
        return None
    
    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray,
                       highlight_indices: Optional[List[int]] = None) -> np.ndarray:
        """ランドマーク描画"""
        output = image.copy()
        
        for i, (x, y) in enumerate(landmarks):
            if highlight_indices and i in highlight_indices:
                color, radius = self.AU_HIGHLIGHT_COLOR, 4
            else:
                color, radius = (0, 255, 0), 2
            cv2.circle(output, (int(x), int(y)), radius, color, -1)
        
        # 接続線
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
    
    def create_analysis_panel(self, image: np.ndarray, result: AnalysisResult) -> np.ndarray:
        """分析パネルを作成"""
        if not result.is_valid:
            return self._create_no_face_panel(image)
        
        # スケール計算
        self._current_scale = ImageScaler.compute_scale(
            image.shape[:2],
            self._layout.target_face_height,
            self._layout.min_face_height,
            self._layout.max_face_height
        )
        
        # 画像とランドマークをスケーリング
        scaled_image = ImageScaler.resize_image(image, self._current_scale)
        scaled_landmarks = ImageScaler.scale_landmarks(result.face_data.landmarks, self._current_scale)
        
        h, w = scaled_image.shape[:2]
        panel_height = max(h, self._layout.min_panel_height)
        
        # 出力画像を作成
        output = np.zeros((panel_height, w + self._layout.panel_width, 3), dtype=np.uint8)
        
        # ランドマーク描画
        highlight_indices = None
        if self._hover_au and self._hover_au in AU_DEFINITIONS:
            highlight_indices = list(AU_DEFINITIONS[self._hover_au].landmarks_involved)
        img_vis = self.draw_landmarks(scaled_image, scaled_landmarks, highlight_indices)
        
        # 画像を配置
        output[:h, :w] = img_vis
        if h < panel_height:
            output[h:, :w] = (30, 30, 30)
        
        # パネルを構築
        self._au_regions.clear()
        panel = self._build_panel(result, panel_height)
        output[:, w:] = panel
        
        return output
    
    def _build_panel(self, result: AnalysisResult, panel_height: int) -> np.ndarray:
        """パネルを構築"""
        builder = PanelBuilder(self._layout.panel_width, panel_height)
        
        # タイトル
        builder.add_title("FACS分析結果")
        
        # FACSコード
        code = result.facs_code[:40] + "..." if len(result.facs_code) > 40 else result.facs_code
        builder.add_text(f"コード: {code}", 12, (0, 255, 255))
        
        # 処理時間
        builder.add_text(f"処理時間: {result.processing_time_ms:.1f}ms", 10, (128, 128, 128))
        builder.add_spacing(5)
        
        # 感情セクション
        builder.add_header("【感情推定】")
        
        for e in result.emotions[:4]:
            if e.confidence >= 0.1:
                builder.add_emotion_bar(e.emotion, e.confidence, e.valence)
        
        # Valence-Arousal
        if result.dominant_emotion:
            v_color = (0, 200, 0) if result.valence > 0 else (0, 0, 200) if result.valence < 0 else (200, 200, 0)
            builder.add_text(f"V: {result.valence:+.2f}  A: {result.arousal:+.2f}", 11, v_color)
        
        builder.add_spacing(10)
        
        # AUセクション
        builder.add_header("【検出AU】ホバーで詳細")
        
        # 表示可能なAU数を計算
        available = builder.available_height
        max_aus = max(1, (available - 180) // 20)  # ポップアップ用に180px確保
        
        for au in result.active_aus[:max_aus]:
            intensity = result.intensity_results.get(au.au_number)
            is_hovered = self._hover_au == au.au_number
            color = self.AU_HIGHLIGHT_COLOR if is_hovered else \
                    self.INTENSITY_COLORS.get(intensity.intensity if intensity else AUIntensity.ABSENT)
            label = intensity.intensity_label if intensity else "-"
            
            region = builder.add_au_row(
                au.au_number, label, au.name, au.confidence, color, is_hovered
            )
            self._au_regions[au.au_number] = region
        
        # ホバー詳細
        if self._hover_au and self._hover_au in result.au_results:
            self._draw_hover_popup(builder, result)
        
        return builder.panel
    
    def _draw_hover_popup(self, builder: PanelBuilder, result: AnalysisResult):
        """ホバーポップアップを描画"""
        au_def = AU_DEFINITIONS.get(self._hover_au)
        au_result = result.au_results.get(self._hover_au)
        intensity = result.intensity_results.get(self._hover_au)
        
        if not au_def or not au_result:
            return
        
        panel = builder.panel
        panel_w = self._layout.panel_width
        
        # ポップアップ位置
        popup_y = builder.current_y + 10
        popup_height = 170
        
        # 背景
        builder.add_popup(5, popup_y, panel_w - 15, popup_height, (45, 45, 45), self.AU_HIGHLIGHT_COLOR)
        
        y = popup_y + 15
        
        # タイトル
        title = f"AU{self._hover_au}: {au_def.name}"
        cv2.putText(panel, title, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.AU_HIGHLIGHT_COLOR, 1)
        y += 20
        
        # 説明（日本語対応はPanelBuilderの_draw_textを使う）
        desc = au_def.description[:35] + "..." if len(au_def.description) > 35 else au_def.description
        cv2.putText(panel, desc, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 18
        
        # 筋肉
        muscle = au_def.muscular_basis[:30] + "..." if len(au_def.muscular_basis) > 30 else au_def.muscular_basis
        cv2.putText(panel, f"Muscle: {muscle}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        y += 20
        
        # 区切り線
        cv2.line(panel, (15, y), (panel_w - 25, y), (80, 80, 80), 1)
        y += 12
        
        # 検出値
        cv2.putText(panel, "Detection Values:", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 18
        
        # Raw Score
        raw_bar_w = int(min(au_result.raw_score, 1.0) * 100)
        cv2.rectangle(panel, (80, y - 8), (80 + raw_bar_w, y), (100, 255, 100), -1)
        cv2.rectangle(panel, (80, y - 8), (180, y), (80, 80, 80), 1)
        cv2.putText(panel, "Raw:", (15, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(panel, f"{au_result.raw_score:.3f}", (185, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 150), 1)
        y += 14
        
        # Confidence
        conf_bar_w = int(au_result.confidence * 100)
        cv2.rectangle(panel, (80, y - 8), (80 + conf_bar_w, y), (100, 200, 255), -1)
        cv2.rectangle(panel, (80, y - 8), (180, y), (80, 80, 80), 1)
        cv2.putText(panel, "Conf:", (15, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(panel, f"{au_result.confidence:.3f}", (185, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 255), 1)
        y += 14
        
        # Intensity
        if intensity:
            int_bar_w = int(intensity.intensity_value / 5.0 * 100)
            int_color = self.INTENSITY_COLORS.get(intensity.intensity, (200, 200, 200))
            cv2.rectangle(panel, (80, y - 8), (80 + int_bar_w, y), int_color, -1)
            cv2.rectangle(panel, (80, y - 8), (180, y), (80, 80, 80), 1)
            cv2.putText(panel, "Int:", (15, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(panel, f"{intensity.intensity_label} ({intensity.intensity_value:.2f})",
                       (185, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, int_color, 1)
    
    def _create_no_face_panel(self, image: np.ndarray) -> np.ndarray:
        """顔未検出時のパネル"""
        scale = ImageScaler.compute_scale(
            image.shape[:2],
            self._layout.target_face_height,
            self._layout.min_face_height,
            self._layout.max_face_height
        )
        scaled_image = ImageScaler.resize_image(image, scale)
        h, w = scaled_image.shape[:2]
        
        output = np.zeros((h, w + self._layout.panel_width, 3), dtype=np.uint8)
        output[:, :w] = scaled_image
        output[:, w:] = (30, 30, 30)
        
        cv2.putText(output, "No face detected", (w + 20, h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        return output


class InteractiveFACSVisualizer(FACSVisualizer):
    """インタラクティブ可視化"""
    
    def __init__(self, layout_config: Optional[LayoutConfig] = None):
        super().__init__(layout_config)
        self._window_name = "FACS Analysis"
        self._panel_x_offset = 0
        self._current_result: Optional[AnalysisResult] = None
        self._current_image: Optional[np.ndarray] = None
    
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            if x >= self._panel_x_offset:
                au = self.get_au_at_position(x, y, self._panel_x_offset)
                if au != self._hover_au:
                    self._hover_au = au
                    self._update_display()
            elif self._hover_au is not None:
                self._hover_au = None
                self._update_display()
    
    def _update_display(self):
        if self._current_image is not None and self._current_result is not None:
            vis = self.create_analysis_panel(self._current_image, self._current_result)
            cv2.imshow(self._window_name, vis)
    
    def show_interactive(self, image: np.ndarray, result: AnalysisResult,
                         window_name: str = "FACS Analysis") -> np.ndarray:
        self._window_name = window_name
        self._current_image = image
        self._current_result = result
        
        scale = ImageScaler.compute_scale(
            image.shape[:2],
            self._layout.target_face_height,
            self._layout.min_face_height,
            self._layout.max_face_height
        )
        self._panel_x_offset = int(image.shape[1] * scale)
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        vis = self.create_analysis_panel(image, result)
        cv2.imshow(window_name, vis)
        
        return vis
