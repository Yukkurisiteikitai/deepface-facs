"""
可視化コンポーネント（SOLID原則に基づく分離）
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import numpy as np
import cv2

from .font_manager import get_text_renderer


@dataclass
class DrawContext:
    """描画コンテキスト"""
    image: np.ndarray
    x: int
    y: int
    width: int
    height: int
    
    def get_region(self) -> np.ndarray:
        """描画領域を取得"""
        return self.image[self.y:self.y+self.height, self.x:self.x+self.width]


class IDrawable(ABC):
    """描画可能オブジェクトのインターフェース"""
    
    @abstractmethod
    def draw(self, image: np.ndarray, x: int, y: int) -> int:
        """描画し、使用した高さを返す"""
        pass


class TextRenderer:
    """テキスト描画ユーティリティ"""
    
    def __init__(self):
        self._renderer = get_text_renderer()
    
    def draw_text(self, image: np.ndarray, text: str, x: int, y: int,
                  font_size: int = 14, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """テキストを描画（インプレース対応）"""
        # PILを使う場合は新しい画像が返されるので、それを元の画像にコピー
        result = self._renderer.put_text(image, text, (x, y), font_size, color)
        if result is not image:
            np.copyto(image, result)
        return image
    
    def draw_text_cv2(self, image: np.ndarray, text: str, x: int, y: int,
                      font_size: int = 14, color: Tuple[int, int, int] = (255, 255, 255)):
        """OpenCVでテキスト描画（インプレース）"""
        scale = font_size / 30.0
        cv2.putText(image, text, (x, y + font_size), cv2.FONT_HERSHEY_SIMPLEX,
                   scale, color, 1, cv2.LINE_AA)


class ProgressBar:
    """プログレスバー描画"""
    
    @staticmethod
    def draw(image: np.ndarray, x: int, y: int, width: int, height: int,
             value: float, max_value: float = 1.0,
             fg_color: Tuple[int, int, int] = (0, 255, 0),
             bg_color: Tuple[int, int, int] = (80, 80, 80)):
        """プログレスバーを描画"""
        # 背景
        cv2.rectangle(image, (x, y), (x + width, y + height), bg_color, 1)
        # 前景
        fill_width = int(width * min(value / max_value, 1.0))
        if fill_width > 0:
            cv2.rectangle(image, (x, y), (x + fill_width, y + height), fg_color, -1)


@dataclass
class PanelSection:
    """パネルセクション"""
    title: str
    start_y: int
    end_y: int = 0


class PanelBuilder:
    """パネル構築クラス（Builder Pattern）"""
    
    def __init__(self, width: int, height: int, bg_color: Tuple[int, int, int] = (30, 30, 30)):
        self._width = width
        self._height = height
        self._panel = np.full((height, width, 3), bg_color, dtype=np.uint8)
        self._current_y = 10
        self._margin = 10
        self._text_renderer = TextRenderer()
        self._sections: List[PanelSection] = []
    
    @property
    def panel(self) -> np.ndarray:
        return self._panel
    
    @property
    def current_y(self) -> int:
        return self._current_y
    
    @property
    def available_height(self) -> int:
        return self._height - self._current_y - 20
    
    def add_spacing(self, height: int = 10) -> 'PanelBuilder':
        """スペースを追加"""
        self._current_y += height
        return self
    
    def add_title(self, text: str, font_size: int = 18,
                  color: Tuple[int, int, int] = (255, 255, 255)) -> 'PanelBuilder':
        """タイトルを追加"""
        self._draw_text(text, self._margin, self._current_y, font_size, color)
        self._current_y += font_size + 10
        return self
    
    def add_text(self, text: str, font_size: int = 12,
                 color: Tuple[int, int, int] = (255, 255, 255),
                 x_offset: int = 0) -> 'PanelBuilder':
        """テキストを追加"""
        self._draw_text(text, self._margin + x_offset, self._current_y, font_size, color)
        self._current_y += font_size + 6
        return self
    
    def add_header(self, text: str, font_size: int = 14,
                   color: Tuple[int, int, int] = (255, 255, 255)) -> 'PanelBuilder':
        """ヘッダーを追加"""
        section = PanelSection(title=text, start_y=self._current_y)
        self._sections.append(section)
        self._draw_text(text, self._margin, self._current_y, font_size, color)
        self._current_y += font_size + 8
        return self
    
    def add_progress_bar(self, label: str, value: float, max_value: float = 1.0,
                         bar_width: int = 150,
                         fg_color: Tuple[int, int, int] = (0, 255, 0),
                         suffix: str = "") -> 'PanelBuilder':
        """ラベル付きプログレスバーを追加"""
        # ラベル
        self._draw_text(label, self._margin, self._current_y, 11, (200, 200, 200))
        
        # バー
        bar_x = self._margin + 70
        bar_y = self._current_y + 2
        bar_height = 10
        ProgressBar.draw(self._panel, bar_x, bar_y, bar_width, bar_height,
                        value, max_value, fg_color)
        
        # 値
        value_x = bar_x + bar_width + 5
        value_text = f"{value:.2f}" if not suffix else f"{value:.2f} {suffix}"
        self._draw_text(value_text, value_x, self._current_y, 10, fg_color)
        
        self._current_y += 16
        return self
    
    def add_emotion_bar(self, emotion: str, confidence: float, valence: float) -> 'PanelBuilder':
        """感情バーを追加"""
        # バーの色を決定
        if valence > 0:
            color = (0, 200, 0)
        elif valence < 0:
            color = (0, 0, 200)
        else:
            color = (200, 200, 0)
        
        # バー
        bar_width = int(confidence * 150)
        cv2.rectangle(self._panel, (self._margin, self._current_y),
                     (self._margin + bar_width, self._current_y + 14), color, -1)
        cv2.rectangle(self._panel, (self._margin, self._current_y),
                     (self._margin + 150, self._current_y + 14), (80, 80, 80), 1)
        
        # ラベル
        label_x = self._margin + 160
        self._draw_text(f"{emotion} ({confidence:.2f})", label_x, self._current_y + 2, 11, (200, 200, 200))
        
        self._current_y += 18
        return self
    
    def add_au_row(self, au_number: int, label: str, name: str, confidence: float,
                   color: Tuple[int, int, int], is_highlighted: bool = False) -> Tuple[int, int, int, int]:
        """AU行を追加し、クリック領域を返す"""
        row_height = 20
        
        # ハイライト背景
        if is_highlighted:
            cv2.rectangle(self._panel, (5, self._current_y - 2),
                         (self._width - 10, self._current_y + row_height - 4), (60, 60, 60), -1)
        
        # テキスト
        text = f"AU{au_number:2d}[{label}] {name[:20]}"
        self._draw_text(text, self._margin, self._current_y, 11, color)
        
        # 信頼度バー
        bar_x = 230
        bar_width = int(confidence * 60)
        cv2.rectangle(self._panel, (bar_x, self._current_y + 2),
                     (bar_x + bar_width, self._current_y + 12), color, -1)
        cv2.rectangle(self._panel, (bar_x, self._current_y + 2),
                     (bar_x + 60, self._current_y + 12), (80, 80, 80), 1)
        
        # クリック領域
        region = (5, self._current_y - 2, self._width - 15, row_height)
        
        self._current_y += row_height
        return region
    
    def add_separator(self, color: Tuple[int, int, int] = (80, 80, 80)) -> 'PanelBuilder':
        """区切り線を追加"""
        cv2.line(self._panel, (self._margin, self._current_y),
                (self._width - self._margin, self._current_y), color, 1)
        self._current_y += 5
        return self
    
    def add_popup(self, x: int, y: int, width: int, height: int,
                  bg_color: Tuple[int, int, int] = (45, 45, 45),
                  border_color: Tuple[int, int, int] = (255, 0, 255)) -> 'PanelBuilder':
        """ポップアップ背景を追加"""
        cv2.rectangle(self._panel, (x, y), (x + width, y + height), bg_color, -1)
        cv2.rectangle(self._panel, (x, y), (x + width, y + height), border_color, 2)
        return self
    
    def _draw_text(self, text: str, x: int, y: int, font_size: int,
                   color: Tuple[int, int, int]):
        """テキストを描画（内部メソッド）"""
        # 日本語チェック
        has_japanese = any('\u3000' <= c <= '\u9fff' or '\uff00' <= c <= '\uffef' for c in text)
        
        if has_japanese:
            # PILで描画
            result = self._text_renderer._renderer.put_text(
                self._panel, text, (x, y), font_size, color
            )
            if result is not self._panel:
                np.copyto(self._panel, result)
        else:
            # OpenCVで描画
            scale = font_size / 30.0
            cv2.putText(self._panel, text, (x, y + int(font_size * 0.8)),
                       cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
