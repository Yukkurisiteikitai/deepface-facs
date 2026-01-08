import platform
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class FontManager:
    """OS別日本語フォント管理"""
    
    # OS別のフォントパス候補
    FONT_PATHS = {
        'Darwin': [  # macOS
            '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
            '/System/Library/Fonts/Hiragino Sans GB.ttc',
            '/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
            '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',
            '/System/Library/Fonts/Helvetica.ttc',
        ],
        'Windows': [  # Windows
            'C:/Windows/Fonts/msgothic.ttc',   # MSゴシック
            'C:/Windows/Fonts/meiryo.ttc',     # メイリオ
            'C:/Windows/Fonts/YuGothM.ttc',    # 游ゴシック
            'C:/Windows/Fonts/msmincho.ttc',   # MS明朝
            'C:/Windows/Fonts/yugothic.ttf',
            'C:/Windows/Fonts/BIZ-UDGothicR.ttc',
        ],
        'Linux': [  # Linux
            '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
            '/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
        ]
    }
    
    def __init__(self):
        self._system = platform.system()
        self._font_path: Optional[str] = None
        self._font_cache: dict = {}
        self._find_font()
    
    def _find_font(self):
        """利用可能な日本語フォントを探す"""
        paths = self.FONT_PATHS.get(self._system, self.FONT_PATHS['Linux'])
        
        for path in paths:
            if os.path.exists(path):
                self._font_path = path
                print(f"日本語フォント: {Path(path).name}")
                return
        
        # 見つからない場合は警告
        print("警告: 日本語フォントが見つかりません。英語表示になります。")
        self._font_path = None
    
    def get_font(self, size: int = 14) -> Optional['ImageFont.FreeTypeFont']:
        """指定サイズのフォントを取得"""
        if not PIL_AVAILABLE or not self._font_path:
            return None
        
        if size not in self._font_cache:
            try:
                self._font_cache[size] = ImageFont.truetype(self._font_path, size)
            except Exception:
                return None
        
        return self._font_cache[size]
    
    @property
    def is_available(self) -> bool:
        """日本語フォントが利用可能か"""
        return PIL_AVAILABLE and self._font_path is not None


class TextRenderer:
    """日本語テキスト描画クラス"""
    
    def __init__(self):
        self._font_manager = FontManager()
    
    def put_text(self, image: np.ndarray, text: str, position: Tuple[int, int],
                 font_size: int = 14, color: Tuple[int, int, int] = (255, 255, 255),
                 bg_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        画像にテキストを描画（日本語対応）
        
        Args:
            image: 描画先画像（BGR）
            text: テキスト
            position: (x, y) 位置
            font_size: フォントサイズ
            color: 文字色 (B, G, R)
            bg_color: 背景色（オプション）
        
        Returns:
            描画後の画像
        """
        # 日本語が含まれているかチェック
        has_japanese = any('\u3000' <= c <= '\u9fff' or '\uff00' <= c <= '\uffef' for c in text)
        
        if has_japanese and self._font_manager.is_available:
            return self._put_text_pil(image, text, position, font_size, color, bg_color)
        else:
            return self._put_text_cv2(image, text, position, font_size, color, bg_color)
    
    def _put_text_pil(self, image: np.ndarray, text: str, position: Tuple[int, int],
                      font_size: int, color: Tuple[int, int, int],
                      bg_color: Optional[Tuple[int, int, int]]) -> np.ndarray:
        """PILを使用して日本語テキストを描画"""
        # BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        font = self._font_manager.get_font(font_size)
        if font is None:
            return self._put_text_cv2(image, text, position, font_size, color, bg_color)
        
        # BGRをRGBに変換
        color_rgb = (color[2], color[1], color[0])
        
        # 背景を描画
        if bg_color:
            bbox = draw.textbbox(position, text, font=font)
            padding = 2
            bg_color_rgb = (bg_color[2], bg_color[1], bg_color[0])
            draw.rectangle(
                [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
                fill=bg_color_rgb
            )
        
        # テキストを描画
        draw.text(position, text, font=font, fill=color_rgb)
        
        # RGB -> BGR
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result
    
    def _put_text_cv2(self, image: np.ndarray, text: str, position: Tuple[int, int],
                      font_size: int, color: Tuple[int, int, int],
                      bg_color: Optional[Tuple[int, int, int]]) -> np.ndarray:
        """OpenCVを使用してテキストを描画（英語のみ）"""
        output = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = font_size / 30.0
        thickness = 1
        
        if bg_color:
            (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
            x, y = position
            cv2.rectangle(output, (x - 2, y - text_h - 2), (x + text_w + 2, y + baseline + 2),
                         bg_color, -1)
        
        cv2.putText(output, text, position, font, scale, color, thickness, cv2.LINE_AA)
        return output
    
    def get_text_size(self, text: str, font_size: int = 14) -> Tuple[int, int]:
        """テキストのサイズを取得"""
        if self._font_manager.is_available:
            font = self._font_manager.get_font(font_size)
            if font:
                try:
                    bbox = font.getbbox(text)
                    return bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    pass
        
        # フォールバック
        scale = font_size / 30.0
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        return w, h


# シングルトンインスタンス
_text_renderer: Optional[TextRenderer] = None

def get_text_renderer() -> TextRenderer:
    """TextRendererのシングルトンを取得"""
    global _text_renderer
    if _text_renderer is None:
        _text_renderer = TextRenderer()
    return _text_renderer
