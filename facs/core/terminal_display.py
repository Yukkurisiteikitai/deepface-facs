"""
ターミナル表示
"""
from typing import Dict, Optional
from .models import AnalysisResult, AUDetectionResult, IntensityResult
from ..config import AU_DEFINITIONS


class TerminalDisplay:
    """ターミナル用表示クラス"""
    
    COLORS = {
        'reset': '\033[0m', 'bold': '\033[1m', 'dim': '\033[2m',
        'red': '\033[91m', 'green': '\033[92m', 'yellow': '\033[93m',
        'blue': '\033[94m', 'magenta': '\033[95m', 'cyan': '\033[96m',
    }
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
    
    def _c(self, text: str, color: str) -> str:
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def print_header(self, title: str, width: int = 60):
        print("\n" + "=" * width)
        print(f"{title:^{width}}")
        print("=" * width)
    
    def print_full_analysis(self, result: AnalysisResult, show_au_details: bool = True):
        if not result.is_valid:
            print(self._c("顔が検出されませんでした", 'red'))
            return
        
        print(f"\n{self._c('FACSコード:', 'bold')} {self._c(result.facs_code, 'cyan')}")
        print(f"{self._c('処理時間:', 'dim')} {result.processing_time_ms:.1f}ms")
        
        print(f"\n{self._c('【感情】', 'bold')}")
        for e in result.emotions[:5]:
            if e.confidence < 0.1:
                continue
            bar = "█" * int(e.confidence * 20) + "░" * (20 - int(e.confidence * 20))
            color = 'green' if e.valence > 0 else 'red' if e.valence < 0 else 'yellow'
            print(f"  {e.emotion:12s} {self._c(bar, color)} {e.confidence:.2f}")
        
        if result.dominant_emotion:
            print(f"\n  主要感情: {self._c(result.dominant_emotion.emotion, 'cyan')}")
            print(f"  V: {result.valence:+.2f} | A: {result.arousal:+.2f}")
        
        if show_au_details and result.active_aus:
            print(f"\n{self._c('【検出AU】', 'bold')}")
            for au in result.active_aus[:10]:
                intensity = result.intensity_results.get(au.au_number)
                label = f"[{intensity.intensity_label}]" if intensity else ""
                print(f"  AU{au.au_number:2d}{label:3s} {au.name:25s} {au.confidence:.2f}")
    
    def print_au_legend(self):
        print(f"\n{self._c('【強度スケール】', 'bold')}")
        print(f"  {self._c('[A]', 'green')} Trace    {self._c('[B]', 'cyan')} Slight")
        print(f"  {self._c('[C]', 'yellow')} Marked   {self._c('[D]', 'magenta')} Severe")
        print(f"  {self._c('[E]', 'red')} Maximum")
