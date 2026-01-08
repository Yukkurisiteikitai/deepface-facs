from typing import Dict, List, Optional
from ..core.models import AnalysisResult, AUDetectionResult, IntensityResult
from ..config import AU_DEFINITIONS

class TerminalDisplay:
    """ターミナル用の詳細表示"""
    
    # ANSIカラーコード
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bg_gray': '\033[100m',
    }
    
    # 強度に対応する色
    INTENSITY_COLORS = {
        'A': 'green',
        'B': 'cyan',
        'C': 'yellow',
        'D': 'magenta',
        'E': 'red',
        '-': 'dim'
    }
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
    
    def _c(self, text: str, color: str) -> str:
        """テキストに色を付ける"""
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def print_header(self, title: str, width: int = 70):
        """ヘッダー表示"""
        print("\n" + "=" * width)
        print(f"{self._c(title, 'bold'):^{width + 8}}")  # カラーコード分の補正
        print("=" * width)
    
    def print_facs_code(self, result: AnalysisResult):
        """FACSコードを表示"""
        print(f"\n{self._c('FACSコード:', 'bold')} {self._c(result.facs_code, 'cyan')}")
        print(f"{self._c('処理時間:', 'dim')} {result.processing_time_ms:.1f}ms")
    
    def print_emotions(self, result: AnalysisResult):
        """感情を表示"""
        print(f"\n{self._c('【感情推定】', 'bold')}")
        
        for emotion in result.emotions[:5]:
            if emotion.confidence < 0.1:
                continue
            
            # プログレスバー
            bar_len = int(emotion.confidence * 25)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            
            # 色を決定
            if emotion.valence > 0.2:
                color = 'green'
            elif emotion.valence < -0.2:
                color = 'red'
            else:
                color = 'yellow'
            
            print(f"  {emotion.emotion:12s} {self._c(bar, color)} {emotion.confidence:.2f}")
            print(f"  {self._c(f'└─ {emotion.description}', 'dim')}")
        
        if result.dominant_emotion:
            print(f"\n  {self._c('主要感情:', 'bold')} {self._c(result.dominant_emotion.emotion, 'cyan')}")
            
            # Valence-Arousal
            v_color = 'green' if result.valence > 0 else 'red' if result.valence < 0 else 'yellow'
            a_color = 'magenta' if result.arousal > 0.5 else 'blue' if result.arousal < -0.5 else 'white'
            print(f"  Valence: {self._c(f'{result.valence:+.2f}', v_color)} | "
                  f"Arousal: {self._c(f'{result.arousal:+.2f}', a_color)}")
    
    def print_active_aus_detailed(self, result: AnalysisResult):
        """Active AUsの詳細を表示"""
        print(f"\n{self._c('【検出されたAction Units】', 'bold')}")
        
        if not result.active_aus:
            print(f"  {self._c('検出されたAUはありません', 'dim')}")
            return
        
        for au in result.active_aus:
            self.print_au_detail(au, result.intensity_results.get(au.au_number))
    
    def print_au_detail(self, au: AUDetectionResult, intensity: Optional[IntensityResult] = None):
        """単一AUの詳細を表示"""
        au_def = AU_DEFINITIONS.get(au.au_number)
        
        # 強度ラベルと色
        label = intensity.intensity_label if intensity else "-"
        color = self.INTENSITY_COLORS.get(label, 'white')
        
        # ヘッダー
        header = f"AU{au.au_number:2d}[{label}]"
        print(f"\n  {self._c(header, color)} {self._c(au.name, 'bold')}")
        
        # 説明
        if au_def:
            print(f"  {self._c('│', 'dim')} {self._c(au_def.description, 'cyan')}")
            print(f"  {self._c('│', 'dim')} 筋肉: {self._c(au_def.muscular_basis, 'dim')}")
        
        # 検出値
        print(f"  {self._c('│', 'dim')} ─────────────────────────────────")
        
        # 信頼度バー
        conf_bar_len = int(au.confidence * 20)
        conf_bar = "█" * conf_bar_len + "░" * (20 - conf_bar_len)
        print(f"  {self._c('│', 'dim')} 信頼度:   {self._c(conf_bar, 'green')} {au.confidence:.3f}")
        
        # Rawスコアバー
        raw_bar_len = int(min(au.raw_score, 1.0) * 20)
        raw_bar = "█" * raw_bar_len + "░" * (20 - raw_bar_len)
        print(f"  {self._c('│', 'dim')} Rawスコア: {self._c(raw_bar, 'blue')} {au.raw_score:.3f}")
        
        # 強度値
        if intensity:
            int_bar_len = int(intensity.intensity_value / 5.0 * 20)
            int_bar = "█" * int_bar_len + "░" * (20 - int_bar_len)
            print(f"  {self._c('│', 'dim')} 強度値:   {self._c(int_bar, color)} {intensity.intensity_value:.2f}/5.0")
        
        # 非対称性
        if abs(au.asymmetry) > 0.1:
            side = "左が強い" if au.asymmetry > 0 else "右が強い"
            asym_color = 'yellow' if abs(au.asymmetry) > 0.3 else 'dim'
            print(f"  {self._c('│', 'dim')} 非対称:   {self._c(f'{au.asymmetry:+.3f} ({side})', asym_color)}")
        
        # 関連ランドマーク
        if au_def:
            landmarks = au_def.landmarks_involved
            print(f"  {self._c('└', 'dim')} 関連点:   {len(landmarks)}点 {self._c(str(landmarks[:8]) + ('...' if len(landmarks) > 8 else ''), 'dim')}")
    
    def print_full_analysis(self, result: AnalysisResult, show_au_details: bool = True):
        """完全な分析結果を表示"""
        self.print_header("FACS分析結果")
        
        if not result.is_valid:
            print(f"\n{self._c('顔が検出されませんでした', 'red')}")
            return
        
        self.print_facs_code(result)
        self.print_emotions(result)
        
        if show_au_details:
            self.print_active_aus_detailed(result)
        
        # サマリー
        print(f"\n{self._c('【サマリー】', 'bold')}")
        print(f"  検出AU数: {len(result.active_aus)}")
        
        if result.active_aus:
            strongest = max(result.active_aus, key=lambda x: x.confidence)
            print(f"  最強AU: AU{strongest.au_number} ({strongest.name}) - {strongest.confidence:.2f}")
        
        print("=" * 70)
    
    def print_au_legend(self):
        """AU強度の凡例を表示"""
        print(f"\n{self._c('【強度スケール (FACS標準)】', 'bold')}")
        print(f"  {self._c('[A]', 'green')} Trace (微弱)     - スコア 0.5-1.5")
        print(f"  {self._c('[B]', 'cyan')} Slight (軽度)    - スコア 1.5-2.5")
        print(f"  {self._c('[C]', 'yellow')} Marked (中程度)  - スコア 2.5-3.5")
        print(f"  {self._c('[D]', 'magenta')} Severe (顕著)    - スコア 3.5-4.5")
        print(f"  {self._c('[E]', 'red')} Maximum (最大)   - スコア 4.5-5.0")
