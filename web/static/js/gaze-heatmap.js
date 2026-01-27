/**
 * 視線ヒートマップ生成モジュール
 * 
 * 視線位置の履歴からヒートマップを生成
 * - リアルタイム/累積ヒートマップ
 * - 統計情報（滞留時間、訪問回数）
 * - エクスポート機能
 */

class GazeHeatmap {
    constructor(options = {}) {
        this.config = {
            // キャンバスサイズ
            width: options.width || window.innerWidth,
            height: options.height || window.innerHeight,
            
            // ヒートマップ設定
            resolution: options.resolution || 2,      // 解像度（1=フル、2=半分）
            radius: options.radius || 50,             // 各ポイントの影響半径
            maxIntensity: options.maxIntensity || 100, // 最大強度
            
            // 色設定
            gradient: options.gradient || {
                0.0: 'rgba(0, 0, 255, 0)',
                0.2: 'rgba(0, 0, 255, 0.5)',
                0.4: 'rgba(0, 255, 255, 0.7)',
                0.6: 'rgba(0, 255, 0, 0.8)',
                0.8: 'rgba(255, 255, 0, 0.9)',
                1.0: 'rgba(255, 0, 0, 1)'
            },
            
            // 減衰設定
            decay: options.decay || 0.995,            // フレームごとの減衰率
            
            ...options
        };
        
        // 内部キャンバス（強度マップ）
        this.intensityCanvas = document.createElement('canvas');
        this.intensityCanvas.width = Math.floor(this.config.width / this.config.resolution);
        this.intensityCanvas.height = Math.floor(this.config.height / this.config.resolution);
        this.intensityCtx = this.intensityCanvas.getContext('2d', { willReadFrequently: true });
        
        // 出力キャンバス（カラーマップ）
        this.outputCanvas = null;
        this.outputCtx = null;
        
        // グラデーションパレット
        this.gradientPalette = this.createGradientPalette();
        
        // 視線データの履歴
        this.gazeHistory = [];
        this.maxHistory = 10000;
        
        // 統計
        this.statistics = {
            totalPoints: 0,
            hotspots: [],
            avgPosition: { x: 0, y: 0 },
            coverage: 0
        };
        
        // AOI（関心領域）
        this.aois = [];
        
        // 初期化
        this.clearIntensityMap();
    }
    
    /**
     * 出力キャンバスを設定
     */
    setOutputCanvas(canvas) {
        this.outputCanvas = canvas;
        this.outputCtx = canvas.getContext('2d');
    }
    
    /**
     * グラデーションパレットを作成
     */
    createGradientPalette() {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 1;
        const ctx = canvas.getContext('2d');
        
        const gradient = ctx.createLinearGradient(0, 0, 256, 0);
        for (const [stop, color] of Object.entries(this.config.gradient)) {
            gradient.addColorStop(parseFloat(stop), color);
        }
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 256, 1);
        
        return ctx.getImageData(0, 0, 256, 1).data;
    }
    
    /**
     * 強度マップをクリア
     */
    clearIntensityMap() {
        this.intensityCtx.fillStyle = 'rgba(0, 0, 0, 1)';
        this.intensityCtx.fillRect(0, 0, this.intensityCanvas.width, this.intensityCanvas.height);
        this.gazeHistory = [];
        this.statistics.totalPoints = 0;
    }
    
    /**
     * 視線ポイントを追加
     */
    addPoint(x, y, intensity = 1, timestamp = performance.now()) {
        // 座標を内部解像度に変換
        const ix = x / this.config.resolution;
        const iy = y / this.config.resolution;
        const radius = this.config.radius / this.config.resolution;
        
        // 放射状のグラデーションで強度を追加
        const gradient = this.intensityCtx.createRadialGradient(ix, iy, 0, ix, iy, radius);
        const alpha = Math.min(1, intensity * 0.1);
        gradient.addColorStop(0, `rgba(255, 255, 255, ${alpha})`);
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
        
        this.intensityCtx.fillStyle = gradient;
        this.intensityCtx.fillRect(ix - radius, iy - radius, radius * 2, radius * 2);
        
        // 履歴に追加
        this.gazeHistory.push({ x, y, timestamp });
        if (this.gazeHistory.length > this.maxHistory) {
            this.gazeHistory.shift();
        }
        
        this.statistics.totalPoints++;
    }
    
    /**
     * 減衰を適用（時間経過で薄くなる）
     */
    applyDecay() {
        const imageData = this.intensityCtx.getImageData(
            0, 0, this.intensityCanvas.width, this.intensityCanvas.height
        );
        const data = imageData.data;
        
        for (let i = 0; i < data.length; i += 4) {
            // RGBを減衰
            data[i] = Math.floor(data[i] * this.config.decay);
            data[i + 1] = Math.floor(data[i + 1] * this.config.decay);
            data[i + 2] = Math.floor(data[i + 2] * this.config.decay);
        }
        
        this.intensityCtx.putImageData(imageData, 0, 0);
    }
    
    /**
     * ヒートマップを描画
     */
    render(targetCanvas = null) {
        const canvas = targetCanvas || this.outputCanvas;
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        // 強度マップを取得
        const intensityData = this.intensityCtx.getImageData(
            0, 0, this.intensityCanvas.width, this.intensityCanvas.height
        );
        
        // 出力用ImageDataを作成
        const outputData = ctx.createImageData(canvas.width, canvas.height);
        
        // 強度をカラーに変換
        for (let y = 0; y < this.intensityCanvas.height; y++) {
            for (let x = 0; x < this.intensityCanvas.width; x++) {
                const intensityIdx = (y * this.intensityCanvas.width + x) * 4;
                const intensity = intensityData.data[intensityIdx]; // R値を強度として使用
                
                // グラデーションパレットから色を取得
                const paletteIdx = Math.min(255, intensity) * 4;
                
                // 出力座標（解像度を考慮）
                for (let dy = 0; dy < this.config.resolution; dy++) {
                    for (let dx = 0; dx < this.config.resolution; dx++) {
                        const outX = x * this.config.resolution + dx;
                        const outY = y * this.config.resolution + dy;
                        
                        if (outX < canvas.width && outY < canvas.height) {
                            const outIdx = (outY * canvas.width + outX) * 4;
                            outputData.data[outIdx] = this.gradientPalette[paletteIdx];
                            outputData.data[outIdx + 1] = this.gradientPalette[paletteIdx + 1];
                            outputData.data[outIdx + 2] = this.gradientPalette[paletteIdx + 2];
                            outputData.data[outIdx + 3] = this.gradientPalette[paletteIdx + 3];
                        }
                    }
                }
            }
        }
        
        ctx.putImageData(outputData, 0, 0);
    }
    
    /**
     * 透過ヒートマップとして描画（背景と合成可能）
     */
    renderOverlay(targetCanvas, opacity = 0.6) {
        const canvas = targetCanvas;
        if (!canvas) return;
        
        // 一時キャンバスにヒートマップを描画
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvas.width;
        tempCanvas.height = canvas.height;
        
        this.render(tempCanvas);
        
        // 透過合成
        const ctx = canvas.getContext('2d');
        ctx.globalAlpha = opacity;
        ctx.drawImage(tempCanvas, 0, 0);
        ctx.globalAlpha = 1;
    }
    
    /**
     * ホットスポット（最も注視された領域）を検出
     */
    detectHotspots(topN = 5) {
        const intensityData = this.intensityCtx.getImageData(
            0, 0, this.intensityCanvas.width, this.intensityCanvas.height
        );
        
        // グリッドに分割して強度を集計
        const gridSize = 10;
        const gridCols = Math.ceil(this.intensityCanvas.width / gridSize);
        const gridRows = Math.ceil(this.intensityCanvas.height / gridSize);
        const grid = [];
        
        for (let gy = 0; gy < gridRows; gy++) {
            for (let gx = 0; gx < gridCols; gx++) {
                let sum = 0;
                let count = 0;
                
                for (let y = gy * gridSize; y < Math.min((gy + 1) * gridSize, this.intensityCanvas.height); y++) {
                    for (let x = gx * gridSize; x < Math.min((gx + 1) * gridSize, this.intensityCanvas.width); x++) {
                        const idx = (y * this.intensityCanvas.width + x) * 4;
                        sum += intensityData.data[idx];
                        count++;
                    }
                }
                
                grid.push({
                    x: (gx + 0.5) * gridSize * this.config.resolution,
                    y: (gy + 0.5) * gridSize * this.config.resolution,
                    intensity: sum / count
                });
            }
        }
        
        // 強度でソートして上位を返す
        grid.sort((a, b) => b.intensity - a.intensity);
        this.statistics.hotspots = grid.slice(0, topN);
        
        return this.statistics.hotspots;
    }
    
    /**
     * 統計情報を計算
     */
    computeStatistics() {
        if (this.gazeHistory.length === 0) {
            return this.statistics;
        }
        
        // 平均位置
        let sumX = 0, sumY = 0;
        this.gazeHistory.forEach(p => {
            sumX += p.x;
            sumY += p.y;
        });
        this.statistics.avgPosition = {
            x: sumX / this.gazeHistory.length,
            y: sumY / this.gazeHistory.length
        };
        
        // カバレッジ（画面のどれくらいを見たか）
        const visited = new Set();
        const cellSize = 50;
        this.gazeHistory.forEach(p => {
            const cellX = Math.floor(p.x / cellSize);
            const cellY = Math.floor(p.y / cellSize);
            visited.add(`${cellX},${cellY}`);
        });
        
        const totalCells = Math.ceil(this.config.width / cellSize) * 
                          Math.ceil(this.config.height / cellSize);
        this.statistics.coverage = visited.size / totalCells;
        
        // ホットスポット検出
        this.detectHotspots();
        
        return this.statistics;
    }
    
    /**
     * AOI（関心領域）を追加
     */
    addAOI(name, x, y, width, height) {
        this.aois.push({ name, x, y, width, height, dwellTime: 0, visits: 0 });
    }
    
    /**
     * AOIの統計を計算
     */
    computeAOIStatistics() {
        // 各AOIの滞留時間を計算
        this.aois.forEach(aoi => {
            aoi.dwellTime = 0;
            aoi.visits = 0;
            let wasInside = false;
            
            this.gazeHistory.forEach((point, i) => {
                const isInside = point.x >= aoi.x && point.x <= aoi.x + aoi.width &&
                                point.y >= aoi.y && point.y <= aoi.y + aoi.height;
                
                if (isInside) {
                    if (i > 0) {
                        const dt = point.timestamp - this.gazeHistory[i - 1].timestamp;
                        aoi.dwellTime += dt;
                    }
                    if (!wasInside) {
                        aoi.visits++;
                    }
                }
                wasInside = isInside;
            });
        });
        
        return this.aois;
    }
    
    /**
     * ヒートマップを画像としてエクスポート
     */
    exportAsImage(format = 'png') {
        const canvas = document.createElement('canvas');
        canvas.width = this.config.width;
        canvas.height = this.config.height;
        
        this.render(canvas);
        
        return canvas.toDataURL(`image/${format}`);
    }
    
    /**
     * データをJSONとしてエクスポート
     */
    exportData() {
        return {
            config: this.config,
            statistics: this.computeStatistics(),
            aois: this.computeAOIStatistics(),
            rawData: this.gazeHistory,
            timestamp: new Date().toISOString()
        };
    }
    
    /**
     * サイズを変更
     */
    resize(width, height) {
        this.config.width = width;
        this.config.height = height;
        
        this.intensityCanvas.width = Math.floor(width / this.config.resolution);
        this.intensityCanvas.height = Math.floor(height / this.config.resolution);
        
        this.clearIntensityMap();
    }
}

// グローバルエクスポート
window.GazeHeatmap = GazeHeatmap;
