# Sync-FlowEdit: Seamless Text-Based Editing of 360° Panoramas

不需訓練、沒有接縫的 360° 全景文字編輯：在等距長方投影（ERP）畫布上維護一份共享的「編輯位移」D = Z_edit − X_src，每一步投影到 N 個透視視角各自算 FlowEdit 的速度差，再用平滑權重反投影回 ERP 平均（同步），最後對 D 做 Euler 更新。

> FlowEdit 核心迴圈是依論文 Algorithm 1 自行重寫（`sfe/flowedit.py`），沒有複製官方程式碼；官方實作 <https://github.com/fallenshock/FlowEdit>（MIT）只用來對照結果。

## 目錄

| 路徑 | 內容 |
| --- | --- |
| `sfe/geometry.py` | ERP ↔ 透視投影、視角配置（`dense18`、`ring8_poles`、`cube6`…）、反投影加權平均 |
| `sfe/models.py` | SD3 / FLUX.1-dev 包裝：VAE（含環狀 padding 的 ERP 編解碼）、速度場、噪聲排程 |
| `sfe/flowedit.py` | 單張影像 FlowEdit |
| `sfe/sync_flowedit.py` | Sync-FlowEdit（`sync="dv"`）、各視角獨立（`sync="none"`）、整張 ERP 直接編輯（`erp_flowedit`） |
| `sfe/metrics.py` | 透視視角 CLIP、LPIPS、接縫比、視角間不一致度 |
| `scripts/edit_pano.py` | 編輯單張全景 |
| `scripts/run_benchmark.py` | 多 GPU 批次跑 benchmark（可中斷續跑） |
| `scripts/summarize.py` | 彙整成 Markdown 表格 |
| `scripts/download_polyhaven.py` | 下載 Poly Haven CC0 全景圖 |
| `tests/` | CPU 單元測試（投影正確性、玩具 backend 跑完整迴圈） |

## 安裝（H100 機器）

```bash
pip install -r requirements.txt
huggingface-cli login   # FLUX.1-dev、SD3 需要先在 HF 網頁同意授權
python tests/test_geometry.py && python tests/test_loop.py
```

## 使用

```bash
python scripts/edit_pano.py --image data/pano/xxx.jpg --model flux --method sync \
  --src "a 360 degree panorama of a living room" --tar "a 360 degree panorama of a living room, snowy winter" --eval
```

`--method`：`sync`（我們的方法）、`independent`（每個視角各自 FlowEdit 再拼接）、`erp`（直接編輯整張 ERP）。

消融選項：`--views {dense18,ring8_poles,ring8,cube6}`、`--weight {cosine,uniform}`、`--noise {view,pano,warp}`（`warp`：保留高斯分布的球面噪聲 warping，`--noise_up` 控制細網格倍率）、`--src_mode {encode,project}`、`--output {views,erp}`、`--view_px`、`--fov`。

3 張 H100 跑 benchmark（我們伺服器的 GPU 0 故障，只用 GPU 1–3；rank 0–2 對應 GPU 1–3）：

```bash
for r in 0 1 2; do CUDA_VISIBLE_DEVICES=$((r+1)) python scripts/run_benchmark.py --config configs/edits.yaml --rank $r --world 3 --model flux & done; wait
python scripts/summarize.py outputs/bench/flux
```

## 尚未在 GPU 上驗證的部分

`sfe/models.py` 依 diffusers 的 SD3 / FLUX API 撰寫，還沒在真正的權重上跑過；第一次上 H100 時請先跑「單張普通照片的 FlowEdit」對照官方 demo 結果，確認速度場、guidance、噪聲排程都正確，再跑全景。
