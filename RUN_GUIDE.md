# 執行教學（H100 伺服器）

## 0. 重要注意

- **GPU 0 壞了，絕對不要用**。每個指令前面都加 `CUDA_VISIBLE_DEVICES=1`（或 2、3）。專案腳本沒設定時預設用 1,2,3，但自己寫的 Python 不會。
- 開跑前先看 `nvidia-smi`，挑沒人在用的卡。一個 FLUX 工作約佔 38 GB，同一張卡塞兩個也放得下，但兩個都會變慢。
- 程式碼在伺服器的 `~/sync_flowedit`，本機 Dropbox 裡的 `sync_flowedit/` 是同一份（改了要記得同步，見第 9 節）。

## 1. 連線與進入環境

在本機終端機（Git Bash 或 PowerShell）：

```bash
ssh <your-server>
```

登入後每次都要先做：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate syncfe && cd ~/sync_flowedit
```

看 GPU 使用狀況：

```bash
nvidia-smi
```

## 2. 先跑測試（CPU，幾秒鐘）

```bash
python tests/test_geometry.py && python tests/test_loop.py
```

全部印出 `ok` 就代表投影和編輯迴圈正常。

## 3. 編輯一張普通照片（單純 FlowEdit）

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/edit_image.py --image outputs/sanity/hotel_room_view.png --src "a photo of a modern hotel room with a bed, a yellow armchair and a desk" --tar "a photo of a modern hotel room with a bed, a red armchair and a desk" --model flux --out outputs/my_test/hotel.png
```

- `--src`：描述原圖的 prompt；`--tar`：把想改的地方換掉的 prompt。兩者越像、越詳細，效果越好。
- `--model`：`flux`（主要）或 `sd3`。約 30 秒（含載入模型）。

## 4. 編輯一張 360° 全景

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/edit_pano.py --image data/pano/ninomaru_teien.jpg --src "a photo of a japanese garden with green trees, a stone path and wooden fences" --tar "a photo of a japanese garden with trees, a stone path and wooden fences, covered in snow in winter" --model flux --method sync --out outputs/my_test --eval
```

輸出在 `outputs/my_test/`：`ninomaru_teien_sync.png`（全景結果）和 `ninomaru_teien_sync.json`（參數、秒數；加 `--eval` 會多 CLIP、LPIPS、接縫比）。

`--method` 三選一：

| 值 | 意思 | 時間（單卡） |
|---|---|---|
| `sync` | 我們的方法 | 約 147 秒 |
| `independent` | 每個視角各自 FlowEdit 再拼接（比較對象） | 約 147 秒 |
| `erp` | 直接編輯整張全景圖（比較對象） | 約 33 秒 |

常用調整參數（只對 `sync`／`independent` 有效的標 *）：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--n_max` | 24 | 編輯強度，越大改越多、結構越容易跑掉 |
| `--g_tar` | 5.5 | 目標 prompt 的 guidance |
| `--views` * | dense18 | 視角配置：dense18（18 個）、ring8_poles（10 個）、cube6（6 個） |
| `--weight_pow` * | 1 | 融合權重的銳利度，越大越少平均 |
| `--noise` * | view | `pano`：在全景抽一次噪聲再投影 |
| `--view_px` * | 768 | 每個視角的解析度 |
| `--erp_h` | 1024 | 全景高度（寬 = 2 倍） |
| `--seed` | 0 | 隨機種子 |

全景圖在 `data/pano/`（80 張），名稱見 `data/pano/list_indoor.txt`、`list_outdoor.txt`；寫好的 prompt 在 `configs/edits.yaml`。

## 5. 跑很久的工作：用 tmux，斷線也不會停

```bash
tmux new -s sfe
```

在 tmux 裡照常進環境、跑指令。要離開但讓它繼續跑：按 `Ctrl+B` 再按 `D`。之後重新接回：

```bash
tmux attach -t sfe
```

或者用 nohup 丟到背景，輸出寫進 log：

```bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/edit_pano.py --image data/pano/hotel_room.jpg --src "a photo of a modern hotel room with a bed, a yellow armchair and a desk" --tar "a photo of a modern hotel room with a bed, a red armchair and a desk" --method sync --out outputs/my_test > logs/my_test.log 2>&1 &
```

看進度：

```bash
tail -f logs/my_test.log
```

## 6. 跑完整 benchmark（124 組 × 3 方法，3 張卡約 4 小時）

```bash
for r in 0 1 2; do CUDA_VISIBLE_DEVICES=$((r+1)) nohup python scripts/run_benchmark.py --config configs/edits.yaml --rank $r --world 3 --model flux --out outputs/bench > logs/bench_rank$r.log 2>&1 < /dev/null & done
```

- 每張卡做完一個就去領下一個（用 `.lock` 檔協調），已完成的會自動跳過，所以中斷後可以重跑同一個指令。**重跑前先刪掉殘留的 lock 檔**：`rm -f outputs/bench/flux/*.lock`
- 失敗的工作記在 `outputs/bench/flux/failed.txt`，其他的會繼續跑。
- 想加參數（例如只測 `weight_pow 4`）：加上 `--methods sync --extra "--weight_pow 4"`，並換一個 `--out` 資料夾。

看進度（完成幾個 / 372）：

```bash
ls outputs/bench/flux/*.json | wc -l
```

彙整成表格：

```bash
python scripts/summarize.py outputs/bench/flux
```

## 7. 小規模驗證（8 組，約 20 分鐘）

`configs/val8.yaml` 是挑出來「各視角獨立有改到、Sync 沒改到」的 8 組，用來快速試新想法：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_benchmark.py --config configs/val8.yaml --methods sync --model flux --out outputs/val/my_idea --extra "--weight_pow 4 --n_max 26"
```

和原本的結果比較（CLIP gain 越高代表編輯越明顯，LPIPS 越低代表結構保留越好）：

```bash
python scripts/compare_runs.py base=outputs/bench/flux:sync my_idea=outputs/val/my_idea/flux:sync
```

## 8. 把結果拿回本機看

在**本機**終端機（不是伺服器上）執行，例如抓單張圖：

```bash
scp <your-server>:~/sync_flowedit/outputs/my_test/ninomaru_teien_sync.png .
```

整個資料夾：

```bash
scp -r <your-server>:~/sync_flowedit/outputs/my_test ./my_test
```

全景圖可以用任何支援等距長方投影（equirectangular）的 360° 檢視器看，例如手機相簿的全景模式；Demo 時再做一個 360° 檢視網頁。

## 9. 改程式後同步到伺服器

在本機 `sync_flowedit/` 資料夾執行（只傳改過的檔案）：

```bash
scp sfe/sync_flowedit.py <your-server>:~/sync_flowedit/sfe/
```

改完先在伺服器跑第 2 節的測試。

## 10. 檔案在哪裡

| 路徑（伺服器 `~/sync_flowedit/`） | 內容 |
|---|---|
| `sfe/` | 核心程式：`geometry.py` 投影、`flowedit.py` FlowEdit、`sync_flowedit.py` 同步、`metrics.py` 指標、`models.py` 模型 |
| `scripts/` | 執行腳本（本文件用到的都在這） |
| `configs/edits.yaml` | 124 組編輯 prompt（由 `scripts/make_edits.py` 產生，要改 prompt 改那支再重跑） |
| `data/pano/` | 80 張 Poly Haven 全景（CC0） |
| `outputs/bench/flux/` | 完整 benchmark 結果 |
| `outputs/val/` | 驗證子集的各種嘗試 |
| `~/flowedit_official/` | 官方 FlowEdit（只用來對照，不要把它的程式複製進我們的程式，老師會用 Moss 比對） |
