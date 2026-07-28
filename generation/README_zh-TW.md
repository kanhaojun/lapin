# 遮罩條件式 DDPM 資料生成

本模組訓練 **遮罩條件式 DDPM（Denoising Diffusion Probabilistic Model）**，用於生成醫學影像，供下游分割實驗擴增資料。

## 流程概覽

1. **準備配對資料** — 以 ImageFolder 格式組織影像與遮罩
2. **訓練 DDPM** — 學習在分割遮罩條件下生成影像
3. **採樣與評估** — 產生合成資料集，可選計算 FID

生成的資料可放入 `data/`，供專案根目錄的分割訓練腳本使用。

## 資料格式

訓練時需使用 ImageFolder 結構：

```
path/to/images/
  0/
    img_001.png
  1/
    img_002.png

path/to/masks/
  0/
    img_001.png
  1/
    img_002.png
```

ISIC 資料可整理為：

```
data/isic2018_gen/
  images/0/ ...
  masks/0/ ...
```

訓練前可用 `generation/resize_images.py` 統一解析度。

## 訓練

在**專案根目錄**執行：

```bash
python -m generation.train \
  --run-name isic18_ddpm \
  --image-path data/isic2018_gen/images \
  --mask-path data/isic2018_gen/masks \
  --num-classes 2 \
  --batch-size 4 \
  --image-size 256 \
  --gpu 0
```

Checkpoint 儲存於 `generation/checkpoints/<run-name>/`。
預覽圖儲存於 `generation/outputs/<run-name>/`。

## 採樣生成

從訓練好的模型生成合成影像-遮罩對：

```bash
python -m generation.sample \
  --run-name isic18_ddpm \
  --mask-path data/isic2018/val/masks \
  --output data/synthetic/isic18_ddpm \
  --num-classes 2 \
  --batch-size 4 \
  --image-size 256 \
  --channels 3 \
  --gpu 0
```

輸出結構：

```
data/synthetic/isic18_ddpm/
  0_0/
    0_0_img.png
    0_0_mask.png
  1_1/
    ...
```

這些資料可合併至集中式或聯邦分割實驗（例如 `sd900combine`）。

## FID 評估

需額外安裝：`pip install pytorch-fid scipy==1.11.1`

```bash
# 單一資料夾
python -m generation.eval_fid path/to/real path/to/generated

# 多子資料夾並計算平均
python -m generation.eval_fid path/to/real_root path/to/generated_root --recursive
```

## 工具

```bash
# 批次調整影像尺寸
python -m generation.resize_images input_dir output_dir --size 256
```

## 模型說明

- **UNet_mask**：4 通道輸入（遮罩 + 含噪 RGB），3 通道噪聲預測
- **EMA**：指數移動平均，穩定採樣品質
- **Classifier-free guidance**：訓練時 10% 機率丟棄標籤（可用 `--cfg-dropout` 調整）

## 與分割模組的整合

典型工作流程：

```
真實影像 + 遮罩
        ↓
  generation.train
        ↓
  generation.sample  →  合成影像/遮罩對
        ↓
  合併至 data/       →  train.py / train_federated.py
```

聯邦實驗中，合成資料常與各客戶端真實資料合併，例如放在 `data/sd900_syn_all_local_relay_diff/`。

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `train.py` | DDPM 訓練 |
| `sample.py` | 遮罩條件採樣 |
| `eval_fid.py` | FID 評估 |
| `resize_images.py` | 批次縮放工具 |
| `diffusion.py` | 噪聲排程與採樣 |
| `modules.py` | UNet_mask 與 EMA |
| `dataset.py` | 影像/遮罩資料載入 |

English documentation: [README.md](README.md)
