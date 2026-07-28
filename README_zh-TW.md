# Lapin

Lapin 是一個用於**醫學影像分割**的研究框架，同時支援**集中式訓練**與**聯邦學習（Federated Learning）**。此外，本專案包含**遮罩條件式 DDPM** 模組，可在分割前先以生成模型合成大量訓練資料。統一的程式碼結構可在 ISIC 2017/2018 等皮膚病灶資料集上，訓練、驗證並比較多種分割模型。

## 主要特色

- 13 種分割架構的統一訓練與評估流程
- 支援 FedAvg、FedProx、SCAFFOLD 三種聯邦學習演算法
- 遮罩條件式 DDPM，用於合成資料擴增
- 標準分割指標：IoU、Dice、Accuracy、Sensitivity、Specificity
- TensorBoard 紀錄，可選用 Weights & Biases
- 模組化設定：模型、資料集、優化器可獨立配置

## 支援模型

| 模型 | 參數 `--model` |
|------|----------------|
| UNet | `unet` |
| VMUNet | `vmunet` |
| VMUNet V2 | `vmunet-v2` |
| HVMUNet | `hvmunet` |
| U²-Net | `u2net` |
| UNet++ | `unetpp` |
| UNet+++ | `unetppp` |
| TransUNet | `tunet` |
| ResUNet | `resunet` |
| ResUNet++ | `resunetpp` |
| Attention UNet | `attu` |
| R2U-Net | `r2u` |
| R2AttU-Net | `attr2u` |

## 專案結構

```
lapin/
├── train.py                 # 集中式分割訓練
├── train_federated.py       # 聯邦分割訓練
├── generation/              # 遮罩條件式 DDPM 資料生成
│   ├── train.py             # 訓練擴散模型
│   ├── sample.py            # 生成合成資料對
│   └── eval_fid.py          # FID 評估
├── engine.py                # 訓練 / 驗證迴圈
├── utils.py                 # 優化器、損失函數、資料增強
├── configs/config.py        # 設定建構器
├── models/                  # 分割模型實作
├── datasets/dataset.py      # 分割資料集載入
├── federated/               # 聯邦資料切分與 SCAFFOLD 工具
├── data/                    # 資料集目錄（需自行放置）
├── pretrained/              # 預訓練權重（需自行下載）
└── results/                 # 實驗輸出
```

## 環境安裝

```bash
git clone https://github.com/your-org/lapin.git
cd lapin

conda create -n lapin python=3.8
conda activate lapin
pip install -r requirements.txt
```

若使用 VMUNet 系列模型，需額外安裝 Mamba 相關套件：

```bash
pip install triton==2.0.0
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
```

預訓練權重說明請參考 [pretrained/README.md](pretrained/README.md)。

## 資料集準備

請將 ISIC 資料集放置於 `data/` 目錄，格式說明見 [data/README.md](data/README.md)。

範例結構：

```
data/isic2018/
  train/images/
  train/masks/
  val/images/
  val/masks/
```

## 集中式訓練

在完整資料集上訓練模型：

```bash
python train.py --model unet --dataset isic18 --gpu 0 --epochs 300
python train.py --model vmunet --dataset isic18 --gpu 0 --epochs 300 --wandb
python train.py --model hvmunet --dataset isic17 --gpu 0 --batch-size 2
```

主要參數：

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--model` | 模型名稱 | `unet` |
| `--dataset` | `isic18` 或 `isic17` | `isic18` |
| `--gpu` | GPU 編號 | `0` |
| `--epochs` | 訓練輪數 | `300` |
| `--batch-size` | 批次大小 | 依模型預設 |
| `--lr` | 學習率 | `0.001` |
| `--wandb` | 啟用 W&B | 關閉 |

訓練結果（checkpoint、log、可視化）會儲存於 `results/`。

## 合成資料生成

分割訓練前，可使用 `generation/` 下的遮罩條件式 DDPM 擴增資料集。詳細說明見 [generation/README_zh-TW.md](generation/README_zh-TW.md)。

```bash
# 1. 以配對影像與遮罩訓練 DDPM
python -m generation.train \
  --run-name isic18_ddpm \
  --image-path data/isic2018_gen/images \
  --mask-path data/isic2018_gen/masks \
  --num-classes 2 --batch-size 4 --image-size 256 --gpu 0

# 2. 採樣合成影像-遮罩對
python -m generation.sample \
  --run-name isic18_ddpm \
  --mask-path data/isic2018/val/masks \
  --output data/synthetic/isic18_ddpm \
  --num-classes 2 --batch-size 4 --gpu 0

# 3. （可選）評估生成品質
python -m generation.eval_fid data/isic2018/val/images data/synthetic/isic18_ddpm/0_0
```

典型流程：**生成合成資料 → 合併至 `data/` → 執行分割訓練**。

## 聯邦學習訓練

以模擬多客戶端方式進行分散式訓練：

```bash
# ISIC18 上的 FedAvg
python train_federated.py --model unet --dataset isic18 --method fedavg --gpu 0

# FedProx，IID 資料切分
python train_federated.py --model unet --dataset isic18 --method fedprox --iid --gpu 0

# SCAFFOLD，工業場景合併資料集
python train_federated.py --model unet --dataset sd900combine --method scaffold --num-clients 23 --gpu 0
```

主要參數：

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--method` | `fedavg`、`fedprox`、`scaffold` | `fedavg` |
| `--num-clients` | 客戶端數量 | `10` |
| `--frac` | 每輪參與比例 | `0.2` |
| `--mu` | FedProx 近端項係數 | `0.01` |
| `--iid` | 使用 IID 切分 | 預設為 non-IID |

## 評估指標

驗證階段會從混淆矩陣計算以下指標：

- **IoU**（二元分割的 mIoU）
- **Dice**（F1 / DSC）
- **Accuracy**（準確率）
- **Sensitivity**（敏感度 / 召回率）
- **Specificity**（特異度）

## 致謝

本專案參考並整合了 UNet 系列、Vision Mamba、TransUNet 及聯邦學習相關開源實作。感謝 VM-UNet、U²-Net、TransUNet 等工作的作者公開程式碼。

## 授權

請參閱 [LICENSE](LICENSE)。

## 引用

若您在研究中使用了 Lapin，請引用：

```bibtex
@software{lapin2026,
  title  = {Lapin: A Unified Framework for Medical Image Segmentation},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/your-org/lapin}
}
```

English documentation: [README.md](README.md)
