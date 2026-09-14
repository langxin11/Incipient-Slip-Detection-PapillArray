# Incipient Slip Detection PapillArray（中文上手版）

论文官方实现：[Robust Learning-Based Incipient Slip Detection using the PapillArray Optical Tactile Sensor for Improved Robotic Gripping](https://arxiv.org/pdf/2307.04011.pdf)（原英文 README 见 [README.en.md](README.en.md)）。

本 fork 在官方代码基础上做了两件事：

1. **环境现代化**：从 Anaconda + Python 3.8 迁移到 **uv + Python 3.12**，一条 `uv sync` 命令即可复现环境（torch 2.10 + CUDA 12.8，CPU 机器见 FAQ 切换方法）。
2. **降低上手门槛**：代码归组为 `papillarray/`（共享库）+ `scripts/`（5 个入口脚本）；新增合成数据生成器，**不下载 2GB 真实数据也能先跑通全流程**；修复了一批阻塞运行的 bug（命令行参数类型、评估器加载预训练模型的维度不匹配等，详见文末「本 fork 修复的问题」）。

## 环境要求

- Python **3.12**（uv 会自动下载管理，无需预装）
- [uv](https://docs.astral.sh/uv/)（包管理器）
- GPU 可选：有 NVIDIA 卡（驱动 ≥ CUDA 12.8）自动用 GPU 训练/推理，没有则自动回退 CPU

## 快速开始（5 分钟冒烟，无需真实数据）

```bash
# 1. 安装 uv（已装可跳过；国内网络建议先看下面 FAQ 的镜像配置）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 一键创建 .venv 并安装全部依赖（含 torch cu128）
uv sync

# 3. 生成合成传感器数据（约 6 个 CSV，写入 datasets/fullset/）
uv run python tools/make_synthetic_data.py

# 4. 完整流水线：柱数据提取 → 数据增强 → 训练 2 个 epoch
uv run python scripts/generate_pillar_data.py
uv run python scripts/augment_dataset.py
uv run python scripts/train.py --epochs 2 --model-num 2 --sample-rate 0.5

# 5. 用仓库自带的预训练模型做离线评估（对合成 slip/stop 数据画滑移概率曲线）
uv run python scripts/evaluate_offline.py \
    --slip-path datasets/fullset/case_1/z=0.9-v=1.0-XY.csv \
    --stop-path datasets/fullset/case_1/z=0.9-v=1.0-XY-stop.csv
```

全部跑通后：训练产物在 `outputs/trained_models/`，评估图在 `outputs/eval/`。看到图就说明环境 OK，可以换真实数据了。

> 注意：**所有命令都在仓库根目录执行**（脚本按 `datasets/...` 相对路径读写数据）。

## 真实数据准备

从 [Zenodo](https://zenodo.org/records/13228084) 下载 `data.zip`（1.5 GB，CC BY 4.0，公开免登录，国内可直接访问），解压后放到 `datasets/` 下；备用镜像：作者的 [Google Drive](https://drive.google.com/drive/folders/1wGuRzLHXnhaB8dtsyesGwO4MZjJLhYRW?usp=drive_link)。目录结构：

```
datasets/
├── fullset/                     # 原始 CSV（完整流水线用）
│   └── <move_name>/
│       ├── z=0.9-v=1-XY.csv         # slip 工况
│       └── z=0.9-v=1-XY-stop.csv    # stop 工况（文件名必须以 -stop.csv 结尾）
├── pillar_data/                 # generate_pillar_data.py 的产出
│   ├── pillar_data_train.npy
│   └── pillar_data_test.npy
└── merge/                       # augment_dataset.py 的产出
    └── dataset_train.npy        # 也可直接下载作者增强好的版本
```

**文件名是硬约定**（解析逻辑依赖）：`z={值}-v={速度}-{方向}.csv`；`v ∈ {0.5, 1.0, 2.0}` 的 stop 文件按 2100 帧处停止解析，其余按 2050 帧。CSV 无表头，73 列（time + 9 柱 × 8 项）。

要点（继承自作者说明）：

- `merge/dataset_train.npy` 已经过完整数据增强，**可直接训练**，作者实测不会过拟合——这是最快复现论文部署模型的路径；
- 想自己重新划分训练/测试集，才需要从 `fullset/` 原始 CSV 走完整流水线；
- 评估抓取实验（gripping）数据需要另行下载，目录约定为 `gripping-data/{物体名}/{trans|rot|trans+rot}/{slip|stop}/f={整数}-v={值}-a={值}/sensor_data.npy`。

## 三条使用路线

| 路线 | 命令 | 说明 |
|---|---|---|
| A. 预训练模型直接评估 | `scripts/evaluate_offline.py` | 仓库自带 `pre-trained-models/`（gripping / hexapod 两套 5 模型集成），默认加载 gripping |
| B. 作者增强数据直接训练 | `scripts/train.py --dataset-path datasets` | 用下载好的 `datasets/merge/dataset_train.npy` |
| C. 原始数据完整流水线 | `generate_pillar_data.py` → `augment_dataset.py` → `train.py` | 见「快速开始」3-4 步 |

各脚本参数说明：`uv run python scripts/xxx.py --help`。

## 目录结构与旧文件名对照

```
papillarray/            共享库（uv sync 后自动可 import）
├── constants.py        列名、滤波窗口等全局常量
├── networks.py         SlipDetectGlobalGru 网络          ← NN_networks.py
├── dataloader.py       训练数据加载                       ← NN_torch_dataloader.py
├── data_handler.py     序列解析/增强采样/滤波             ← DATA_handler.py
├── median_filter.py    在线中值滤波器                     ← DATA_filter.py
└── utils.py            参数处理/模型加载/绘图工具          ← utils.py
scripts/
├── generate_pillar_data.py   提取柱数据                   ← DATA_pillar_data_generator.py
├── augment_dataset.py        数据增强                     ← DATA_augmentor.py
├── train.py                  训练                         ← NN_train.py
├── evaluate_offline.py       离线评估（画滑移概率曲线）    ← NN_offline_evaluator.py
└── evaluate_gripping.py      抓取实验批量评估              ← NN_gripping_evaluator.py
tools/make_synthetic_data.py  合成数据冒烟（新增）
pre-trained-models/     自带预训练模型（勿往里写东西）
outputs/                训练与评估产物（gitignore）
```

## FAQ

**数据在哪下？国内能直连吗？**
主链接是 [Zenodo](https://zenodo.org/records/13228084)：公开、免登录、国内一般可直接访问，就一个 `data.zip`（1.5 GB，CC BY 4.0）。Google Drive 只是备用镜像，访问受限时可先用 Zenodo。

**国内网络加速依赖下载？**
给 uv 配置 PyPI 镜像后再 `uv sync`：
```bash
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
```
注意：CUDA 版 torch 走的是 PyTorch 官方源（`download.pytorch.org`），镜像不覆盖；下载慢或超时可 `export UV_HTTP_TIMEOUT=600` 后重试。

**没有 N 卡 / 想装 CPU 版 torch？**
编辑 `pyproject.toml`，删掉 `[tool.uv.sources]` 中 `torch = { index = "pytorch-cu128" }` 一行，重新 `uv sync`（torch 2.10.0 会从 PyPI 装 CPU 轮子，体积小得多）。脚本会自动检测并回退 CPU。

**Windows 能用吗？**
可以。训练输出目录的时间戳已改为无冒号格式（原代码在 Windows 上会因非法目录名报错）。

**预训练模型和自己训的模型输入维度不一样？**
是的：预训练模型是 8 柱 16 特征（640 维输入），本仓库训练流水线产出 9 柱 18 特征（720 维）。`evaluate_offline.py` 和 `utils.load_model` 会按 checkpoint 实际宽度自适应构造网络，两种模型都能评估。

## 本 fork 修复的问题

原仓库直接 `pip install -r requirements.txt` 会失败（`tdqm` 拼写错误），且以下问题会阻塞运行，均已修复：

1. `requirements.txt`：删掉拼错的 `tdqm`（未使用）、`torchvision`（未使用）、`argparse`（标准库，不该安装），补上实际用到的 `matplotlib`、`seaborn`；
2. 5 个入口脚本的 argparse 参数全部缺失 `type=`，任何命令行传参都会变成字符串直接崩溃；
3. `NN_gripping_evaluator.py` 参数定义本身损坏（选项名末尾多空格、参数名与下游不一致、内部重复解析 sys.argv），原仓库唯一跑法是改代码写死路径；
4. `utils.load_model` 用模型数量而非 checkpoint 编号拼文件名，无法加载自带的 `ckpt_final_model_*.pth`；
5. 网络输入维度写死 18 特征，与预训练模型的 16 特征不匹配（改为按权重自适应，见 FAQ）；
6. `torch.load` 显式 `weights_only=True`（torch 2.6+ 默认值变更的兼容声明）；
7. 训练日志除零保护（小数据集 + 高跳采样率时触发）、评估结果被旧文件覆盖、pandas 切片加列告警、输出目录时间戳冒号（Windows）。

## 引用

```bibtex
@article{incipient-slip-papillarray,
  title={Robust Learning-Based Incipient Slip Detection using the PapillArray Optical Tactile Sensor for Improved Robotic Gripping},
  author={Wang, Qiang and Ulloa, Pablo Martinez and Burke, Robert and Bulens, David Cordova and Redmond, Stephen J},
  journal={arXiv preprint arXiv:2307.04011},
  year={2023}
}
```

许可证：MIT（见 [LICENSE](LICENSE)）。
