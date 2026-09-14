# 模型架构说明（SlipDetectGlobalGru）

一句话总结：**MLP 特征提取（projector）→ 单步 GRU 记忆（GRUCell）→ MLP 分类头（predictor）的三段式网络，以 40 ms 为节拍流式运行，5 个独立训练的模型集成取平均**。定义在 `papillarray/networks.py`（另有论文对比用的 LSTM 变体 `SlipDetectGlobalLstm`，仓库主线未使用）。

## 1. 物理背景与输入约定

- PapillArray 传感器是 3×3 方阵的 9 根弹性硅胶柱（P0 为中心柱），以 1 kHz 输出每柱 x-y-z 三向力。
- 特征只用切向力 **FX/FY**，不用 FZ（论文假设 z 力对初始滑移判别作用有限，且数据中 z 近似恒定，防止网络学偏）。
- 原始力先过 21 点中值滤波（`constants.py` 的 `window_size=21`），再**每 40 个采样点打包一窗**（`stack_samples=40`）→ 网络节拍 25 Hz，每 40 ms 出一次滑移概率。

每个时间步的输入向量是「每柱特征数 × 40 点」摊平的一维向量，存在两种配置：

| 配置 | 柱范围 | 每步特征 | 输入维度 | 出处 |
|---|---|---|---|---|
| **部署/预训练模型** | 外圈 8 柱（P1–P8） | 16 | **640** | `pre-trained-models/`（checkpoint 首层实测 640） |
| **本仓库训练流水线** | 全部 9 柱（P0–P8） | 18 | **720** | `augment_dataset.py` 拼特征 + `utils.args_handler` 写死 18 |

差异的来龙去脉见 §5。

## 2. 网络结构与维度流（以预训练模型为例）

```
传感器 1kHz 采样
  │  中值滤波(21点) → 每 40 点打包一窗
  ▼
[输入] 640 维 = 16 特征 × 40 点
  ▼
┌─ projector（特征 MLP）─────────────────────┐
│  Linear 640→1024 → BatchNorm1d → ReLU      │
│  Linear 1024→512 → BatchNorm1d → ReLU      │
└──────────────────┬─────────────────────────┘
                   ▼ (1, 512)   当前窗口的嵌入
┌─ rnn（GRUCell，单步）──────────────────────┐
│  h_new = GRUCell(x, h_old)                 │
│  h 跨窗口传递，编码整条序列的受力历史        │
└──────────────────┬─────────────────────────┘
                   ▼ (1, 512)
┌─ predictor（分类头）───────────────────────┐
│  Linear 512→256 → BN → ReLU                │
│  Linear 256→128 → BN → ReLU                │
│  Linear 128→2  → BN → Softmax              │
└──────────────────┬─────────────────────────┘
                   ▼
      [P(无滑移), P(滑移)]，5 个模型取平均
```

`forward` 就是这三行的串联（`papillarray/networks.py`）：

```python
def forward(self, inp, h):
    x = self.projector(inp)     # 一窗原始力 → hidden 维嵌入
    h = self.rnn(x, h)          # 更新序列记忆
    x = self.predictor(h)       # 记忆 → 二类概率
    return x, h
```

## 3. 参数量（实测）

| 配置 | projector | GRUCell | predictor | 合计 |
|---|---|---|---|---|
| 预训练 gripping/hexapod（640 输入，hidden 512） | 1,184,256 | 1,575,936 | 165,254 | **2,925,446** |
| 默认训练配置（720 输入，hidden 128） | 871,808 | 99,072 | 66,950 | **1,037,830** |

注意参数大头在 projector 和 GRUCell；hidden_dim 对参数量影响极大（GRUCell 参数 ∝ 3·hidden²）。

## 4. 训练与推理的运行方式

这是理解本代码的关键——网络被设计成**单步流式**，不是 `nn.GRU` 那种整条序列一次吃入：

- **训练**（`scripts/train.py`）：DataLoader 按 batch 取出序列后转置成时间步优先；对每个 40 ms 时间步都执行一次前向、算一次 loss 并立刻 `sub_loss.backward()`。`model_num=5` 个模型从不同随机初始化各自训练（论文集成规模 Z=5）。
- **推理**（评估器）：逐窗喂入，隐状态 `h` 手工传递；每条新序列开始时 `h` 清零（训练每 batch 重建、评估每序列重建）。5 个模型的滑移概率取平均，**平均概率首次 ≥ 50% 的时刻即初始滑移检出时刻**。

### 两个必须知道的坑

1. **双重 Softmax**：模型末尾自带 `Softmax`，而训练默认损失是 `CrossEntropyLoss`（期望 logits）——上游就是在这个配置下训出来的，权重有效但概率被压平。自己重训若难收敛，可改用 BCE 或去掉末尾 Softmax。
2. **BatchNorm1d 要求 batch>1**：训练态下 batch=1 直接报错；所有评估/在线推理必须 `model.eval()` 切到 running statistics。

## 5. 为什么有 16 / 18 两种特征配置

- **论文描述**：输入为 9 柱 × FX/FY = 18 维；"只用外圈八柱"那句原文指的是**旋转工况的标签标注**——旋转绕中心柱进行，中心柱按定义永不滑移。
- **实际部署**：仓库代码把中心柱从在线流里去掉了——`constants.py` 的 `online_raw` 注释掉了 P0 两列、`data_handler.reshape_sequence` 检测滑移只遍历 `range(1,9)`、预训练 checkpoint 首层实测 640（= 8 柱 × 2 × 40）。仓库未写明原因，合理推断是中心柱对滑移起点判别信息量最小（承接论文那句话），在线流省掉两维。
- **本仓库的训练流水线**仍拼 9 柱 18 特征（720 维），因此**自己训出的模型与自带预训练模型输入宽度不一致**。`scripts/evaluate_offline.py` 与 `papillarray/utils.py` 的 `load_model` 已按 checkpoint 实际输入宽度自适应构造网络，两种模型都能评估。

## 6. 动手查看结构的工具

```bash
# 层级表 + 参数量（临时装 torchinfo，不动项目依赖）
uv run --with torchinfo python -c "
import torch
from torchinfo import summary
from papillarray.networks import SlipDetectGlobalGru
class A: pass
a = A(); a.input_dim, a.hidden_dim, a.categories = 640, 512, 2
summary(SlipDetectGlobalGru(a), input_size=[(1, 640), (1, 512)])   # forward 是 (inp, h) 双输入
"

# 导出 ONNX 后拖进 netron.app 看交互式计算图（torch 2.10 需补 onnxscript）
uv run --with onnxscript python -c "
import torch
from papillarray.networks import SlipDetectGlobalGru
class A: pass
a = A(); a.input_dim, a.hidden_dim, a.categories = 640, 512, 2
m = SlipDetectGlobalGru(a).eval()
x, h = torch.randn(1, 640), torch.zeros(1, 512)
torch.onnx.export(m, (x, h), '/tmp/slip_gru.onnx', input_names=['window','h'], output_names=['prob','h_new'])
"
uv run --with netron netron /tmp/slip_gru.onnx    # 打开 http://localhost:8080
```

（写论文配图建议手绘/draw.io，三段式结构手绘比自动生成图清晰。）

## 7. 相关文件索引

| 文件 | 角色 |
|---|---|
| `papillarray/networks.py` | 网络定义（GRU/LSTM 两个变体） |
| `papillarray/dataloader.py` | 训练数据切分（特征/标签按 input_dim 划分、按 stack_samples 重排） |
| `scripts/train.py` | 训练入口（逐步损失、集成、ckpt 命名 `ckpt_{epoch\|final}_model_{i}.pth`） |
| `scripts/evaluate_offline.py` / `scripts/evaluate_gripping.py` | 流式推理 + 集成平均 + 阈值检出 |
| `papillarray/utils.py` | `args_handler`（input_dim 推导）、`reload_args`/`reload_info.npy`、`load_model`（按 ckpt 宽度自适应） |
| `papillarray/constants.py` | `online_raw`（在线 8 柱列定义，P0 被注释）、滤波窗口、标签常量 |
