# VAD 模型剪枝与重训练工具使用指南

## 概述

`analysis_struct.py` 是一个用于 VAD 模型剪枝、微调和重训练的工具。支持单卡和多卡分布式训练。

## 功能特性

1. **非结构化剪枝**: L1 范数剪枝，减少模型权重
2. **结构化剪枝**: 基于通道重要性的剪枝
3. **微调模式**: 使用较小学习率在剪枝模型上继续训练
4. **重训练模式**: 从头开始训练剪枝模型（不加载优化器状态）
5. **多卡并行训练**: 支持单卡、多卡 DataParallel 和分布式训练

## 基本用法

### 1. 仅剪枝（不训练）

```bash
python tools/analysis_tools/analysis_struct.py \
    projects/configs/VAD/VAD_tiny_stage_2.py \
    ckpts/VAD_tiny.pth \
    --pruning-ratio 0.1 \
    --output-dir ckpts/pruned
```

**输出**:
- `ckpts/pruned/vad_pruned.pth` - 非结构化剪枝模型
- `ckpts/pruned/vad_structured_pruned.pth` - 结构化剪枝模型

---

### 2. 剪枝 + 单卡微调

```bash
python tools/analysis_tools/analysis_struct.py \
    projects/configs/VAD/VAD_tiny_stage_2.py \
    ckpts/VAD_tiny.pth \
    --pruning-ratio 0.1 \
    --output-dir ckpts/pruned \
    --fine-tune \
    --fine-tune-epochs 6 \
    --gpus 1 \
    --gpu-ids 0
```

**说明**:
- 使用较小学习率（原始学习率 × 0.1）
- 继承优化器状态（如果有）
- 适合轻微调整剪枝后的模型

---

### 3. 剪枝 + 单卡重训练（推荐）

```bash
python tools/analysis_tools/analysis_struct.py \
    projects/configs/VAD/VAD_tiny_stage_2.py \
    ckpts/VAD_tiny.pth \
    --pruning-ratio 0.1 \
    --output-dir ckpts/pruned \
    --fine-tune \
    --retrain \
    --fine-tune-epochs 12 \
    --gpus 1 \
    --gpu-ids 0
```

**说明**:
- 使用中等学习率（原始学习率 × 0.5）
- 从 epoch 0 开始，不加载优化器状态
- 适合剪枝比例较大的情况

---

### 4. 剪枝 + 多卡微调（3 卡）

```bash
# 使用 torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=29500 \
    tools/analysis_tools/analysis_struct.py \
    projects/configs/VAD/VAD_tiny_stage_2.py \
    ckpts/VAD_tiny.pth \
    --pruning-ratio 0.1 \
    --output-dir ckpts/pruned \
    --fine-tune \
    --fine-tune-epochs 6 \
    --gpus 3 \
    --gpu-ids 0 1 2 \
    --launcher pytorch
```

**说明**:
- 使用 PyTorch 分布式训练
- 自动使用 `MMDistributedDataParallel`
- 需要指定 `--launcher pytorch`

---

### 5. 剪枝 + 多卡重训练（3 卡，完整训练）

```bash
# 使用 torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=29500 \
    tools/analysis_tools/analysis_struct.py \
    projects/configs/VAD/VAD_tiny_stage_2.py \
    ckpts/VAD_tiny.pth \
    --pruning-ratio 0.1 \
    --output-dir ckpts/pruned \
    --fine-tune \
    --retrain \
    --fine-tune-epochs 18 \
    --gpus 3 \
    --gpu-ids 0 1 2 \
    --launcher pytorch
```

**说明**:
- 完整的重训练流程
- 适合较大剪枝比例（如 0.3）
- 可恢复剪枝模型的性能

---

### 6. 使用 torch.distributed.run（PyTorch 1.9+）

```bash
# 推荐使用 torch.distributed.run 替代 torch.distributed.launch
python -m torch.distributed.run \
    --nproc_per_node=3 \
    --master_port=29500 \
    tools/analysis_tools/analysis_struct.py \
    projects/configs/VAD/VAD_tiny_stage_2.py \
    ckpts/VAD_tiny.pth \
    --pruning-ratio 0.1 \
    --output-dir ckpts/pruned \
    --fine-tune \
    --retrain \
    --fine-tune-epochs 18 \
    --gpus 3 \
    --gpu-ids 0 1 2 \
    --launcher pytorch
```

---

## 参数说明

### 必选参数

- `config`: 配置文件路径（如 `projects/configs/VAD/VAD_tiny_stage_2.py`）
- `checkpoint`: 预训练模型路径（如 `ckpts/VAD_tiny.pth`）

### 剪枝参数

- `--pruning-ratio`: 剪枝比例，范围 [0.0, 1.0]，默认 0.3
  - 0.1: 轻度剪枝，适合保持性能
  - 0.3: 中度剪枝，平衡大小和性能
  - 0.5: 激进剪枝，可能需要较长重训练
- `--output-dir`: 输出目录，默认 `./pruned_models`

### 训练参数

- `--fine-tune`: 是否进行训练（微调或重训练）
- `--retrain`: 是否从头重训练（配合 `--fine-tune` 使用）
- `--fine-tune-epochs`: 训练轮数，默认 6
  - 微调: 建议 6-12 轮
  - 重训练: 建议 12-18 轮

### GPU 参数

- `--gpus`: GPU 数量，默认 1
- `--gpu-ids`: 指定 GPU ID 列表，如 `--gpu-ids 0 1 2`
- `--launcher`: 启动器类型
  - `none`: 单 GPU 或 DataParallel（默认）
  - `pytorch`: PyTorch 分布式（**多卡必须**）
  - `slurm`: Slurm 集群
  - `mpi`: MPI

---

## 训练模式对比

| 模式 | 学习率倍率 | Warmup | Epoch 起点 | 适用场景 |
|------|-----------|--------|-----------|---------|
| **微调** | 0.1× | 500 iters | 继续 | 小幅剪枝（<0.2），保持性能 |
| **重训练** | 0.5× | 1000 iters | 0 | 中大剪枝（≥0.2），恢复性能 |

---

## 输出目录结构

```
ckpts/pruned/
├── vad_pruned.pth                    # 非结构化剪枝模型
├── vad_structured_pruned.pth         # 结构化剪枝模型
├── fine_tuned/                       # 微调输出（如果使用 --fine-tune）
│   ├── VAD_tiny_stage_2.py          # 配置文件
│   ├── latest.pth                    # 最新 checkpoint
│   ├── epoch_1.pth
│   ├── epoch_2.pth
│   └── ...
└── retrained/                        # 重训练输出（如果使用 --retrain）
    ├── VAD_tiny_stage_2.py
    ├── latest.pth
    └── ...
```

---

## 多卡训练注意事项

### 1. 必须使用 `torch.distributed.launch` 或 `torch.distributed.run`

```bash
# ✅ 正确
python -m torch.distributed.run --nproc_per_node=3 tools/analysis_tools/analysis_struct.py ... --launcher pytorch

# ❌ 错误（多卡会自动改为单卡）
python tools/analysis_tools/analysis_struct.py ... --gpus 3 --launcher none
```

### 2. `--launcher` 必须设置为 `pytorch`

```bash
# ✅ 正确
--launcher pytorch

# ❌ 错误（会被强制改为 pytorch）
--launcher none
```

### 3. GPU ID 与 CUDA_VISIBLE_DEVICES

脚本会自动设置 `CUDA_VISIBLE_DEVICES`，或者手动设置：

```bash
# 方法 1: 使用 --gpu-ids
python -m torch.distributed.run --nproc_per_node=3 ... --gpu-ids 0 1 2 --launcher pytorch

# 方法 2: 手动设置环境变量
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 ... --launcher pytorch
```

### 4. 端口冲突

如果 master_port 被占用，更换端口：

```bash
# 检查端口占用
lsof -i :29500

# 更换端口
python -m torch.distributed.run --master_port=29501 ...
```

---

## 常见问题

### Q1: 多卡训练时出现 "Address already in use"？

**A**: 更换 master_port 或杀死占用端口的进程：

```bash
lsof -i :29500
kill -9 <PID>
```

### Q2: 剪枝后模型精度下降严重？

**A**:
1. 降低剪枝比例（如从 0.3 改为 0.1）
2. 使用 `--retrain` 模式从头训练
3. 增加训练轮数（如 18-24 轮）

### Q3: 微调和重训练该如何选择？

**A**:
- **剪枝比例 < 0.2**: 使用微调（6-12 轮）
- **剪枝比例 ≥ 0.2**: 使用重训练（12-24 轮）

### Q4: 如何检查训练是否使用了多卡？

**A**: 查看日志中的分布式信息：

```
分布式训练初始化成功，world_size: 3
使用GPU数量: 3, GPU ID: range(0, 3)
```

---

## 完整示例

### 场景: 剪枝 30% 并在 3 张 GPU 上重训练 18 轮

```bash
# 1. 设置环境变量（可选）
export CUDA_VISIBLE_DEVICES=0,1,2

# 2. 运行剪枝和重训练
python -m torch.distributed.run \
    --nproc_per_node=3 \
    --master_port=29500 \
    tools/analysis_tools/analysis_struct.py \
    projects/configs/VAD/VAD_tiny_stage_2.py \
    ckpts/VAD_tiny.pth \
    --pruning-ratio 0.3 \
    --output-dir ckpts/pruned_0.3 \
    --fine-tune \
    --retrain \
    --fine-tune-epochs 18 \
    --gpus 3 \
    --gpu-ids 0 1 2 \
    --launcher pytorch

# 3. 评估剪枝和重训练后的模型
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    projects/configs/VAD/VAD_tiny_stage_2.py \
    ckpts/pruned_0.3/retrained/latest.pth \
    --launcher none \
    --eval bbox \
    --tmpdir tmp
```

---

## 参考

- 原始训练脚本: [tools/train.py](../train.py)
- 评估脚本: [tools/test.py](../test.py)
- Wilson 笔记: [wilson_read.md](../../wilson_read.md)
