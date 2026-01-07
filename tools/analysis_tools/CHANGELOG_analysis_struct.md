# analysis_struct.py 更新说明

## 更新日期
2026-01-07

## 主要改进

### 1. ✅ 支持剪枝后重训练模式
- **新增 `--retrain` 参数**: 支持从头重训练剪枝后的模型
- **微调 vs 重训练**:
  - **微调模式** (`--fine-tune`): 使用小学习率 (0.1×)，继承优化器状态
  - **重训练模式** (`--fine-tune --retrain`): 使用中等学习率 (0.5×)，从 epoch 0 开始

### 2. ✅ 完善的多卡并行训练支持
- **自动检测和初始化分布式训练**
- **强制要求**: 多卡训练必须使用 `--launcher pytorch`
- **兼容 PyTorch 分布式**: 支持 `torch.distributed.run` 和 `torch.distributed.launch`
- **自动设置环境变量**: `RANK`, `WORLD_SIZE`, `LOCAL_RANK`

### 3. ✅ 智能的 checkpoint 加载策略
- **重训练模式**: 仅加载模型权重 (`load_from`)，不恢复训练状态
- **微调模式**: 检查 epoch 信息，自动选择 `resume_from` 或 `load_from`
- **防止 epoch 冲突**: 自动处理 checkpoint epoch >= 目标 epoch 的情况

### 4. ✅ 灵活的学习率调度
```python
# 重训练模式
lr = original_lr × 0.5
warmup_iters = 1000

# 微调模式
lr = original_lr × 0.1
warmup_iters = 500
```

### 5. ✅ 改进的输出目录管理
- **微调**: `{output_dir}/fine_tuned/`
- **重训练**: `{output_dir}/retrained/`

## 新增参数

### `--retrain`
- **类型**: `action='store_true'`
- **作用**: 启用重训练模式（必须配合 `--fine-tune` 使用）
- **示例**:
```bash
python tools/analysis_tools/analysis_struct.py \
    config.py checkpoint.pth \
    --fine-tune --retrain \
    --fine-tune-epochs 18
```

## 使用场景推荐

| 剪枝比例 | 推荐模式 | 训练轮数 | 命令示例 |
|---------|---------|---------|---------|
| < 0.2 | 微调 | 6-12 | `--fine-tune --fine-tune-epochs 6` |
| 0.2-0.4 | 重训练 | 12-18 | `--fine-tune --retrain --fine-tune-epochs 18` |
| > 0.4 | 重训练 | 18-24 | `--fine-tune --retrain --fine-tune-epochs 24` |

## 多卡训练示例

### 单卡
```bash
python tools/analysis_tools/analysis_struct.py \
    projects/configs/VAD/VAD_tiny_stage_2.py \
    ckpts/VAD_tiny.pth \
    --pruning-ratio 0.1 \
    --output-dir ckpts/pruned \
    --fine-tune --retrain \
    --fine-tune-epochs 18 \
    --gpus 1 --gpu-ids 0
```

### 多卡 (3 GPUs)
```bash
python -m torch.distributed.run \
    --nproc_per_node=3 \
    --master_port=29500 \
    tools/analysis_tools/analysis_struct.py \
    projects/configs/VAD/VAD_tiny_stage_2.py \
    ckpts/VAD_tiny.pth \
    --pruning-ratio 0.1 \
    --output-dir ckpts/pruned \
    --fine-tune --retrain \
    --fine-tune-epochs 18 \
    --gpus 3 --gpu-ids 0 1 2 \
    --launcher pytorch
```

## 向后兼容性

✅ **完全兼容旧命令**: 不带 `--retrain` 参数时行为与之前完全一致

```bash
# 旧命令仍然有效
python tools/analysis_tools/analysis_struct.py \
    config.py checkpoint.pth \
    --fine-tune --fine-tune-epochs 6
```

## 技术细节

### checkpoint 加载逻辑
```python
if retrain_from_scratch:
    cfg.load_from = pruned_checkpoint  # 仅加载权重
    cfg.resume_from = None             # 不恢复训练状态
else:
    # 微调模式：智能选择
    if checkpoint_epoch >= target_epochs:
        cfg.load_from = pruned_checkpoint
        cfg.resume_from = None
    else:
        cfg.resume_from = pruned_checkpoint
```

### 分布式训练初始化
```python
if len(gpu_ids) > 1:
    if launcher == 'none':
        launcher = 'pytorch'  # 自动切换
    init_dist(launcher, **cfg.dist_params)
```

## 已知限制

1. **workers_per_gpu 设置为 0**: 为避免序列化问题，当前禁用了多进程数据加载
2. **多卡必须使用分布式**: `launcher=none` 时会自动降级为单卡

## 文档和示例

- 📖 **详细文档**: [README_pruning.md](./README_pruning.md)
- 🚀 **示例脚本**: [run_pruning_examples.sh](./run_pruning_examples.sh)

## 快速开始

查看所有示例：
```bash
bash tools/analysis_tools/run_pruning_examples.sh
```

运行推荐示例（多卡重训练）：
```bash
bash tools/analysis_tools/run_pruning_examples.sh 5
```
