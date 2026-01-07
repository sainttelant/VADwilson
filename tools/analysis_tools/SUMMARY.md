# analysis_struct.py 修改总结

## 修改概述

已成功修改 `tools/analysis_tools/analysis_struct.py`，使其支持**剪枝后模型的完整重训练**和**多卡并行训练**。

## 核心改进

### 1. 新增重训练模式 ✨
- 添加 `--retrain` 参数，支持从头重训练剪枝模型
- 重训练时使用 `load_from`（仅加载权重），不使用 `resume_from`（会恢复优化器状态）
- 自动从 epoch 0 开始，避免 epoch 冲突

### 2. 完善多卡并行训练 🚀
- 自动检测多卡训练，强制使用分布式
- 兼容 `torch.distributed.run` 和 `torch.distributed.launch`
- 正确初始化分布式环境变量

### 3. 智能学习率调度 📊
- **重训练**: lr = original_lr × 0.5, warmup = 1000 iters
- **微调**: lr = original_lr × 0.1, warmup = 500 iters

### 4. 灵活的输出管理 📁
- 微调输出: `{output_dir}/fine_tuned/`
- 重训练输出: `{output_dir}/retrained/`

## 关键代码修改

### 函数签名更新
```python
def fine_tune_pruned_model(config_file, pruned_checkpoint, output_dir,
                          fine_tune_epochs=6, gpus=1, gpu_ids=None, launcher='none',
                          retrain_from_scratch=False):  # 新增参数
```

### Checkpoint 加载逻辑
```python
if retrain_from_scratch:
    # 重训练：仅加载权重
    cfg.load_from = args.resume_from
    cfg.resume_from = None
else:
    # 微调：智能恢复
    if checkpoint_epoch >= fine_tune_epochs:
        cfg.load_from = args.resume_from
        cfg.resume_from = None
    else:
        cfg.resume_from = args.resume_from
```

### 分布式训练初始化
```python
if len(cfg.gpu_ids) > 1:
    if args.launcher == 'none':
        args.launcher = 'pytorch'  # 自动切换
    init_dist(args.launcher, **cfg.dist_params)
```

## 使用方法

### 基础用法
```bash
# 单卡重训练
python tools/analysis_tools/analysis_struct.py \
    projects/configs/VAD/VAD_tiny_stage_2.py \
    ckpts/VAD_tiny.pth \
    --pruning-ratio 0.1 \
    --output-dir ckpts/pruned \
    --fine-tune --retrain \
    --fine-tune-epochs 18 \
    --gpus 1 --gpu-ids 0
```

### 多卡重训练（推荐）
```bash
# 3 卡并行重训练
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

## 新增文档

### 1. README_pruning.md
- 📖 完整的使用指南
- 包含所有参数说明
- 多种使用场景示例
- 常见问题解答

### 2. run_pruning_examples.sh
- 🚀 7 个即用示例脚本
- 涵盖单卡、多卡、微调、重训练
- 可配置的参数模板

### 3. CHANGELOG_analysis_struct.md
- 📝 详细的更新说明
- 技术实现细节
- 向后兼容性说明

## 参数对比

| 参数 | 微调模式 | 重训练模式 |
|------|---------|-----------|
| `--fine-tune` | ✅ | ✅ |
| `--retrain` | ❌ | ✅ |
| 学习率 | 0.1× original | 0.5× original |
| Warmup | 500 iters | 1000 iters |
| Epoch 起点 | 继续 | 0 |
| 加载策略 | `resume_from` | `load_from` |
| 推荐场景 | 小剪枝 (<0.2) | 大剪枝 (≥0.2) |

## 快速开始

### 查看所有示例
```bash
bash tools/analysis_tools/run_pruning_examples.sh
```

### 运行推荐配置（多卡重训练）
```bash
bash tools/analysis_tools/run_pruning_examples.sh 5
```

### 阅读完整文档
```bash
cat tools/analysis_tools/README_pruning.md
```

## 测试建议

### 1. 单卡测试
```bash
python tools/analysis_tools/analysis_struct.py \
    projects/configs/VAD/VAD_tiny_stage_2.py \
    ckpts/VAD_tiny.pth \
    --pruning-ratio 0.1 \
    --output-dir ckpts/test_pruned \
    --fine-tune --retrain \
    --fine-tune-epochs 2 \
    --gpus 1 --gpu-ids 0
```

### 2. 多卡测试
```bash
python -m torch.distributed.run \
    --nproc_per_node=3 \
    --master_port=29500 \
    tools/analysis_tools/analysis_struct.py \
    projects/configs/VAD/VAD_tiny_stage_2.py \
    ckpts/VAD_tiny.pth \
    --pruning-ratio 0.1 \
    --output-dir ckpts/test_pruned \
    --fine-tune --retrain \
    --fine-tune-epochs 2 \
    --gpus 3 --gpu-ids 0 1 2 \
    --launcher pytorch
```

### 3. 验证分布式训练
查看日志中是否包含：
```
分布式训练初始化成功，world_size: 3
使用GPU数量: 3, GPU ID: range(0, 3)
```

## 兼容性

✅ **完全向后兼容**: 不使用 `--retrain` 时行为与修改前完全一致

## 相关文件

- 主脚本: [analysis_struct.py](./analysis_struct.py)
- 使用指南: [README_pruning.md](./README_pruning.md)
- 示例脚本: [run_pruning_examples.sh](./run_pruning_examples.sh)
- 更新日志: [CHANGELOG_analysis_struct.md](./CHANGELOG_analysis_struct.md)

## 总结

✅ 支持剪枝后重训练
✅ 支持多卡并行训练
✅ 智能学习率调度
✅ 完善的文档和示例
✅ 向后兼容

现在可以高效地对 VAD 模型进行剪枝和重训练了！
