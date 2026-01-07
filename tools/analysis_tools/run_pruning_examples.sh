#!/bin/bash

# VAD 模型剪枝与重训练脚本示例
# 使用前请根据实际情况修改配置文件路径、模型路径和 GPU 配置

# ============================================
# 配置参数
# ============================================
CONFIG="projects/configs/VAD/VAD_tiny_stage_2.py"
CHECKPOINT="ckpts/VAD_tiny.pth"
PRUNING_RATIO=0.1
OUTPUT_DIR="ckpts/pruned"
EPOCHS=18
MASTER_PORT=29500

# ============================================
# 示例 1: 仅剪枝（不训练）
# ============================================
example_prune_only() {
    echo "=== 示例 1: 仅剪枝 ==="
    python tools/analysis_tools/analysis_struct.py \
        ${CONFIG} \
        ${CHECKPOINT} \
        --pruning-ratio ${PRUNING_RATIO} \
        --output-dir ${OUTPUT_DIR}
}

# ============================================
# 示例 2: 剪枝 + 单卡微调
# ============================================
example_single_gpu_finetune() {
    echo "=== 示例 2: 剪枝 + 单卡微调 ==="
    python tools/analysis_tools/analysis_struct.py \
        ${CONFIG} \
        ${CHECKPOINT} \
        --pruning-ratio ${PRUNING_RATIO} \
        --output-dir ${OUTPUT_DIR} \
        --fine-tune \
        --fine-tune-epochs 6 \
        --gpus 1 \
        --gpu-ids 0
}

# ============================================
# 示例 3: 剪枝 + 单卡重训练
# ============================================
example_single_gpu_retrain() {
    echo "=== 示例 3: 剪枝 + 单卡重训练 ==="
    python tools/analysis_tools/analysis_struct.py \
        ${CONFIG} \
        ${CHECKPOINT} \
        --pruning-ratio ${PRUNING_RATIO} \
        --output-dir ${OUTPUT_DIR} \
        --fine-tune \
        --retrain \
        --fine-tune-epochs ${EPOCHS} \
        --gpus 1 \
        --gpu-ids 0
}

# ============================================
# 示例 4: 剪枝 + 多卡微调 (3 GPUs)
# ============================================
example_multi_gpu_finetune() {
    echo "=== 示例 4: 剪枝 + 多卡微调 (3 GPUs) ==="
    python -m torch.distributed.run \
        --nproc_per_node=3 \
        --master_port=${MASTER_PORT} \
        tools/analysis_tools/analysis_struct.py \
        ${CONFIG} \
        ${CHECKPOINT} \
        --pruning-ratio ${PRUNING_RATIO} \
        --output-dir ${OUTPUT_DIR} \
        --fine-tune \
        --fine-tune-epochs 6 \
        --gpus 3 \
        --gpu-ids 0 1 2 \
        --launcher pytorch
}

# ============================================
# 示例 5: 剪枝 + 多卡重训练 (3 GPUs) - 推荐
# ============================================
example_multi_gpu_retrain() {
    echo "=== 示例 5: 剪枝 + 多卡重训练 (3 GPUs) - 推荐 ==="
    python -m torch.distributed.run \
        --nproc_per_node=3 \
        --master_port=${MASTER_PORT} \
        tools/analysis_tools/analysis_struct.py \
        ${CONFIG} \
        ${CHECKPOINT} \
        --pruning-ratio ${PRUNING_RATIO} \
        --output-dir ${OUTPUT_DIR} \
        --fine-tune \
        --retrain \
        --fine-tune-epochs ${EPOCHS} \
        --gpus 3 \
        --gpu-ids 0 1 2 \
        --launcher pytorch
}

# ============================================
# 示例 6: 使用 CUDA_VISIBLE_DEVICES 指定 GPU
# ============================================
example_with_cuda_visible_devices() {
    echo "=== 示例 6: 使用 CUDA_VISIBLE_DEVICES ==="
    CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run \
        --nproc_per_node=3 \
        --master_port=${MASTER_PORT} \
        tools/analysis_tools/analysis_struct.py \
        ${CONFIG} \
        ${CHECKPOINT} \
        --pruning-ratio ${PRUNING_RATIO} \
        --output-dir ${OUTPUT_DIR} \
        --fine-tune \
        --retrain \
        --fine-tune-epochs ${EPOCHS} \
        --gpus 3 \
        --launcher pytorch
}

# ============================================
# 示例 7: 完整流程 - 剪枝 -> 重训练 -> 评估
# ============================================
example_full_pipeline() {
    echo "=== 示例 7: 完整流程 ==="

    # 步骤 1: 剪枝和重训练
    echo "步骤 1: 剪枝和重训练..."
    python -m torch.distributed.run \
        --nproc_per_node=3 \
        --master_port=${MASTER_PORT} \
        tools/analysis_tools/analysis_struct.py \
        ${CONFIG} \
        ${CHECKPOINT} \
        --pruning-ratio 0.3 \
        --output-dir ${OUTPUT_DIR} \
        --fine-tune \
        --retrain \
        --fine-tune-epochs 18 \
        --gpus 3 \
        --gpu-ids 0 1 2 \
        --launcher pytorch

    # 步骤 2: 评估重训练后的模型
    echo "步骤 2: 评估重训练后的模型..."
    CUDA_VISIBLE_DEVICES=0 python tools/test.py \
        ${CONFIG} \
        ${OUTPUT_DIR}/retrained/latest.pth \
        --launcher none \
        --eval bbox \
        --tmpdir tmp

    # 步骤 3: 可视化结果（可选）
    echo "步骤 3: 可视化结果..."
    # python tools/analysis_tools/visualization.py \
    #     --result-path ${OUTPUT_DIR}/retrained/results.pkl \
    #     --save-path ${OUTPUT_DIR}/retrained/visualizations
}

# ============================================
# 帮助信息
# ============================================
show_help() {
    cat << EOF
VAD 模型剪枝与重训练脚本示例

用法:
    bash $(basename $0) [示例编号]

示例列表:
    1 - 仅剪枝（不训练）
    2 - 剪枝 + 单卡微调
    3 - 剪枝 + 单卡重训练
    4 - 剪枝 + 多卡微调 (3 GPUs)
    5 - 剪枝 + 多卡重训练 (3 GPUs) - 推荐
    6 - 使用 CUDA_VISIBLE_DEVICES 指定 GPU
    7 - 完整流程 (剪枝 -> 重训练 -> 评估)

配置参数 (在脚本开头修改):
    CONFIG          = ${CONFIG}
    CHECKPOINT      = ${CHECKPOINT}
    PRUNING_RATIO   = ${PRUNING_RATIO}
    OUTPUT_DIR      = ${OUTPUT_DIR}
    EPOCHS          = ${EPOCHS}
    MASTER_PORT     = ${MASTER_PORT}

示例:
    bash $(basename $0) 5     # 运行示例 5
    bash $(basename $0)       # 显示帮助信息
EOF
}

# ============================================
# 主程序
# ============================================
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    case $1 in
        1)
            example_prune_only
            ;;
        2)
            example_single_gpu_finetune
            ;;
        3)
            example_single_gpu_retrain
            ;;
        4)
            example_multi_gpu_finetune
            ;;
        5)
            example_multi_gpu_retrain
            ;;
        6)
            example_with_cuda_visible_devices
            ;;
        7)
            example_full_pipeline
            ;;
        -h|--help|help)
            show_help
            ;;
        *)
            echo "错误: 未知的示例编号 '$1'"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 运行主程序
main "$@"
