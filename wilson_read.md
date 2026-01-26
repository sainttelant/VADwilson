# train script in original version

python -m torch.distributed.run --nproc_per_node=3 --master_port=2333 tools/train.py projects/configs/VAD/VAD_tiny_stage_2.py --launcher pytorch --deterministic --work-dir ckpts/retrain 

lsof -i :2333
and then you may kill the thread above in order to run the next one

# 生成pkl 文件
 python tools/data_converter/vad_nuscenes_converter.py nuscenes --root-path ./data --out-dir ./data/ --extra-tag vad_nuscenes --version v1.0 --canbus ./data

# eval results
CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/VAD/VAD_tiny_stage_2.py ckpts/VAD_tiny.pth --launcher none --eval bbox --tmpdir tmp

# visualize results
python tools/analysis_tools/visualization.py --result-path <>--save-path <>

# prune model
# 使用单块GPU（例如GPU 0）
python tools/analysis_tools/analysis_struct.py projects/configs/VAD/VAD_tiny_stage_2.py ckpts/VAD_tiny.pth --pruning-ratio 0.1 --output-dir ckpts/ --fine-tune --gpus 1 --gpu-ids 0

# 使用多块GPU（例如GPU 0和1）
python tools/analysis_tools/analysis_struct.py projects/configs/VAD/VAD_tiny_stage_2.py ckpts/VAD_tiny.pth --pruning-ratio 0.1 --output-dir ckpts/ --fine-tune --gpus 2 --gpu-ids 0 1

# 使用分布式训练（所有可用GPU）
python tools/analysis_tools/analysis_struct.py projects/configs/VAD/VAD_tiny_stage_2.py ckpts/VAD_tiny.pth --pruning-ratio 0.1 --output-dir ckpts/ --fine-tune --fine-tune-epochs 18 --gpus 3 --gpu-ids 0 1 2 --launcher none

# 使用特定GPU的分布式训练
CUDA_VISIBLE_DEVICES=0,1,2 python tools/analysis_tools/analysis_struct.py projects/configs/VAD/VAD_tiny_stage_2.py ckpts/VAD_tiny.pth --pruning-ratio 0.1 --output-dir ckpts/ --fine-tune --launcher pytorch --gpus 3



# try to infer it with pruned model
CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/VAD/VAD_tiny_stage_2.py ckpts/vad_structured_pruned.pth --launcher none --eval bbox --tmpdir tmp

# 开始转化剪枝后重新训练的模型到onnx
