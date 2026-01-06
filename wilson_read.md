# 生成pkl 文件

 python tools/data_converter/vad_nuscenes_converter.py nuscenes --root-path ./data --out-dir ./data/ --extra-tag vad_nuscenes --version v1.0 --canbus ./data

# eval results
CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/VAD/VAD_tiny_stage_2.py ckpts/VAD_tiny.pth --launcher none --eval bbox --tmpdir tmp

# visualize results
python tools/analysis_tools/visualization.py --result-path <>--save-path <>

# prune model
python tools/analysis_tools/analysis_struct.py projects/configs/VAD/VAD_tiny_stage_2.py ckpts/VAD_tiny.pth --pruning-ratio 0.1 --output-dir ckpts/ --fine-tune

# try to infer it with pruned model
CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/VAD/VAD_tiny_stage_2.py ckpts/vad_structured_pruned.pth --launcher none --eval bbox --tmpdir tmp