import torch
import torch.nn.utils.prune as prune
from mmcv import Config
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint, save_checkpoint
import os
import argparse
import sys

# 添加项目路径到Python路径
sys.path.insert(0, '/workspace/Bev/VAD')

# 分析模型参数量分布
def analyze_model_structure(model):
    total_params = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            params = module.weight.numel()
            total_params += params
            print(f"{name}: {params/1e6:.2f}M params")
    print(f"Total: {total_params/1e6:.2f}M params")

def register_vad_model(cfg):
    """注册VAD模型，模拟test.py中的注册过程"""
    # 导入自定义模块，这样VAD模型就会被注册
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(f"导入插件模块: {_module_path}")
            plg_lib = importlib.import_module(_module_path)
        else:
            # 如果plugin_dir不存在，使用默认路径
            _module_path = 'projects.mmdet3d_plugin'
            print(f"导入默认插件模块: {_module_path}")
            plg_lib = importlib.import_module(_module_path)

def prune_vad_model(config_file, checkpoint_file, output_file, pruning_ratio=0.3):
    """VAD模型剪枝主函数"""
    
    print(f"开始加载配置: {config_file}")
    # 加载配置和模型
    cfg = Config.fromfile(config_file)
    print("配置加载完成")
    
    # 注册VAD模型
    print("注册VAD模型...")
    register_vad_model(cfg)
    
    print("开始构建模型...")
    # 设置train_cfg为None，与test.py中保持一致
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    print("模型构建完成")
    
    # 加载预训练权重
    print(f"开始加载预训练权重: {checkpoint_file}")
    checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
    print("预训练权重加载完成")
    
    # 设置CLASSES属性，与test.py中保持一致
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        # 如果checkpoint中没有CLASSES信息，需要从配置中获取
        class_names = cfg.get('class_names', [])
        if class_names:
            model.CLASSES = class_names
        else:
            print("警告: 无法确定模型类别")
    
    # 分析模型结构
    print("=== 模型结构分析 ===")
    analyze_model_structure(model)
    
    # 剪枝策略
    pruning_targets = []
    
    print("开始识别剪枝目标...")
    # 1. 剪枝骨干网络（ResNet-50）
    if hasattr(model, 'img_backbone'):
        for name, module in model.img_backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                pruning_targets.append((module, 'weight'))
                print(f"添加剪枝目标: img_backbone.{name}")
    
    # 2. 剪枝FPN颈部网络
    if hasattr(model, 'img_neck'):
        for name, module in model.img_neck.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                pruning_targets.append((module, 'weight'))
                print(f"添加剪枝目标: img_neck.{name}")
    
    # 3. 剪枝Transformer中的线性层（谨慎处理）
    transformer_modules = []
    if hasattr(model, 'pts_bbox_head') and hasattr(model.pts_bbox_head, 'transformer'):
        for name, module in model.pts_bbox_head.transformer.named_modules():
            if isinstance(module, torch.nn.Linear):
                transformer_modules.append((module, 'weight'))
                print(f"添加Transformer剪枝目标: {name}")
    
    print(f"总共找到 {len(pruning_targets)} 个卷积层和 {len(transformer_modules)} 个线性层")
    
    if not pruning_targets and not transformer_modules:
        print("警告: 没有找到可剪枝的层")
        return model
    
    # 应用全局剪枝（优先剪枝卷积层）
    print(f"=== 应用剪枝，比例: {pruning_ratio} ===")
    if pruning_targets:
        prune.global_unstructured(
            pruning_targets,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
        print("卷积层剪枝完成")
    
    # 对Transformer层使用更保守的剪枝
    if transformer_modules:
        prune.global_unstructured(
            transformer_modules,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio * 0.5,  # 更保守的比例
        )
        print("Transformer层剪枝完成")
    
    # 永久移除剪枝掩码
    print("永久移除剪枝掩码...")
    for module, name in pruning_targets:
        prune.remove(module, name)
    for module, name in transformer_modules:
        prune.remove(module, name)
    
    # 保存剪枝后的模型
    print(f"保存剪枝后的模型到: {output_file}")
    
    # 创建新的checkpoint，包含必要的元数据
    new_checkpoint = {
        'meta': checkpoint.get('meta', {}),
        'state_dict': model.state_dict()
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    save_checkpoint(model, output_file, meta=new_checkpoint['meta'])
    
    # 计算压缩率
    if os.path.exists(checkpoint_file):
        original_size = os.path.getsize(checkpoint_file) / (1024 * 1024)
        pruned_size = os.path.getsize(output_file) / (1024 * 1024)
        compression_ratio = (1 - pruned_size/original_size) * 100
        
        print(f"原始模型: {original_size:.1f}MB")
        print(f"剪枝后: {pruned_size:.1f}MB")
        print(f"压缩率: {compression_ratio:.1f}%")
    else:
        print("警告: 无法找到原始模型文件进行大小比较")
    
    return model

def structured_pruning_vad(config_file, checkpoint_file, output_file):
    """结构化剪枝 - 针对通道维度"""
    print("开始结构化剪枝...")
    
    cfg = Config.fromfile(config_file)
    
    # 注册VAD模型
    register_vad_model(cfg)
    
    # 设置train_cfg为None，与test.py中保持一致
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
    
    # 设置CLASSES属性
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    
    # 基于重要性的结构化剪枝
    def channel_importance(weight):
        # 计算每个通道的L2范数作为重要性指标
        return torch.norm(weight, p=2, dim=[1, 2, 3])
    
    pruning_ratio = 0.3
    
    pruned_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if module.groups == 1:  # 避免剪枝分组卷积
                importance = channel_importance(module.weight.data)
                num_pruned = int(module.out_channels * pruning_ratio)
                
                if num_pruned > 0 and num_pruned < module.out_channels:
                    # 找到最不重要的通道
                    _, indices = torch.topk(importance, num_pruned, largest=False)
                    
                    # 创建掩码
                    mask = torch.ones_like(module.weight.data)
                    mask[indices] = 0
                    
                    # 应用剪枝
                    module.weight.data *= mask
                    pruned_layers += 1
                    print(f"结构化剪枝层: {name}, 剪枝通道数: {num_pruned}")
    
    print(f"总共剪枝了 {pruned_layers} 个卷积层")
    
    # 保存模型
    new_checkpoint = {
        'meta': checkpoint.get('meta', {}),
        'state_dict': model.state_dict()
    }
    
    save_checkpoint(model, output_file, meta=new_checkpoint['meta'])
    print(f"结构化剪枝完成，模型保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='VAD模型剪枝工具')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='预训练模型路径')
    parser.add_argument('--pruning-ratio', type=float, default=0.3, help='剪枝比例')
    parser.add_argument('--output-dir', default='./pruned_models', help='输出目录')
    parser.add_argument('--fine-tune', action='store_true', help='是否进行微调')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("VAD模型剪枝工具启动")
    print(f"配置文件: {args.config}")
    print(f"模型文件: {args.checkpoint}")
    print(f"剪枝比例: {args.pruning_ratio}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 50)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print("输出目录已创建！")
    
    # 基础剪枝
    pruned_model_path = os.path.join(args.output_dir, 'vad_pruned.pth')
    print(f"开始基础剪枝，输出路径: {pruned_model_path}")
    
    try:
        model = prune_vad_model(args.config, args.checkpoint, pruned_model_path, args.pruning_ratio)
        print("基础剪枝完成！")
    except Exception as e:
        print(f"基础剪枝失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 可选：结构化剪枝
    structured_path = os.path.join(args.output_dir, 'vad_structured_pruned.pth')
    print(f"开始结构化剪枝，输出路径: {structured_path}")
    
    try:
        structured_pruning_vad(args.config, pruned_model_path, structured_path)
        print("结构化剪枝完成！")
    except Exception as e:
        print(f"结构化剪枝失败: {e}")
        import traceback
        traceback.print_exc()
        # 继续执行，不返回
    
    # 可选：微调
    if args.fine_tune:
        fine_tuned_path = os.path.join(args.output_dir, 'vad_fine_tuned.pth')
        print(f"开始微调，输出路径: {fine_tuned_path}")
        try:
            # 这里需要根据您的实际训练代码进行调整
            print("微调功能需要根据实际训练代码实现")
        except Exception as e:
            print(f"微调失败: {e}")
    
    print("=" * 50)
    print("剪枝流程完成！")
    print("=" * 50)

if __name__ == '__main__':
    main()