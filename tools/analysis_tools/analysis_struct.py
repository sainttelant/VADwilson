import torch
import torch.nn.utils.prune as prune
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmcv.runner import load_checkpoint, save_checkpoint
import os
import argparse
import sys
import copy
import mmcv
import time
import warnings
from mmcv.runner import get_dist_info, init_dist
from os import path as osp
from mmdet3d.datasets import build_dataset
from mmdet3d.utils import get_root_logger
from mmdet.apis import set_random_seed

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)

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

def fine_tune_pruned_model(config_file, pruned_checkpoint, output_dir,
                          fine_tune_epochs=6, gpus=1, gpu_ids=None, launcher='none',
                          retrain_from_scratch=False):
    """剪枝后微调或重训练

    Args:
        config_file: 配置文件路径
        pruned_checkpoint: 剪枝后的checkpoint路径
        output_dir: 输出目录
        fine_tune_epochs: 训练轮数
        gpus: GPU数量
        gpu_ids: GPU ID列表
        launcher: 启动器类型 (none/pytorch/slurm/mpi)
        retrain_from_scratch: 是否从头重训练（不加载优化器状态和epoch）
    """
    mode_str = "重训练" if retrain_from_scratch else "微调"
    print(f"开始{mode_str}剪枝后的模型...")

    # 设置GPU环境变量
    if gpu_ids:
        gpu_ids_str = ','.join(str(x) for x in gpu_ids)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
        print(f"{mode_str}环节设置可见GPU: {gpu_ids_str}")
    
    # 解析参数
    class Args:
        def __init__(self):
            self.config = config_file
            self.work_dir = output_dir
            self.resume_from = pruned_checkpoint
            self.no_validate = False
            self.gpus = gpus
            self.gpu_ids = list(range(gpus)) if gpu_ids is None else gpu_ids
            self.seed = 0
            self.deterministic = False
            self.cfg_options = None
            self.launcher = launcher
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.autoscale_lr = False
    
    args = Args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # 导入自定义模块
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    # 导入插件模块
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
            _module_dir = os.path.dirname(args.config)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(f"导入默认插件模块: {_module_path}")
            plg_lib = importlib.import_module(_module_path)
    
    # 设置cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    # 设置工作目录
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        mode_suffix = 'retrain' if retrain_from_scratch else 'fine_tune'
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0] + f'_{mode_suffix}')

    # 关键修改：处理checkpoint加载策略
    if args.resume_from is not None and osp.isfile(args.resume_from):
        if retrain_from_scratch:
            # 重训练模式：只加载模型权重，不恢复训练状态
            print(f"重训练模式：仅加载模型权重，从epoch 0开始")
            cfg.load_from = args.resume_from
            cfg.resume_from = None
        else:
            # 微调模式：检查checkpoint状态
            checkpoint = torch.load(args.resume_from, map_location='cpu')
            if 'meta' in checkpoint and 'epoch' in checkpoint['meta']:
                resumed_epoch = checkpoint['meta']['epoch']
                print(f"检测到checkpoint的epoch: {resumed_epoch}")

                # 如果恢复的epoch >= 目标epoch，需要调整
                if resumed_epoch >= fine_tune_epochs:
                    print(f"警告: 恢复的epoch({resumed_epoch}) >= 目标epoch({fine_tune_epochs})")
                    print("将使用load_from而不是resume_from，重新开始训练")
                    cfg.load_from = args.resume_from
                    cfg.resume_from = None
                else:
                    cfg.resume_from = args.resume_from
            else:
                # 如果没有epoch信息，使用load_from
                cfg.load_from = args.resume_from
                cfg.resume_from = None
    else:
        cfg.resume_from = None
        cfg.load_from = None
    
    # 设置GPU
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    
    print(f"使用GPU数量: {len(cfg.gpu_ids)}, GPU ID: {cfg.gpu_ids}")
    
    # 调整训练配置以适应剪枝后模型
    cfg.total_epochs = fine_tune_epochs
    cfg.runner.max_epochs = fine_tune_epochs

    # 修改学习率策略
    if hasattr(cfg, 'optimizer'):
        original_lr = cfg.optimizer.lr
        if retrain_from_scratch:
            # 重训练：使用原始学习率或稍小的学习率
            cfg.optimizer.lr = original_lr * 0.5
            print(f"重训练学习率: {cfg.optimizer.lr} (原始: {original_lr})")
        else:
            # 微调：使用更小的学习率
            cfg.optimizer.lr = original_lr * 0.1
            print(f"微调学习率: {cfg.optimizer.lr} (原始: {original_lr})")

    # 修改学习率调度器
    if hasattr(cfg, 'lr_config'):
        if retrain_from_scratch:
            # 重训练：使用完整的warmup
            if 'warmup' in cfg.lr_config:
                cfg.lr_config.warmup_iters = 1000
        else:
            # 微调：使用较短的warmup
            if 'warmup' in cfg.lr_config:
                cfg.lr_config.warmup_iters = 500
    
    # 修复：添加device属性到cfg
    if torch.cuda.is_available():
        cfg.device = 'cuda'
    else:
        cfg.device = 'cpu'
    
    # 修复：避免多进程数据加载的序列化问题
    # 设置workers_per_gpu为0，避免多进程数据加载
    if hasattr(cfg, 'data') and hasattr(cfg.data, 'workers_per_gpu'):
        original_workers = cfg.data.workers_per_gpu
        cfg.data.workers_per_gpu = 0  # 禁用多进程数据加载
        print(f"禁用多进程数据加载，workers_per_gpu: {original_workers} -> 0")
    
    # 修复分布式训练初始化
    if len(cfg.gpu_ids) > 1:
        # 多GPU训练必须使用分布式训练
        if args.launcher == 'none':
            print("警告: 多GPU训练需要使用分布式训练，自动将launcher设置为pytorch")
            args.launcher = 'pytorch'
        
        # 设置分布式训练所需的环境变量
        if 'RANK' not in os.environ:
            os.environ['RANK'] = '0'
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = str(len(cfg.gpu_ids))
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = '0'
    
    if args.launcher == 'none':
        distributed = False
        # 单GPU训练
        if len(cfg.gpu_ids) > 1:
            print("警告: MMDataParallel只支持单GPU训练，将使用第一块GPU")
            cfg.gpu_ids = [cfg.gpu_ids[0]]
    else:
        distributed = True
        try:
            init_dist(args.launcher, **cfg.dist_params)
            _, world_size = get_dist_info()
            cfg.gpu_ids = range(world_size)
            print(f"分布式训练初始化成功，world_size: {world_size}")
        except Exception as e:
            print(f"分布式训练初始化失败: {e}")
            print("将回退到单GPU训练")
            distributed = False
            cfg.gpu_ids = [cfg.gpu_ids[0]] if cfg.gpu_ids else [0]
    
    # 创建工作目录
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    # 保存配置
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    
    # 初始化日志
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    
    # 确定logger名称
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)
    
    # 初始化元数据
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    
    # 记录基本信息
    logger.info(f'分布式训练: {distributed}')
    logger.info(f'使用GPU: {cfg.gpu_ids}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    
    # 设置随机种子
    if args.seed is not None:
        logger.info(f'设置随机种子: {args.seed}, 确定性: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)
    
    # 构建模型
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    
    logger.info(f'Model:\n{model}')
    
    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]
    
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    
    # 设置checkpoint配置
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__import__('mmdet').__version__,
            mmdet3d_version=__import__('mmdet3d').__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE if hasattr(datasets[0], 'PALETTE') else None)
    
    # 设置CLASSES属性
    model.CLASSES = datasets[0].CLASSES
    
    # 导入自定义训练函数
    try:
        from projects.mmdet3d_plugin.VAD.apis.train import custom_train_model
        use_custom_train = True
    except ImportError as e:
        print(f"无法导入custom_train_model: {e}")
        print("将使用标准训练流程")
        from mmdet.apis import train_detector
        use_custom_train = False
    
    # 开始训练
    print(f"开始{mode_str}训练...")

    # 使用自定义训练函数
    if use_custom_train:
        custom_train_model(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)
    else:
        from mmdet.apis import train_detector
        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)

    print(f"{mode_str}完成！模型保存在: {cfg.work_dir}")
def main():
    parser = argparse.ArgumentParser(description='VAD模型剪枝工具')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='预训练模型路径')
    parser.add_argument('--pruning-ratio', type=float, default=0.3, help='剪枝比例')
    parser.add_argument('--output-dir', default='./pruned_models', help='输出目录')
    parser.add_argument('--fine-tune', action='store_true', help='是否进行微调')
    parser.add_argument('--fine-tune-epochs', type=int, default=6, help='微调轮数')
    
    # 添加GPU相关参数
    parser.add_argument('--gpus', type=int, default=1, help='使用的GPU数量')
    parser.add_argument('--gpu-ids', type=int, nargs='+', help='指定使用的GPU ID列表，例如：--gpu-ids 0 1')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                       default='none', help='分布式启动方式')

    # 添加重训练参数
    parser.add_argument('--retrain', action='store_true',
                       help='剪枝后从头重训练（不加载优化器状态），而不是微调')

    args = parser.parse_args()
    
    # 设置CUDA_VISIBLE_DEVICES环境变量
    if args.gpu_ids:
        gpu_ids_str = ','.join(str(x) for x in args.gpu_ids)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
        print(f"设置可见GPU: {gpu_ids_str}")
        # 更新gpus数量为实际指定的GPU数量
        args.gpus = len(args.gpu_ids)
    
    print("=" * 50)
    print("VAD模型剪枝工具启动")
    print(f"配置文件: {args.config}")
    print(f"模型文件: {args.checkpoint}")
    print(f"剪枝比例: {args.pruning_ratio}")
    print(f"输出目录: {args.output_dir}")
    print(f"微调: {args.fine_tune}")
    if args.fine_tune:
        print(f"训练轮数: {args.fine_tune_epochs}")
        print(f"训练模式: {'重训练 (从头开始)' if args.retrain else '微调'}")
        print(f"GPU数量: {args.gpus}")
        if args.gpu_ids:
            print(f"GPU ID: {args.gpu_ids}")
        print(f"启动器: {args.launcher}")
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
    
    # 可选：微调或重训练
    if args.fine_tune:
        mode_str = "重训练" if args.retrain else "微调"
        output_subdir = 'retrained' if args.retrain else 'fine_tuned'
        output_train_dir = os.path.join(args.output_dir, output_subdir)
        print(f"开始{mode_str}，输出目录: {output_train_dir}")
        try:
            fine_tune_pruned_model(
                args.config,
                structured_path,
                output_train_dir,
                args.fine_tune_epochs,
                gpus=args.gpus,
                gpu_ids=args.gpu_ids,
                launcher=args.launcher,
                retrain_from_scratch=args.retrain
            )
            print(f"{mode_str}完成！")
        except Exception as e:
            print(f"{mode_str}失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()