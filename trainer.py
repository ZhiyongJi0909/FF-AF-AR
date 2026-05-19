# trainer.py
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import os
import copy
import numpy as np
import logging
import time

from early_stopping import EarlyStopping
from model_utils import create_ema_model, load_pretrained_vsoarnet, freeze_vsoarnet
from optimization import configure_optimizer, configure_scheduler
from loss_utils import adjust_loss_weights
from training import train_epoch, validate
from utils import setup_logger, plot_training_history, update_dropout, update_weight_decay

logger = logging.getLogger(__name__)


class MultimodalTrainer:
    """多模态融合模型训练器（完整实现）"""
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 test_loader=None,
                 pretrained_vsoarnet_path=None,
                 device='cuda',
                 lr=5e-4,
                 imu_lr=1e-3,
                 weight_decay=5e-4,
                 num_epochs=50,
                 log_dir='./logs_fusion',
                 warmup_epochs=5,
                 freeze_epochs=20,
                 patience=10,
                 ema_decay=0.999,
                 align_weight=0.3,
                 use_wandb=False,
                 min_delta=0.3,
                 project_name="Multimodal-HAR",
                 use_ema=True):  # 新增参数控制是否使用EMA
        """
        初始化多模态训练器
        
        Args:
            model: 多模态融合模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            pretrained_vsoarnet_path: VSOARnet预训练权重路径
            device: 训练设备
            lr: VSOARnet学习率
            imu_lr: IMU模型学习率
            weight_decay: 权重衰减
            num_epochs: 总训练轮数
            log_dir: 日志目录
            warmup_epochs: 预热轮数
            freeze_epochs: 冻结阶段轮数
            patience: 早停耐心值
            ema_decay: EMA衰减率
            align_weight: 特征对齐损失权重
            use_wandb: 是否使用W&B
            project_name: W&B项目名称
            early_stop_acc_threshold: 验证集准确率阈值，达到或超过时提前停止训练
            use_ema: 是否使用EMA模型
        """
        self.device = device
        self.log_dir = log_dir
        self.patience = patience
        os.makedirs(log_dir, exist_ok=True)
        self.num_epochs = num_epochs
        self.min_delta = min_delta
        # 调整初始权重，使验证集分类损失比例更接近训练集
        self.cls_weight = 4.0  # 增加分类权重的初始值
    
        self.decorr_weight = 0.1  # 保持去相关权重较低
        # 多模态训练参数
        self.warmup_epochs = warmup_epochs
        self.freeze_epochs = freeze_epochs
        self.imu_lr = imu_lr
        self.align_weight = align_weight
        self.use_wandb = use_wandb
        self.use_ema = use_ema  # 新增参数控制是否使用EMA
        
        # 初始化动态正则化参数
        self.initial_dropout = 0.3  # 初始dropout值
        self.current_dropout = self.initial_dropout  # 当前dropout值
        self.dropout_decay = 0.95  # dropout衰减率，每个epoch减少5%
        
        self.initial_weight_decay = weight_decay  # 初始权重衰减
        self.current_weight_decay = self.initial_weight_decay  # 当前权重衰减
        self.weight_decay_growth = 1.05  # 权重衰减增长率，每个epoch增加5%

        # 初始化模型
        self.model = model.to(device)
        load_pretrained_vsoarnet(self.model, pretrained_vsoarnet_path, device)
        
        # 创建EMA模型（仅当use_ema为True时）
        self.ema_model = None
        if self.use_ema:
            self.ema_model = create_ema_model(self.model, device)
        
        # self.dropout_schedule = np.linspace(0.1, 0.3, self.num_epochs)
        self.dropout_schedule = {
            'vision': np.linspace(0.4, 0.15, self.num_epochs),  # 视觉分支：0.4→0.15
            'imu': np.linspace(0.2, 0.05, self.num_epochs),     # IMU分支：0.2→0.05
            'fusion': np.linspace(0.3, 0.1, self.num_epochs)    # 融合层：0.3→0.1
        }
        self.weight_decay_schedule = np.linspace(1e-5, 5e-4, self.num_epochs)
        # 训练历史
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_cls_loss': [],
            'train_align_loss': [],
            'train_decorr_loss': [],  # 添加特征去相关损失历史记录
            'train_acc': [],
            'val_loss': [],
            'val_cls_loss': [],  # 新增
            'val_align_loss': [],  # 新增
            'val_decorr_loss': [],  # 新增
            'val_acc': []
        }
        
        # 优化器和学习率调度器
        self.optimizer = configure_optimizer(self.model, lr, self.imu_lr, self.current_weight_decay)  
     
        self.scheduler = configure_scheduler(self.optimizer, self.num_epochs, self.warmup_epochs)
        
        # 混合精度训练（重新启用）
        self.scaler = GradScaler()
        
        # 初始化W&B
        if use_wandb:
            import wandb
            wandb.init(project=project_name, config={
                "lr": lr,
                "imu_lr": imu_lr,
                "weight_decay": weight_decay,
                "batch_size": train_loader.batch_size,
                "num_epochs": num_epochs,
                "warmup_epochs": warmup_epochs,
                "freeze_epochs": freeze_epochs,
                "align_weight": align_weight
            })
            wandb.watch(self.model, log="all", log_freq=100)
        
        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        
        # EMA设置
        self.ema_decay = ema_decay
        self.best_acc = 0.0
        self.best_model = None
        
        # 设置日志
        setup_logger(log_dir)
    
        logger.info("✅ 多模态训练器初始化完成")

    def train(self):
        """开始训练过程"""
        logger.info("🚀 开始训练...")
        
        early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta)
        
        for epoch in range(self.num_epochs):
            logger.info(f"\n📅 Epoch {epoch+1}/{self.num_epochs}")
            update_dropout(self.model, epoch, self.dropout_schedule)
            self.current_weight_decay = update_weight_decay(self.optimizer, epoch, 
                                                           self.initial_weight_decay, 
                                                           self.weight_decay_growth)
            # 1. 训练阶段
            train_loss, train_cls_loss, train_align_loss, train_decorr_loss, train_acc = train_epoch(
                self.model, self.train_loader, self.optimizer, self.scaler, self.device,
                self.cls_weight, self.align_weight, self.decorr_weight,
                self.current_dropout, self.current_weight_decay,
                self.use_ema, self.ema_model, self.ema_decay
            )
            
            # 2. 验证阶段
            val_loss, val_cls_loss, val_align_loss, val_decorr_loss, val_acc, _, _ = validate(
                self.model, self.val_loader, self.device,
                self.cls_weight, self.align_weight, self.decorr_weight, epoch
            )
            
            # 3. 动态调整损失权重
            self.cls_weight, self.align_weight, self.decorr_weight = adjust_loss_weights(
                epoch, self.cls_weight, self.align_weight, self.decorr_weight,
                train_cls_loss, train_align_loss, train_decorr_loss,
                val_cls_loss, val_align_loss, val_decorr_loss
            )
            
            # 4. 更新学习率
            self.scheduler.step()
            
            # 5. 记录训练历史
            self.history['epoch'].append(epoch+1)
            self.history['train_loss'].append(train_loss)
            self.history['train_cls_loss'].append(train_cls_loss)
            self.history['train_align_loss'].append(train_align_loss)
            self.history['train_decorr_loss'].append(train_decorr_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_cls_loss'].append(val_cls_loss)
            self.history['val_align_loss'].append(val_align_loss)
            self.history['val_decorr_loss'].append(val_decorr_loss)
            self.history['val_acc'].append(val_acc)
    
            
            # 6. 检查早停条件
            early_stopping(val_acc, self.model, epoch)
            if early_stopping.early_stop:
                logger.info(f"⏸️  早停触发: 验证集准确率已连续{self.patience}个epoch未提升{self.min_delta}%")
                logger.info(f"🏆 最佳验证集准确率: {early_stopping.best_score:.2f}% (Epoch {early_stopping.best_epoch+1})")
                break
            
            # 7. 更新最佳模型
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_model = copy.deepcopy(self.model)
                logger.info(f"📈 最佳模型更新: 验证集准确率 {val_acc:.2f}%")
        
        # 训练结束后加载最佳权重
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model.state_dict())
        
        plot_training_history(self.history, self.log_dir, self.use_wandb)
        logger.info("🏁 训练完成!")
        return self.history

    def evaluate_final_model(self):
        """评估最终模型"""
        if self.test_loader is None:
            logger.warning("⚠️ 测试集加载器未提供，跳过测试")
            return 0.0
        
        logger.info("📊 评估最终模型...")
        # 使用验证方法评估测试集
        test_loss, test_cls_loss, test_align_loss, test_decorr_loss, test_acc, all_preds, all_targets = validate(
            self.model, self.test_loader, self.device,
            self.cls_weight, self.align_weight, self.decorr_weight, 
            epoch=-1, log_confusion=True
        )
        
        logger.info(f"\n=== 最终测试结果 ===")
        logger.info(f"测试集损失: {test_loss:.4f}")
        logger.info(f"测试集分类损失: {test_cls_loss:.4f}")
        logger.info(f"测试集对齐损失: {test_align_loss:.4f}")
        logger.info(f"测试集去相关损失: {test_decorr_loss:.4f}")
        logger.info(f"测试集准确率: {test_acc:.2f}%")
        
        return test_acc

    def get_best_model(self):
        """获取最佳模型"""
        return self.best_model
