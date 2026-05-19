# training.py
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, optimizer, scaler, device, cls_weight, align_weight, decorr_weight, 
               current_dropout, current_weight_decay, use_ema=False, ema_model=None, ema_decay=0.999):
    """训练一个epoch（优化版本）"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_align_loss = 0
    total_decorr_loss = 0
    correct = 0
    total = 0
    
    # 进度条
    pbar = tqdm(train_loader, desc=f'[Train]')
    
    # 训练循环函数（避免代码重复）
    def train_loop_body(batch_idx, batch):
        # 声明使用外部函数的变量
        nonlocal total_loss, total_cls_loss, total_align_loss, total_decorr_loss, correct, total
        
        # 检查batch类型并提取所需数据
        try:
            if isinstance(batch, tuple):
                # 处理元组格式，确保至少有3个元素
                if len(batch) >= 3:
                    rgb_clips, imu_data, labels = batch[:3]
                else:
                    raise ValueError(f"Batch tuple has insufficient elements: {len(batch)}. Expected at least 3.")
            elif isinstance(batch, list):
                # 处理列表格式，转换为元组处理
                batch_tuple = tuple(batch)
                if len(batch_tuple) >= 3:
                    rgb_clips, imu_data, labels = batch_tuple[:3]
                else:
                    raise ValueError(f"Batch list has insufficient elements: {len(batch_tuple)}. Expected at least 3.")
            elif isinstance(batch, dict):
                # 尝试多种可能的键名，提高兼容性
                rgb_clips = batch.get('rgb', batch.get('video', None))
                imu_data = batch.get('imu', batch.get('inertial', None))
                labels = batch.get('label', batch.get('target', None))
                # 检查是否所有必要的键都存在
                if rgb_clips is None or imu_data is None or labels is None:
                    missing_keys = []
                    if rgb_clips is None: missing_keys.append('rgb/video')
                    if imu_data is None: missing_keys.append('imu/inertial')
                    if labels is None: missing_keys.append('label/target')
                    raise KeyError(f"Batch missing required keys: {', '.join(missing_keys)}. Available keys: {list(batch.keys())}")
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}. Expected tuple, list or dict.")
        except Exception as e:
            logger.error(f"Batch processing error (batch {batch_idx}): {str(e)}")
            logger.error(f"Batch type: {type(batch)}, content preview: {str(batch)[:500]}")
            raise  # 重新抛出异常以停止训练
        
        # 数据转移到设备（使用非阻塞传输）
        videos = rgb_clips.to(device, non_blocking=True)
        imus = imu_data.to(device, non_blocking=True)
        targets = labels.to(device, non_blocking=True)
        
        # 前向传播（使用混合精度）
        with autocast():
            output_dict = model(videos, imus)
                        
            loss_dict = model.get_loss(
                output_dict, targets,
                cls_weight=cls_weight, 
                align_weight=align_weight,
                decorr_weight=decorr_weight
            )
            # 获取当前批次的损失
            batch_total_loss = loss_dict['total_loss']
            batch_cls_loss = loss_dict['cls_loss']
            batch_align_loss = loss_dict['align_loss']
            batch_decorr_loss = loss_dict['decorr_loss']
            
            # 获取logits用于计算准确率
            logits = output_dict['fusion_logits']
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)  # 优化：使用set_to_none减少内存
        scaler.scale(batch_total_loss).backward()
        
        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器更新
        scaler.step(optimizer)
        scaler.update()
        
        # 更新EMA（只在部分批次更新，减少内存）
        if use_ema and ema_model is not None and batch_idx % 5 == 0:  # 每5个批次更新一次EMA
            with torch.no_grad():
                for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data = (ema_param.data * ema_decay + model_param.data * (1 - ema_decay))
        
        # 统计
        total_loss += batch_total_loss.item()
        total_cls_loss += batch_cls_loss.item()
        total_align_loss += batch_align_loss.item()
        total_decorr_loss += batch_decorr_loss.item()
        
        # 计算准确率（使用inplace操作减少内存）
        with torch.no_grad():
            predicted = logits.argmax(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{total_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total if total > 0 else 0.0:.2f}%',
            'Cls': f'{total_cls_loss/(batch_idx+1):.4f}',
            'Align': f'{total_align_loss/(batch_idx+1):.4f}',
            'Decorr': f'{total_decorr_loss/(batch_idx+1):.4f}',
            'Dropout': f'{current_dropout:.4f}',
            'WD': f'{current_weight_decay:.4f}'
        })
        
        # 清理不需要的中间变量
        del videos, imus, targets, logits, predicted, output_dict, loss_dict
        torch.cuda.empty_cache()
    
    # 执行训练循环
    for batch_idx, batch in enumerate(train_loader):
        train_loop_body(batch_idx, batch)
    
    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_align_loss = total_align_loss / len(train_loader)
    avg_decorr_loss = total_decorr_loss / len(train_loader)
    accuracy = 100. * correct / total if total > 0 else 0.0
    logger.info(f"训练集损失: {avg_loss:.4f}, 分类损失: {avg_cls_loss:.4f}, 对齐损失: {avg_align_loss:.4f}, 去相关损失: {avg_decorr_loss:.4f}, 准确率: {accuracy:.2f}%")
    return avg_loss, avg_cls_loss, avg_align_loss, avg_decorr_loss, accuracy


def validate(model, val_loader, device, cls_weight, align_weight, decorr_weight, epoch=-1, log_confusion=False):
    """验证模型并返回详细损失信息"""
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_align_loss = 0
    total_decorr_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        # 进度条
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        for batch_idx, batch in enumerate(val_loader):
            # 检查batch类型并提取所需数据
            try:
                if isinstance(batch, tuple):
                    if len(batch) >= 3:
                        rgb_clips, imu_data, labels = batch[:3]
                    else:
                        raise ValueError(f"Batch tuple has insufficient elements: {len(batch)}. Expected at least 3.")
                elif isinstance(batch, list):
                    batch_tuple = tuple(batch)
                    if len(batch_tuple) >= 3:
                        rgb_clips, imu_data, labels = batch_tuple[:3]
                    else:
                        raise ValueError(f"Batch list has insufficient elements: {len(batch_tuple)}. Expected at least 3.")
                elif isinstance(batch, dict):
                    rgb_clips = batch.get('rgb', batch.get('video', None))
                    imu_data = batch.get('imu', batch.get('inertial', None))
                    labels = batch.get('label', batch.get('target', None))
                    if rgb_clips is None or imu_data is None or labels is None:
                        missing_keys = []
                        if rgb_clips is None: missing_keys.append('rgb/video')
                        if imu_data is None: missing_keys.append('imu/inertial')
                        if labels is None: missing_keys.append('label/target')
                        raise KeyError(f"Batch missing required keys: {', '.join(missing_keys)}. Available keys: {list(batch.keys())}")
                else:
                    raise TypeError(f"Unsupported batch type: {type(batch)}. Expected tuple, list or dict.")
            except Exception as e:
                logger.error(f"Batch processing error (batch {batch_idx}): {str(e)}")
                logger.error(f"Batch type: {type(batch)}, content preview: {str(batch)[:500]}")
                raise
            
            # 数据转移到设备
            videos = rgb_clips.to(device)
            imus = imu_data.to(device)
            targets = labels.to(device)
            
            # 前向传播（使用混合精度）
            with autocast():
                output_dict = model(videos, imus)
                
                # 计算损失
                loss_dict = model.get_loss(
                    output_dict, targets,
                    cls_weight=cls_weight, 
                    align_weight=align_weight,
                    decorr_weight=decorr_weight
                )
                
                batch_total_loss = loss_dict['total_loss']
                batch_cls_loss = loss_dict['cls_loss']
                batch_align_loss = loss_dict['align_loss']
                batch_decorr_loss = loss_dict['decorr_loss']
                
                # 获取logits用于计算准确率
                logits = output_dict['fusion_logits']
            
            # 统计
            total_loss += batch_total_loss.item()
            total_cls_loss += batch_cls_loss.item()
            total_align_loss += batch_align_loss.item()
            total_decorr_loss += batch_decorr_loss.item()
            
            # 计算准确率
            predicted = logits.argmax(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 收集预测结果用于混淆矩阵
            if log_confusion:
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            
            # 清理不需要的中间变量
            del videos, imus, targets, logits, predicted, output_dict, loss_dict
            torch.cuda.empty_cache()
        
        pbar.close()
    
    avg_loss = total_loss / len(val_loader)
    avg_cls_loss = total_cls_loss / len(val_loader)
    avg_align_loss = total_align_loss / len(val_loader)
    avg_decorr_loss = total_decorr_loss / len(val_loader)
    accuracy = 100. * correct / total if total > 0 else 0.0
    
    logger.info(f"验证集损失: {avg_loss:.4f}, 分类损失: {avg_cls_loss:.4f}, 对齐损失: {avg_align_loss:.4f}, 去相关损失: {avg_decorr_loss:.4f}, 准确率: {accuracy:.2f}%")
    
    return avg_loss, avg_cls_loss, avg_align_loss, avg_decorr_loss, accuracy, all_preds, all_targets
