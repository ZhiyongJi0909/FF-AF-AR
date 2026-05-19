# data_utd_v3.py - 优化版UTD-METHOD数据集加载器
import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
from scipy.io import loadmat
import pandas as pd
from torchvision import transforms
from collections import defaultdict, OrderedDict
import logging
class ModalityDropout:
    """模态级随机丢弃，增强单模态鲁棒性"""
    def __init__(self, p_vision=0.1, p_imu=0.15, p_both=0.05):
        self.p_vision = p_vision  # 视觉丢弃概率
        self.p_imu = p_imu        # IMU丢弃概率
        self.p_both = p_both      # 双模态同时丢弃概率
    
    def __call__(self, vision_clips, imu_data):
        batch_size = vision_clips.size(0)
        # 生成随机掩码
        mask = torch.rand(batch_size, 2)  # [B, 2] 每样本两个模态
        
        # 应用概率：视觉丢弃 | IMU丢弃 | 双模态丢弃
        vision_mask = (mask[:, 0] > self.p_vision) & (mask[:, 1] > self.p_both)
        imu_mask = (mask[:, 1] > self.p_imu) & (mask[:, 0] > self.p_both)
        
        # 零化被丢弃模态
        vision_clips = vision_clips * vision_mask.float().view(-1, 1, 1, 1, 1)
        imu_data = imu_data * imu_mask.float().view(-1, 1, 1)
        
        return vision_clips, imu_data
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def custom_collate_fn(batch):
    """
    自定义collate_fn，用于过滤无效样本
    Args:
        batch: 数据批次，可能包含None值
    Returns:
        过滤后的批次数据，确保所有样本都是有效的
    """
    # 过滤掉无效样本（None值）
    valid_samples = [sample for sample in batch if sample is not None]
    
    if not valid_samples:
        # 如果所有样本都无效，返回空批次
        return None
    
    # 使用默认的collate处理有效样本
    return torch.utils.data.dataloader.default_collate(valid_samples)
class InvalidSampleError(Exception):
    """当样本无效或加载失败时抛出此异常"""
    pass
class UTDMethodDataset(Dataset):
    """
    UTD-METHOD多模态数据集加载器（优化版V3）
    
    目录结构：
    root/
    ├── ClassName/
    │   ├── depth/
    │   │   ├── aX_sY_tZ_depth.mat
    │   └── RGB/
    │       ├── aX_sY_tZ_color/
    │       │   ├── image_00001.jpg
    │       │   └── n_frames
    """
    
    def __init__(self, 
                 root_dir,
                 class_to_idx=None,  # 类别映射
                 split='train',
                 frames_per_clip=16,
                 img_size=112,
                 imu_seq_len=200,
                 balance_strategy='weighted',  # 类别平衡策略: 'weighted', 'oversample', 'undersample', None
                 frame_sampling_mode='uniform',  # 帧采样模式: 'uniform', 'random', 'keyframe'
                 enable_cache=True,
                 cache_size=500):
        """初始化数据集"""
        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.img_size = img_size
        self.imu_seq_len = imu_seq_len
        self.class_to_idx = class_to_idx
        self.split = split
        self.balance_strategy = balance_strategy
        self.frame_sampling_mode = frame_sampling_mode
        self.enable_cache = enable_cache
        self.modality_dropout =  None
        # 加载类别映射（若未提供）
        if self.class_to_idx is None:
            self.class_to_idx = self._load_class_mapping()
        
        # 训练/验证/测试划分标准（改进版）
        self.train_subjects = list(range(1, 7))  # 1-6训练
        self.val_subjects = [7]                 # 7验证
        self.test_subjects = [8]                # 8测试
        
        # 加载样本
        self.samples = self._load_samples(split)
        self._analyze_class_distribution()
        
        # 数据增强
        self._setup_transforms(split)
        
        # 缓存设置
        self.cache = OrderedDict() if enable_cache else {}
        self.cache_size = cache_size
        
        # 类别权重计算（用于损失函数和采样）
        self.class_weights = self._compute_class_weights()
        
        logger.info(f"✅ 数据集初始化完成: {split} - {len(self.samples)} 样本, {len(self.class_to_idx)} 类别")
    
    def _load_class_mapping(self):
        """从classInd.txt加载类别映射"""
        class_file = os.path.join(self.root_dir, '../utdTrainTestlist', 'classInd.txt')
        if os.path.exists(class_file):
            df = pd.read_csv(class_file, delimiter=' ', header=None, names=['id', 'class_name'])
            class_to_idx = {row[1]: row[0] - 1 for _, row in df.iterrows()}
            logger.info(f"✅ 加载类别映射: {len(class_to_idx)} 类")
            return class_to_idx
        else:
            # 自动生成映射（从目录名）
            logger.warning(f"⚠️ 未找到classInd.txt，从目录生成映射")
            classes = sorted([d for d in os.listdir(self.root_dir) 
                            if os.path.isdir(os.path.join(self.root_dir, d))])
            return {cls: idx for idx, cls in enumerate(classes)}
    
    def _compute_class_weights(self):
        """计算类别权重（用于交叉熵损失）"""
        if not self.samples:
            return None
            
        class_counts = defaultdict(int)
        for sample in self.samples:
            class_counts[sample['label']] += 1
        
        total_samples = sum(class_counts.values())
        weights = torch.zeros(len(self.class_to_idx))
        
        for cls, count in class_counts.items():
            weights[cls] = total_samples / count
        
        logger.info(f"⚖️ 类别权重: {weights.tolist()}")
        return weights
    
    def _setup_transforms(self, split):
        """配置数据增强（增强版）"""
        if split == 'train':
            # 视频数据增强
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),  # 增加裁剪范围和比例变化
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),  # 增加垂直翻转（针对某些动作）
                transforms.RandomRotation(20),  # 增加旋转角度
                transforms.ColorJitter(
                    brightness=0.4, 
                    contrast=0.4, 
                    saturation=0.4, 
                    hue=0.2
                ),  # 增强颜色抖动
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1), 
                    scale=(0.9, 1.1)
                ),  # 增强仿射变换
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.4, 
                    scale=(0.02, 0.4), 
                    ratio=(0.3, 3.3),
                    value='random'
                ),  # 增强随机擦除
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            # 测试/验证时的变换（保持一致）
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def _parse_trial_name(self, trial_name):
        """
        解析试验名称
        Args:
            trial_name: 'a6_s1_t1_color' 或 'a6_s1_t1_inertial.mat'
        Returns:
            action_id, subject_id, trial_id
        """
        match = re.match(r'a(\d+)_s(\d+)_t(\d+)', trial_name)
        if not match:
            raise ValueError(f"无法解析试验名称: {trial_name}")
        
        action_id = int(match.group(1))
        subject_id = int(match.group(2))
        trial_id = int(match.group(3))
        
        return action_id, subject_id, trial_id
    
    def _load_samples(self, split):
        """加载样本列表（改进版）"""
        samples = []
        
        # 确定目标subjects
        if split == 'train':
            target_subjects = self.train_subjects
        elif split == 'val':
            target_subjects = self.val_subjects
        elif split == 'test':
            target_subjects = self.test_subjects
        else:
            raise ValueError(f"无效的split值: {split}")
        
        # 遍历类别目录
        for class_name in sorted(self.class_to_idx.keys()):
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(self.root_dir, class_name)
            
            if not os.path.isdir(class_dir):
                logger.warning(f"⚠️ 跳过类别 {class_name}: 目录不存在")
                continue
            
            rgb_dir = os.path.join(class_dir, 'RGB')
            inertial_dir = os.path.join(class_dir, 'inertial')
            
            if not os.path.exists(rgb_dir) or not os.path.exists(inertial_dir):
                logger.warning(f"⚠️ 跳过类别 {class_name}: 缺少RGB或inertial目录")
                continue
            
            # 遍历RGB文件夹
            for trial_folder in sorted(os.listdir(rgb_dir)):
                if not trial_folder.endswith('_color'):
                    continue
                
                # 解析试验信息
                try:
                    action_id, subject_id, trial_id = self._parse_trial_name(trial_folder)
                except ValueError as e:
                    logger.warning(f"⚠️ 跳过 {trial_folder}: {str(e)}")
                    continue
                
                # 根据被试ID划分训练/验证/测试
                if subject_id not in target_subjects:
                    continue
                
                # 构建RGB路径
                rgb_folder = os.path.join(rgb_dir, trial_folder)
                n_frames_file = os.path.join(rgb_folder, 'n_frames')
                
                if not os.path.exists(n_frames_file):
                    logger.warning(f"⚠️ 跳过 {trial_folder}: 缺少n_frames文件")
                    continue
                
                # 查找对应的IMU文件
                imu_filename = f"a{action_id}_s{subject_id}_t{trial_id}_inertial.mat"
                imu_file = os.path.join(inertial_dir, imu_filename)
                
                if not os.path.exists(imu_file):
                    logger.warning(f"⚠️ 跳过 {trial_folder}: 缺少IMU文件 {imu_filename}")
                    continue
                
                # 添加样本
                samples.append({
                    'rgb_folder': rgb_folder,
                    'imu_file': imu_file,
                    'label': class_idx,
                    'subject': subject_id,
                    'trial': trial_id,
                    'action': action_id
                })
        
        if len(samples) == 0:
            raise RuntimeError(f"未加载到任何{split}样本！")
        
        # 应用类别平衡策略
        if self.balance_strategy and split == 'train':
            samples = self._apply_balance_strategy(samples)
        
        logger.info(f"📥 {split}: 加载 {len(samples)} 个样本")
        return samples
    
    def _analyze_class_distribution(self):
        """分析类别分布"""
        class_counts = defaultdict(int)
        for sample in self.samples:
            class_counts[sample['label']] += 1
        
        self.class_distribution = dict(class_counts)
        self.min_class_count = min(class_counts.values())
        self.max_class_count = max(class_counts.values())
        self.median_class_count = np.median(list(class_counts.values()))
        
        logger.info(f"📊 类别分布统计 ({self.split}):")
        logger.info(f"   样本总数: {len(self.samples)}")
        logger.info(f"   类别数: {len(class_counts)}")
        logger.info(f"   最小样本数: {self.min_class_count}")
        logger.info(f"   最大样本数: {self.max_class_count}")
        logger.info(f"   中位数样本数: {self.median_class_count}")
    
    def _apply_balance_strategy(self, samples):
        """应用类别平衡策略"""
        if self.balance_strategy is None:
            return samples
        
        logger.info(f"⚖️ 应用类别平衡策略: {self.balance_strategy}")
        
        # 统计类别分布
        class_samples = defaultdict(list)
        for sample in samples:
            class_samples[sample['label']].append(sample)
        
        balanced_samples = []
        
        if self.balance_strategy == 'oversample':
            # 过采样少样本类别
            target_count = self.median_class_count
            
            for cls, cls_samples in class_samples.items():
                if len(cls_samples) >= target_count:
                    balanced_samples.extend(cls_samples)
                else:
                    # 过采样到目标数量
                    oversampled = cls_samples * (int(target_count / len(cls_samples)) + 1)
                    balanced_samples.extend(oversampled[:int(target_count)])
        
        elif self.balance_strategy == 'undersample':
            # 欠采样多样本类别
            target_count = self.min_class_count
            
            for cls, cls_samples in class_samples.items():
                if len(cls_samples) <= target_count:
                    balanced_samples.extend(cls_samples)
                else:
                    # 随机欠采样到目标数量
                    np.random.shuffle(cls_samples)
                    balanced_samples.extend(cls_samples[:int(target_count)])
        
        elif self.balance_strategy == 'weighted':
            # 加权采样（通过sampler实现，这里不改变原始样本列表）
            balanced_samples = samples
        
        logger.info(f"⚖️ 平衡后样本数: {len(balanced_samples)}")
        return balanced_samples
    
    def _sample_frames(self, n_frames, mode='uniform'):
        """
        帧采样策略（改进版）
        Args:
            n_frames: 总帧数
            mode: 采样模式: 'uniform', 'random', 'keyframe'
        Returns:
            采样的帧索引列表
        """
        if n_frames <= self.frames_per_clip:
            return np.arange(n_frames)
        
        if mode == 'uniform':
            # 均匀采样
            return np.linspace(0, n_frames-1, self.frames_per_clip, dtype=int)
        
        elif mode == 'random':
            # 随机采样
            return np.random.choice(n_frames, self.frames_per_clip, replace=False)
        
        elif mode == 'keyframe':
            # 关键帧采样（基于动作变化）
            # 简化版：均匀采样基础上增加随机偏移
            base_indices = np.linspace(0, n_frames-1, self.frames_per_clip, dtype=int)
            max_offset = max(1, int(n_frames * 0.05))  # 最大偏移量为总帧数的5%
            offsets = np.random.randint(-max_offset, max_offset+1, size=self.frames_per_clip)
            indices = np.clip(base_indices + offsets, 0, n_frames-1)
            return np.unique(indices)[:self.frames_per_clip]  # 去重并确保数量
        
        else:
            logger.warning(f"⚠️ 未知的采样模式: {mode}, 使用默认的均匀采样")
            return np.linspace(0, n_frames-1, self.frames_per_clip, dtype=int)
    
    def _load_video(self, folder):
        """加载视频帧（优化版）"""
        # 读取n_frames
        n_frames_path = os.path.join(folder, 'n_frames')
        with open(n_frames_path, 'r') as f:
            n_frames = int(f.read().strip())
        
        # 帧采样
        indices = self._sample_frames(n_frames, self.frame_sampling_mode)
        
        frames = []
        for idx in indices:
            frame_path = os.path.join(folder, f"image_{idx+1:05d}.jpg")
            frame = cv2.imread(frame_path)
            
            if frame is None:
                logger.warning(f"⚠️ 无法读取帧: {frame_path}")
                continue
            
            # 优化：先调整大小再转换颜色空间
            frame = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 应用变换
            frame = self.transform(frame)
            frames.append(frame)
        
        # 确保有足够的帧
        if len(frames) < self.frames_per_clip:
            # 填充缺失的帧
            while len(frames) < self.frames_per_clip:
                frames.append(frames[-1])  # 重复最后一帧
        
        # 将帧列表转换为张量 [frames_per_clip, C, H, W] -> [C, frames_per_clip, H, W]
        return torch.stack(frames).permute(1, 0, 2, 3)
    
    def _augment_imu(self, imu_data):
        """Enhanced IMU data augmentation with cross-modal synchronization"""
        augmented = imu_data.copy()
        
        # 1. Random scaling
        scale_factor = np.random.uniform(0.9, 1.1)
        augmented *= scale_factor
        
        # 2. Random Gaussian noise with adaptive std
        noise_std = np.random.uniform(0, 0.05)
        augmented += np.random.normal(0, noise_std, augmented.shape)
        
        # 3. Time warping (simulate different sampling rates)
        if np.random.random() < 0.3:
            warping_factor = np.random.uniform(0.95, 1.05)
            new_length = int(len(augmented) * warping_factor)
            if new_length > 0:
                old_indices = np.arange(len(augmented))
                new_indices = np.linspace(0, len(augmented)-1, new_length)
                warped = np.zeros((new_length, augmented.shape[1]))
                for i in range(augmented.shape[1]):
                    warped[:, i] = np.interp(new_indices, old_indices, augmented[:, i])
                # Crop or pad to original length
                if len(warped) > len(augmented):
                    augmented = warped[:len(augmented)]
                else:
                    augmented[:len(warped)] = warped
                    augmented[len(warped):] = warped[-1]
        
        # 4. Random channel dropout (simulate sensor failure)
        if np.random.random() < 0.1:
            dropout_channel = np.random.randint(0, 6)
            augmented[:, dropout_channel] = 0
        
        return augmented
    
    def _load_imu(self, file_path, apply_augmentation=True):
        """加载IMU数据（优化版）"""
        try:
            data = loadmat(file_path)
            imu_data = data.get('d_iner')
            
            if imu_data is None:
                raise KeyError(f"IMU数据加载失败：{file_path}")
            
            # 规范形状
            if len(imu_data.shape) == 1:
                imu_data = imu_data.reshape(-1, 6)
            
            # 确保数据形状正确
            if imu_data.shape[1] != 6:
                raise ValueError(f"IMU数据维度不正确：期望6个通道，实际{imu_data.shape[1]}个通道")
            
            # 重采样（使用NumPy实现）
            if len(imu_data) != self.imu_seq_len:
                old_indices = np.arange(len(imu_data))
                new_indices = np.linspace(0, len(imu_data)-1, self.imu_seq_len)
                resampled_imu = np.zeros((self.imu_seq_len, 6))
                
                for i in range(6):
                    resampled_imu[:, i] = np.interp(new_indices, old_indices, imu_data[:, i])
                
                imu_data = resampled_imu
            
            # 标准化
            mean = imu_data.mean(axis=0, keepdims=True)
            std = imu_data.std(axis=0, keepdims=True) + 1e-8
            imu_data = (imu_data - mean) / std
            
            # 应用增强
            if apply_augmentation and self.transform is not None and self.split == 'train':
                imu_data = self._augment_imu(imu_data)
            
            return torch.FloatTensor(imu_data)  # [seq_len, 6]
        
        except Exception as e:
            logger.error(f"❌ 加载IMU数据失败：{file_path}，错误：{str(e)}")
            # 返回零矩阵作为默认值
            return torch.zeros(self.imu_seq_len, 6, dtype=torch.float32)
    
    def _get_from_cache(self, idx):
        """从缓存获取数据"""
        if idx in self.cache:
            # 更新缓存顺序（LRU策略）
            item = self.cache.pop(idx)
            self.cache[idx] = item
            return item
        return None
    
    def _add_to_cache(self, idx, data):
        """添加数据到缓存"""
        if not self.enable_cache:
            return
        
        if idx in self.cache:
            self.cache.pop(idx)
        elif len(self.cache) >= self.cache_size:
            # 移除最旧的缓存项
            self.cache.popitem(last=False)
        
        self.cache[idx] = data
    
    def __len__(self):
        """返回样本数量"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 检查缓存
        cached_data = self._get_from_cache(idx)
        if cached_data is not None:
            return cached_data
        
        sample = self.samples[idx]
        
        max_retries = 3
        for retry in range(max_retries):
            try:
                rgb = self._load_video(sample['rgb_folder'])
                imu = self._load_imu(sample['imu_file'], apply_augmentation=(self.split == 'train'))
                
                # 验证数据有效性
                if torch.isnan(rgb).any() or torch.isinf(rgb).any():
                    raise ValueError("RGB数据包含NaN或Inf值")
                if torch.isnan(imu).any() or torch.isinf(imu).any():
                    raise ValueError("IMU数据包含NaN或Inf值")
                
                label = sample['label']
                if self.modality_dropout is not None:
                    rgb, imu = self.modality_dropout(rgb, imu)
                data = {
                    'rgb': rgb,
                    'imu': imu,
                    'label': torch.LongTensor([label])[0],
                    'subject': sample['subject'],
                    'trial': sample['trial'],
                    'action': sample['action']
                }
                
                # 添加到缓存
                self._add_to_cache(idx, data)
                
                return data
            
            except Exception as e:
                logger.warning(f"❌ 获取样本 {idx} 失败 (重试 {retry+1}/{max_retries}): {str(e)}")
                # 随机选择另一个样本作为替代
                if retry < max_retries - 1:
                    idx = np.random.randint(0, len(self.samples))
                    sample = self.samples[idx]
    
        # 所有重试都失败，抛出异常
        logger.critical(f"❌ 样本 {idx} 多次尝试加载失败，返回默认值")
        
        # 返回有效的默认值
        return {
            'rgb': torch.zeros(3, self.frames_per_clip, self.img_size, self.img_size),
            'imu': torch.zeros(self.imu_seq_len, 6),
            'label': torch.LongTensor([0])[0],
            'subject': 0,
            'trial': 0,
            'action': 0
        }

def get_utd_dataloaders(root_dir, 
                       batch_size=8, 
                       num_workers=4, 
                       balance_strategy='weighted',
                       frame_sampling_mode='uniform',
                       **kwargs):
    """
    获取UTD数据加载器（优化版）
    Args:
        root_dir: 数据集根目录
        batch_size: 批次大小
        num_workers: 工作线程数
        balance_strategy: 类别平衡策略
        frame_sampling_mode: 帧采样模式
        **kwargs: 其他参数
    Returns:
        train_loader, val_loader, test_loader, class_to_idx
    """
    
    # 加载类别映射
    class_file = os.path.join(root_dir, '../utdTrainTestlist', 'classInd.txt')
    if os.path.exists(class_file):
        df = pd.read_csv(class_file, delimiter=' ', header=None, names=['id', 'class_name'])
        class_to_idx = {row[1]: row[0] - 1 for _, row in df.iterrows()}
    else:
        # 从目录自动生成
        logger.warning(f"⚠️ 未找到classInd.txt，从目录生成映射")
        classes = sorted([d for d in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir, d))])
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # 创建数据集
    logger.info("📂 创建训练数据集...")
    train_dataset = UTDMethodDataset(
        root_dir=root_dir,
        class_to_idx=class_to_idx,
        split='train',
        balance_strategy=balance_strategy,
        frame_sampling_mode=frame_sampling_mode,
        **kwargs
    )
    
    logger.info("📂 创建验证数据集...")
    val_dataset = UTDMethodDataset(
        root_dir=root_dir,
        class_to_idx=class_to_idx,
        split='val',
        balance_strategy=None,  # 验证集不使用平衡策略
        frame_sampling_mode='uniform',  # 验证集使用均匀采样
        **kwargs
    )
    
    logger.info("📂 创建测试数据集...")
    test_dataset = UTDMethodDataset(
        root_dir=root_dir,
        class_to_idx=class_to_idx,
        split='test',
        balance_strategy=None,  # 测试集不使用平衡策略
        frame_sampling_mode='uniform',  # 测试集使用均匀采样
        **kwargs
    )
    
    # 为训练集创建加权采样器（如果需要）
    sampler = None
    if balance_strategy == 'weighted' and len(train_dataset) > 0:
        logger.info("⚖️ 创建加权采样器...")
        # 计算样本权重
        sample_weights = []
        for sample in train_dataset.samples:
            sample_weights.append(train_dataset.class_weights[sample['label']])
            weight = max(weight, 0.1)  # 避免权重过小
            weight = min(weight, 10.0)  # 避免权重过大
            sample_weights.append(weight)
        
        sample_weights = np.array(sample_weights)
        sample_weights /= sample_weights.sum()
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # 优化数据加载器配置
    num_workers = min(8, num_workers, os.cpu_count())  # 限制worker数量
    prefetch_factor = 2  # 减少预取数量
    persistent_workers = False  # 禁用持久化worker
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=custom_collate_fn
    )
    
    logger.info(f"\n📊 数据加载统计:")
    logger.info(f"   训练样本: {len(train_dataset)}")
    logger.info(f"   验证样本: {len(val_dataset)}")
    logger.info(f"   测试样本: {len(test_dataset)}")
    logger.info(f"   类别数: {len(class_to_idx)}")
    logger.info(f"   批次大小: {batch_size}")
    logger.info(f"   工作线程: {num_workers}")
    
    return train_loader, val_loader, test_loader, class_to_idx

def test_dataset():
    """功能测试代码"""
    logger.info("🚀 开始数据集功能测试...")
    
    # 测试路径
    test_root = r'E:\Academic\artical\mutilmodel\code\pre-train\UTD-MHAD'
    try:
        # 测试数据加载器
        train_loader, val_loader, test_loader, class_map = get_utd_dataloaders(
            test_root, 
            batch_size=4, 
            frames_per_clip=8,
            img_size=112,
            imu_seq_len=100,
            balance_strategy='weighted',
            frame_sampling_mode='uniform',
            num_workers=2
        )
        
        # 测试训练集
        logger.info("\n🧪 测试训练集加载...")
        batch = next(iter(train_loader))
        logger.info(f"   训练集批次:")
        logger.info(f"   RGB shape: {batch['rgb'].shape}")
        logger.info(f"   IMU shape: {batch['imu'].shape}")
        logger.info(f"   Labels: {batch['label']}")
        logger.info(f"   Subjects: {batch['subject']}")
        
        # 测试验证集
        logger.info("\n🧪 测试验证集加载...")
        batch = next(iter(val_loader))
        logger.info(f"   验证集批次:")
        logger.info(f"   RGB shape: {batch['rgb'].shape}")
        logger.info(f"   IMU shape: {batch['imu'].shape}")
        logger.info(f"   Labels: {batch['label']}")
        
        # 测试测试集
        logger.info("\n🧪 测试测试集加载...")
        batch = next(iter(test_loader))
        logger.info(f"   测试集批次:")
        logger.info(f"   RGB shape: {batch['rgb'].shape}")
        logger.info(f"   IMU shape: {batch['imu'].shape}")
        logger.info(f"   Labels: {batch['label']}")
        
        # 测试类别映射
        logger.info(f"\n📋 类别映射: {class_map}")
        
        logger.info("\n✅ 数据集功能测试通过！")
        
    except Exception as e:
        logger.error(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

# 快速测试
if __name__ == '__main__':
    test_dataset()