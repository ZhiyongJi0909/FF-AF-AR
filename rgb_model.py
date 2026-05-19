# VSOARnet model — Video-based Special Operations Action Recognition network
# Lightweight RepViT-based visual feature extractor for edge deployment.
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import torch
import numpy as np
import math

class TemporalGlobalPool3d(nn.Module):
    """
    修正后的3D全局池化（兼容2D/3D输入）
    Input: [B,C,T,H,W] or [B,C,H,W] -> Output: [B,C]
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        if x.dim() == 5:  # 3D输入 [B,C,T,H,W]
            return self.pool(x).flatten(1)  # [B,C]
        else:  # 2D输入 [B,C,H,W] -> 视为单帧3D
            return self.pool(x.unsqueeze(2)).flatten(1)  # [B,C]


class ECABlock(nn.Module):
    """
    高效通道注意力（ECA）
    通过1D卷积捕获跨通道交互，避免降维
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs(math.log2(channels) / gamma + b))
        k = k if k % 2 == 1 else k + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B,C,H,W]
        y = self.avg_pool(x)  # [B,C,1,1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B,1,C]
        y = self.conv(y)  # [B,1,C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B,C,1,1]
        return x * self.sigmoid(y)

class CFFBlock(nn.Module):
    """
    卷积前馈网络（CFF）
    深度卷积编码空间位置 + GLU门控机制
    """
    def __init__(self, dim, hidden_dim=None, drop=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim
        # 深度可分离卷积：位置感知
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        # 门控线性单元
        self.pwconv1 = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1)
        self.glu_act = nn.Sigmoid()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        residual = x
        # 深度卷积编码几何结构
        x = self.dwconv(x)
        x = self.norm(x)
        # GLU: gating机制筛选关键特征
        x = self.pwconv1(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = x1 * self.glu_act(x2)  # 逐元素门控
        x = self.pwconv2(x)
        x = self.drop(x)
        return x + residual

class RepVITBlock(nn.Module):
    """
    重参数化视觉Transformer块
    训练时：三分支并行（1x1, 3x3dw, Identity）
    推理时：融合为单3x3卷积
    """
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        # 训练时分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.branch3 = nn.BatchNorm2d(dim)  # 纯BN分支
        self.act = nn.ReLU(inplace=True)
        self.eca = ECABlock(dim)
        self.cff = CFFBlock(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        # 推理用融合卷积
        self.fused_conv = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.deploy = False

    def forward(self, x):
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
            is_5d = True
        else:
            is_5d = False
        
        identity = x
        if self.deploy:
            # 推理模式：单卷积
            out = self.act(self.fused_conv(x))
        else:
            # 训练模式：多分支
            out = self.act(
                self.branch1(x) + 
                self.branch2(x) + 
                self.branch3(x)
            )
        out = self.eca(out)
        out = self.cff(out)
        out = self.drop_path(out)
        if is_5d:
            out = out.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
            identity = identity.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return out + identity

    def switch_to_deploy(self):
        """融合多分支为单卷积（核心修正版）"""
        if self.deploy:
            return

        dim = self.branch1[0].in_channels
        # 1. 融合Conv+BN
        k1, b1 = self._fuse_conv_bn(self.branch1[0], self.branch1[1])
        k2, b2 = self._fuse_conv_bn(self.branch2[0], self.branch2[1])
        # 2. 融合纯BN分支（生成单位卷积核）
        k3, b3 = self._fuse_conv_bn(None, self.branch3)
        # 3. 构建3x3融合核
        fused_kernel = torch.zeros(dim, dim, 3, 3, device=k1.device)
        fused_bias = torch.zeros(dim, device=b1.device)
        # branch1: 1x1 -> 中心点
        fused_kernel[:, :, 1:2, 1:2] += k1.unsqueeze(-1).unsqueeze(-1)
        fused_bias += b1
        # branch2: 3x3 depthwise -> 仅对角通道
        for i in range(dim):
            fused_kernel[i, i, :, :] += k2[i, 0, :, :]
        fused_bias += b2
        # branch3: Identity BN -> 对角线单位阵
        diag_indices = torch.arange(dim)
        fused_kernel[diag_indices, diag_indices, 1, 1] += 1.0
        fused_bias += b3
        # 4. 更新融合卷积
        self.fused_conv.weight = nn.Parameter(fused_kernel)
        self.fused_conv.bias = nn.Parameter(fused_bias)
        # 5. 删除训练时分支
        self.__delattr__('branch1')
        self.__delattr__('branch2')
        self.__delattr__('branch3')
        self.deploy = True

    def _fuse_conv_bn(self, conv, bn):
        """融合Conv和BN层"""
        if conv is None:
            # 对于纯BN分支，创建单位卷积
            kernel = torch.zeros(bn.num_features, bn.num_features, 3, 3)
            for i in range(bn.num_features):
                kernel[i, i, 1, 1] = 1.0
            return kernel, torch.zeros(bn.num_features)
        
        # 融合Conv和BN
        std = (bn.running_var + bn.eps).sqrt()
        t = bn.weight / std
        weights = conv.weight * t.reshape(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * bn.weight / std
        
        return weights, bias



class VSOARnet(nn.Module):
    """
    VSOARnet (Video-based Special Operations Action Recognition network)
    Lightweight RepViT-based visual feature extractor for edge deployment.
    """
    def __init__(self, num_classes=27, pretrained=False, drop_path_rate=0.1):
        """
        Args:
            num_classes: 类别数
            pretrained: 是否加载预训练权重
            drop_path_rate: 随机深度丢弃率
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = 3  # 保持与预训练模型一致
        
        # 通道配置 (与UTD预训练模型对齐)
        embed_dims = [48, 96, 192]  # 修改为预训练模型的通道数
        num_blocks = [1, 1, 1]  # 保持与预训练模型一致
        self.dropout = nn.Dropout(drop_path_rate * 1.5)
        # 构建stem层 (与预训练模型完全一致)
        self.stem = nn.Sequential(
            nn.Conv3d(3, embed_dims[0]//2, kernel_size=(3,3,3), 
                     stride=(1,2,2), padding=(1,1,1)),  # 48//2=24
            nn.BatchNorm3d(embed_dims[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dims[0]//2, embed_dims[0], kernel_size=(3,3,3), 
                     stride=(1,2,2), padding=(1,1,1)),  # 24→48
            nn.BatchNorm3d(embed_dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # 构建各个阶段
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            # 创建RepVITBlock序列 (保持与预训练模型一致)
            stage_blocks = nn.Sequential(
                *[RepVITBlock(
                    dim=embed_dims[i],
                    drop_path=drop_path_rate * (sum(num_blocks[:i]) + j) / sum(num_blocks)
                ) for j in range(num_blocks[i])]
            )
            self.stages.append(stage_blocks)
            
            # 在阶段之间添加下采样块（除最后一个阶段外）
            if i < self.num_stages - 1:
                downsample = DownsampleBlock(
                    in_channels=embed_dims[i],
                    out_channels=embed_dims[i+1]
                )
                self.stages.append(downsample)
        
        self.deploy = False
        # 全局池化 (特征提取关键)
        self.global_pool = TemporalGlobalPool3d()
        
        # 特征维度 (用于融合)
        self.feature_dim = embed_dims[-1]
        
        # 分类头
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        self._initialize_weights()
        
       
    
    def _process_input(self, x):
        """Normalize input dimensions to [B, C, T, H, W]."""
        if x.dim() == 5:
            if x.shape[1] == 3:
                pass
            elif x.shape[2] == 3:
                x = x.permute(0, 2, 1, 3, 4)
            else:
                raise ValueError(f"输入张量的通道数不正确！期望3通道，但得到{x.shape[1]}或{x.shape[2]}通道。")
        elif x.dim() == 4:
            x = x.unsqueeze(2)
        else:
            raise ValueError(f"输入张量的维度不正确！期望4D或5D，但得到{x.dim()}D。")
        return x

    def forward_features_with_frames(self, x):
        """
        Extract frame-level and pooled visual features for FFUA fusion.
        Args:
            x: video input [B, C, T, H, W] or [B, T, C, H, W]
        Returns:
            frame_features: [B, feature_dim, T] after spatial pooling
            pooled_features: [B, feature_dim] after temporal + spatial pooling
        """
        x = self._process_input(x)
        x = self.stem(x)  # [B, 48, T, 56, 56]
        for stage in self.stages:
            x = stage(x)  # [B, 192, T, 14, 14]

        # Spatial pooling per frame: [B, C, T, H, W] -> [B, C, T]
        frame_features = x.mean(dim=[3, 4])
        # Temporal pooling: [B, C, T] -> [B, C]
        pooled_features = frame_features.mean(dim=2)
        pooled_features = self.dropout(pooled_features)
        return frame_features, pooled_features

    def forward_features(self, x):
        """
        提取视觉特征 (不经过分类头)
        Args:
            x: 视频输入 [batch_size, channels, frames, height, width] or [batch_size, frames, channels, height, width]
        Returns:
            视觉特征 [batch_size, feature_dim]
        """
        _, features = self.forward_features_with_frames(x)
        return features
    
    def forward(self, x):
        """
        完整前向传播 (含分类)
        Args:
            x: 视频输入 [batch_size, channels, frames, height, width]
        Returns:
            分类结果 [batch_size, num_classes]
        """
        features = self.forward_features(x)
        return self.classifier(features)
    def test_forward(self, batch_size=8, frames=16, height=224, width=224):
        """测试前向传播是否产生正确维度"""
        x = torch.randn(batch_size, 3, frames, height, width)

        # 测试stem
        stem_out = self.stem(x)
        print(f"Stem输出: {stem_out.shape} (期望: [{batch_size}, 48, {frames}, 56, 56])")

        # 测试stages
        stages_out = stem_out
        for i, stage in enumerate(self.stages):
            stages_out = stage(stages_out)
            print(f"Stage {i}输出: {stages_out.shape}")

        # 测试帧级特征提取
        frame_features, pooled_features = self.forward_features_with_frames(x)
        print(f"帧级特征: {frame_features.shape} (期望: [{batch_size}, {self.feature_dim}, T])")
        print(f"池化特征: {pooled_features.shape} (期望: [{batch_size}, {self.feature_dim}])")

        # 测试完整前向
        output = self(x)
        print(f"分类输出: {output.shape} (期望: [{batch_size}, {self.num_classes}])")
    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
class DownsampleBlock(nn.Module):
    """深度可分离下采样块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                stride=2, padding=1, groups=in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        is_5d = False
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
            is_5d = True#torch.Size([128, 48, 56, 56])
        x = self.dw_conv(x)#torch.Size([128, 48, 28, 28])
        x = self.pw_conv(x)#发生变化torch.Size([128, 96, 28, 28])
        x = self.norm(x)
        if is_5d:
             _, C_out, H_out, W_out = x.shape  # 获取正确通道数和空间尺寸
             x = x.reshape(B, T, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)
        return self.act(x)
    



if __name__ == "__main__":
    model = VSOARnet(num_classes=27)
    model.test_forward()