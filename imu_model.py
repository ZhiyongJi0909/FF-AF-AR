import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import kaiserord, firwin
import scipy.io as sio
import os
import re
import pywt
class WaveletDenoiser(nn.Module):
    """Wavelet denoising for IMU data"""
    def __init__(self, wavelet='db4', level=3):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        
    def forward(self, x):
        # 简化的小波去噪实现（使用软阈值）
        batch_size, seq_len, channels = x.shape
        denoised = torch.zeros_like(x)
        
        for b in range(batch_size):
            for c in range(channels):
                # 计算小波系数
                coeffs = pywt.wavedec(x[b, :, c].cpu().numpy(), self.wavelet, level=self.level)
                
                # 计算阈值（基于噪声水平）
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(seq_len))
                
                # 软阈值处理
                coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
                
                # 重构信号
                denoised[b, :, c] = torch.from_numpy(pywt.waverec(coeffs, self.wavelet)).to(x.device)
        
        return denoised
class KaiserFilter(nn.Module):
    """Kaiser window low-pass filter for IMU data preprocessing"""
    def __init__(self, cutoff_freq=10.0, fs=100.0, beta=8.0):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.beta = beta
        self.register_buffer('filter_coeffs', self._design_filter())
        
    def _design_filter(self):
        """Design Kaiser window FIR filter"""
        nyq_rate = self.fs / 2.0
        width = 5.0 / nyq_rate
        ripple_db = 60.0
        
        N, beta = kaiserord(ripple_db, width)
        if N % 2 == 0:
            N += 1
        
        taps = firwin(N, self.cutoff_freq / nyq_rate, window=('kaiser', beta))
        return torch.tensor(taps, dtype=torch.float32)
    
    def forward(self, x):
        """Apply low-pass filter to IMU data"""
        batch_size, seq_len, channels = x.shape
        
        x_reshaped = x.permute(0, 2, 1).reshape(batch_size * channels, seq_len)
        
        filter_len = self.filter_coeffs.shape[0]
        padding = (filter_len - 1) // 2
        
        filtered = F.conv1d(
            x_reshaped.unsqueeze(1), 
            self.filter_coeffs.view(1, 1, -1),
            padding=padding
        ).squeeze(1)
        
        filtered = filtered[:, :seq_len].view(batch_size, channels, seq_len).permute(0, 2, 1)
        
        return filtered

class ISOARnet(nn.Module):
    """ISOARnet (Inertial-based Special Operations Action Recognition network)
    Multiscale CNN-BiGRU inertial feature extractor with channel attention.
    """
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Kaiser low-pass filter
        self.kaiser_filter = KaiserFilter(cutoff_freq=10.0, fs=100.0)
        # 新增小波去噪模块
        # self.wavelet_denoiser = WaveletDenoiser()
        # Simplified multi-scale feature extraction with 2 branches
        self.multi_scale_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
        ])
        
        # Bi-directional GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=32 * 2,  # Multi-scale features concatenated (2 branches)
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Simplified channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Sigmoid()
        )
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # GRU initialization
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # CNN initialization
        for conv_layer in self.multi_scale_conv:
            for m in conv_layer:
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: IMU data [batch_size, seq_len, 6]
        Returns:
            frame_features: [batch_size, seq_len, hidden_dim*2] GRU outputs for fluctuation analysis
            pooled_features: [batch_size, output_dim] aggregated features for fusion
        """
        batch_size, seq_len, _ = x.shape

        # Apply Kaiser low-pass filter
        x = self.kaiser_filter(x)
        # Multi-scale feature extraction
        multi_scale_features = []
        x_permuted = x.permute(0, 2, 1)  # [B, C, T]

        for conv in self.multi_scale_conv:
            conv_out = conv(x_permuted)
            multi_scale_features.append(conv_out)

        # Concatenate multi-scale features
        cnn_features = torch.cat(multi_scale_features, dim=1)  # [B, 32*2, T]
        cnn_features = cnn_features.permute(0, 2, 1)  # [B, T, 64]

        # GRU processing
        gru_output, _ = self.gru(cnn_features)  # [B, T, hidden_dim*2]

        # Apply temporal pooling with channel attention
        attention_weights = self.channel_attention(gru_output.mean(dim=1)).unsqueeze(1)
        h_attended = (gru_output * attention_weights).sum(dim=1)

        # Project to target dimension
        pooled_features = self.projection(h_attended)

        return gru_output, pooled_features  # frame_features, pooled_features

def load_imu_data(imu_file, seq_len=200):
    """Load IMU data from .mat file"""
    data = sio.loadmat(imu_file)
    imu_data = data['sensor_fusion']  # Shape: [6, T]
    
    # Transpose to [T, 6]
    imu_data = imu_data.T
    
    # Normalize each axis independently
    for i in range(6):
        mean = np.mean(imu_data[:, i])
        std = np.std(imu_data[:, i])
        if std > 0:
            imu_data[:, i] = (imu_data[:, i] - mean) / std
    
    # Crop or pad to target length
    T = imu_data.shape[0]
    if T >= seq_len:
        start_idx = (T - seq_len) // 2
        imu_data = imu_data[start_idx:start_idx + seq_len, :]
    else:
        pad_len = seq_len - T
        pad_before = pad_len // 2
        pad_after = pad_len - pad_before
        imu_data = np.pad(imu_data, ((pad_before, pad_after), (0, 0)), 'constant')
    
    return imu_data.astype(np.float32)