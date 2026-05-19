import torch
import torch.nn as nn
import torch.nn.functional as F
from imu_model import ISOARnet
from rgb_model import VSOARnet
import logging

logger = logging.getLogger(__name__)


class FFUA(nn.Module):
    """
    Feature Fluctuation-Driven Uncertainty-Aware (FFUA) fusion module.

    Implements the three-stage uncertainty estimation from the paper:
    1) Extracting multidimensional fluctuation features (temporal variance,
       Top-1 probability, Shannon entropy)
    2) Nonlinear variance aggregation via lightweight MLP
    3) Adaptive inverse-variance weighting for multimodal fusion
    """

    def __init__(self, vision_dim=192, inertial_dim=128, num_classes=27, hidden_dim=16):
        super().__init__()
        self.vision_dim = vision_dim
        self.inertial_dim = inertial_dim
        self.num_classes = num_classes

        # Auxiliary projection heads for logits (shared with main L_cls)
        self.visual_logits_head = nn.Linear(vision_dim, num_classes)
        self.inertial_logits_head = nn.Linear(inertial_dim, num_classes)

        # Lightweight uncertainty MLPs: input size 3 (variance, entropy, top1_prob)
        self.visual_uncertainty_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.inertial_uncertainty_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, v_frame, v_pooled, i_frame, i_pooled):
        """
        Args:
            v_frame: visual frame-level features [B, vision_dim, T]
            v_pooled: visual pooled features [B, vision_dim]
            i_frame: inertial frame-level features [B, T, inertial_hidden_dim]
            i_pooled: inertial pooled features [B, inertial_dim]
        Returns:
            z: fused feature [B, fused_dim]
            p_v: visual logits [B, num_classes]
            p_i: inertial logits [B, num_classes]
            u_v: visual uncertainty [B, 1]
            u_i: inertial uncertainty [B, 1]
            w_v: visual fusion weight [B, 1]
            w_i: inertial fusion weight [B, 1]
        """
        # ---- Stage 1: Auxiliary logits ----
        p_v = self.visual_logits_head(v_pooled)   # [B, num_classes]
        p_i = self.inertial_logits_head(i_pooled) # [B, num_classes]

        # ---- Stage 2: Extract multidimensional fluctuation features ----
        # Visual: [B, C, T] -> temporal variance over T, averaged over C
        sigma2_v = v_frame.var(dim=2, unbiased=False).mean(dim=1, keepdim=True)  # [B, 1]
        # Inertial: [B, T, H] -> temporal variance over T, averaged over H
        sigma2_i = i_frame.var(dim=1, unbiased=False).mean(dim=1, keepdim=True)  # [B, 1]

        def _dist_stats(logits):
            prob = F.softmax(logits, dim=1)
            p_max = prob.max(dim=1, keepdim=True).values
            entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1, keepdim=True)
            return p_max, entropy

        p_v_max, H_v = _dist_stats(p_v)
        p_i_max, H_i = _dist_stats(p_i)

        u_v_raw = torch.cat([sigma2_v, H_v, p_v_max], dim=1)  # [B, 3]
        u_i_raw = torch.cat([sigma2_i, H_i, p_i_max], dim=1)  # [B, 3]

        # ---- Stage 3: Nonlinear variance aggregation ----
        u_v = torch.exp(self.visual_uncertainty_mlp(u_v_raw))   # [B, 1]
        u_i = torch.exp(self.inertial_uncertainty_mlp(u_i_raw)) # [B, 1]

        # ---- Stage 4: Adaptive inverse-variance weighting ----
        inv_sum = 1.0 / (u_v + 1e-8) + 1.0 / (u_i + 1e-8)
        w_v = (1.0 / (u_v + 1e-8)) / inv_sum  # [B, 1]
        w_i = (1.0 / (u_i + 1e-8)) / inv_sum  # [B, 1]

        # ---- Stage 5: Weighted fusion ----
        # Dimensions may differ; project to common space or use separate projections
        z = w_v * v_pooled + w_i * i_pooled  # [B, ...]

        return z, p_v, p_i, u_v, u_i, w_v, w_i


class FF_UA_MF(nn.Module):
    """
    Feature Fluctuation-Driven Uncertainty-Aware Method for Multimodal Fusion.
    Paper: "FF-UA-MF for Special-Operations Monitoring"

    Architecture:
      - VSOARnet (visual branch)
      - ISOARnet (inertial branch)
      - FFUA (uncertainty-aware fusion)
      - Final fusion classifier
    """

    def __init__(self, num_classes=27, vision_pretrained=False,
                 align_weight=0.3, decorr_weight=0.05, cls_weight=1.0,
                 fusion_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.align_weight = align_weight
        self.decorr_weight = decorr_weight
        self.cls_weight = cls_weight

        # ---- Dual-stream feature extractors ----
        self.vision_model = VSOARnet(pretrained=vision_pretrained)
        self.inertial_model = ISOARnet(
            input_dim=6, hidden_dim=256, output_dim=128, num_layers=2, dropout=0.3
        )

        # Freeze part of the vision model stem/first stage for transfer learning stability
        self._freeze_vision_model()

        # ---- Feature projection to common fusion space ----
        vision_dim = self.vision_model.feature_dim      # 192
        inertial_dim = 128                               # ISOARnet projection output

        self.vision_fusion_proj = nn.Sequential(
            nn.Linear(vision_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.inertial_fusion_proj = nn.Sequential(
            nn.Linear(inertial_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # ---- FFUA fusion module ----
        self.ffua = FFUA(
            vision_dim=fusion_dim,
            inertial_dim=fusion_dim,
            num_classes=num_classes,
            hidden_dim=16
        )

        # ---- Final fusion classifier ----
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # ---- Modality-specific classifiers (for evaluation / ablation) ----
        self.vision_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_classes)
        )
        self.inertial_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_classes)
        )

        # ---- Alignment projection heads (for L_align) ----
        self.align_proj_v = nn.Linear(fusion_dim, 128)
        self.align_proj_i = nn.Linear(fusion_dim, 128)

        self.dropout = nn.Dropout(p=0.4)

        logger.info(
            f"FF_UA_MF initialized: cls={cls_weight}, align={align_weight}, "
            f"decorr={decorr_weight}, fusion_dim={fusion_dim}"
        )

    def _freeze_vision_model(self):
        """Freeze stem and first stage of vision model."""
        for name, param in self.vision_model.named_parameters():
            if 'stem' in name or 'stages.0' in name:
                param.requires_grad = False

    def forward(self, vision_input, imu_input):
        # ---- Handle multi-dimensional vision inputs ----
        if vision_input.dim() == 6 and vision_input.shape[2] == 3:
            vision_input = vision_input[:, :, 1, :, :, :]  # [B, 3, T, H, W]
        elif vision_input.dim() == 6 and vision_input.shape[1] == 3:
            vision_input = vision_input[:, 1, :, :, :, :]  # [B, 3, T, H, W]

        # ---- Extract features from each modality ----
        v_frame, v_pooled = self.vision_model.forward_features_with_frames(vision_input)
        i_frame, i_pooled = self.inertial_model(imu_input)

        # Apply dropout
        v_pooled = self.dropout(v_pooled)
        i_pooled = self.dropout(i_pooled)

        # Project to common fusion dimension
        v_proj = self.vision_fusion_proj(v_pooled)      # [B, fusion_dim]
        i_proj = self.inertial_fusion_proj(i_pooled)    # [B, fusion_dim]

        # ---- FFUA fusion ----
        z, p_v, p_i, u_v, u_i, w_v, w_i = self.ffua(v_frame, v_proj, i_frame, i_proj)

        # ---- Classification ----
        fusion_logits = self.fusion_classifier(z)
        vision_logits = self.vision_classifier(v_proj)
        inertial_logits = self.inertial_classifier(i_proj)

        # ---- Alignment embeddings ----
        vision_align = self.align_proj_v(v_proj)
        inertial_align = self.align_proj_i(i_proj)

        return {
            'fusion_logits': fusion_logits,
            'vision_logits': vision_logits,
            'inertial_logits': inertial_logits,
            'vision_features': v_proj,
            'inertial_features': i_proj,
            'vision_embed': vision_align,
            'inertial_embed': inertial_align,
            'fused_features': z,
            'uncertainty_v': u_v,
            'uncertainty_i': u_i,
            'fusion_weights_v': w_v,
            'fusion_weights_i': w_i,
        }

    def get_loss(self, outputs, labels, cls_weight=None, align_weight=None, decorr_weight=None):
        """Compute joint multitask loss: L_total = L_cls + L_align + L_decorr."""
        cls_w = cls_weight if cls_weight is not None else self.cls_weight
        align_w = align_weight if align_weight is not None else self.align_weight
        decorr_w = decorr_weight if decorr_weight is not None else self.decorr_weight

        ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        # L_cls: classification loss on fusion, visual, and inertial logits
        cls_loss = ce_loss(outputs['fusion_logits'], labels)
        cls_loss += ce_loss(outputs['vision_logits'], labels)
        cls_loss += ce_loss(outputs['inertial_logits'], labels)
        cls_loss = cls_loss / 3.0

        # L_align: InfoNCE with same-class masking
        vision_embed = outputs['vision_embed']
        inertial_embed = outputs['inertial_embed']
        align_loss = self._compute_contrastive_loss(vision_embed, inertial_embed, labels)

        # L_decorr: Frobenius norm of cross-modal covariance
        try:
            decorr_loss = self._feature_decorr_loss(
                outputs['vision_features'], outputs['inertial_features']
            )
        except Exception as e:
            logger.warning(f"Decorrelation loss computation failed: {e}")
            decorr_loss = torch.tensor(0.0, device=outputs['fusion_logits'].device)

        total_loss = cls_w * cls_loss + align_w * align_loss + decorr_w * decorr_loss

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'align_loss': align_loss,
            'decorr_loss': decorr_loss,
        }

    def _compute_contrastive_loss(self, vision_embed, imu_embed, labels, temperature=0.1):
        """InfoNCE alignment loss with same-class sample masking."""
        batch_size = vision_embed.shape[0]

        # Normalize features
        vision_embed = F.normalize(vision_embed, dim=1)
        imu_embed = F.normalize(imu_embed, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(vision_embed, imu_embed.T) / temperature

        # Create mask: positive pairs have the same class
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(similarity_matrix.device)

        # Remove self-contrast cases
        mask = mask - torch.eye(batch_size, device=similarity_matrix.device)

        # Compute InfoNCE loss
        exp_similarity = torch.exp(similarity_matrix)
        sum_exp = torch.sum(
            exp_similarity * (1 - torch.eye(batch_size, device=similarity_matrix.device)),
            dim=1, keepdim=True
        )
        sum_exp = torch.clamp(sum_exp, min=1e-8)

        log_prob = similarity_matrix - torch.log(sum_exp)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        loss = -mean_log_prob_pos.mean()
        loss = torch.clamp(loss, min=0.0)
        return loss

    def _feature_decorr_loss(self, vision_embed, imu_embed):
        """Decorrelation loss: minimize Frobenius norm of cross-modal covariance."""
        # Flatten features
        vision_flat = vision_embed.view(vision_embed.shape[0], -1)
        imu_flat = imu_embed.view(imu_embed.shape[0], -1)

        # Normalize features to have unit variance
        vision_flat = F.normalize(vision_flat, p=2, dim=1)
        imu_flat = F.normalize(imu_flat, p=2, dim=1)

        # Compute covariance matrix
        combined = torch.cat([vision_flat, imu_flat], dim=1)
        cov_matrix = torch.cov(combined.T)

        # Extract off-diagonal blocks (cross-modal correlation)
        vision_dim = vision_flat.shape[1]
        cross_cov = cov_matrix[:vision_dim, vision_dim:]

        # Compute Frobenius norm of cross-covariance matrix
        loss = torch.norm(cross_cov, p='fro')
        return loss
