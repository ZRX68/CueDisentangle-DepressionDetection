# ====================== 环境配置 ======================
import os

# 禁用 Hugging Face 符号链接警告
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# ====================== 导入库 ======================
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from math import sqrt
import numpy as np

class _GradReverseFn(torch.autograd.Function):
    """梯度反转：前向恒等，反向乘以 -alpha"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha: float = 1.0):
    return _GradReverseFn.apply(x, alpha)

# ====================== AstroModel 编码器模块 ======================
class AstroModel_gaze(nn.Module):
    def __init__(self) -> None:
        super(AstroModel_gaze, self).__init__()
        self.conv = nn.Conv1d(12, 12, 1)
        self.dropout = nn.Dropout(0.2)

        self.conv1_1 = nn.Sequential(
            nn.Conv1d(12, 12, 3, padding=2, dilation=2),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv1d(12, 12, 3, padding=2, dilation=2),
            nn.ReLU()
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv1d(12, 12, 3, padding=4, dilation=4),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(12, 12, 3, padding=4, dilation=4),
            nn.ReLU()
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv1d(12, 12, 3, padding=8, dilation=8),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv1d(12, 12, 3, padding=8, dilation=8),
            nn.ReLU()
        )

        self.conv4_1 = nn.Sequential(
            nn.Conv1d(12, 12, 3, padding=16, dilation=16),
            nn.ReLU()
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv1d(12, 12, 3, padding=16, dilation=16),
            nn.ReLU()
        )

    def forward(self, x):
        # Block 1
        residual = x
        x = self.conv1_1(x)
        x = self.dropout(x)
        x = self.conv1_2(x)
        x = x + residual

        # Block 2
        residual = x
        x = self.conv2_1(x)
        x = self.dropout(x)
        x = self.conv2_2(x)
        x = x + residual

        # Block 3
        residual = x
        x = self.conv3_1(x)
        x = self.dropout(x)
        x = self.conv3_2(x)
        x = x + residual

        # Block 4
        residual = x
        x = self.conv4_1(x)
        x = self.dropout(x)
        x = self.conv4_2(x)
        x = x + residual

        return x


class AstroModel_pose(nn.Module):
    def __init__(self) -> None:
        super(AstroModel_pose, self).__init__()
        self.conv = nn.Conv1d(6, 6, 1)
        self.dropout = nn.Dropout(0.2)

        self.conv1_1 = nn.Sequential(
            nn.Conv1d(6, 6, 3, padding=2, dilation=2),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv1d(6, 6, 3, padding=2, dilation=2),
            nn.ReLU()
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv1d(6, 6, 3, padding=4, dilation=4),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(6, 6, 3, padding=4, dilation=4),
            nn.ReLU()
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv1d(6, 6, 3, padding=8, dilation=8),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv1d(6, 6, 3, padding=8, dilation=8),
            nn.ReLU()
        )

        self.conv4_1 = nn.Sequential(
            nn.Conv1d(6, 6, 3, padding=16, dilation=16),
            nn.ReLU()
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv1d(6, 6, 3, padding=16, dilation=16),
            nn.ReLU()
        )

    def forward(self, x):
        # Block 1
        residual = x
        x = self.conv1_1(x)
        x = self.dropout(x)
        x = self.conv1_2(x)
        x = x + residual

        # Block 2
        residual = x
        x = self.conv2_1(x)
        x = self.dropout(x)
        x = self.conv2_2(x)
        x = x + residual

        # Block 3
        residual = x
        x = self.conv3_1(x)
        x = self.dropout(x)
        x = self.conv3_2(x)
        x = x + residual

        # Block 4
        residual = x
        x = self.conv4_1(x)
        x = self.dropout(x)
        x = self.conv4_2(x)
        x = x + residual

        return x


class AstroModel_landmark(nn.Module):
    def __init__(self) -> None:
        super(AstroModel_landmark, self).__init__()
        self.conv = nn.Conv1d(136, 136, 1)
        self.dropout = nn.Dropout(0.2)

        self.conv1_1 = nn.Sequential(
            nn.Conv1d(136, 136, 3, padding=2, dilation=2),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv1d(136, 136, 3, padding=2, dilation=2),
            nn.ReLU()
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv1d(136, 136, 3, padding=4, dilation=4),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(136, 136, 3, padding=4, dilation=4),
            nn.ReLU()
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv1d(136, 136, 3, padding=8, dilation=8),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv1d(136, 136, 3, padding=8, dilation=8),
            nn.ReLU()
        )

        self.conv4_1 = nn.Sequential(
            nn.Conv1d(136, 136, 3, padding=16, dilation=16),
            nn.ReLU()
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv1d(136, 136, 3, padding=16, dilation=16),
            nn.ReLU()
        )

    def forward(self, x):
        # Block 1
        residual = x
        x = self.conv1_1(x)
        x = self.dropout(x)
        x = self.conv1_2(x)
        x = x + residual

        # Block 2
        residual = x
        x = self.conv2_1(x)
        x = self.dropout(x)
        x = self.conv2_2(x)
        x = x + residual

        # Block 3
        residual = x
        x = self.conv3_1(x)
        x = self.dropout(x)
        x = self.conv3_2(x)
        x = x + residual

        # Block 4
        residual = x
        x = self.conv4_1(x)
        x = self.dropout(x)
        x = self.conv4_2(x)
        x = x + residual

        return x


class AstroModel_au(nn.Module):
    def __init__(self) -> None:
        super(AstroModel_au, self).__init__()
        self.conv = nn.Conv1d(17, 17, 1)
        self.dropout = nn.Dropout(0.2)

        self.conv1_1 = nn.Sequential(
            nn.Conv1d(17, 17, 3, padding=2, dilation=2),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv1d(17, 17, 3, padding=2, dilation=2),
            nn.ReLU()
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv1d(17, 17, 3, padding=4, dilation=4),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(17, 17, 3, padding=4, dilation=4),
            nn.ReLU()
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv1d(17, 17, 3, padding=8, dilation=8),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv1d(17, 17, 3, padding=8, dilation=8),
            nn.ReLU()
        )

        self.conv4_1 = nn.Sequential(
            nn.Conv1d(17, 17, 3, padding=16, dilation=16),
            nn.ReLU()
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv1d(17, 17, 3, padding=16, dilation=16),
            nn.ReLU()
        )

    def forward(self, x):
        # Block 1
        residual = x
        x = self.conv1_1(x)
        x = self.dropout(x)
        x = self.conv1_2(x)
        x = x + residual

        # Block 2
        residual = x
        x = self.conv2_1(x)
        x = self.dropout(x)
        x = self.conv2_2(x)
        x = x + residual

        # Block 3
        residual = x
        x = self.conv3_1(x)
        x = self.dropout(x)
        x = self.conv3_2(x)
        x = x + residual

        # Block 4
        residual = x
        x = self.conv4_1(x)
        x = self.dropout(x)
        x = self.conv4_2(x)
        x = x + residual

        return x


# ====================== 线索级特征分离模块 ======================
class CueLevelFeatureDiverger(nn.Module):
    """
    线索级特征分离模块：为每种行为线索分别进行病理特征与个性化特征分离
    """

    def __init__(self, encoded_dim=128, pathological_dim=32, personalized_dim=32):
        super(CueLevelFeatureDiverger, self).__init__()
        self.cue_names = ['gaze', 'pose', 'landmark', 'au']
        self.encoded_dim = encoded_dim  # 经过编码器后的统一维度
        self.pathological_dim = pathological_dim
        self.personalized_dim = personalized_dim

        # 为每种线索创建独立的特征分离器
        self.cue_divergers = nn.ModuleDict()

        for cue_name in self.cue_names:
            self.cue_divergers[cue_name] = nn.ModuleDict({
                # 病理特征提取器
                'pathological_extractor': nn.Sequential(
                    nn.Linear(encoded_dim, pathological_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(pathological_dim * 2, pathological_dim),
                    nn.Tanh()
                ),
                # 个性化特征提取器
                'personalized_extractor': nn.Sequential(
                    nn.Linear(encoded_dim, personalized_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(personalized_dim * 2, personalized_dim),
                    nn.Tanh()
                ),
                # 特征重构器
                'reconstructor': nn.Sequential(
                    nn.Linear(pathological_dim + personalized_dim, encoded_dim),
                    nn.ReLU(),
                    nn.Linear(encoded_dim, encoded_dim)
                )
            })

    def forward(self, cue_features_list):
        """
        Args:
            cue_features_list: [gaze_feat, pose_feat, landmark_feat, au_feat]
                每个元素形状为 [batch_size, seq_len, cue_dim]
        Returns:
            pathological_features: 各线索的病理特征字典
            personalized_features: 各线索的个性化特征字典
            reconstruction_losses: 各线索的重构损失
            orthogonal_losses: 各线索的正交损失
        """
        pathological_features = {}
        personalized_features = {}
        reconstruction_losses = {}
        orthogonal_losses = {}

        for i, (cue_name, cue_feat) in enumerate(zip(self.cue_names, cue_features_list)):
            batch_size, seq_len, cue_dim = cue_feat.shape

            # 展平处理
            cue_flat = cue_feat.reshape(-1, cue_dim)

            # 提取病理特征和个性化特征
            path_flat = self.cue_divergers[cue_name]['pathological_extractor'](cue_flat)
            pers_flat = self.cue_divergers[cue_name]['personalized_extractor'](cue_flat)

            # 重构原始特征
            combined_flat = torch.cat([path_flat, pers_flat], dim=1)
            reconstructed_flat = self.cue_divergers[cue_name]['reconstructor'](combined_flat)

            # 恢复形状
            pathological_features[cue_name] = path_flat.reshape(batch_size, seq_len, self.pathological_dim)
            personalized_features[cue_name] = pers_flat.reshape(batch_size, seq_len, self.personalized_dim)
            reconstructed_feat = reconstructed_flat.reshape(batch_size, seq_len, cue_dim)

            # 计算重构损失
            reconstruction_losses[cue_name] = F.mse_loss(reconstructed_feat, cue_feat)

            # 计算正交约束损失
            path_norm = F.normalize(path_flat, dim=1)
            pers_norm = F.normalize(pers_flat, dim=1)
            orthogonal_losses[cue_name] = torch.mean(torch.abs(torch.sum(path_norm * pers_norm, dim=1)))

        return pathological_features, personalized_features, reconstruction_losses, orthogonal_losses


# ====================== 多线索对比学习模块 ======================
class MultiCueContrastiveLearning(nn.Module):
    """
    多线索对比学习模块：在线索级别进行对比学习
    """

    def __init__(self, cue_names=['gaze', 'pose', 'landmark', 'au'],
                 pathological_dim=32, personalized_dim=32, temperature=0.07):
        super(MultiCueContrastiveLearning, self).__init__()
        self.cue_names = cue_names
        self.temperature = temperature
        self.pathological_dim = pathological_dim
        self.personalized_dim = personalized_dim

        # 为每种线索创建投影头
        self.cue_projectors = nn.ModuleDict()
        for cue_name in cue_names:
            self.cue_projectors[cue_name] = nn.ModuleDict({
                'pathological_projector': nn.Sequential(
                    nn.Linear(pathological_dim, pathological_dim),
                    nn.ReLU(),
                    nn.Linear(pathological_dim, pathological_dim // 2)
                ),
                'personalized_projector': nn.Sequential(
                    nn.Linear(personalized_dim, personalized_dim),
                    nn.ReLU(),
                    nn.Linear(personalized_dim, personalized_dim // 2)
                )
            })

    def compute_cue_level_contrastive_loss(self, pathological_features, personalized_features, labels):
        """
        计算线索级对比学习损失（简化版）

        (1) 线索内对比（Intra-Cue）：
            - 病理特征：不同抑郁样本的相同线索 → 靠近
            - 个性化特征：不同抑郁样本的相同线索 → 远离

        (2) 线索间对比（Inter-Cue）：
            - 病理特征：同一样本内的所有线索 → 靠近
            - 个性化特征：同一样本内的所有线索 → 远离
        """
        loss_components = {}

        # ==================== (1) 线索内对比学习 ====================
        intra_cue_loss = self.compute_intra_cue_contrastive(
            pathological_features, personalized_features, labels
        )
        loss_components['intra_cue_contrastive'] = intra_cue_loss

        # ==================== (2) 线索间对比学习 ====================
        inter_cue_loss = self.compute_inter_cue_contrastive(
            pathological_features, personalized_features, labels
        )
        loss_components['inter_cue_contrastive'] = inter_cue_loss

        # 总对比学习损失
        total_loss = intra_cue_loss + inter_cue_loss

        # 确保总损失是有限的
        if not torch.isfinite(total_loss):
            total_loss = torch.tensor(0.0, device=total_loss.device, requires_grad=True)

        return total_loss, loss_components

    def compute_intra_cue_contrastive(self, pathological_features, personalized_features, labels):
        """
        (1) 线索内对比学习：对于不同抑郁样本的相同线索
            - 病理特征：应该互相靠近
            - 个性化特征：应该远离
        """
        total_loss = 0.0
        depression_mask = (labels == 1)

        # 只对抑郁样本计算
        if torch.sum(depression_mask) <= 1:
            return torch.tensor(0.0, device=labels.device, requires_grad=True)

        for cue_name in self.cue_names:
            # 获取当前线索的特征
            path_feat = pathological_features[cue_name]  # [batch, seq_len, path_dim]
            pers_feat = personalized_features[cue_name]  # [batch, seq_len, pers_dim]

            # 时序维度平均池化
            path_pooled = torch.mean(path_feat, dim=1)  # [batch, path_dim]
            pers_pooled = torch.mean(pers_feat, dim=1)  # [batch, pers_dim]

            # 投影到对比学习空间
            path_projected = self.cue_projectors[cue_name]['pathological_projector'](path_pooled)
            pers_projected = self.cue_projectors[cue_name]['personalized_projector'](pers_pooled)

            # 筛选抑郁样本
            dep_path = path_projected[depression_mask]
            dep_pers = pers_projected[depression_mask]

            if dep_path.shape[0] > 1:
                # ✅ 病理特征：不同抑郁样本应该靠近（聚类）
                path_attract_loss = self.info_nce_loss(dep_path, dep_path)
                if torch.isfinite(path_attract_loss):
                    total_loss += path_attract_loss

                # ✅ 个性化特征：不同抑郁样本应该远离（保持个体差异）
                pers_repel_loss = self.repulsion_loss(dep_pers)
                if torch.isfinite(pers_repel_loss):
                    total_loss += pers_repel_loss

        return total_loss / len(self.cue_names)

    def compute_inter_cue_contrastive(self, pathological_features, personalized_features, labels):
        """
        (2) 线索间对比学习：在一个样本内，其中的所有线索
            - 病理特征：应该互相靠近（一致性）
            - 个性化特征：应该远离（互补性）
        """
        total_loss = 0.0
        batch_size = labels.shape[0]

        # 收集所有线索的特征
        all_path_features = []
        all_pers_features = []

        for cue_name in self.cue_names:
            path_feat = torch.mean(pathological_features[cue_name], dim=1)  # [batch, path_dim]
            pers_feat = torch.mean(personalized_features[cue_name], dim=1)  # [batch, pers_dim]

            # 投影
            path_proj = self.cue_projectors[cue_name]['pathological_projector'](path_feat)
            pers_proj = self.cue_projectors[cue_name]['personalized_projector'](pers_feat)

            all_path_features.append(path_proj)  # [batch, proj_dim]
            all_pers_features.append(pers_proj)  # [batch, proj_dim]

        # 转换为 [batch, num_cues, proj_dim]
        path_features_stacked = torch.stack(all_path_features, dim=1)  # [batch, 4, proj_dim]
        pers_features_stacked = torch.stack(all_pers_features, dim=1)  # [batch, 4, proj_dim]

        # ✅ 病理特征：同一样本内的所有线索应该靠近（一致性）
        # 对每个样本，计算其4个线索之间的两两距离，鼓励靠近
        for i in range(len(self.cue_names)):
            for j in range(i + 1, len(self.cue_names)):
                feat_i = path_features_stacked[:, i, :]  # [batch, proj_dim]
                feat_j = path_features_stacked[:, j, :]  # [batch, proj_dim]

                # MSE损失：鼓励靠近
                path_attract_loss = F.mse_loss(
                    F.normalize(feat_i, dim=1),
                    F.normalize(feat_j, dim=1)
                )
                if torch.isfinite(path_attract_loss):
                    total_loss += path_attract_loss

        # ✅ 个性化特征：同一样本内的所有线索应该远离（互补性）
        # 对每个样本，计算其4个线索之间的两两距离，鼓励远离
        for i in range(len(self.cue_names)):
            for j in range(i + 1, len(self.cue_names)):
                feat_i = pers_features_stacked[:, i, :]  # [batch, proj_dim]
                feat_j = pers_features_stacked[:, j, :]  # [batch, proj_dim]

                # 负MSE损失：鼓励远离
                pers_repel_loss = -F.mse_loss(
                    F.normalize(feat_i, dim=1),
                    F.normalize(feat_j, dim=1)
                )
                if torch.isfinite(pers_repel_loss):
                    total_loss += pers_repel_loss

        # 归一化：共C(4,2)=6对
        num_pairs = len(self.cue_names) * (len(self.cue_names) - 1) // 2
        return total_loss / num_pairs

    def info_nce_loss(self, features1, features2):
        """
        InfoNCE损失函数 - 用于特征聚类（靠近）
        让features1中的每个样本与features2中对应位置的样本靠近
        """
        if features1.shape[0] <= 1:
            return torch.tensor(0.0, device=features1.device, requires_grad=True)

        features1 = F.normalize(features1, dim=1, eps=1e-8)
        features2 = F.normalize(features2, dim=1, eps=1e-8)

        similarity_matrix = torch.matmul(features1, features2.T) / self.temperature
        # 数值稳定性：限制相似度矩阵的范围
        similarity_matrix = torch.clamp(similarity_matrix, min=-10, max=10)

        batch_size = features1.shape[0]
        labels = torch.arange(batch_size).to(features1.device)

        return F.cross_entropy(similarity_matrix, labels)

    def repulsion_loss(self, features):
        """
        排斥损失 - 用于特征分散（远离）
        让features中的不同样本互相远离

        原理：最大化样本间的平均距离 = 最小化样本间的平均相似度
        """
        if features.shape[0] <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # 归一化
        features_norm = F.normalize(features, dim=1, eps=1e-8)

        # 计算相似度矩阵 [batch, batch]
        similarity_matrix = torch.matmul(features_norm, features_norm.T)

        # 去除对角线（自己和自己的相似度）
        batch_size = features.shape[0]
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)

        # 平均相似度（不包括对角线）
        avg_similarity = similarity_matrix[mask].mean()

        # 损失 = 平均相似度（最小化相似度 = 最大化距离）
        # 为了数值稳定性，限制范围
        avg_similarity = torch.clamp(avg_similarity, min=-1.0, max=1.0)

        return avg_similarity


# ====================== 自适应加权融合模块 ======================
class AdaptiveFeatureWeighting(nn.Module):
    """
    自适应特征加权融合模块：基于特征内容自动学习病理特征和个性化特征的最优权重分配
    不依赖标签，而是通过特征本身的模式来决定权重
    """

    def __init__(self, pathological_dim=128, personalized_dim=128, hidden_dim=64):
        super(AdaptiveFeatureWeighting, self).__init__()
        self.pathological_dim = pathological_dim
        self.personalized_dim = personalized_dim

        # ===== 核心改进：基于特征内容的权重预测器 =====
        # 分析病理特征的一致性程度
        self.pathological_analyzer = nn.Sequential(
            nn.Linear(pathological_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # 输出病理特征的重要性分数
            nn.Sigmoid()
        )

        # 分析个性化特征的差异性程度
        self.personalized_analyzer = nn.Sequential(
            nn.Linear(personalized_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # 输出个性化特征的重要性分数
            nn.Sigmoid()
        )

        # 联合分析器：同时考虑两种特征的关系
        self.joint_weight_predictor = nn.Sequential(
            nn.Linear(pathological_dim + personalized_dim + 2, hidden_dim),  # +2是两个重要性分数
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 输出两个权重值
            nn.Softmax(dim=-1)  # 确保权重和为1
        )

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(pathological_dim + personalized_dim, pathological_dim + personalized_dim),
            nn.LayerNorm(pathological_dim + personalized_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 用于训练时的辅助分类器（预测样本是否为抑郁）
        # 这帮助模型学习什么样的特征模式对应什么样的权重
        self.auxiliary_classifier = nn.Sequential(
            nn.Linear(pathological_dim + personalized_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, pathological_features, personalized_features, labels=None):
        """
        Args:
            pathological_features: [batch_size, seq_len, pathological_dim]
            personalized_features: [batch_size, seq_len, personalized_dim]
            labels: [batch_size] - 仅用于训练时的辅助损失，不用于决定权重
        Returns:
            weighted_features: [batch_size, seq_len, pathological_dim + personalized_dim]
            weights: [batch_size, 2] - 用于可视化的权重值
            aux_logits: [batch_size, 2] - 辅助分类器的输出（训练时使用）
        """
        batch_size, seq_len, _ = pathological_features.shape

        # 1. 对序列维度进行平均池化，获取全局特征表示
        path_pooled = torch.mean(pathological_features, dim=1)  # [batch_size, pathological_dim]
        pers_pooled = torch.mean(personalized_features, dim=1)  # [batch_size, personalized_dim]

        # 2. 分析每种特征的重要性
        path_importance = self.pathological_analyzer(path_pooled)  # [batch_size, 1]
        pers_importance = self.personalized_analyzer(pers_pooled)  # [batch_size, 1]

        # 3. 基于特征内容和重要性分数预测权重
        # 拼接所有信息：病理特征 + 个性化特征 + 两个重要性分数
        weight_input = torch.cat([
            path_pooled,
            pers_pooled,
            path_importance,
            pers_importance
        ], dim=-1)  # [batch_size, path_dim + pers_dim + 2]

        # 预测权重
        weights = self.joint_weight_predictor(weight_input)  # [batch_size, 2]

        # 4. 扩展权重到序列维度
        path_weight = weights[:, 0:1].unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, 1]
        pers_weight = weights[:, 1:2].unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, 1]

        # 5. 应用权重
        weighted_path = pathological_features * path_weight  # [batch_size, seq_len, pathological_dim]
        weighted_pers = personalized_features * pers_weight  # [batch_size, seq_len, personalized_dim]

        # 6. 拼接加权后的特征
        weighted_combined = torch.cat([weighted_path, weighted_pers],
                                      dim=-1)  # [batch_size, seq_len, path_dim + pers_dim]

        # 7. 特征融合
        fused_features = self.fusion_layer(
            weighted_combined.reshape(-1, self.pathological_dim + self.personalized_dim)
        ).reshape(batch_size, seq_len, self.pathological_dim + self.personalized_dim)

        # 8. 辅助分类（仅在训练时使用，帮助学习权重分配策略）
        aux_logits = None
        if self.training:
            # 使用池化后的加权特征进行辅助分类
            combined_pooled = torch.cat([path_pooled, pers_pooled], dim=-1)
            aux_logits = self.auxiliary_classifier(combined_pooled)  # [batch_size, 2]

        return fused_features, weights, aux_logits


# ====================== 交叉特征注意力交互模块 ======================
class CrossCharacteristicAttention(nn.Module):
    """
    交叉特征注意力交互模块：用于多线索特征的深度交互和融合（优化版）
    """

    def __init__(self, hidden_size=256, num_heads=4, use_lightweight=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_lightweight = use_lightweight

        if use_lightweight:
            # 轻量级版本：减少计算复杂度
            # 直接融合四种线索
            self.fusion_linear = nn.Linear(hidden_size * 4, hidden_size)

            # 单层注意力
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                batch_first=True
            )

            self.norm = nn.LayerNorm(hidden_size)

        else:
            # 完整版本：保留原有复杂结构
            # BiLSTM层用于学习线索间的依赖关系
            self.bilstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size // 2,
                num_layers=1,
                bidirectional=True,
                batch_first=True
            )

            # 线性变换层用于特征投影
            self.linear_q = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(5)
            ])
            self.linear_k = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(5)
            ])
            self.linear_v = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(5)
            ])

            # 多头注意力机制
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                batch_first=True
            )

            # 融合层
            self.fusion_linear = nn.Linear(hidden_size * 5, hidden_size)
            self.final_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                batch_first=True
            )

            self.linear_cues = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, cues):
        """
        cues: 四种线索的特征列表 [gaze, pose, landmark, au]，形状为 [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = cues[0].shape

        if self.use_lightweight:
            # 轻量级版本
            # 1. 拼接四种线索
            cues_combined = torch.cat(cues, dim=2)  # [batch, seq_len, hidden_size*4]

            # 2. 线性融合
            fused = self.fusion_linear(cues_combined)  # [batch, seq_len, hidden_size]

            # 3. 自注意力
            attn_output, _ = self.attn(fused, fused, fused)

            # 4. 残差连接 + 归一化
            output = self.norm(attn_output + fused)

            return output

        else:
            # 完整版本（原有逻辑）
            num_cues = len(cues)

            # 1. 使用BiLSTM学习线索间的依赖关系
            cues_combined = torch.cat(cues, dim=2)  # [batch, seq_len, hidden_size*4]
            cues_combined = cues_combined.reshape(batch_size * seq_len, -1)
            cues_combined = self.linear_cues(cues_combined)
            cues_combined = cues_combined.reshape(batch_size, seq_len, hidden_size)

            # 通过BiLSTM
            lstm_output, _ = self.bilstm(cues_combined)  # [batch, seq_len, hidden_size]

            # 分离回四种线索
            cues_lstm = []
            for i in range(num_cues):
                cues_lstm.append(lstm_output + cues[i])  # 残差连接

            cues_lstm.append(lstm_output)

            # 2. 以每种线索作为Q，其他线索作为K和V进行注意力交互
            fused_cues = []
            for q_idx in range(5):
                q = self.linear_q[q_idx](cues_lstm[q_idx])  # [batch, seq_len, hidden_size]
                k = torch.cat([self.linear_k[i](cues_lstm[i]) for i in range(5)],
                              dim=1)  # [batch, seq_len*5, hidden_size]
                v = torch.cat([self.linear_v[i](cues_lstm[i]) for i in range(5)],
                              dim=1)  # [batch, seq_len*5, hidden_size]

                # 多头注意力
                attn_output, att_weights = self.multihead_attn(q, k, v)  # [batch, seq_len, hidden_size]
                fused_cues.append(attn_output + cues_lstm[q_idx])  # 残差连接

            # 3. 拼接五种融合结果
            combined_fused = torch.cat(fused_cues, dim=2)  # [batch, seq_len, hidden_size*5]

            # 4. 线性变换降维
            combined_fused = self.fusion_linear(combined_fused)  # [batch, seq_len, hidden_size]

            # 5. 自注意力提取关键信息
            attn_output, _ = self.final_attn(combined_fused, combined_fused, combined_fused)
            final_fusion = attn_output + combined_fused  # 残差连接

            return final_fusion  # [batch, seq_len, hidden_size]


# ====================== 集成的线索级对比学习模型（改进版）======================
class CueLevelContrastiveBCDCIE(nn.Module):
    """
    集成线索级对比学习的BCDCIE模型（改进版）
    
    改进点：
    1. 去掉交叉注意力融合，保留线索独立性
    2. 采用分线索加权，每个线索独立决定病理/个性化权重
    3. 简单拼接融合 + 线索重要性加权
    4. 增强可解释性，可追溯每个线索的贡献
    """

    def __init__(self, pathological_dim=32, personalized_dim=32):
        super(CueLevelContrastiveBCDCIE, self).__init__()
        
        self.cue_names = ['gaze', 'pose', 'landmark', 'au']

        # AstroModel 编码器
        self.video_astro_gaze = AstroModel_gaze()
        self.video_astro_pose = AstroModel_pose()
        self.video_astro_landmark = AstroModel_landmark()
        self.video_astro_au = AstroModel_au()

        # 投影层：将各线索特征投影到统一维度
        self.proj_gaze = nn.Linear(12, 128)
        self.proj_pose = nn.Linear(6, 128)
        self.proj_landmark = nn.Linear(136, 128)
        self.proj_au = nn.Linear(17, 128)

        # 线索级特征分离模块
        self.cue_diverger = CueLevelFeatureDiverger(
            encoded_dim=128,  # 经过编码器后的统一维度
            pathological_dim=pathological_dim,
            personalized_dim=personalized_dim
        )

        # 多线索对比学习模块
        self.multi_cue_contrastive = MultiCueContrastiveLearning(
            pathological_dim=pathological_dim,
            personalized_dim=personalized_dim
        )

        # 病理特征投影层：将病理特征投影回128维
        self.path_projectors = nn.ModuleDict({
            'gaze': nn.Linear(pathological_dim, 128),
            'pose': nn.Linear(pathological_dim, 128),
            'landmark': nn.Linear(pathological_dim, 128),
            'au': nn.Linear(pathological_dim, 128)
        })

        # 个性化特征投影层：投影到128维（与病理特征维度一致）
        self.pers_projectors = nn.ModuleDict({
            'gaze': nn.Linear(personalized_dim, 128),
            'pose': nn.Linear(personalized_dim, 128),
            'landmark': nn.Linear(personalized_dim, 128),
            'au': nn.Linear(personalized_dim, 128)
        })

        # ========== 核心改进1：分线索加权模块 ==========
        # 为每个线索创建独立的自适应加权模块
        self.cue_adaptive_weightings = nn.ModuleDict()
        for cue_name in self.cue_names:
            self.cue_adaptive_weightings[cue_name] = AdaptiveFeatureWeighting(
                pathological_dim=128,  # 投影后的病理特征维度
                personalized_dim=128,  # 投影后的个性化特征维度
                hidden_dim=64
            )
        
        # ========== 核心改进2：简单融合（替代交叉注意力）==========
        # 方式1：直接拼接融合
        self.simple_fusion = nn.Sequential(
            nn.Linear(256 * 4, 512),  # 4个线索 × 256维（128病理+128个性化）
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        self.cross_cue_attn_path = CrossCharacteristicAttention(hidden_size=256, num_heads=4)
        
        # 方式2：线索重要性加权（可选，增强可解释性）
        self.cue_importance_predictor = nn.Sequential(
            nn.Linear(256 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
            nn.Softmax(dim=-1)  # 输出4个线索的重要性 [gaze, pose, landmark, au]
        )

        # 序列长度调整层：从915降到186
        self.seq_conv = nn.Conv1d(in_channels=915, out_channels=186, kernel_size=1, padding=0, stride=1)

        # 音频处理模块
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        # 多头交叉注意力（视频-音频交互）
        self.video_audio_attn = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

        # LayerNorm
        self.norm2 = nn.LayerNorm(256)
        self.norm1 = nn.LayerNorm(256)

        # 自适应池化
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # 分类器（直接基于视听融合特征）
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # ================== 【新增】对偶监督头 (Dual Supervision) ==================
        # 接收 4 个线索的拼接特征 (32 * 4 = 128维)
        path_sup_in_dim = pathological_dim * len(self.cue_names)
        pers_sup_in_dim = personalized_dim * len(self.cue_names)

        # 病理监督头 (正向)
        self.path_depression_head = nn.Sequential(
            nn.Linear(path_sup_in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        # 个性化对抗头 (反向)
        self.pers_depression_head = nn.Sequential(
            nn.Linear(pers_sup_in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

        # GRL 对抗强度，允许在训练脚本中动态更新 (退火策略)
        self.grl_alpha = 1.0

        # 损失权重
        self.classification_weight = 1.0
        self.reconstruction_weight = 0.3
        self.contrastive_weight = 0.5
        self.orthogonal_weight = 0.1
        self.auxiliary_weight = 0.2          # 辅助分类损失权重
        self.path_supervision_weight = 1.0
        self.pers_adversarial_weight = 0.5

    def forward(self, video_input, audio_input, labels=None):
        """
        前向传播（改进版）
        """
        # ========== 1. 提取原始线索特征 ==========
        video_part_gaze = video_input[:, :, :12].transpose(1, 2)
        video_part_pose = video_input[:, :, 12:18].transpose(1, 2)
        video_part_landmark = video_input[:, :, 18:154].transpose(1, 2)
        video_part_au = video_input[:, :, 154:].transpose(1, 2)

        # 编码四种线索
        gaze_feat = self.video_astro_gaze(video_part_gaze).transpose(1, 2)
        pose_feat = self.video_astro_pose(video_part_pose).transpose(1, 2)
        landmark_feat = self.video_astro_landmark(video_part_landmark).transpose(1, 2)
        au_feat = self.video_astro_au(video_part_au).transpose(1, 2)

        # 投影到统一维度 128
        batch_size, seq_len, _ = gaze_feat.shape
        gaze_feat = self.proj_gaze(gaze_feat.reshape(-1, 12)).reshape(batch_size, seq_len, 128)
        pose_feat = self.proj_pose(pose_feat.reshape(-1, 6)).reshape(batch_size, seq_len, 128)
        landmark_feat = self.proj_landmark(landmark_feat.reshape(-1, 136)).reshape(batch_size, seq_len, 128)
        au_feat = self.proj_au(au_feat.reshape(-1, 17)).reshape(batch_size, seq_len, 128)

        cue_features_list = [gaze_feat, pose_feat, landmark_feat, au_feat]

        # ========== 2. 线索级特征分离 ==========
        pathological_features, personalized_features, reconstruction_losses, orthogonal_losses = \
            self.cue_diverger(cue_features_list)

        # ========== 3. 对比学习损失 ==========
        contrastive_loss = 0
        loss_components = {}
        if labels is not None and self.training:
            contrastive_loss, loss_components = self.multi_cue_contrastive.compute_cue_level_contrastive_loss(
                pathological_features, personalized_features, labels
            )

        # ========== 4. 投影特征 ==========
        path_features_dict = {}
        pers_features_dict = {}
        
        for cue_name in self.cue_names:
            # 投影病理特征到128维
            path_feat = pathological_features[cue_name]
            batch_size, seq_len, path_dim = path_feat.shape
            path_feat_proj = self.path_projectors[cue_name](
                path_feat.reshape(-1, path_dim)
            ).reshape(batch_size, seq_len, 128)
            path_features_dict[cue_name] = path_feat_proj

            # 投影个性化特征到128维
            pers_feat = personalized_features[cue_name]
            pers_feat_proj = self.pers_projectors[cue_name](
                pers_feat.reshape(-1, pers_feat.shape[-1])
            ).reshape(batch_size, seq_len, 128)
            pers_features_dict[cue_name] = pers_feat_proj

        # ========== 5. 核心改进：分线索加权 ==========
        cue_weighted_features = []
        cue_weights_dict = {}
        cue_aux_logits_dict = {}
        
        for cue_name in self.cue_names:
            # 每个线索独立进行自适应加权
            weighted_feat, weights, aux_logits = self.cue_adaptive_weightings[cue_name](
                pathological_features=path_features_dict[cue_name],  # [batch, seq, 128]
                personalized_features=pers_features_dict[cue_name],  # [batch, seq, 128]
                labels=labels
            )
            # weighted_feat: [batch, seq, 256] (128病理 + 128个性化)
            # weights: [batch, 2] (病理权重, 个性化权重)
            
            cue_weighted_features.append(weighted_feat)
            cue_weights_dict[cue_name] = weights  # 保存每个线索的权  重
            if aux_logits is not None:
                cue_aux_logits_dict[cue_name] = aux_logits

        # ========== 6. 核心改进：简单融合（替代交叉注意力）==========
        # 拼接所有线索的加权特征
        all_cues_concat = torch.cat(cue_weighted_features, dim=-1)
        # [batch, seq, 256*4=1024]

        Att_fused_features = self.cross_cue_attn_path(cue_weighted_features)
        
        # 方式B：线索重要性加权（增强可解释性）
        all_cues_pooled = torch.mean(all_cues_concat, dim=1)  # [batch, 1024]
        cue_importance = self.cue_importance_predictor(all_cues_pooled)
        # cue_importance: [batch, 4] 例如 [0.25, 0.15, 0.45, 0.15]
        
        # 按重要性加权每个线索
        importance_weighted_features = []
        for i, cue_feat in enumerate(cue_weighted_features):
            importance = cue_importance[:, i:i+1].unsqueeze(1)  # [batch, 1, 1]
            weighted = cue_feat * importance  # [batch, seq, 256]
            importance_weighted_features.append(weighted)
        
        # 融合加权后的特征
        importance_fused = torch.sum(torch.stack(importance_weighted_features, dim=0), dim=0)
        # [batch, seq, 256]
        
        # 混合两种融合方式
        fused_video_features = 0.5 * Att_fused_features + 0.5 * importance_fused
        # [batch, seq, 256]

        # fused_video_features = importance_fused
        
        # 降维到128维
        fused_video_features = fused_video_features[:, :, :128]  # [batch, seq, 128]

        # ========== 7. 调整序列长度 ==========
        fused_video_features = self.seq_conv(fused_video_features)  # [batch, 186, 128]

        # ========== 8. 音频处理 ==========
        audio_features = audio_input  # [batch, 186, 128]

        # ========== 9. 视频-音频交叉注意力 ==========
        video_attn, _ = self.video_audio_attn(
            fused_video_features, audio_features, audio_features
        )
        audio_attn, _ = self.video_audio_attn(
            audio_features, fused_video_features, fused_video_features
        )

        # ========== 10. 特征融合 ==========
        combined_features = torch.cat([video_attn, audio_attn], dim=2)  # [batch, 186, 256]

        # 前馈网络 + 残差
        combined_features = self.norm2(combined_features)
        ffn_output = self.ffn(combined_features.reshape(-1, 256)).reshape(batch_size, 186, 256)
        combined_features = self.norm2(ffn_output + combined_features)

        # ========== 11. 全局池化 ==========
        pooled_features = self.pooling(combined_features.transpose(1, 2)).squeeze(-1)

        # ========== 12. 分类 ==========
        classification_output = self.classifier(pooled_features)
        # ========== 13. 【新增】对偶监督（Dual Supervision）==========
        # 用 P_c / E_c 的原始特征直接预测抑郁标签
        # 将各线索在时序维度平均池化，然后拼接
        path_pooled_list = [torch.mean(pathological_features[c], dim=1) for c in self.cue_names]  # each [B, 32]
        pers_pooled_list = [torch.mean(personalized_features[c], dim=1) for c in self.cue_names]  # each [B, 32]

        path_all = torch.cat(path_pooled_list, dim=-1)  # [B, 128]
        pers_all = torch.cat(pers_pooled_list, dim=-1)  # [B, 128]

        # 13.1 正向监督：强迫病理特征懂抑郁
        path_logits = self.path_depression_head(path_all)

        # 13.2 对抗监督：套上 GRL，强迫个性化特征不懂抑郁
        pers_logits = self.pers_depression_head(grad_reverse(pers_all, self.grl_alpha))

        # ========== 14. 损失计算 ==========
        total_loss = 0
        aux_loss_total = 0

        path_sup_loss = torch.tensor(0.0, device=classification_output.device)
        pers_adv_loss = torch.tensor(0.0, device=classification_output.device)

        if labels is not None:
            labels = labels.long()

            # 分类损失
            classification_loss = F.cross_entropy(classification_output, labels)

            # 重构损失
            total_reconstruction_loss = sum(reconstruction_losses.values())

            # 正交损失
            total_orthogonal_loss = sum(orthogonal_losses.values())

            # 辅助分类损失（每个线索的辅助分类器）
            if cue_aux_logits_dict:
                for cue_name, aux_logits in cue_aux_logits_dict.items():
                    aux_loss_total += F.cross_entropy(aux_logits, labels)
                aux_loss_total *= self.auxiliary_weight

            path_sup_loss = F.cross_entropy(path_logits, labels)
            pers_adv_loss = F.cross_entropy(pers_logits, labels)

            total_loss = (self.classification_weight * classification_loss +
                          self.reconstruction_weight * total_reconstruction_loss +
                          self.contrastive_weight * contrastive_loss +
                          self.orthogonal_weight * total_orthogonal_loss +
                          aux_loss_total+
                          self.path_supervision_weight * path_sup_loss +
                          self.pers_adversarial_weight * pers_adv_loss)

        # ========== 14. 返回结果（增强可解释性）==========
        result = {
            'classification_output': classification_output,
            'pathological_features': pathological_features,
            'personalized_features': personalized_features,
            'final_fused_features': pooled_features,
            'total_loss': total_loss,
            'losses': {
                'classification': F.cross_entropy(classification_output, labels) if labels is not None else 0,
                'reconstruction': sum(reconstruction_losses.values()),
                'contrastive': contrastive_loss,
                'orthogonal': sum(orthogonal_losses.values()),
                'auxiliary': aux_loss_total if labels is not None else 0,
                'path_supervision': path_sup_loss,  # 【新增】
                'pers_adversarial': pers_adv_loss,  # 【新增】
            },
            'path_logits': path_logits,  # 诊断用
            'pers_logits': pers_logits,  # 诊断用
            # ✅ 可解释性信息
            'cue_weights': cue_weights_dict,  # 每个线索的权重 {'gaze': [0.85, 0.15], ...}
            'cue_importance': cue_importance,  # 线索重要性 [0.25, 0.15, 0.45, 0.15]
        }
        
        # 计算不同类别的平均权重（用于监控）
        if labels is not None:
            depression_mask = (labels == 1)
            normal_mask = (labels == 0)
            
            # 为每个线索计算平均权重
            avg_weights_depression = {}
            avg_weights_normal = {}
            
            for cue_name in self.cue_names:
                if depression_mask.any():
                    avg_weights_depression[cue_name] = cue_weights_dict[cue_name][depression_mask].mean(dim=0)
                if normal_mask.any():
                    avg_weights_normal[cue_name] = cue_weights_dict[cue_name][normal_mask].mean(dim=0)
            
            result['avg_weights_depression'] = avg_weights_depression
            result['avg_weights_normal'] = avg_weights_normal
            
            # 线索重要性的平均值
            if depression_mask.any():
                result['avg_cue_importance_depression'] = cue_importance[depression_mask].mean(dim=0)
            if normal_mask.any():
                result['avg_cue_importance_normal'] = cue_importance[normal_mask].mean(dim=0)

        return result

    # ====================== 因果干预接口 ======================
    def forward_intervention(self, audio_input, pathological_features, personalized_features):
        """
        用于因果干预实验：跳过编码与解耦阶段，直接接受外部注入的 P_c / E_c 字典，
        运行后续的投影、加权、融合、分类阶段。

        Args:
            audio_input:            [B, 186, 128]
            pathological_features:  dict, key in {'gaze','pose','landmark','au'},
                                    value: [B, T, path_dim]
            personalized_features:  dict, key同上, value: [B, T, pers_dim]

        Returns:
            classification_output:  [B, 2]
        """
        device = audio_input.device
        first_cue = pathological_features[self.cue_names[0]]
        batch_size, seq_len, _ = first_cue.shape

        # ---- 4. 投影 P/E 到 128 维 ----
        path_features_dict = {}
        pers_features_dict = {}
        for cue_name in self.cue_names:
            path_feat = pathological_features[cue_name]
            B, T, pd = path_feat.shape
            path_features_dict[cue_name] = self.path_projectors[cue_name](
                path_feat.reshape(-1, pd)
            ).reshape(B, T, 128)

            pers_feat = personalized_features[cue_name]
            pers_features_dict[cue_name] = self.pers_projectors[cue_name](
                pers_feat.reshape(-1, pers_feat.shape[-1])
            ).reshape(B, T, 128)

        # ---- 5. 分线索自适应加权 ----
        cue_weighted_features = []
        for cue_name in self.cue_names:
            weighted_feat, _, _ = self.cue_adaptive_weightings[cue_name](
                pathological_features=path_features_dict[cue_name],
                personalized_features=pers_features_dict[cue_name],
                labels=None
            )
            cue_weighted_features.append(weighted_feat)

        # ---- 6. 融合（注意力支路 + 线索重要性支路）----
        all_cues_concat = torch.cat(cue_weighted_features, dim=-1)
        Att_fused_features = self.cross_cue_attn_path(cue_weighted_features)

        all_cues_pooled = torch.mean(all_cues_concat, dim=1)
        cue_importance = self.cue_importance_predictor(all_cues_pooled)

        importance_weighted_features = []
        for i, cue_feat in enumerate(cue_weighted_features):
            importance = cue_importance[:, i:i + 1].unsqueeze(1)
            importance_weighted_features.append(cue_feat * importance)

        importance_fused = torch.sum(torch.stack(importance_weighted_features, dim=0), dim=0)
        fused_video_features = 0.5 * Att_fused_features + 0.5 * importance_fused
        fused_video_features = fused_video_features[:, :, :128]

        # ---- 7. 序列长度对齐 ----
        fused_video_features = self.seq_conv(fused_video_features)

        # ---- 8-10. 视听交叉注意力 + FFN ----
        audio_features = audio_input
        video_attn, _ = self.video_audio_attn(
            fused_video_features, audio_features, audio_features
        )
        audio_attn, _ = self.video_audio_attn(
            audio_features, fused_video_features, fused_video_features
        )
        combined_features = torch.cat([video_attn, audio_attn], dim=2)
        combined_features = self.norm2(combined_features)
        ffn_output = self.ffn(combined_features.reshape(-1, 256)).reshape(batch_size, 186, 256)
        combined_features = self.norm2(ffn_output + combined_features)

        # ---- 11-12. 池化 + 分类 ----
        pooled_features = self.pooling(combined_features.transpose(1, 2)).squeeze(-1)
        classification_output = self.classifier(pooled_features)
        return classification_output


# ====================== 使用示例 ======================
if __name__ == '__main__':

    # 创建线索级对比学习模型
    cue_contrastive_model = CueLevelContrastiveBCDCIE().cuda()

    # 模拟数据
    batch_size = 4
    video_data = torch.randn(batch_size, 915, 171).cuda()
    audio_data = torch.randn(batch_size, 186, 128).cuda()
    labels = torch.randint(0, 2, (batch_size,)).cuda()

    # 前向传播
    cue_contrastive_model.train()
    results = cue_contrastive_model(video_data, audio_data, labels)

    print("=" * 60)
    print("分类输出形状:", results['classification_output'].shape)
    print("分类输出:", results['classification_output'])
    
    print("\n" + "=" * 60)
    print("【可解释性分析】")
    print("=" * 60)
    
    # 显示每个线索的权重
    print("\n各线索的病理/个性化权重分布:")
    for cue_name, weights in results['cue_weights'].items():
        weights_mean = weights.mean(dim=0)
        print(f"  {cue_name:10s}: 病理 {weights_mean[0]:.3f}, 个性化 {weights_mean[1]:.3f}")
    
    # 显示线索重要性
    print("\n线索重要性分布:")
    cue_importance_mean = results['cue_importance'].mean(dim=0)
    for i, cue_name in enumerate(['gaze', 'pose', 'landmark', 'au']):
        print(f"  {cue_name:10s}: {cue_importance_mean[i]:.3f} ({cue_importance_mean[i]*100:.1f}%)")
    
    # 显示不同类别的权重
    if 'avg_weights_depression' in results:
        print("\n抑郁样本的平均权重:")
        for cue_name, weights in results['avg_weights_depression'].items():
            print(f"  {cue_name:10s}: 病理 {weights[0]:.3f}, 个性化 {weights[1]:.3f}")
    
    if 'avg_weights_normal' in results:
        print("\n正常样本的平均权重:")
        for cue_name, weights in results['avg_weights_normal'].items():
            print(f"  {cue_name:10s}: 病理 {weights[0]:.3f}, 个性化 {weights[1]:.3f}")
    
    print("\n" + "=" * 60)
    print("【损失信息】")
    print("=" * 60)
    print("总损失:", results['total_loss'].item())
    print("详细损失:")
    for loss_name, loss_value in results['losses'].items():
        if torch.is_tensor(loss_value):
            print(f"  {loss_name:15s}: {loss_value.item():.4f}")
        else:
            print(f"  {loss_name:15s}: {loss_value:.4f}")
    
    print("\n" + "=" * 60)
    print("【模型架构改进说明】")
    print("=" * 60)
    # print("1. ✅ 删除了交叉注意力模块，保留线索独立性")
    # print("2. ✅ 采用分线索加权，每个线索独立决定权重")
    print("3. ✅ 简单融合 + 线索重要性加权")
    print("4. ✅ 增强可解释性，可追溯每个线索的贡献")
    print("=" * 60)
