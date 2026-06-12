from collections import OrderedDict
import torch.nn as nn
import torch
import torch.nn.functional as F
from math import sqrt

class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时，梯度乘以 -alpha
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return _GradReverseFn.apply(x, alpha)

# ====================== 核心模块 ======================
class AstroModel(nn.Module):
    """时序特征编码器"""

    def __init__(self) -> None:
        super(AstroModel, self).__init__()
        self.dropout = nn.Dropout(0.2)

        self.conv1_1 = nn.Sequential(nn.Conv1d(25, 25, 3, padding=2, dilation=2), nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv1d(25, 25, 3, padding=2, dilation=2), nn.ReLU())
        self.conv2_1 = nn.Sequential(nn.Conv1d(25, 25, 3, padding=4, dilation=4), nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv1d(25, 25, 3, padding=4, dilation=4), nn.ReLU())
        self.conv3_1 = nn.Sequential(nn.Conv1d(25, 25, 3, padding=8, dilation=8), nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv1d(25, 25, 3, padding=8, dilation=8), nn.ReLU())
        self.conv4_1 = nn.Sequential(nn.Conv1d(25, 25, 3, padding=16, dilation=16), nn.ReLU())
        self.conv4_2 = nn.Sequential(nn.Conv1d(25, 25, 3, padding=16, dilation=16), nn.ReLU())

    def forward(self, x):
        features = []
        residual = x
        x = self.conv1_1(x)
        x = self.dropout(x)
        x = self.conv1_2(x)
        features.append(x)
        x = x + residual

        residual = x
        x = self.conv2_1(x)
        x = self.dropout(x)
        x = self.conv2_2(x)
        features.append(x)
        x = x + residual

        residual = x
        x = self.conv3_1(x)
        x = self.dropout(x)
        x = self.conv3_2(x)
        features.append(x)
        x = x + residual

        residual = x
        x = self.conv4_1(x)
        x = self.dropout(x)
        x = self.conv4_2(x)
        features.append(x)
        x = x + residual
        return x, features


# ====================== 特征分离模块 ======================
class FeatureDisentangler(nn.Module):
    """将特征分离为病理特征和个性化特征"""

    def __init__(self, input_dim=25, pathological_dim=15, personalized_dim=10):
        super().__init__()
        self.pathological_encoder = nn.Sequential(
            nn.Conv1d(input_dim, pathological_dim * 2, 1),
            nn.BatchNorm1d(pathological_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(pathological_dim * 2, pathological_dim, 1),
            nn.BatchNorm1d(pathological_dim),
        )
        self.personalized_encoder = nn.Sequential(
            nn.Conv1d(input_dim, personalized_dim * 2, 1),
            nn.BatchNorm1d(personalized_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(personalized_dim * 2, personalized_dim, 1),
            nn.BatchNorm1d(personalized_dim),
        )

    def forward(self, x):
        return self.pathological_encoder(x), self.personalized_encoder(x)


# ====================== 特征重建模块 ======================
class FeatureReconstructor(nn.Module):
    def __init__(self, pathological_dim=15, personalized_dim=10, output_dim=25):
        super().__init__()
        self.reconstructor = nn.Sequential(
            nn.Conv1d(pathological_dim + personalized_dim, output_dim * 2, 1),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(output_dim * 2, output_dim, 1),
        )

    def forward(self, pathological, personalized):
        combined = torch.cat([pathological, personalized], dim=1)
        return self.reconstructor(combined)


# ====================== 【新增】自适应特征加权模块 ======================
class AdaptiveFeatureWeighting(nn.Module):
    """
    基于特征内容，自适应计算病理和个性化特征的权重并融合
    输入形状: [B, L, C] (注意：这里在内部处理时使用 Linear，所以需要 L 在前或转置)
    """

    def __init__(self, pathological_dim, personalized_dim, hidden_dim=32):
        super(AdaptiveFeatureWeighting, self).__init__()
        self.pathological_dim = pathological_dim
        self.personalized_dim = personalized_dim

        # 病理特征分析器
        self.pathological_analyzer = nn.Sequential(
            nn.Linear(pathological_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # 个性化特征分析器
        self.personalized_analyzer = nn.Sequential(
            nn.Linear(personalized_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # 联合权重预测器
        self.joint_weight_predictor = nn.Sequential(
            nn.Linear(pathological_dim + personalized_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(pathological_dim + personalized_dim, pathological_dim + personalized_dim),
            nn.LayerNorm(pathological_dim + personalized_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, pathological_features, personalized_features):
        """
        Args:
            pathological_features: [B, L, path_dim]
            personalized_features: [B, L, pers_dim]
        Returns:
            fused_features: [B, L, path_dim + pers_dim]
            weights: [B, 2]
        """
        batch_size, seq_len, _ = pathological_features.shape

        # 全局池化用于计算权重
        path_pooled = torch.mean(pathological_features, dim=1)  # [B, path_dim]
        pers_pooled = torch.mean(personalized_features, dim=1)  # [B, pers_dim]

        # 分析重要性
        path_imp = self.pathological_analyzer(path_pooled)
        pers_imp = self.personalized_analyzer(pers_pooled)

        # 预测权重
        weight_input = torch.cat([path_pooled, pers_pooled, path_imp, pers_imp], dim=-1)
        weights = self.joint_weight_predictor(weight_input)  # [B, 2]

        # 扩展权重
        path_weight = weights[:, 0:1].unsqueeze(1)  # [B, 1, 1]
        pers_weight = weights[:, 1:2].unsqueeze(1)  # [B, 1, 1]

        # 加权
        weighted_path = pathological_features * path_weight
        weighted_pers = personalized_features * pers_weight

        # 拼接与融合
        weighted_combined = torch.cat([weighted_path, weighted_pers], dim=-1)  # [B, L, total_dim]
        fused_features = self.fusion_layer(weighted_combined)

        return fused_features, weights


# ====================== 交叉注意力模块 (Stream A) ======================
class Multi_CrossAttention(nn.Module):
    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num

        self.linear_q = nn.Linear(hidden_size, all_head_size)
        self.linear_k = nn.Linear(hidden_size, all_head_size)
        self.linear_v = nn.Linear(hidden_size, all_head_size)

    def forward(self, x, y):
        # x, y: [B, L, hidden_size]
        batch_size = x.size(0)

        # Video 查询 Audio
        qx = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        ky = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        vy = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # Audio 查询 Video
        qy = self.linear_q(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        kx = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        vx = self.linear_v(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        attention_x = (qx @ ky.transpose(-2, -1)) / sqrt(self.h_size)
        attention_x = torch.softmax(attention_x, dim=-1) @ vy

        attention_y = (qy @ kx.transpose(-2, -1)) / sqrt(self.h_size)
        attention_y = torch.softmax(attention_y, dim=-1) @ vx

        attention_x = attention_x.transpose(1, 2).contiguous().view(batch_size, -1, self.all_head_size)
        attention_y = attention_y.transpose(1, 2).contiguous().view(batch_size, -1, self.all_head_size)

        return attention_x + x, attention_y + y


# ====================== 损失函数 ======================
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features1, features2, labels=None, same_modality=False):
        features1 = F.adaptive_avg_pool1d(features1, 1).squeeze(-1)
        features2 = F.adaptive_avg_pool1d(features2, 1).squeeze(-1)
        features1 = F.normalize(features1, dim=1, eps=1e-8)
        features2 = F.normalize(features2, dim=1, eps=1e-8)
        batch_size = features1.shape[0]
        similarity_matrix = torch.matmul(features1, features2.T) / self.temperature
        similarity_matrix = torch.clamp(similarity_matrix, min=-10, max=10)

        if same_modality and labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(features1.device)
            mask = mask - torch.eye(batch_size).to(features1.device)
            exp_sim = torch.exp(similarity_matrix)
            log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
            loss = -mean_log_prob_pos.mean()
        else:
            labels_contrastive = torch.arange(batch_size).to(features1.device)
            loss = F.cross_entropy(similarity_matrix, labels_contrastive)
        return loss


class RepulsiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, features1, features2):
        features1 = F.adaptive_avg_pool1d(features1, 1).squeeze(-1)
        features2 = F.adaptive_avg_pool1d(features2, 1).squeeze(-1)
        features1 = F.normalize(features1, dim=1, eps=1e-8)
        features2 = F.normalize(features2, dim=1, eps=1e-8)
        similarity = F.cosine_similarity(features1, features2, dim=1)
        loss = torch.clamp(similarity + self.margin, min=0).mean()
        return loss


# ====================== 视频预处理模块 ======================
class Conv1d(nn.Module):
    def __init__(self) -> None:
        super(Conv1d, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv1d(136, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv1d(64, 25, 3, padding=1), nn.BatchNorm1d(25), nn.ReLU())

    def forward(self, input):
        return self.layer3(self.layer2(self.layer1(input)))


# ====================== 主网络 (集成加权与混合融合) ======================
class Net(nn.Module):
    def __init__(self, pathological_dim=15, personalized_dim=10):
        super().__init__()

        # 1. 基础编码器
        self.video_conv = Conv1d()
        self.video_astro = AstroModel()
        self.audio_astro = AstroModel()

        # 2. 特征分离
        self.video_disentangler = FeatureDisentangler(25, pathological_dim, personalized_dim)
        self.audio_disentangler = FeatureDisentangler(25, pathological_dim, personalized_dim)
        self.video_reconstructor = FeatureReconstructor(pathological_dim, personalized_dim, 25)
        self.audio_reconstructor = FeatureReconstructor(pathological_dim, personalized_dim, 25)

        # 3. 自适应特征加权模块
        self.video_adaptive_weighting = AdaptiveFeatureWeighting(pathological_dim, personalized_dim)
        self.audio_adaptive_weighting = AdaptiveFeatureWeighting(pathological_dim, personalized_dim)

        self.weighted_dim = pathological_dim + personalized_dim

        # 4. 【Stream A】拼接自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.weighted_dim,
            num_heads=5,
            batch_first=True,
            dropout=0.1
        )
        self.norm_sa = nn.LayerNorm(self.weighted_dim)

        # 5. 【Stream B】重要性加权流
        self.importance_predictor = nn.Sequential(
            nn.Linear(self.weighted_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

        # 6. 主分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.weighted_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

        # 7. 【新增】对偶监督头 (Dual Supervision Heads)
        # 接收拼接后的视听病理特征 (15+15=30维)
        self.path_depression_head = nn.Sequential(
            nn.Linear(pathological_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
        # 接收拼接后的视听个性化特征 (15+15=30维)
        self.pers_depression_head = nn.Sequential(
            nn.Linear(personalized_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
        # GRL 对抗强度，外部可调
        self.grl_alpha = 1.0

        # 损失函数
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        self.repulsive_loss = RepulsiveLoss(margin=0.5)
        self.recon_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()

        self.video_norm = nn.LayerNorm(self.weighted_dim)
        self.audio_norm = nn.LayerNorm(self.weighted_dim)

    def forward(self, video_input, audio_input, labels=None, return_losses=False):
        batch_size = video_input.size(0)

        # ========== 1. 编码与分离 ==========
        video_conv = self.video_conv(video_input.transpose(1, 2))
        video_encoded, _ = self.video_astro(video_conv)
        audio_encoded, _ = self.audio_astro(audio_input.transpose(1, 2))

        # 形状: [B, Dim, L]
        video_path, video_pers = self.video_disentangler(video_encoded)
        audio_path, audio_pers = self.audio_disentangler(audio_encoded)

        # 重建
        video_recon = self.video_reconstructor(video_path, video_pers)
        audio_recon = self.audio_reconstructor(audio_path, audio_pers)

        # ========== 【新增】对偶监督预测 ==========
        # 时序维度求均值，转换形状为 [B, Dim]
        v_p_pool = torch.mean(video_path, dim=2)
        a_p_pool = torch.mean(audio_path, dim=2)
        v_s_pool = torch.mean(video_pers, dim=2)
        a_s_pool = torch.mean(audio_pers, dim=2)

        # 拼接视听模态的病理和个性化特征
        path_all = torch.cat([v_p_pool, a_p_pool], dim=1)  # [B, 30]
        pers_all = torch.cat([v_s_pool, a_s_pool], dim=1)  # [B, 30]

        # 1. 正向监督病理特征 (强迫 P 懂抑郁)
        path_logits = self.path_depression_head(path_all)

        # 2. 梯度反转对抗个性化特征 (强迫 E 不懂抑郁)
        pers_logits = self.pers_depression_head(grad_reverse(pers_all, self.grl_alpha))

        # ========== 2. 自适应加权阶段 ==========
        # 转置为 [B, L, Dim] 以适应线性层和注意力机制
        video_path_t = video_path.transpose(1, 2)
        video_pers_t = video_pers.transpose(1, 2)
        audio_path_t = audio_path.transpose(1, 2)
        audio_pers_t = audio_pers.transpose(1, 2)

        # 在交互之前进行加权
        video_weighted, _ = self.video_adaptive_weighting(video_path_t, video_pers_t)
        audio_weighted, _ = self.audio_adaptive_weighting(audio_path_t, audio_pers_t)

        # ========== 3. 双流混合融合阶段 ==========
        video_len = video_weighted.size(1)
        audio_len = audio_weighted.size(1)

        video_weighted = self.video_norm(video_weighted)
        audio_weighted = self.audio_norm(audio_weighted)

        # --- Stream A: 拼接后自注意力 ---
        concat_seq = torch.cat([video_weighted, audio_weighted], dim=1)
        attn_output, _ = self.self_attention(concat_seq, concat_seq, concat_seq)
        concat_seq = self.norm_sa(concat_seq + attn_output)

        att_video = concat_seq[:, :video_len, :]
        att_audio = concat_seq[:, video_len:, :]
        stream_a_fused = torch.cat([torch.mean(att_video, dim=1), torch.mean(att_audio, dim=1)], dim=1)

        # --- Stream B: 全局重要性 ---
        pool_video = torch.mean(video_weighted, dim=1)
        pool_audio = torch.mean(audio_weighted, dim=1)
        modality_importance = self.importance_predictor(torch.cat([pool_video, pool_audio], dim=1))

        stream_b_fused = torch.cat([
            pool_video * modality_importance[:, 0:1],
            pool_audio * modality_importance[:, 1:2]
        ], dim=1)

        # --- 最终混合 ---
        final_features = 0.5 * stream_a_fused + 0.5 * stream_b_fused

        # ========== 4. 主分类 ==========
        cls_output = self.classifier(final_features)

        # ========== 5. 损失计算 ==========
        losses = {}
        if return_losses and labels is not None:
            labels_long = labels.long()

            # 基础损失
            losses['recon_loss'] = self.recon_loss(video_recon, video_encoded) + \
                                   self.recon_loss(audio_recon, audio_encoded)
            losses['classification_loss'] = self.cls_loss(cls_output, labels_long)

            # 对比与排斥损失
            losses['cross_modal_path_loss'] = self.contrastive_loss(video_path, audio_path, labels)
            losses['personalized_repulsive_loss'] = self.repulsive_loss(video_pers, audio_pers)

            if torch.sum(labels == 1) > 1:
                losses['same_modal_path_loss'] = self.contrastive_loss(video_path, video_path, labels, True) + \
                                                 self.contrastive_loss(audio_path, audio_path, labels, True)
            else:
                losses['same_modal_path_loss'] = torch.tensor(0.0, device=video_input.device)

            v_p_flat = F.adaptive_avg_pool1d(video_path, 1).squeeze(-1)
            v_s_flat = F.adaptive_avg_pool1d(video_pers, 1).squeeze(-1)
            a_p_flat = F.adaptive_avg_pool1d(audio_path, 1).squeeze(-1)
            a_s_flat = F.adaptive_avg_pool1d(audio_pers, 1).squeeze(-1)

            losses['orthogonal_loss'] = torch.mean(
                torch.abs(torch.sum(F.normalize(v_p_flat, dim=1) * F.normalize(v_s_flat, dim=1), dim=1))) + \
                                        torch.mean(torch.abs(
                                            torch.sum(F.normalize(a_p_flat, dim=1) * F.normalize(a_s_flat, dim=1),
                                                      dim=1)))

            # 【新增】计算对偶监督损失
            losses['path_sup_loss'] = self.cls_loss(path_logits, labels_long)
            losses['pers_adv_loss'] = self.cls_loss(pers_logits, labels_long)

        return (cls_output, losses) if return_losses else cls_output


# ====================== 测试代码 ======================
if __name__ == '__main__':
    # 初始化模型
    # 注意：pathological_dim + personalized_dim 就是加权后的特征维度
    model = Net(pathological_dim=15, personalized_dim=15).cuda()

    # 模拟输入数据
    video_data = torch.randn(4, 596, 136).cuda()
    audio_data = torch.randn(4, 596, 25).cuda()
    labels = torch.tensor([0, 1, 0, 1]).cuda()

    # 前向传播
    cls_output, losses = model(video_data, audio_data, labels, return_losses=True)

    print("=" * 50)
    print("模型结构检查:")
    print(f"加权模块输出维度: {model.weighted_dim}")  # 应该是 15+15=30
    print(f"分类器输入维度: {model.weighted_dim * 2}")  # 应该是 60
    print("=" * 50)
    print("分类输出形状:", cls_output.shape)
    print("=" * 50)
    print("各项损失:")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value.item():.6f}")