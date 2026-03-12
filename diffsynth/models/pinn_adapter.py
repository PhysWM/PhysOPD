"""
Physics-Informed Adapter Module
物理约束适配器 - 作为插件连接到原始模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsAdapter(nn.Module):
    """
    物理约束适配器层
    
    将物理信息注入到速度场预测中，而不改变原始模型
    类似于 LoRA 的思想，但专门用于物理约束
    """
    
    def __init__(
        self,
        latent_dim=16,
        hidden_dim=64,
        num_phenomena=17,
        n_numeric_dim=12,
        q_input_dim=64,
        n_text_vocab_size=2048,
        shared_expert_weight=0.3,
        moe_top_k=4,
        pde_residuals=None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_phenomena = num_phenomena
        self.n_numeric_dim = n_numeric_dim
        self.q_input_dim = q_input_dim
        self.n_text_vocab_size = n_text_vocab_size
        self.shared_expert_weight = shared_expert_weight
        self.moe_top_k = max(1, min(int(moe_top_k), num_phenomena))
        self.router_temperature = 1.0
        self.router_label_bias = 2.0
        self.use_moe = True
        self.label_only_mode = False
        self.lightweight_cache = False
        self.pde_residuals = pde_residuals  # Reference to PDE residuals module
        self.apply_constraints_in_forward = True  # Enable constraint application
        self.constraint_step_size = 0.01  # Small step size for constraint enforcement

        # 物理特征提取器（轻量级）
        self.physics_encoder = nn.Sequential(
            nn.Conv3d(latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        
        # 旧路径校正层：metadata 缺失时回退使用
        self.physics_correction_fallback = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, latent_dim, kernel_size=1),
        )

        # 条件编码器：n(规则数值) + n(文本) + q(多标签)
        self.n_numeric_proj = nn.Sequential(
            nn.Linear(n_numeric_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.n_text_embedding = nn.Embedding(n_text_vocab_size, hidden_dim)
        self.q_proj = nn.Sequential(
            nn.Linear(q_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.expert_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_phenomena),
        )

        # MoE 专家：在物理特征空间中做 top-k 路由
        self.phenomenon_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(hidden_dim * 2, hidden_dim, kernel_size=1),
                nn.SiLU(),
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
            )
            for _ in range(num_phenomena)
        ])
        self.shared_expert = nn.Sequential(
            nn.Conv3d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        # 专家输出 + 基础特征汇合为统一校正
        self.shared_merge_head = nn.Sequential(
            nn.Conv3d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, latent_dim, kernel_size=1),
        )

        # 一致性正则：专家特征可重建条件嵌入
        self.condition_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.register_buffer(
            "expert_usage_ema",
            torch.full((num_phenomena,), 1.0 / max(num_phenomena, 1))
        )
        
        # 可学习的缩放因子（初始化为0，不影响原始输出）
        self.scale = nn.Parameter(torch.zeros(1))
        
        # 缓存最近一次 forward 的中间结果，供外部可视化使用
        self._cache = {}

    def _route_label_ids(self, metadata, batch_size, device):
        if not isinstance(metadata, dict) or "label_id" not in metadata:
            return torch.zeros(batch_size, device=device, dtype=torch.long)
        label_ids = metadata["label_id"].to(device=device, dtype=torch.long).view(-1)
        if label_ids.numel() == 1:
            label_ids = label_ids.repeat(batch_size)
        elif label_ids.numel() != batch_size:
            label_ids = label_ids[:1].repeat(batch_size)
        return torch.clamp(label_ids, min=0, max=self.num_phenomena - 1)

    def _fit_2d(self, tensor, target_dim, batch_size, device, dtype):
        if tensor is None:
            return torch.zeros(batch_size, target_dim, device=device, dtype=dtype)
        tensor = tensor.to(device=device, dtype=dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.repeat(batch_size, 1)
        if tensor.shape[0] != batch_size:
            tensor = tensor[:1].repeat(batch_size, 1)
        feat_dim = tensor.shape[1]
        if feat_dim > target_dim:
            return tensor[:, :target_dim]
        if feat_dim < target_dim:
            pad = torch.zeros(batch_size, target_dim - feat_dim, device=device, dtype=dtype)
            return torch.cat([tensor, pad], dim=1)
        return tensor

    def _encode_condition(self, metadata, batch_size, device, dtype):
        if not isinstance(metadata, dict):
            return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        if self.label_only_mode:
            return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        n_numeric = self._fit_2d(
            metadata.get("n_numeric"), self.n_numeric_dim, batch_size, device, dtype
        )
        q_vector = self._fit_2d(
            metadata.get("q_vector"), self.q_input_dim, batch_size, device, dtype
        )
        n_text_ids = metadata.get("n_text_ids")
        if n_text_ids is None:
            n_text_ids = torch.zeros(batch_size, 3, device=device, dtype=torch.long)
        else:
            n_text_ids = n_text_ids.to(device=device, dtype=torch.long)
            if n_text_ids.ndim == 1:
                n_text_ids = n_text_ids.unsqueeze(0)
            if n_text_ids.shape[0] == 1 and batch_size > 1:
                n_text_ids = n_text_ids.repeat(batch_size, 1)
            if n_text_ids.shape[0] != batch_size:
                n_text_ids = n_text_ids[:1].repeat(batch_size, 1)
        n_text_ids = torch.clamp(n_text_ids, min=0, max=self.n_text_vocab_size - 1)

        n_numeric_feat = self.n_numeric_proj(n_numeric)
        n_text_feat = self.n_text_embedding(n_text_ids).mean(dim=1)
        q_feat = self.q_proj(q_vector)
        cond_feat = self.condition_fuse(torch.cat([n_numeric_feat, n_text_feat, q_feat], dim=-1))
        return cond_feat.to(dtype=dtype)

    def _route_experts(self, cond_feat, metadata, batch_size, device):
        label_ids = self._route_label_ids(metadata, batch_size, device)
        route_logits = self.expert_router(cond_feat)
        label_prior = torch.zeros_like(route_logits)
        label_prior.scatter_(1, label_ids.unsqueeze(1), self.router_label_bias)
        route_logits = route_logits + label_prior

        top_k = min(self.moe_top_k, self.num_phenomena)
        topk_logits, topk_indices = torch.topk(route_logits, k=top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits / max(self.router_temperature, 1e-6), dim=-1)
        dominant_expert = topk_indices[:, 0]
        return route_logits, topk_indices, topk_weights, dominant_expert, label_ids
    
    def forward(self, v_original, z_t, metadata=None):
        """
        Forward pass: Apply physics-informed corrections via MoE expert routing.

        Args:
            v_original: 原始模型预测的速度场 [B, C, T, H, W]
            z_t: 当前的latent [B, C, T, H, W]
            metadata: 条件输入字典（label_id/label_name/n/q）

        Returns:
            v_corrected: 物理校正后的速度场 [B, C, T, H, W]
        """
        B = v_original.shape[0]
        
        physics_feat = self.physics_encoder(z_t)  # [B, hidden_dim, T, H, W]

        if metadata is None or not self.use_moe:
            raw_correction = self.physics_correction_fallback(physics_feat)
            v_corrected = v_original + self.scale * raw_correction
            topk_shape = (B, self.moe_top_k)
            branch_feat_shape = (B, self.moe_top_k, *physics_feat.shape[1:])
            branch_latent_shape = (B, self.moe_top_k, *v_original.shape[1:])
            zero_indices = torch.zeros(topk_shape, dtype=torch.long, device=v_original.device)
            zero_weights = torch.zeros(topk_shape, dtype=physics_feat.dtype, device=v_original.device)
            zero_branch_outputs = torch.zeros(branch_feat_shape, dtype=physics_feat.dtype, device=v_original.device)
            zero_branch_corrections = torch.zeros(
                branch_latent_shape, dtype=v_original.dtype, device=v_original.device
            )
            self._cache = {
                "physics_feat": physics_feat.detach(),
                "raw_correction": raw_correction.detach(),
                "expert_output": torch.zeros_like(physics_feat).detach(),
                "cond_feat": torch.zeros(B, self.hidden_dim, device=v_original.device).detach(),
                "expert_output_live": torch.zeros_like(physics_feat),
                "cond_feat_live": torch.zeros(B, self.hidden_dim, device=v_original.device, dtype=physics_feat.dtype),
                "label_ids": torch.zeros(B, dtype=torch.long, device=v_original.device),
                "active_expert_indices": zero_indices,
                "active_expert_weights": zero_weights,
                "branch_physics_features": zero_branch_outputs.detach(),
                "branch_physics_features_live": zero_branch_outputs,
                "branch_expert_outputs": zero_branch_outputs.detach(),
                "branch_expert_outputs_live": zero_branch_outputs,
                "branch_raw_corrections": zero_branch_corrections.detach(),
                "branch_raw_corrections_live": zero_branch_corrections,
                "branch_v_corrected": zero_branch_corrections.detach(),
                "branch_v_corrected_live": zero_branch_corrections,
                "dominant_expert": torch.zeros(B, dtype=torch.long, device=v_original.device),
                "route_logits": torch.zeros(B, self.num_phenomena, device=v_original.device),
            }
            return v_corrected

        cond_feat = self._encode_condition(metadata, B, v_original.device, physics_feat.dtype)
        cond_map = cond_feat.view(B, -1, 1, 1, 1).expand(-1, -1, *physics_feat.shape[2:])
        expert_input = torch.cat([physics_feat, cond_map], dim=1)

        route_logits, topk_indices, topk_weights, dominant_expert, label_ids = self._route_experts(
            cond_feat=cond_feat,
            metadata=metadata,
            batch_size=B,
            device=v_original.device,
        )
        expert_output = torch.zeros_like(physics_feat)
        branch_expert_outputs = torch.zeros(
            B,
            topk_indices.shape[1],
            *physics_feat.shape[1:],
            device=physics_feat.device,
            dtype=physics_feat.dtype,
        )
        branch_raw_corrections = torch.zeros(
            B,
            topk_indices.shape[1],
            self.latent_dim,
            *v_original.shape[2:],
            device=v_original.device,
            dtype=v_original.dtype,
        )
        branch_v_corrected = torch.zeros(
            B,
            topk_indices.shape[1],
            *v_original.shape[1:],
            device=v_original.device,
            dtype=v_original.dtype,
        )
        for i in range(B):
            routed_feat = torch.zeros_like(physics_feat[i:i + 1])
            for j in range(topk_indices.shape[1]):
                expert_idx = int(topk_indices[i, j].item())
                expert_weight = topk_weights[i, j].to(dtype=physics_feat.dtype)
                branch_feat = self.phenomenon_experts[expert_idx](expert_input[i:i + 1])
                branch_expert_outputs[i:i + 1, j] = branch_feat
                branch_merged = torch.cat([physics_feat[i:i + 1], branch_feat], dim=1)
                branch_raw_correction = self.shared_merge_head(branch_merged)
                branch_raw_corrections[i:i + 1, j] = branch_raw_correction
                branch_v_corrected[i:i + 1, j] = (
                    v_original[i:i + 1] + self.scale * branch_raw_correction
                )
                routed_feat = routed_feat + expert_weight * branch_feat
            shared_feat = self.shared_expert(expert_input[i:i + 1])
            mixed_feat = (1.0 - self.shared_expert_weight) * routed_feat + self.shared_expert_weight * shared_feat
            expert_output[i:i + 1] = mixed_feat

        merged = torch.cat([physics_feat, expert_output], dim=1)
        raw_correction = self.shared_merge_head(merged)
        v_corrected = v_original + self.scale * raw_correction

        with torch.no_grad():
            hist = torch.zeros(self.num_phenomena, device=v_original.device, dtype=torch.float32)
            hist.scatter_add_(0, topk_indices.reshape(-1), topk_weights.reshape(-1).float())
            hist = hist / max(float(B), 1.0)
            self.expert_usage_ema.mul_(0.99).add_(0.01 * hist)
        branch_physics_features_live = branch_expert_outputs
        branch_expert_outputs_live = branch_expert_outputs
        branch_raw_corrections_live = branch_raw_corrections
        if self.lightweight_cache:
            branch_physics_features_live = branch_expert_outputs.detach()
            branch_expert_outputs_live = branch_expert_outputs.detach()
            branch_raw_corrections_live = branch_raw_corrections.detach()

        # 缓存中间结果（detach 避免影响梯度计算图）
        self._cache = {
            "physics_feat": physics_feat.detach(),    # [B, hidden_dim, T, H, W]
            "raw_correction": raw_correction.detach(), # [B, C, T, H, W] (未乘 scale)
            "expert_output": expert_output.detach(),
            "cond_feat": cond_feat.detach(),
            "expert_output_live": expert_output,
            "cond_feat_live": cond_feat,
            "label_ids": label_ids.detach(),
            "active_expert_indices": topk_indices.detach(),
            "active_expert_weights": topk_weights.detach(),
            "branch_physics_features": branch_expert_outputs.detach(),
            "branch_physics_features_live": branch_physics_features_live,
            "branch_expert_outputs": branch_expert_outputs.detach(),
            "branch_expert_outputs_live": branch_expert_outputs_live,
            "branch_raw_corrections": branch_raw_corrections.detach(),
            "branch_raw_corrections_live": branch_raw_corrections_live,
            "branch_v_corrected": branch_v_corrected.detach(),
            "branch_v_corrected_live": branch_v_corrected,
            "dominant_expert": dominant_expert.detach(),
            "route_logits": route_logits.detach(),
        }

        return v_corrected

    def set_pde_residuals(self, pde_residuals):
        """设置PDE残差模块（用于约束计算）"""
        self.pde_residuals = pde_residuals

    def set_constraint_mode(self, enabled=True, step_size=0.01):
        """设置物理约束应用模式"""
        self.apply_constraints_in_forward = bool(enabled)
        self.constraint_step_size = max(float(step_size), 0.0)

    def set_ablation_modes(self, use_moe=True, label_only_mode=False):
        """设置消融模式开关"""
        self.use_moe = bool(use_moe)
        self.label_only_mode = bool(label_only_mode)

    def set_cache_mode(self, lightweight=False):
        """设置训练缓存模式，lightweight=True 时减少重型 live 张量缓存。"""
        self.lightweight_cache = bool(lightweight)

    def compute_auxiliary_losses(self):
        """返回 MoE 相关辅助损失"""
        device = self.scale.device
        expert_balance = torch.mean(
            (self.expert_usage_ema - 1.0 / max(self.num_phenomena, 1)) ** 2
        ).to(device=device)

        if "expert_output_live" not in self._cache or "cond_feat_live" not in self._cache:
            condition_consistency = torch.zeros((), device=device)
        else:
            expert_output = self._cache["expert_output_live"]
            cond_feat = self._cache["cond_feat_live"]
            expert_pool = expert_output.mean(dim=(2, 3, 4))
            recon_cond = self.condition_reconstructor(expert_pool)
            condition_consistency = F.mse_loss(recon_cond, cond_feat)

        return {
            "expert_balance": expert_balance,
            "condition_consistency": condition_consistency,
        }
