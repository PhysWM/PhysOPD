"""
Physics-Informed Adapter Module
物理约束适配器 - 作为插件连接到原始模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    """
    3D Residual Block with Conv3d + GroupNorm + SiLU
    支持残差连接，输入输出维度可以不同（通过 1x1 conv 调整）
    """
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, padding=padding)
        self.gn1 = nn.GroupNorm(8, out_dim)
        self.conv2 = nn.Conv3d(out_dim, out_dim, kernel_size=kernel_size, padding=padding)
        self.gn2 = nn.GroupNorm(8, out_dim)

        # 如果维度不同，使用 1x1 conv 调整
        self.shortcut = nn.Conv3d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = F.silu(x + residual)  # 残差连接
        return x


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
        num_phenomena=10,
        n_numeric_dim=12,
        q_input_dim=64,
        n_text_vocab_size=2048,
        shared_expert_weight=0.3,
        moe_top_k=4,
        pde_residuals=None,
        physics_state_mode="x0_hat",
        use_sigma_gate=True,
        sigma_gate_curve="quadratic",
        use_sigma_conditioning=True,
        sigma_conditioning_dim=None,
        sigma_gate_floor=0.05,
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
        self.router_temperature = 2.0  # Increased for smoother routing
        self.router_label_bias = 2.0
        self.use_moe = True
        self.label_only_mode = False
        self.lightweight_cache = False
        self.pde_residuals = pde_residuals  # Reference to PDE residuals module
        self.apply_constraints_in_forward = True  # Enable constraint application
        self.constraint_step_size = 0.01  # Small step size for constraint enforcement
        self.physics_state_mode = physics_state_mode
        self.use_sigma_gate = bool(use_sigma_gate)
        self.sigma_gate_curve = sigma_gate_curve
        self.use_sigma_conditioning = bool(use_sigma_conditioning)
        self.sigma_conditioning_dim = (
            hidden_dim if sigma_conditioning_dim is None else max(int(sigma_conditioning_dim), 1)
        )
        self.sigma_gate_floor = min(max(float(sigma_gate_floor), 0.0), 1.0)

        # 共享物理特征提取器（4层残差网络）
        self.physics_encoder_shared = nn.Sequential(
            ResBlock3D(latent_dim, hidden_dim),
            ResBlock3D(hidden_dim, hidden_dim),
            ResBlock3D(hidden_dim, hidden_dim),
            ResBlock3D(hidden_dim, hidden_dim),
        )

        # 每个专家独立的物理编码器（4层残差网络）
        self.expert_physics_encoders = nn.ModuleList([
            nn.Sequential(
                ResBlock3D(latent_dim, hidden_dim),
                ResBlock3D(hidden_dim, hidden_dim),
                ResBlock3D(hidden_dim, hidden_dim),
                ResBlock3D(hidden_dim, hidden_dim),
            )
            for _ in range(num_phenomena)
        ])

        # 每个专家独立的物理解码器（2层 + 残差）
        self.expert_physics_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(hidden_dim * 2, hidden_dim, kernel_size=1),
                nn.SiLU(),
                ResBlock3D(hidden_dim, hidden_dim),
                nn.Conv3d(hidden_dim, latent_dim, kernel_size=1),
            )
            for _ in range(num_phenomena)
        ])

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
        self.sigma_condition_proj = nn.Sequential(
            nn.Linear(1, self.sigma_conditioning_dim),
            nn.SiLU(),
            nn.Linear(self.sigma_conditioning_dim, hidden_dim),
        )
        self.expert_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_phenomena),
        )

        # MoE 专家：处理各自编码器提取的物理特征（4层残差网络）
        self.phenomenon_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(hidden_dim * 2, hidden_dim, kernel_size=1),
                nn.SiLU(),
                ResBlock3D(hidden_dim, hidden_dim),
                ResBlock3D(hidden_dim, hidden_dim),
                ResBlock3D(hidden_dim, hidden_dim),
            )
            for _ in range(num_phenomena)
        ])
        self.shared_expert = nn.Sequential(
            nn.Conv3d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.SiLU(),
            ResBlock3D(hidden_dim, hidden_dim),
            ResBlock3D(hidden_dim, hidden_dim),
            ResBlock3D(hidden_dim, hidden_dim),
        )

        # 共享解码器（用于 fallback 和最终输出融合）- 2层 + 残差
        self.shared_merge_head = nn.Sequential(
            nn.Conv3d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.SiLU(),
            ResBlock3D(hidden_dim, hidden_dim),
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

        # 初始化：将共享编码器/解码器的权重复制到每个专家
        # 这样专家从相同的起点开始，避免随机初始化导致的不稳定
        self._init_expert_weights_from_shared()
        
        # 可学习的缩放因子（初始化为0，不影响原始输出）
        self.scale = nn.Parameter(torch.zeros(1))
        
        # 缓存最近一次 forward 的中间结果，供外部可视化使用
        self._cache = {}

    def _init_expert_weights_from_shared(self):
        """
        将共享编码器/解码器的权重复制到每个专家的编码器/解码器。
        这确保所有专家从相同的起点开始训练，避免随机初始化导致的数值不稳定。
        """
        # 复制共享编码器权重到每个专家编码器
        shared_encoder_state = self.physics_encoder_shared.state_dict()
        for expert_encoder in self.expert_physics_encoders:
            expert_encoder.load_state_dict(shared_encoder_state)

        # 复制共享解码器权重到每个专家解码器
        shared_decoder_state = self.shared_merge_head.state_dict()
        for expert_decoder in self.expert_physics_decoders:
            expert_decoder.load_state_dict(shared_decoder_state)

    def _check_nan(self, tensor, name="tensor"):
        """检查张量是否包含 NaN 或 Inf，用于调试"""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return True
        return False

    def _safe_correction(self, correction, max_norm=10.0):
        """
        对校正值进行安全处理，防止数值爆炸。
        - 裁剪过大的值
        - 将 NaN/Inf 替换为 0
        """
        # 替换 NaN 和 Inf
        if torch.isnan(correction).any() or torch.isinf(correction).any():
            correction = torch.where(
                torch.isnan(correction) | torch.isinf(correction),
                torch.zeros_like(correction),
                correction
            )
        # 裁剪过大的值
        correction = torch.clamp(correction, min=-max_norm, max=max_norm)
        return correction

    def _prepare_sigma_values(self, sigma, batch_size, device, dtype):
        if sigma is None:
            return None

        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, device=device, dtype=dtype)
        else:
            sigma = sigma.to(device=device, dtype=dtype)

        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        sigma = sigma.reshape(-1)
        if sigma.shape[0] == 1 and batch_size > 1:
            sigma = sigma.repeat(batch_size)
        if sigma.shape[0] != batch_size:
            sigma = sigma[:1].repeat(batch_size)
        return torch.clamp(sigma, 0.0, 1.0)

    def _sigma_gate(self, sigma, ref_tensor):
        if (not self.use_sigma_gate) or sigma is None:
            return torch.ones(
                ref_tensor.shape[0], 1, 1, 1, 1,
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )

        sigma = self._prepare_sigma_values(
            sigma,
            batch_size=ref_tensor.shape[0],
            device=ref_tensor.device,
            dtype=ref_tensor.dtype,
        )

        if self.sigma_gate_curve == "linear":
            base_gate = 1.0 - sigma
        elif self.sigma_gate_curve == "hard":
            base_gate = torch.clamp((0.5 - sigma) / 0.5, min=0.0, max=1.0)
        else:
            base_gate = (1.0 - sigma) ** 2
        gate = self.sigma_gate_floor + (1.0 - self.sigma_gate_floor) * base_gate
        return gate.view(ref_tensor.shape[0], 1, 1, 1, 1)

    def _sigma_embedding(self, sigma, batch_size, device, dtype):
        if (not self.use_sigma_conditioning) or sigma is None:
            return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

        sigma_values = self._prepare_sigma_values(
            sigma,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        sigma_feat = self.sigma_condition_proj(sigma_values.view(batch_size, 1))
        return sigma_feat.to(dtype=dtype)

    def _route_label_ids(self, metadata, batch_size, device):
        """
        从 metadata 中提取多标签 ID 列表。
        优先使用 label_ids（多标签列表），回退到 label_id（单标签）。
        返回的是标签索引列表的列表（每个 batch 元素可能对应多个标签）。
        """
        if not isinstance(metadata, dict):
            return [[0] for _ in range(batch_size)]

        # 优先使用多标签列表
        label_ids_list = metadata.get("label_ids")
        if label_ids_list is not None and len(label_ids_list) > 0:
            # label_ids_list 是列表，为每个 batch 元素复制相同的标签列表
            labels_per_sample = [
                [int(lid) for lid in label_ids_list]
                for _ in range(batch_size)
            ]
            return labels_per_sample

        # 回退到单标签
        label_id = metadata.get("label_id", 0)
        if isinstance(label_id, torch.Tensor):
            label_id = int(label_id.item())
        else:
            label_id = int(label_id) if label_id is not None else 0
        return [[label_id] for _ in range(batch_size)]

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
        """
        多标签专家路由：所有标签对应的专家都会获得 bias 加成。
        """
        labels_per_sample = self._route_label_ids(metadata, batch_size, device)
        route_logits = self.expert_router(cond_feat)
        label_prior = torch.zeros_like(route_logits)

        # 为每个样本的每个标签对应的位置加上 bias
        for sample_idx, label_list in enumerate(labels_per_sample):
            for label_id in label_list:
                label_id = int(label_id)
                if 0 <= label_id < self.num_phenomena:
                    label_prior[sample_idx, label_id] += self.router_label_bias

        route_logits = route_logits + label_prior

        top_k = min(self.moe_top_k, self.num_phenomena)
        topk_logits, topk_indices = torch.topk(route_logits, k=top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits / max(self.router_temperature, 1e-6), dim=-1)
        dominant_expert = topk_indices[:, 0]

        # 返回主标签（每个样本的第一个标签）用于后续记录
        primary_label_ids = torch.tensor(
            [labels[0] if labels else 0 for labels in labels_per_sample],
            device=device, dtype=torch.long
        )
        return route_logits, topk_indices, topk_weights, dominant_expert, primary_label_ids
    
    def forward(self, v_original, state_for_physics, sigma=None, metadata=None):
        """
        Forward pass: Apply physics-informed corrections via MoE expert routing.

        每个专家拥有独立的物理编码器和解码器，允许不同物理现象
        学习各自特有的物理场特征表示。

        Args:
            v_original: 原始模型预测的速度场 [B, C, T, H, W]
            state_for_physics: 用于物理建模的状态 [B, C, T, H, W]
            metadata: 条件输入字典（label_id/label_name/n/q）

        Returns:
            v_corrected: 物理校正后的速度场 [B, C, T, H, W]
        """
        B = v_original.shape[0]
        sigma_gate = self._sigma_gate(sigma, v_original)
        sigma_embed = self._sigma_embedding(
            sigma,
            batch_size=B,
            device=v_original.device,
            dtype=v_original.dtype,
        )
        sigma_map = sigma_embed.view(B, self.hidden_dim, 1, 1, 1)
        gate_scale = self.scale * sigma_gate

        # 使用共享编码器获取基础物理特征（用于 fallback 和条件编码）
        physics_feat_shared = self.physics_encoder_shared(state_for_physics)  # [B, hidden_dim, T, H, W]
        physics_feat_shared = physics_feat_shared + sigma_map

        if metadata is None or not self.use_moe:
            raw_correction = self.physics_correction_fallback(physics_feat_shared)
            raw_correction = self._safe_correction(raw_correction)
            gated_correction = gate_scale * raw_correction
            v_corrected = v_original + gated_correction
            topk_shape = (B, self.moe_top_k)
            branch_feat_shape = (B, self.moe_top_k, *physics_feat_shared.shape[1:])
            branch_latent_shape = (B, self.moe_top_k, *v_original.shape[1:])
            zero_indices = torch.zeros(topk_shape, dtype=torch.long, device=v_original.device)
            zero_weights = torch.zeros(topk_shape, dtype=physics_feat_shared.dtype, device=v_original.device)
            zero_branch_outputs = torch.zeros(branch_feat_shape, dtype=physics_feat_shared.dtype, device=v_original.device)
            zero_branch_corrections = torch.zeros(
                branch_latent_shape, dtype=v_original.dtype, device=v_original.device
            )
            self._cache = {
                "using_shared_fallback": True,  # 标记：使用共享编码器/解码器回退模式
                "fallback_reason": "no_metadata" if metadata is None else "moe_disabled",
                "physics_feat": physics_feat_shared.detach(),
                "raw_correction": raw_correction.detach(),
                "expert_output": torch.zeros_like(physics_feat_shared).detach(),
                "cond_feat": torch.zeros(B, self.hidden_dim, device=v_original.device).detach(),
                "sigma_embedding": sigma_embed.detach(),
                "expert_output_live": torch.zeros_like(physics_feat_shared),
                "cond_feat_live": torch.zeros(B, self.hidden_dim, device=v_original.device, dtype=physics_feat_shared.dtype),
                "sigma_embedding_live": sigma_embed,
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
                "sigma_gate": sigma_gate.detach(),
                "effective_scale": gate_scale.detach().view(B),
                "raw_correction_norm": raw_correction.detach().float().reshape(B, -1).norm(dim=1),
                "gated_correction_norm": gated_correction.detach().float().reshape(B, -1).norm(dim=1),
            }
            return v_corrected

        cond_feat = self._encode_condition(metadata, B, v_original.device, physics_feat_shared.dtype)
        cond_feat = cond_feat + sigma_embed.to(dtype=cond_feat.dtype)

        route_logits, topk_indices, topk_weights, dominant_expert, label_ids = self._route_experts(
            cond_feat=cond_feat,
            metadata=metadata,
            batch_size=B,
            device=v_original.device,
        )

        # 使用共享特征形状来初始化输出张量
        expert_output = torch.zeros_like(physics_feat_shared)
        branch_physics_features = torch.zeros(
            B,
            topk_indices.shape[1],
            *physics_feat_shared.shape[1:],
            device=physics_feat_shared.device,
            dtype=physics_feat_shared.dtype,
        )
        branch_expert_outputs = torch.zeros(
            B,
            topk_indices.shape[1],
            *physics_feat_shared.shape[1:],
            device=physics_feat_shared.device,
            dtype=physics_feat_shared.dtype,
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

        # 为每个样本处理其 top-k 专家
        for i in range(B):
            routed_feat = torch.zeros_like(physics_feat_shared[i:i + 1])
            routed_correction = torch.zeros_like(v_original[i:i + 1])

            for j in range(topk_indices.shape[1]):
                expert_idx = int(topk_indices[i, j].item())
                expert_weight = topk_weights[i, j].to(dtype=physics_feat_shared.dtype)

                # 每个专家使用自己的编码器提取物理特征
                expert_physics_feat = self.expert_physics_encoders[expert_idx](state_for_physics[i:i + 1])
                expert_physics_feat = expert_physics_feat + sigma_map[i:i + 1].to(dtype=expert_physics_feat.dtype)
                branch_physics_features[i:i + 1, j] = expert_physics_feat

                # 将条件特征与专家编码的物理特征拼接
                cond_map = cond_feat[i:i + 1].view(1, -1, 1, 1, 1).expand(-1, -1, *expert_physics_feat.shape[2:])
                expert_input = torch.cat([expert_physics_feat, cond_map], dim=1)

                # 专家处理
                branch_feat = self.phenomenon_experts[expert_idx](expert_input)
                branch_expert_outputs[i:i + 1, j] = branch_feat

                # 每个专家使用自己的解码器生成校正
                branch_merged = torch.cat([expert_physics_feat, branch_feat], dim=1)
                branch_raw_correction = self.expert_physics_decoders[expert_idx](branch_merged)
                # 安全处理每个分支的校正
                branch_raw_correction = self._safe_correction(branch_raw_correction)
                branch_raw_corrections[i:i + 1, j] = branch_raw_correction
                branch_v_corrected[i:i + 1, j] = (
                    v_original[i:i + 1] + gate_scale[i:i + 1] * branch_raw_correction
                )

                # 加权累加专家输出
                routed_feat = routed_feat + expert_weight * branch_feat
                routed_correction = routed_correction + expert_weight * branch_raw_correction

            # 共享专家使用共享编码器的特征
            cond_map_shared = cond_feat[i:i + 1].view(1, -1, 1, 1, 1).expand(-1, -1, *physics_feat_shared.shape[2:])
            shared_expert_input = torch.cat([physics_feat_shared[i:i + 1], cond_map_shared], dim=1)
            shared_feat = self.shared_expert(shared_expert_input)
            shared_merged = torch.cat([physics_feat_shared[i:i + 1], shared_feat], dim=1)
            shared_correction = self.shared_merge_head(shared_merged)

            # 混合路由专家和共享专家的结果
            mixed_feat = (1.0 - self.shared_expert_weight) * routed_feat + self.shared_expert_weight * shared_feat
            expert_output[i:i + 1] = mixed_feat

        # 最终校正：混合路由专家和共享专家的校正
        # 使用加权平均的方式融合各专家的校正
        final_correction = torch.zeros_like(v_original)
        for i in range(B):
            routed_correction = torch.zeros_like(v_original[i:i + 1])
            for j in range(topk_indices.shape[1]):
                expert_weight = topk_weights[i, j].to(dtype=v_original.dtype)
                routed_correction = routed_correction + expert_weight * branch_raw_corrections[i:i + 1, j]

            # 共享专家的校正
            cond_map_shared = cond_feat[i:i + 1].view(1, -1, 1, 1, 1).expand(-1, -1, *physics_feat_shared.shape[2:])
            shared_expert_input = torch.cat([physics_feat_shared[i:i + 1], cond_map_shared], dim=1)
            shared_feat = self.shared_expert(shared_expert_input)
            shared_merged = torch.cat([physics_feat_shared[i:i + 1], shared_feat], dim=1)
            shared_correction = self.shared_merge_head(shared_merged)

            final_correction[i:i + 1] = (
                (1.0 - self.shared_expert_weight) * routed_correction +
                self.shared_expert_weight * shared_correction
            )

        raw_correction = final_correction
        # 安全处理：防止 NaN/Inf 和数值爆炸
        raw_correction = self._safe_correction(raw_correction)
        gated_correction = gate_scale * raw_correction
        v_corrected = v_original + gated_correction

        with torch.no_grad():
            hist = torch.zeros(self.num_phenomena, device=v_original.device, dtype=torch.float32)
            hist.scatter_add_(0, topk_indices.reshape(-1), topk_weights.reshape(-1).float())
            hist = hist / max(float(B), 1.0)
            self.expert_usage_ema.mul_(0.99).add_(0.01 * hist)
        branch_physics_features_live = branch_physics_features
        branch_expert_outputs_live = branch_expert_outputs
        branch_raw_corrections_live = branch_raw_corrections
        if self.lightweight_cache:
            branch_physics_features_live = branch_physics_features.detach()
            branch_expert_outputs_live = branch_expert_outputs.detach()
            branch_raw_corrections_live = branch_raw_corrections.detach()

        # 缓存中间结果（detach 避免影响梯度计算图）
        self._cache = {
            "using_shared_fallback": False,  # 标记：使用每个专家独立的编码器/解码器
            "fallback_reason": None,
            "physics_feat": physics_feat_shared.detach(),    # [B, hidden_dim, T, H, W] 共享编码器特征
            "raw_correction": raw_correction.detach(), # [B, C, T, H, W] (未乘 scale)
            "expert_output": expert_output.detach(),
            "cond_feat": cond_feat.detach(),
            "sigma_embedding": sigma_embed.detach(),
            "expert_output_live": expert_output,
            "cond_feat_live": cond_feat,
            "sigma_embedding_live": sigma_embed,
            "label_ids": label_ids.detach(),
            "active_expert_indices": topk_indices.detach(),
            "active_expert_weights": topk_weights.detach(),
            "branch_physics_features": branch_physics_features.detach(),  # 每个专家编码器的特征
            "branch_physics_features_live": branch_physics_features_live,
            "branch_expert_outputs": branch_expert_outputs.detach(),
            "branch_expert_outputs_live": branch_expert_outputs_live,
            "branch_raw_corrections": branch_raw_corrections.detach(),
            "branch_raw_corrections_live": branch_raw_corrections_live,
            "branch_v_corrected": branch_v_corrected.detach(),
            "branch_v_corrected_live": branch_v_corrected,
            "dominant_expert": dominant_expert.detach(),
            "route_logits": route_logits.detach(),
            "sigma_gate": sigma_gate.detach(),
            "effective_scale": gate_scale.detach().view(B),
            "raw_correction_norm": raw_correction.detach().float().reshape(B, -1).norm(dim=1),
            "gated_correction_norm": gated_correction.detach().float().reshape(B, -1).norm(dim=1),
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

    def is_using_shared_fallback(self):
        """
        检查上一次 forward 是否使用了共享编码器/解码器回退模式。

        Returns:
            bool: True 表示使用共享模式（metadata 缺失或 MoE 被禁用）
                  False 表示使用每个专家独立的编码器/解码器
        """
        return self._cache.get("using_shared_fallback", True)

    def get_fallback_reason(self):
        """
        获取回退到共享模式的原因。

        Returns:
            str or None: "no_metadata" - metadata 为 None
                        "moe_disabled" - use_moe 被设为 False
                        None - 未使用回退模式（正常 MoE 路径）
        """
        return self._cache.get("fallback_reason", None)

    def get_mode_info(self):
        """
        获取当前编码器/解码器模式的详细信息。

        Returns:
            dict: 包含模式状态的字典
        """
        return {
            "using_shared_fallback": self._cache.get("using_shared_fallback", True),
            "fallback_reason": self._cache.get("fallback_reason", None),
            "use_moe_setting": self.use_moe,
            "num_experts": self.num_phenomena,
            "active_expert_indices": self._cache.get("active_expert_indices", None),
            "active_expert_weights": self._cache.get("active_expert_weights", None),
        }

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
