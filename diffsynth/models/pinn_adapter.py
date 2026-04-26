"""
Physics-Informed Adapter Module
物理约束适配器 - 作为插件连接到原始模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pinn_contracts import (
    EXPERT_FIELD_RECIPES,
    PHENOMENON_LABELS,
    PHYSICS_ATTRIBUTE_CONTRACT,
    PHYSICS_ATTR_DIM,
    attribute_mask,
    split_attribute_bank,
)

ONLY_U_RECOVERY_PHASES = ("core", "alpha", "T", "j", "D", "psi")
ONLY_U_RECOVERY_FIELDS = {
    "core": ("d",),
    "alpha": ("d", "alpha"),
    "T": ("d", "alpha", "T"),
    "j": ("d", "alpha", "T", "j"),
    "D": ("d", "alpha", "T", "j", "D"),
    "psi": ("d", "alpha", "T", "j", "D", "psi"),
}

CORE_ABLATION_MODES = (
    "full",
    "generic_latent_correction",
    "wo_explicit_physical_interface",
    "wo_pde_residuals",
    "wo_phenomenon_specific_operators",
    "wo_learned_expert_routing",
    "wo_physics_to_flow_injection",
)


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
        physics_attr_dim=None,
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
        use_adaptive_condition_injection=True,
        adaptive_conditioning_dim=None,
        adaptive_conditioning_strength=0.5,
        adaptive_conditioning_gate_floor=0.05,
        enable_rl_expert_optimization=True,
        rl_hidden_dim=None,
        rl_reward_decay=0.95,
        strict_physical_state_contract=False,
        excluded_expert_names=None,
        core_ablation_mode="full",
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.physics_attr_dim = int(PHYSICS_ATTR_DIM if physics_attr_dim is None else physics_attr_dim)
        self.num_phenomena = num_phenomena
        self.n_numeric_dim = n_numeric_dim
        self.q_input_dim = q_input_dim
        self.n_text_vocab_size = n_text_vocab_size
        self.shared_expert_weight = shared_expert_weight
        self.moe_top_k = max(0, min(int(moe_top_k), num_phenomena))
        self.router_temperature = 2.0  # Increased for smoother routing
        self.router_label_bias = 2.0
        self.use_moe = True
        self.label_only_mode = False
        self.core_ablation_mode = "full"
        self.interpret_attribute_bank_as_physical = True
        self.use_phenomenon_specific_operators = True
        self.physics_to_flow_injection_enabled = True
        self.force_label_only_routing = False
        self.set_core_ablation_mode(core_ablation_mode)
        if self.physics_attr_dim != PHYSICS_ATTR_DIM:
            raise ValueError(
                f"Explicit attribute-bank v2 requires physics_attr_dim={PHYSICS_ATTR_DIM}, "
                f"got {self.physics_attr_dim}."
            )
        if self.num_phenomena > len(PHENOMENON_LABELS):
            raise ValueError(
                f"PhysicsAdapter only defines {len(PHENOMENON_LABELS)} explicit experts, "
                f"got num_phenomena={self.num_phenomena}."
            )
        self.phenomenon_labels = list(PHENOMENON_LABELS[: self.num_phenomena])
        self.expert_field_recipes = {
            label: tuple(EXPERT_FIELD_RECIPES[label]) for label in self.phenomenon_labels
        }
        self.excluded_expert_names = []
        self.excluded_expert_indices = set()
        self.set_excluded_experts(excluded_expert_names)
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
        self.use_adaptive_condition_injection = False
        self.adaptive_conditioning_dim = (
            hidden_dim if adaptive_conditioning_dim is None else max(int(adaptive_conditioning_dim), 8)
        )
        self.adaptive_conditioning_strength = max(float(adaptive_conditioning_strength), 0.0)
        self.adaptive_conditioning_gate_floor = min(
            max(float(adaptive_conditioning_gate_floor), 0.0), 1.0
        )
        self.enable_rl_expert_optimization = False
        self.rl_hidden_dim = hidden_dim if rl_hidden_dim is None else max(int(rl_hidden_dim), 8)
        self.rl_reward_decay = min(max(float(rl_reward_decay), 0.0), 0.999)
        self.strict_physical_state_contract = bool(strict_physical_state_contract)
        self.export_expert_attention = False
        self.expert_attention_apply_router_weight = True

        # 唯一共享 encoder：所有物理语义都先进入同一套共享特征空间。
        self.physics_encoder_shared = nn.Sequential(
            ResBlock3D(latent_dim, hidden_dim),
            ResBlock3D(hidden_dim, hidden_dim),
            ResBlock3D(hidden_dim, hidden_dim),
            ResBlock3D(hidden_dim, hidden_dim),
        )

        # 共享显式属性库头：定义唯一的 32 维物理属性合同。
        self.shared_attribute_head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, self.physics_attr_dim, kernel_size=1),
        )
        observable_hidden_dim = max(hidden_dim // 2, 8)
        self.u_head = nn.Sequential(
            nn.Conv3d(4, observable_hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(observable_hidden_dim, 2, kernel_size=1),
        )
        self.d_head = nn.Sequential(
            nn.Conv3d(4, observable_hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(observable_hidden_dim, 4, kernel_size=1),
        )
        self.prho_constructor = nn.Sequential(
            nn.Conv3d(hidden_dim + 5, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, observable_hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(observable_hidden_dim, 4, kernel_size=1),
        )
        self.alpha_head = self._make_field_head(hidden_dim + 5, 1, observable_hidden_dim)
        self.T_head = self._make_field_head(hidden_dim + 4, 1, observable_hidden_dim)
        self.j_head = self._make_field_head(hidden_dim + 14, 2, observable_hidden_dim)
        self.D_head = self._make_field_head(hidden_dim + 14, 1, observable_hidden_dim)
        self.psi_head = self._make_field_head(hidden_dim + 1, 1, observable_hidden_dim)
        self.ablation_preset = "legacy_direct_bank"
        self.secondary_field_strategy = "legacy_direct_bank"
        self.active_field_set = "legacy"
        self.field_enable_schedule = "none"
        self.field_recovery_phase = "core"
        self.only_u_rho_floor = 1e-4
        self.only_u_sigma_mu = 1.0
        self.only_u_sigma_lambda = 1.0

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

        # Operator MoE：专家只在显式共享字段上输出 residual updates。
        operator_input_dim = hidden_dim * 2 + self.physics_attr_dim
        self.operator_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(operator_input_dim, hidden_dim, kernel_size=1),
                nn.SiLU(),
                ResBlock3D(hidden_dim, hidden_dim),
                nn.Conv3d(hidden_dim, self.physics_attr_dim, kernel_size=1),
            )
            for _ in range(num_phenomena)
        ])

        # 单一共享 decoder：最终校正只能由共享属性库和原始 v_pred 共同生成。
        self.shared_decoder = nn.Sequential(
            nn.Conv3d(hidden_dim * 2 + self.physics_attr_dim + latent_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            ResBlock3D(hidden_dim, hidden_dim),
            nn.Conv3d(hidden_dim, latent_dim, kernel_size=1),
        )

        # 旧私有物理路径在 v1 中彻底退役，不再注册可训练参数。
        self.expert_physics_encoders = None
        self.expert_state_heads = None
        self.expert_velocity_heads = None
        self.expert_physics_decoders = None
        self.physics_correction_fallback = None
        self.adaptive_condition_expert_embedding = None
        self.adaptive_condition_state_proj = None
        self.adaptive_condition_modulator = None
        self.shared_adaptive_condition_modulator = None
        self.rl_expert_embedding = None
        self.rl_state_proj = None
        self.rl_policy_head = None
        self.phenomenon_experts = None
        self.shared_expert = None
        self.shared_merge_head = None
        self.condition_reconstructor = None

        self.register_buffer(
            "expert_usage_ema",
            torch.full((num_phenomena,), 1.0 / max(num_phenomena, 1))
        )
        self.register_buffer(
            "rl_reward_ema",
            torch.zeros(num_phenomena, dtype=torch.float32)
        )
        expert_field_masks = torch.stack(
            [
                attribute_mask(
                    self.expert_field_recipes[label],
                    dtype=torch.float32,
                )
                for label in self.phenomenon_labels
            ],
            dim=0,
        )
        self.register_buffer("expert_field_masks", expert_field_masks)

        # 为兼容训练/推理统计逻辑保留 scale 字段，但它不再参与最终输出抑制。
        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)
        
        # 缓存最近一次 forward 的中间结果，供外部可视化使用
        self._cache = {}

    @staticmethod
    def _make_field_head(input_dim, output_dim, hidden_dim):
        return nn.Sequential(
            nn.Conv3d(input_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, output_dim, kernel_size=1),
        )

    def _init_expert_weights_from_shared(self):
        """
        v1 共享槽位架构不再维护私有 expert encoder/decoder 权重镜像。
        """
        return None

    def _check_nan(self, tensor, name="tensor"):
        """检查张量是否包含 NaN 或 Inf，用于调试"""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return True
        return False

    @staticmethod
    def _tensor_debug_summary(tensor):
        if not isinstance(tensor, torch.Tensor):
            return "value=<non-tensor>"
        detached = tensor.detach().float()
        finite_mask = torch.isfinite(detached)
        finite_count = int(finite_mask.sum().item())
        nan_count = int(torch.isnan(detached).sum().item())
        inf_count = int(torch.isinf(detached).sum().item())
        if finite_count > 0:
            finite_values = detached[finite_mask]
            min_value = float(finite_values.min().item())
            max_value = float(finite_values.max().item())
            mean_value = float(finite_values.mean().item())
        else:
            min_value = float("nan")
            max_value = float("nan")
            mean_value = float("nan")
        return (
            f"shape={tuple(detached.shape)} "
            f"min={min_value:.6f} "
            f"max={max_value:.6f} "
            f"mean={mean_value:.6f} "
            f"finite_count={finite_count} "
            f"nan_count={nan_count} "
            f"inf_count={inf_count}"
        )

    @staticmethod
    def _format_debug_value(value):
        if isinstance(value, torch.Tensor):
            detached = value.detach().cpu()
            if detached.numel() == 1:
                return f"{float(detached.reshape(-1)[0].item()):.6f}"
            if detached.ndim <= 1 and detached.numel() <= 8:
                return str([float(x) for x in detached.reshape(-1).tolist()])
            return PhysicsAdapter._tensor_debug_summary(detached)
        if isinstance(value, (list, tuple)):
            if len(value) <= 8:
                return str(list(value))
            return str(list(value[:8]) + ["..."])
        if isinstance(value, dict):
            keys = list(value.keys())
            if len(keys) <= 8:
                return str({key: value[key] for key in keys})
            return str({key: value[key] for key in keys[:8]}) + " ..."
        return str(value)

    def set_debug_context(self, **kwargs):
        context = getattr(self, "_debug_context", {})
        if not isinstance(context, dict):
            context = {}
        for key, value in kwargs.items():
            if value is None:
                context.pop(key, None)
            else:
                context[key] = value
        self._debug_context = context

    def clear_debug_context(self):
        self._debug_context = {}

    def _debug_context_suffix(self):
        context = getattr(self, "_debug_context", None)
        if not isinstance(context, dict) or len(context) == 0:
            return ""
        ordered_keys = (
            "step",
            "training_stage",
            "phase",
            "phenomenon",
            "timestep_id",
            "sigma",
            "metadata",
        )
        parts = []
        seen = set()
        for key in ordered_keys:
            if key in context:
                parts.append(f"{key}={self._format_debug_value(context[key])}")
                seen.add(key)
        for key in sorted(context.keys()):
            if key in seen:
                continue
            parts.append(f"{key}={self._format_debug_value(context[key])}")
        return " context: " + " ".join(parts)

    def _raise_invalid_tensor(self, name, tensor):
        raise FloatingPointError(
            f"Invalid values detected in `{name}`. "
            f"{self._tensor_debug_summary(tensor)}"
            f"{self._debug_context_suffix()}"
        )

    def _require_tensor(self, tensor, name, expected_channels=None):
        if tensor is None:
            raise RuntimeError(f"Missing required tensor `{name}` in PhysicsAdapter.")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected `{name}` to be a torch.Tensor, got {type(tensor)!r}.")
        if expected_channels is not None:
            if tensor.ndim < 2:
                raise RuntimeError(
                    f"Tensor `{name}` must have at least 2 dims, got shape {tuple(tensor.shape)}."
                )
            if tensor.shape[1] != expected_channels:
                raise RuntimeError(
                    f"Tensor `{name}` channel mismatch: expected {expected_channels}, got {tensor.shape[1]}."
                )
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            self._raise_invalid_tensor(name, tensor)
        return tensor

    def _validate_named_parameters_finite(self, prefixes=None):
        for name, value in self.named_parameters():
            if prefixes is not None and not any(name.startswith(prefix) for prefix in prefixes):
                continue
            if not isinstance(value, torch.Tensor):
                continue
            if torch.isnan(value).any() or torch.isinf(value).any():
                raise FloatingPointError(
                    f"Invalid values detected in parameter `{name}`. "
                    f"{self._tensor_debug_summary(value)}"
                    f"{self._debug_context_suffix()}"
                )

    def validate_monitored_parameters_finite(self, prefixes=None):
        self._validate_named_parameters_finite(prefixes=prefixes)

    def _safe_physical_tensor(self, tensor, name, max_abs=10.0):
        tensor = self._require_tensor(tensor, name)
        return torch.clamp(tensor, min=-max_abs, max=max_abs)

    def _safe_correction(self, correction, max_norm=10.0):
        """
        对校正值进行安全处理，防止数值爆炸。
        - 裁剪过大的值
        - 将 NaN/Inf 替换为 0
        """
        if torch.isnan(correction).any() or torch.isinf(correction).any():
            if self.strict_physical_state_contract:
                raise FloatingPointError("Invalid values detected in correction tensor.")
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

    def _prepare_motion_values(self, metadata, batch_size, device, dtype):
        if not isinstance(metadata, dict):
            if self.strict_physical_state_contract:
                raise RuntimeError("PhysicsAdapter requires metadata with motion_mask in strict mode.")
            return torch.zeros(batch_size, device=device, dtype=dtype)
        motion_mask = metadata.get("motion_mask")
        if not isinstance(motion_mask, torch.Tensor):
            if self.strict_physical_state_contract:
                raise RuntimeError("PhysicsAdapter requires `motion_mask` tensor in metadata under strict mode.")
            return torch.zeros(batch_size, device=device, dtype=dtype)
        motion_mask = motion_mask.to(device=device, dtype=dtype)
        if motion_mask.ndim <= 1:
            motion_values = motion_mask.reshape(-1)
        else:
            motion_values = motion_mask.reshape(motion_mask.shape[0], -1).mean(dim=-1)
        if motion_values.shape[0] == 1 and batch_size > 1:
            motion_values = motion_values.repeat(batch_size)
        if motion_values.shape[0] != batch_size:
            motion_values = motion_values[:1].repeat(batch_size)
        return torch.clamp(motion_values, 0.0, 1.0)

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

        try:
            proj_dtype = next(self.sigma_condition_proj.parameters()).dtype
        except StopIteration:
            proj_dtype = dtype
        sigma_values = self._prepare_sigma_values(
            sigma,
            batch_size=batch_size,
            device=device,
            dtype=proj_dtype,
        )
        sigma_values = self._require_tensor(sigma_values, "sigma_values")
        sigma_input = sigma_values.view(batch_size, 1)
        sigma_input = self._require_tensor(sigma_input, "sigma_condition_proj_input")
        sigma_feat = self.sigma_condition_proj(sigma_input)
        sigma_feat = self._require_tensor(sigma_feat, "sigma_condition_proj_output_raw")
        sigma_feat = sigma_feat.to(dtype=dtype)
        return self._require_tensor(sigma_feat, "sigma_condition_proj_output")

    def _build_adaptive_condition_state(self, cond_feat, physics_feat_shared, sigma, metadata):
        batch_size = cond_feat.shape[0]
        device = cond_feat.device
        dtype = cond_feat.dtype
        if not self.use_adaptive_condition_injection:
            return torch.zeros(
                batch_size, self.adaptive_conditioning_dim, device=device, dtype=dtype
            )

        pooled_physics = physics_feat_shared.mean(dim=(2, 3, 4)).to(dtype=dtype)
        sigma_values = self._prepare_sigma_values(
            sigma,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        if sigma_values is None:
            sigma_values = torch.zeros(batch_size, device=device, dtype=dtype)
        motion_values = self._prepare_motion_values(
            metadata,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        usage_context = self.expert_usage_ema.to(device=device, dtype=dtype).unsqueeze(0)
        usage_context = usage_context.expand(batch_size, -1)
        state_input = torch.cat(
            [
                cond_feat,
                pooled_physics,
                usage_context,
                sigma_values.view(batch_size, 1),
                motion_values.view(batch_size, 1),
            ],
            dim=-1,
        )
        return self.adaptive_condition_state_proj(state_input)

    def _unpack_adaptive_condition_params(self, raw_params):
        hidden_dim = self.hidden_dim
        strength = self.adaptive_conditioning_strength
        gate_floor = self.adaptive_conditioning_gate_floor
        encoder_scale = 1.0 + strength * torch.tanh(raw_params[..., :hidden_dim])
        encoder_bias = strength * torch.tanh(raw_params[..., hidden_dim:hidden_dim * 2])
        feature_scale = 1.0 + strength * torch.tanh(raw_params[..., hidden_dim * 2:hidden_dim * 3])
        feature_bias = strength * torch.tanh(raw_params[..., hidden_dim * 3:hidden_dim * 4])
        condition_gate = gate_floor + (1.0 - gate_floor) * torch.sigmoid(raw_params[..., hidden_dim * 4])
        correction_gate = gate_floor + (1.0 - gate_floor) * torch.sigmoid(raw_params[..., hidden_dim * 4 + 1])
        return {
            "encoder_scale": encoder_scale,
            "encoder_bias": encoder_bias,
            "feature_scale": feature_scale,
            "feature_bias": feature_bias,
            "condition_gate": condition_gate,
            "correction_gate": correction_gate,
        }

    def _compute_adaptive_condition_params(self, adaptive_state, topk_indices, topk_weights, route_logits):
        batch_size, top_k = topk_indices.shape
        device = adaptive_state.device
        dtype = adaptive_state.dtype
        if (not self.use_adaptive_condition_injection) or topk_indices.numel() == 0:
            ones = torch.ones(batch_size, top_k, self.hidden_dim, device=device, dtype=dtype)
            zeros = torch.zeros_like(ones)
            gates = torch.ones(batch_size, top_k, device=device, dtype=dtype)
            return {
                "encoder_scale": ones,
                "encoder_bias": zeros,
                "feature_scale": ones,
                "feature_bias": zeros,
                "condition_gate": gates,
                "correction_gate": gates,
            }

        expert_embed = self.adaptive_condition_expert_embedding(topk_indices).to(dtype=dtype)
        gathered_route_logits = route_logits.gather(1, topk_indices).to(dtype=dtype)
        modulator_input = torch.cat(
            [
                adaptive_state.unsqueeze(1).expand(-1, top_k, -1),
                expert_embed,
                topk_weights.unsqueeze(-1),
                gathered_route_logits.unsqueeze(-1),
            ],
            dim=-1,
        )
        raw_params = self.adaptive_condition_modulator(modulator_input)
        return self._unpack_adaptive_condition_params(raw_params)

    def _compute_shared_adaptive_condition_params(self, adaptive_state):
        batch_size = adaptive_state.shape[0]
        device = adaptive_state.device
        dtype = adaptive_state.dtype
        if not self.use_adaptive_condition_injection:
            ones = torch.ones(batch_size, self.hidden_dim, device=device, dtype=dtype)
            zeros = torch.zeros_like(ones)
            gates = torch.ones(batch_size, device=device, dtype=dtype)
            return {
                "encoder_scale": ones,
                "encoder_bias": zeros,
                "feature_scale": ones,
                "feature_bias": zeros,
                "condition_gate": gates,
                "correction_gate": gates,
            }

        raw_params = self.shared_adaptive_condition_modulator(adaptive_state)
        return self._unpack_adaptive_condition_params(raw_params)

    @staticmethod
    def _expand_modulation_param(param, ref_tensor):
        expanded = param.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        while expanded.ndim < ref_tensor.ndim:
            expanded = expanded.unsqueeze(-1)
        return expanded

    def _apply_feature_modulation(self, feat, scale, bias):
        if not self.use_adaptive_condition_injection:
            return feat
        scale = self._expand_modulation_param(scale, feat)
        bias = self._expand_modulation_param(bias, feat)
        return feat * scale + bias

    def _route_label_ids(self, metadata, batch_size, device):
        """
        从 metadata 中提取多标签 ID 列表。
        优先使用 label_ids（多标签列表），回退到 label_id（单标签）。
        返回的是标签索引列表的列表（每个 batch 元素可能对应多个标签）。
        """
        if not isinstance(metadata, dict):
            if self.strict_physical_state_contract:
                raise RuntimeError("PhysicsAdapter requires metadata dict for expert routing in strict mode.")
            return [[0] for _ in range(batch_size)]

        # 优先使用多标签列表
        label_ids_list = metadata.get("label_ids")
        if label_ids_list is not None and len(label_ids_list) > 0:
            # label_ids_list 是列表，为每个 batch 元素复制相同的标签列表
            filtered_label_ids = self._filter_excluded_label_ids(label_ids_list)
            labels_per_sample = [
                list(filtered_label_ids)
                for _ in range(batch_size)
            ]
            return labels_per_sample

        # 回退到单标签
        label_id = metadata.get("label_id", 0)
        if self.strict_physical_state_contract and "label_id" not in metadata:
            raise RuntimeError("PhysicsAdapter requires `label_id` or `label_ids` for strict routing.")
        if isinstance(label_id, torch.Tensor):
            label_id = int(label_id.item())
        else:
            label_id = int(label_id) if label_id is not None else 0
        filtered_label_ids = self._filter_excluded_label_ids([label_id])
        return [list(filtered_label_ids) for _ in range(batch_size)]

    def _fit_2d(self, tensor, target_dim, batch_size, device, dtype):
        if tensor is None:
            if self.strict_physical_state_contract:
                raise RuntimeError(
                    f"PhysicsAdapter expected metadata tensor with dim {target_dim}, but got None."
                )
            return torch.zeros(batch_size, target_dim, device=device, dtype=dtype)
        tensor = tensor.to(device=device, dtype=dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.repeat(batch_size, 1)
        if tensor.shape[0] != batch_size:
            if self.strict_physical_state_contract:
                raise RuntimeError(
                    f"Metadata batch mismatch: expected {batch_size}, got {tensor.shape[0]}."
                )
            tensor = tensor[:1].repeat(batch_size, 1)
        feat_dim = tensor.shape[1]
        if feat_dim > target_dim:
            if self.strict_physical_state_contract:
                raise RuntimeError(
                    f"Metadata feature dim mismatch: expected {target_dim}, got {feat_dim}."
                )
            return tensor[:, :target_dim]
        if feat_dim < target_dim:
            if self.strict_physical_state_contract:
                raise RuntimeError(
                    f"Metadata feature dim mismatch: expected {target_dim}, got {feat_dim}."
                )
            pad = torch.zeros(batch_size, target_dim - feat_dim, device=device, dtype=dtype)
            return torch.cat([tensor, pad], dim=1)
        return tensor

    def _encode_condition(self, metadata, batch_size, device, dtype):
        if not isinstance(metadata, dict):
            if self.strict_physical_state_contract:
                raise RuntimeError("PhysicsAdapter requires metadata dict in strict mode.")
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
            if self.strict_physical_state_contract:
                raise RuntimeError("PhysicsAdapter requires `n_text_ids` in metadata under strict mode.")
            n_text_ids = torch.zeros(batch_size, 3, device=device, dtype=torch.long)
        else:
            n_text_ids = n_text_ids.to(device=device, dtype=torch.long)
            if n_text_ids.ndim == 1:
                n_text_ids = n_text_ids.unsqueeze(0)
            if n_text_ids.shape[0] == 1 and batch_size > 1:
                n_text_ids = n_text_ids.repeat(batch_size, 1)
            if n_text_ids.shape[0] != batch_size:
                if self.strict_physical_state_contract:
                    raise RuntimeError(
                        f"`n_text_ids` batch mismatch: expected {batch_size}, got {n_text_ids.shape[0]}."
                    )
                n_text_ids = n_text_ids[:1].repeat(batch_size, 1)
        n_text_ids = torch.clamp(n_text_ids, min=0, max=self.n_text_vocab_size - 1)

        n_numeric_feat = self.n_numeric_proj(n_numeric)
        n_text_feat = self.n_text_embedding(n_text_ids).mean(dim=1)
        q_feat = self.q_proj(q_vector)
        cond_feat = self.condition_fuse(torch.cat([n_numeric_feat, n_text_feat, q_feat], dim=-1))
        return cond_feat.to(dtype=dtype)

    def _decode_physical_state(self, physics_feat, state_head, velocity_head, state_name, velocity_name):
        physics_feat = self._require_tensor(physics_feat, f"{state_name}_source")
        x_phys = state_head(physics_feat)
        v_phys = velocity_head(physics_feat)
        x_phys = self._safe_physical_tensor(x_phys, state_name)
        v_phys = self._safe_physical_tensor(v_phys, velocity_name)
        if x_phys.shape != v_phys.shape:
            raise RuntimeError(
                f"Physical state shape mismatch between `{state_name}`={tuple(x_phys.shape)} "
                f"and `{velocity_name}`={tuple(v_phys.shape)}."
            )
        return x_phys, v_phys

    def _decode_attribute_bank(self, physics_feat, attribute_name):
        physics_feat = self._require_tensor(physics_feat, f"{attribute_name}_source")
        attribute_bank = self.shared_attribute_head(physics_feat)
        attribute_bank = self._safe_physical_tensor(attribute_bank, attribute_name)
        if attribute_bank.shape[1] != self.physics_attr_dim:
            raise RuntimeError(
                f"Attribute bank channel mismatch: expected {self.physics_attr_dim}, "
                f"got {attribute_bank.shape[1]}."
            )
        return attribute_bank

    def phenomenon_name_from_index(self, expert_idx):
        expert_idx = int(expert_idx)
        if expert_idx < 0 or expert_idx >= len(self.phenomenon_labels):
            raise IndexError(
                f"Expert index {expert_idx} is out of range for {len(self.phenomenon_labels)} experts."
            )
        return self.phenomenon_labels[expert_idx]

    def _parse_excluded_expert_names(self, excluded_expert_names):
        if excluded_expert_names is None:
            return []
        if isinstance(excluded_expert_names, str):
            parts = [
                part.strip()
                for part in excluded_expert_names.replace(";", ",").split(",")
                if part.strip()
            ]
        else:
            parts = [str(part).strip() for part in excluded_expert_names if str(part).strip()]
        lookup = {label.casefold(): label for label in self.phenomenon_labels}
        unknown = [part for part in parts if part.casefold() not in lookup]
        if unknown:
            raise ValueError(
                f"Unknown excluded expert names: {unknown}. "
                f"Known experts: {self.phenomenon_labels}."
            )
        normalized = []
        seen = set()
        for part in parts:
            canonical = lookup[part.casefold()]
            if canonical not in seen:
                normalized.append(canonical)
                seen.add(canonical)
        return normalized

    def set_excluded_experts(self, excluded_expert_names=None):
        names = self._parse_excluded_expert_names(excluded_expert_names)
        self.excluded_expert_names = names
        self.excluded_expert_indices = {
            idx for idx, label in enumerate(self.phenomenon_labels) if label in names
        }

    def _default_allowed_label_id(self):
        for preferred in ("Fluid", "Rigid Body"):
            if preferred in self.phenomenon_labels:
                idx = self.phenomenon_labels.index(preferred)
                if idx not in self.excluded_expert_indices:
                    return idx
        for idx in range(self.num_phenomena):
            if idx not in self.excluded_expert_indices:
                return idx
        return 0

    def _filter_excluded_label_ids(self, label_ids):
        filtered = []
        seen = set()
        for label_id in label_ids:
            label_id = int(label_id)
            if label_id < 0 or label_id >= self.num_phenomena:
                continue
            if label_id in self.excluded_expert_indices:
                continue
            if label_id not in seen:
                filtered.append(label_id)
                seen.add(label_id)
        if not filtered:
            filtered = [self._default_allowed_label_id()]
        return filtered

    def _available_expert_count(self):
        return max(self.num_phenomena - len(self.excluded_expert_indices), 0)

    def expert_field_mask(self, expert_idx, device=None, dtype=None):
        mask = self.expert_field_masks[int(expert_idx)]
        if device is not None or dtype is not None:
            mask = mask.to(
                device=mask.device if device is None else device,
                dtype=mask.dtype if dtype is None else dtype,
            )
        return mask

    @staticmethod
    def _labels_to_padded_tensor(labels_per_sample, device):
        if not isinstance(labels_per_sample, (list, tuple)):
            return None
        batch_size = len(labels_per_sample)
        if batch_size == 0:
            return torch.empty(0, 0, device=device, dtype=torch.long)
        max_labels = max((len(sample_labels) for sample_labels in labels_per_sample), default=0)
        if max_labels <= 0:
            return torch.empty(batch_size, 0, device=device, dtype=torch.long)
        label_tensor = torch.full((batch_size, max_labels), -1, device=device, dtype=torch.long)
        for sample_idx, sample_labels in enumerate(labels_per_sample):
            if not sample_labels:
                continue
            label_tensor[sample_idx, : len(sample_labels)] = torch.tensor(
                [int(label_id) for label_id in sample_labels],
                device=device,
                dtype=torch.long,
            )
        return label_tensor

    @staticmethod
    def _offlabel_selected_experts(topk_indices, labels_per_sample):
        if not isinstance(topk_indices, torch.Tensor):
            return None
        offlabel = torch.full_like(topk_indices, -1)
        batch_size = int(topk_indices.shape[0]) if topk_indices.ndim > 0 else 0
        for sample_idx in range(batch_size):
            label_set = set(int(label_id) for label_id in labels_per_sample[sample_idx])
            for slot_idx, expert_idx in enumerate(topk_indices[sample_idx].tolist()):
                expert_idx = int(expert_idx)
                if expert_idx not in label_set:
                    offlabel[sample_idx, slot_idx] = expert_idx
        return offlabel

    def attribute_bank_to_field_dict(self, attribute_bank):
        return split_attribute_bank(attribute_bank)

    def set_field_policy(
        self,
        *,
        ablation_preset="legacy_direct_bank",
        secondary_field_strategy="legacy_direct_bank",
        active_field_set="legacy",
        field_enable_schedule="none",
        field_recovery_phase="core",
    ):
        self.ablation_preset = str(ablation_preset or "legacy_direct_bank")
        self.secondary_field_strategy = str(secondary_field_strategy or "legacy_direct_bank")
        self.active_field_set = str(active_field_set or "legacy")
        self.field_enable_schedule = str(field_enable_schedule or "none")
        phase = str(field_recovery_phase or "core")
        if phase not in ONLY_U_RECOVERY_PHASES:
            raise ValueError(
                f"Unsupported field_recovery_phase={phase!r}. "
                f"Expected one of {ONLY_U_RECOVERY_PHASES}."
            )
        self.field_recovery_phase = phase

    def _uses_only_u_policy(self):
        return str(self.ablation_preset).startswith("u_only_")

    @staticmethod
    def _finite_difference_height(field):
        if field.shape[-2] <= 1:
            return torch.zeros_like(field)
        diff = field[..., 1:, :] - field[..., :-1, :]
        return F.pad(diff, (0, 0, 0, 1))

    @staticmethod
    def _finite_difference_width(field):
        if field.shape[-1] <= 1:
            return torch.zeros_like(field)
        diff = field[..., :, 1:] - field[..., :, :-1]
        return F.pad(diff, (0, 1, 0, 0))

    def _velocity_kinematics(self, u):
        u = self._require_tensor(u, "physical_u", expected_channels=2)
        ux = u[:, 0:1]
        uy = u[:, 1:2]
        dux_dx = self._finite_difference_width(ux)
        dux_dy = self._finite_difference_height(ux)
        duy_dx = self._finite_difference_width(uy)
        duy_dy = self._finite_difference_height(uy)
        div_u = dux_dx + duy_dy
        curl_u = duy_dx - dux_dy
        eps_xy = 0.5 * (dux_dy + duy_dx)
        eps = torch.cat([dux_dx, eps_xy, eps_xy, duy_dy], dim=1)
        return {"div_u": div_u, "curl_u": curl_u, "eps": eps}

    def _normalize_pressure_density(self, p_raw, rho_raw):
        if p_raw.shape[1] != 1:
            p_raw = p_raw.mean(dim=1, keepdim=True)
        if rho_raw.shape[1] != 1:
            rho_raw = rho_raw.mean(dim=1, keepdim=True)
        p = p_raw - p_raw.mean(dim=(1, 2, 3, 4), keepdim=True)
        rho = F.softplus(rho_raw) + self.only_u_rho_floor
        return p, rho

    def _build_sigma(self, p, eps):
        p_scalar = p.mean(dim=1, keepdim=True)
        trace_eps = eps[:, 0:1] + eps[:, 3:4]
        sigma_xx = -p_scalar + 2.0 * self.only_u_sigma_mu * eps[:, 0:1] + self.only_u_sigma_lambda * trace_eps
        sigma_xy = 2.0 * self.only_u_sigma_mu * eps[:, 1:2]
        sigma_yx = 2.0 * self.only_u_sigma_mu * eps[:, 2:3]
        sigma_yy = -p_scalar + 2.0 * self.only_u_sigma_mu * eps[:, 3:4] + self.only_u_sigma_lambda * trace_eps
        return torch.cat([sigma_xx, sigma_xy, sigma_yx, sigma_yy], dim=1)

    @staticmethod
    def _repeat_channels(field, out_channels):
        if field.shape[1] == out_channels:
            return field
        repeat_factor = max((out_channels + max(field.shape[1], 1) - 1) // max(field.shape[1], 1), 1)
        return field.repeat(1, repeat_factor, 1, 1, 1)[:, :out_channels]

    @staticmethod
    def _scalarize_field(field):
        if field.shape[1] == 1:
            return field
        return field.mean(dim=1, keepdim=True)

    def _resolved_recovery_phase(self, field_recovery_phase=None):
        phase = str(field_recovery_phase or self.field_recovery_phase or "core")
        if phase not in ONLY_U_RECOVERY_PHASES:
            raise ValueError(
                f"Unsupported field_recovery_phase={phase!r}. "
                f"Expected one of {ONLY_U_RECOVERY_PHASES}."
            )
        return phase

    def _available_only_u_fields(self, field_recovery_phase=None):
        if not self._uses_only_u_policy():
            return set(PHYSICS_ATTRIBUTE_CONTRACT.keys())
        phase = self._resolved_recovery_phase(field_recovery_phase)
        available = {"u", "p", "rho", "eps", "sigma"}
        available.update(ONLY_U_RECOVERY_FIELDS.get(phase, ()))
        active_spec = {
            item.strip()
            for item in str(self.active_field_set).split(",")
            if item.strip() and item.strip().lower() not in {"legacy", "auto"}
        }
        if active_spec:
            available = {
                name for name in available
                if name in active_spec or name in {"u", "p", "rho", "eps", "sigma"}
            }
        return available

    def _inactive_only_u_fields(self, field_recovery_phase=None):
        if not self._uses_only_u_policy():
            return set()
        active = self._available_only_u_fields(field_recovery_phase=field_recovery_phase)
        return {name for name in PHYSICS_ATTRIBUTE_CONTRACT if name not in active}

    def _frame_time_grid(self, reference, metadata=None):
        batch_size = reference.shape[0]
        time_dim = reference.shape[2]
        device = reference.device
        dtype = reference.dtype
        if isinstance(metadata, dict):
            frame_time_grid = metadata.get("frame_time_grid")
            if isinstance(frame_time_grid, torch.Tensor) and frame_time_grid.numel() > 0:
                frame_time_grid = frame_time_grid.to(device=device, dtype=dtype)
                if frame_time_grid.ndim == 1:
                    frame_time_grid = frame_time_grid.unsqueeze(0)
                elif frame_time_grid.ndim > 2:
                    frame_time_grid = frame_time_grid.view(frame_time_grid.shape[0], -1)
                if frame_time_grid.shape[0] == 1 and batch_size > 1:
                    frame_time_grid = frame_time_grid.repeat(batch_size, 1)
                if frame_time_grid.shape[0] != batch_size:
                    frame_time_grid = frame_time_grid[:1].repeat(batch_size, 1)
                if frame_time_grid.shape[1] == time_dim:
                    return frame_time_grid
        if time_dim <= 1:
            return torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return torch.linspace(0.0, 1.0, steps=time_dim, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)

    def _integrate_displacement_from_velocity(self, u, metadata=None):
        u = self._require_tensor(u, "u_for_displacement", expected_channels=2)
        batch_size, _, time_dim, _, _ = u.shape
        d = torch.zeros_like(u)
        if time_dim <= 1:
            return d
        time_grid = self._frame_time_grid(u, metadata=metadata)
        time_deltas = time_grid[:, 1:] - time_grid[:, :-1]
        anchor_idx = time_dim // 2
        for time_idx in range(anchor_idx + 1, time_dim):
            dt = time_deltas[:, time_idx - 1].view(batch_size, 1, 1, 1)
            midpoint_velocity = 0.5 * (u[:, :, time_idx - 1] + u[:, :, time_idx])
            d[:, :, time_idx] = d[:, :, time_idx - 1] + dt * midpoint_velocity
        for time_idx in range(anchor_idx - 1, -1, -1):
            dt = time_deltas[:, time_idx].view(batch_size, 1, 1, 1)
            midpoint_velocity = 0.5 * (u[:, :, time_idx] + u[:, :, time_idx + 1])
            d[:, :, time_idx] = d[:, :, time_idx + 1] - dt * midpoint_velocity
        return d

    def _accumulate_damage(self, damage_init, damage_drive, metadata=None, eta=0.2):
        damage_init = self._scalarize_field(damage_init).sigmoid()
        damage_drive = self._scalarize_field(damage_drive)
        batch_size, _, time_dim, _, _ = damage_init.shape
        damage = torch.zeros_like(damage_init)
        damage[:, :, 0] = damage_init[:, :, 0]
        if time_dim <= 1:
            return damage.clamp(0.0, 1.0)
        time_grid = self._frame_time_grid(damage_init, metadata=metadata)
        time_deltas = time_grid[:, 1:] - time_grid[:, :-1]
        for time_idx in range(1, time_dim):
            dt = time_deltas[:, time_idx - 1].view(batch_size, 1, 1, 1)
            damage[:, :, time_idx] = torch.clamp(
                damage[:, :, time_idx - 1] + eta * dt * damage_drive[:, :, time_idx - 1],
                0.0,
                1.0,
            )
        return damage

    def _build_recovered_scalar_and_vector_fields(
        self,
        *,
        phase,
        physics_feat_shared,
        u,
        p_scalar,
        rho_scalar,
        eps,
        sigma,
        d_phys,
        metadata=None,
    ):
        batch_size, _, time_dim, height, width = physics_feat_shared.shape
        device = physics_feat_shared.device
        dtype = physics_feat_shared.dtype
        zeros_scalar = torch.zeros(batch_size, 1, time_dim, height, width, device=device, dtype=dtype)
        zeros_vector = torch.zeros(batch_size, 2, time_dim, height, width, device=device, dtype=dtype)

        alpha_scalar = zeros_scalar
        T_scalar = zeros_scalar
        j_phys = zeros_vector
        D_scalar = zeros_scalar
        psi_scalar = zeros_scalar

        if phase in {"alpha", "T", "j", "D", "psi"}:
            alpha_input = torch.cat([physics_feat_shared, u, rho_scalar, d_phys], dim=1)
            alpha_scalar = torch.sigmoid(self.alpha_head(alpha_input))
        if phase in {"T", "j", "D", "psi"}:
            t_input = torch.cat([physics_feat_shared, u, rho_scalar, alpha_scalar], dim=1)
            T_scalar = self.T_head(t_input)
        if phase in {"j", "D", "psi"}:
            j_input = torch.cat([physics_feat_shared, d_phys, u, p_scalar, rho_scalar, eps, sigma], dim=1)
            j_phys = self.j_head(j_input)
        if phase in {"D", "psi"}:
            damage_input = torch.cat([physics_feat_shared, d_phys, u, eps, sigma, j_phys], dim=1)
            raw_damage = self.D_head(damage_input)
            damage_drive = torch.relu(
                torch.linalg.vector_norm(j_phys, dim=1, keepdim=True)
                + eps.abs().mean(dim=1, keepdim=True)
                - 0.1
            )
            D_scalar = self._accumulate_damage(raw_damage, damage_drive, metadata=metadata)
        if phase == "psi":
            psi_input = torch.cat([physics_feat_shared, alpha_scalar], dim=1)
            psi_scalar = self.psi_head(psi_input)

        return {
            "alpha_scalar": alpha_scalar,
            "T_scalar": T_scalar,
            "j_phys": j_phys,
            "D_scalar": D_scalar,
            "psi_scalar": psi_scalar,
        }

    def _zero_contract_field(self, reference, channels):
        return reference.new_zeros(reference.shape[0], channels, *reference.shape[2:])

    def _pack_field_dict_to_bank(self, field_dict, reference_bank):
        bank_chunks = []
        for field_name, field_channels in PHYSICS_ATTRIBUTE_CONTRACT.items():
            field_value = field_dict.get(field_name)
            if field_value is None:
                field_value = self._zero_contract_field(reference_bank, field_channels)
            bank_chunks.append(
                self._require_tensor(
                    field_value,
                    f"field_dict[{field_name}]",
                    expected_channels=field_channels,
                )
            )
        return torch.cat(bank_chunks, dim=1)

    def build_physical_field_dict(self, attribute_bank, physics_feat_shared, metadata=None, field_recovery_phase=None):
        attribute_bank = self._require_tensor(
            attribute_bank,
            "attribute_bank",
            expected_channels=self.physics_attr_dim,
        )
        physics_feat_shared = self._require_tensor(
            physics_feat_shared,
            "physics_feat_shared",
            expected_channels=self.hidden_dim,
        )
        latent_field_dict = self.attribute_bank_to_field_dict(attribute_bank)
        u_latent = self._require_tensor(latent_field_dict["u"], "u_latent", expected_channels=4)
        u = self.u_head(u_latent)
        kinematics = self._velocity_kinematics(u)
        phase = self._resolved_recovery_phase(field_recovery_phase)

        strategy = str(self.secondary_field_strategy or "direct_bank")
        if strategy == "legacy_direct_bank":
            strategy = "direct_bank"
        if strategy == "direct_bank":
            p_raw = self._require_tensor(latent_field_dict["p"], "p_raw", expected_channels=2)
            rho_raw = self._require_tensor(latent_field_dict["rho"], "rho_raw", expected_channels=2)
        elif strategy in {"u_first_constructor", "u_first_constructor_detach"}:
            detach_inputs = strategy.endswith("_detach")
            u_for_constructor = u.detach() if detach_inputs else u
            div_for_constructor = kinematics["div_u"].detach() if detach_inputs else kinematics["div_u"]
            curl_for_constructor = kinematics["curl_u"].detach() if detach_inputs else kinematics["curl_u"]
            u_mag = torch.linalg.vector_norm(u_for_constructor, dim=1, keepdim=True)
            constructor_input = torch.cat(
                [
                    physics_feat_shared,
                    u_for_constructor,
                    u_mag,
                    div_for_constructor,
                    curl_for_constructor,
                ],
                dim=1,
            )
            p_raw, rho_raw = torch.chunk(self.prho_constructor(constructor_input), 2, dim=1)
        else:
            raise ValueError(f"Unsupported secondary_field_strategy: {strategy}")

        p_scalar, rho_scalar = self._normalize_pressure_density(p_raw, rho_raw)
        p = self._repeat_channels(p_scalar, 2)
        rho = self._repeat_channels(rho_scalar, 2)
        eps = kinematics["eps"]
        sigma = self._build_sigma(p_scalar, eps)
        d_phys = self._integrate_displacement_from_velocity(u, metadata=metadata)
        recovered = self._build_recovered_scalar_and_vector_fields(
            phase=phase,
            physics_feat_shared=physics_feat_shared,
            u=u,
            p_scalar=p_scalar,
            rho_scalar=rho_scalar,
            eps=eps,
            sigma=sigma,
            d_phys=d_phys,
            metadata=metadata,
        )
        d_bank = self._repeat_channels(d_phys, 4)
        alpha = self._repeat_channels(recovered["alpha_scalar"], 2)
        temperature = self._repeat_channels(recovered["T_scalar"], 2)
        j = self._repeat_channels(recovered["j_phys"], 4)
        damage = self._repeat_channels(recovered["D_scalar"], 2)
        psi = self._repeat_channels(recovered["psi_scalar"], 2)

        final_field_dict = {
            "u_latent": u_latent,
            "u": u,
            "u_phys": u,
            "p": p,
            "p_scalar": p_scalar,
            "rho": rho,
            "rho_scalar": rho_scalar,
            "eps": eps,
            "sigma": sigma,
            "d": d_bank,
            "d_phys": d_phys,
            "alpha": alpha,
            "alpha_scalar": recovered["alpha_scalar"],
            "T": temperature,
            "T_scalar": recovered["T_scalar"],
            "j": j,
            "j_phys": recovered["j_phys"],
            "D": damage,
            "D_scalar": recovered["D_scalar"],
            "psi": psi,
            "psi_scalar": recovered["psi_scalar"],
            "div_u": kinematics["div_u"],
            "curl_u": kinematics["curl_u"],
        }
        contract_field_dict = dict(latent_field_dict)
        contract_field_dict["u"] = u_latent
        contract_field_dict["p"] = p
        contract_field_dict["rho"] = rho
        contract_field_dict["d"] = d_bank
        contract_field_dict["alpha"] = alpha
        contract_field_dict["T"] = temperature
        contract_field_dict["eps"] = eps
        contract_field_dict["sigma"] = sigma
        contract_field_dict["j"] = j
        contract_field_dict["D"] = damage
        contract_field_dict["psi"] = psi
        for field_name in self._inactive_only_u_fields(field_recovery_phase=phase):
            contract_field_dict[field_name] = self._zero_contract_field(
                attribute_bank,
                PHYSICS_ATTRIBUTE_CONTRACT[field_name],
            )

        final_bank = self._pack_field_dict_to_bank(contract_field_dict, attribute_bank)
        diagnostics = {
            "rho_mean": rho.mean(),
            "rho_min": rho.min(),
            "p_abs_mean": p.abs().mean(),
            "div_u_abs_mean": kinematics["div_u"].abs().mean(),
            "alpha_mean": recovered["alpha_scalar"].mean(),
            "T_mean": recovered["T_scalar"].mean(),
            "j_abs_mean": recovered["j_phys"].abs().mean(),
            "D_mean": recovered["D_scalar"].mean(),
            "psi_mean": recovered["psi_scalar"].mean(),
        }
        return final_field_dict, final_bank, diagnostics

    def observable_fields_from_bank(self, attribute_bank):
        field_dict = self.attribute_bank_to_field_dict(attribute_bank)
        return {
            "d": field_dict["d"],
            "u": field_dict["u"],
        }

    def observable_heads_from_fields(self, observable_fields):
        d_field = self._require_tensor(observable_fields.get("d"), "observable.d", expected_channels=4)
        u_field = self._require_tensor(observable_fields.get("u"), "observable.u", expected_channels=4)
        return {
            "flow": self.u_head(u_field),
            "deformation": self.d_head(d_field),
        }

    def predict_observable_proxies(self, attribute_bank):
        observable_fields = self.observable_fields_from_bank(attribute_bank)
        observable_outputs = self.observable_heads_from_fields(observable_fields)
        return observable_fields, observable_outputs

    def forward_observable_pretrain(self, state_for_physics, sigma=None):
        state_for_physics = self._require_tensor(
            state_for_physics,
            "state_for_physics",
            expected_channels=self.latent_dim,
        )
        batch_size = state_for_physics.shape[0]
        sigma_embed = self._sigma_embedding(
            sigma,
            batch_size=batch_size,
            device=state_for_physics.device,
            dtype=state_for_physics.dtype,
        )
        sigma_embed = self._require_tensor(sigma_embed, "sigma_embedding")
        sigma_map = sigma_embed.view(batch_size, self.hidden_dim, 1, 1, 1)
        physics_feat_shared = self.physics_encoder_shared(state_for_physics)
        physics_feat_shared = self._require_tensor(
            physics_feat_shared,
            "physics_feat_shared_pre_sigma",
            expected_channels=self.hidden_dim,
        )
        physics_feat_shared = physics_feat_shared + sigma_map
        physics_feat_shared = self._require_tensor(
            physics_feat_shared,
            "physics_feat_shared_post_sigma",
            expected_channels=self.hidden_dim,
        )
        shared_attribute_bank = self._decode_attribute_bank(
            physics_feat_shared,
            "shared_attribute_bank",
        )
        observable_fields, observable_outputs = self.predict_observable_proxies(shared_attribute_bank)

        self._cache = {
            "training_path": "observable_pretrain",
            "physics_feat": physics_feat_shared.detach(),
            "physics_feat_live": physics_feat_shared,
            "sigma_embedding": sigma_embed.detach(),
            "sigma_embedding_live": sigma_embed,
            "shared_attribute_bank": shared_attribute_bank.detach(),
            "shared_attribute_bank_live": shared_attribute_bank,
            "observable_d": observable_fields["d"].detach(),
            "observable_d_live": observable_fields["d"],
            "observable_u": observable_fields["u"].detach(),
            "observable_u_live": observable_fields["u"],
            "observable_flow": observable_outputs["flow"].detach(),
            "observable_flow_live": observable_outputs["flow"],
            "observable_deformation": observable_outputs["deformation"].detach(),
            "observable_deformation_live": observable_outputs["deformation"],
            "using_shared_fallback": True,
            "fallback_reason": "observable_pretrain",
        }
        return {
            "shared_attribute_bank": shared_attribute_bank,
            "observable_fields": observable_fields,
            "observable_outputs": observable_outputs,
        }

    def _decode_operator_deltas(self, operator_output, delta_x_name, delta_v_name):
        operator_output = self._require_tensor(operator_output, f"{delta_x_name}_source")
        if operator_output.shape[1] != self.latent_dim * 2:
            raise RuntimeError(
                f"Operator expert output channel mismatch: expected {self.latent_dim * 2}, "
                f"got {operator_output.shape[1]}."
            )
        delta_x, delta_v = torch.chunk(operator_output, 2, dim=1)
        delta_x = self._safe_physical_tensor(delta_x, delta_x_name)
        delta_v = self._safe_physical_tensor(delta_v, delta_v_name)
        return delta_x, delta_v

    def _decode_operator_attribute_update(self, operator_output, expert_idx, update_name):
        operator_output = self._require_tensor(operator_output, f"{update_name}_source")
        if operator_output.shape[1] != self.physics_attr_dim:
            raise RuntimeError(
                f"Operator expert output channel mismatch: expected {self.physics_attr_dim}, "
                f"got {operator_output.shape[1]}."
            )
        if not self.interpret_attribute_bank_as_physical:
            return self._safe_physical_tensor(operator_output, update_name)
        expert_mask = self.expert_field_mask(
            expert_idx,
            device=operator_output.device,
            dtype=operator_output.dtype,
        ).view(1, self.physics_attr_dim, 1, 1, 1)
        masked_update = operator_output * expert_mask
        return self._safe_physical_tensor(masked_update, update_name)

    def _build_decoder_input(self, encoder_feat, decoded_feat, x_phys, v_phys, name):
        encoder_feat = self._require_tensor(encoder_feat, f"{name}.encoder_feat")
        decoded_feat = self._require_tensor(decoded_feat, f"{name}.decoded_feat")
        x_phys = self._require_tensor(x_phys, f"{name}.x_phys", expected_channels=self.latent_dim)
        v_phys = self._require_tensor(v_phys, f"{name}.v_phys", expected_channels=self.latent_dim)
        spatial_shape = encoder_feat.shape[2:]
        if decoded_feat.shape[2:] != spatial_shape or x_phys.shape[2:] != spatial_shape or v_phys.shape[2:] != spatial_shape:
            raise RuntimeError(f"Decoder input spatial mismatch detected for `{name}`.")
        return torch.cat([encoder_feat, decoded_feat, x_phys, v_phys], dim=1)

    def _route_experts(self, cond_feat, metadata, batch_size, device):
        """
        多标签专家路由：所有标签对应的专家都会获得 bias 加成。
        """
        cond_feat = self._require_tensor(cond_feat, "cond_feat", expected_channels=self.hidden_dim)
        labels_per_sample = self._route_label_ids(metadata, batch_size, device)
        active_label_ids = self._labels_to_padded_tensor(labels_per_sample, device=device)
        if active_label_ids is not None:
            self.set_debug_context(active_label_ids=active_label_ids.detach())
        if self.label_only_mode or self.force_label_only_routing:
            route_logits = torch.zeros(
                batch_size,
                self.num_phenomena,
                device=device,
                dtype=cond_feat.dtype,
            )
        else:
            route_logits = self.expert_router(cond_feat)
        route_logits = self._require_tensor(route_logits, "route_logits_raw", expected_channels=self.num_phenomena)
        label_prior = torch.zeros_like(route_logits)

        # 为每个样本的每个标签对应的位置加上 bias
        for sample_idx, label_list in enumerate(labels_per_sample):
            for label_id in label_list:
                label_id = int(label_id)
                if 0 <= label_id < self.num_phenomena:
                    label_prior[sample_idx, label_id] += self.router_label_bias

        route_logits = route_logits + label_prior
        route_logits = self._require_tensor(route_logits, "route_logits", expected_channels=self.num_phenomena)
        if self.excluded_expert_indices:
            excluded_mask = torch.zeros(
                self.num_phenomena,
                device=route_logits.device,
                dtype=torch.bool,
            )
            excluded_mask[list(sorted(self.excluded_expert_indices))] = True
            route_logits = route_logits.masked_fill(
                excluded_mask.unsqueeze(0),
                torch.finfo(route_logits.dtype).min,
            )
            route_logits = self._require_tensor(
                route_logits,
                "route_logits_excluded_masked",
                expected_channels=self.num_phenomena,
            )

        top_k = min(self.moe_top_k, self._available_expert_count())
        if top_k > 0:
            topk_logits, topk_indices = torch.topk(route_logits, k=top_k, dim=-1)
            topk_logits = self._require_tensor(topk_logits, "topk_logits", expected_channels=top_k)
            topk_weights = torch.softmax(topk_logits / max(self.router_temperature, 1e-6), dim=-1)
            topk_weights = self._require_tensor(topk_weights, "topk_weights", expected_channels=top_k)
            dominant_expert = topk_indices[:, 0]
        else:
            topk_indices = torch.empty(batch_size, 0, device=device, dtype=torch.long)
            topk_weights = torch.empty(batch_size, 0, device=device, dtype=route_logits.dtype)
            dominant_expert = torch.full(
                (batch_size,),
                -1,
                device=device,
                dtype=torch.long,
            )
        offlabel_selected_experts = self._offlabel_selected_experts(topk_indices, labels_per_sample)
        self.set_debug_context(
            selected_topk_experts=topk_indices.detach(),
            selected_topk_weights=topk_weights.detach(),
            active_label_ids=active_label_ids.detach() if isinstance(active_label_ids, torch.Tensor) else None,
            offlabel_selected_experts=(
                offlabel_selected_experts.detach()
                if isinstance(offlabel_selected_experts, torch.Tensor)
                else None
            ),
            dominant_expert=dominant_expert.detach(),
        )

        # 返回主标签（每个样本的第一个标签）用于后续记录
        primary_label_ids = torch.tensor(
            [labels[0] if labels else 0 for labels in labels_per_sample],
            device=device, dtype=torch.long
        )
        return (
            route_logits,
            topk_indices,
            topk_weights,
            dominant_expert,
            primary_label_ids,
            active_label_ids,
            offlabel_selected_experts,
        )

    def _compute_rl_policy_weights(
        self,
        cond_feat,
        physics_feat_shared,
        sigma,
        topk_indices,
        topk_weights,
        route_logits,
    ):
        if (not self.enable_rl_expert_optimization) or topk_indices.numel() == 0:
            zeros = torch.zeros_like(topk_weights)
            return topk_weights, zeros, torch.zeros(
                cond_feat.shape[0],
                self.rl_hidden_dim,
                device=cond_feat.device,
                dtype=cond_feat.dtype,
            )

        batch_size, top_k = topk_indices.shape
        pooled_physics = physics_feat_shared.mean(dim=(2, 3, 4)).to(dtype=cond_feat.dtype)
        sigma_values = self._prepare_sigma_values(
            sigma,
            batch_size=batch_size,
            device=cond_feat.device,
            dtype=cond_feat.dtype,
        )
        if sigma_values is None:
            sigma_values = torch.zeros(batch_size, device=cond_feat.device, dtype=cond_feat.dtype)
        usage_context = self.expert_usage_ema.to(
            device=cond_feat.device, dtype=cond_feat.dtype
        ).unsqueeze(0).expand(batch_size, -1)
        state_input = torch.cat(
            [
                cond_feat,
                pooled_physics,
                usage_context,
                sigma_values.view(batch_size, 1),
            ],
            dim=-1,
        )
        rl_state = self.rl_state_proj(state_input)
        expert_embed = self.rl_expert_embedding(topk_indices).to(dtype=cond_feat.dtype)
        gathered_route_logits = route_logits.gather(1, topk_indices).to(dtype=cond_feat.dtype)
        policy_input = torch.cat(
            [
                rl_state.unsqueeze(1).expand(-1, top_k, -1),
                expert_embed,
                topk_weights.unsqueeze(-1),
                gathered_route_logits.unsqueeze(-1),
            ],
            dim=-1,
        )
        policy_logits = self.rl_policy_head(policy_input).squeeze(-1)
        fused_logits = torch.log(topk_weights.clamp_min(1e-6)) + policy_logits
        policy_weights = torch.softmax(fused_logits, dim=-1)
        return policy_weights, policy_logits, rl_state

    def _update_rl_reward_ema(self, expert_indices, rewards, valid_mask):
        if not self.enable_rl_expert_optimization:
            return
        if expert_indices is None or rewards is None or valid_mask is None:
            return
        valid = valid_mask > 0
        if not torch.any(valid):
            return

        with torch.no_grad():
            ema_device = self.rl_reward_ema.device
            ema_dtype = self.rl_reward_ema.dtype
            idx = expert_indices[valid].reshape(-1).to(device=ema_device, dtype=torch.long)
            rew = rewards[valid].detach().reshape(-1).to(device=ema_device, dtype=ema_dtype)
            reward_sum = torch.zeros_like(self.rl_reward_ema, device=ema_device, dtype=ema_dtype)
            reward_count = torch.zeros_like(self.rl_reward_ema, device=ema_device, dtype=ema_dtype)
            reward_sum.scatter_add_(0, idx, rew)
            reward_count.scatter_add_(0, idx, torch.ones_like(rew, device=ema_device, dtype=ema_dtype))
            reward_mean = torch.where(
                reward_count > 0,
                reward_sum / reward_count.clamp_min(1.0),
                self.rl_reward_ema,
            )
            updated = torch.where(
                reward_count > 0,
                self.rl_reward_decay * self.rl_reward_ema
                + (1.0 - self.rl_reward_decay) * reward_mean,
                self.rl_reward_ema,
            )
            self.rl_reward_ema.copy_(updated)
    
    def forward(self, v_original, state_for_physics, sigma=None, metadata=None):
        """
        Forward pass:
        x_hat -> shared attribute bank -> operator MoE field updates -> shared decoder -> v_corrected.

        Args:
            v_original: 原始模型预测的速度场 [B, C, T, H, W]
            state_for_physics: 用于物理建模的状态 [B, C, T, H, W]
            metadata: 条件输入字典（label_id/label_name/n/q）

        Returns:
            v_corrected: 物理校正后的速度场 [B, C, T, H, W]
        """
        v_original = self._require_tensor(v_original, "v_original")
        state_for_physics = self._require_tensor(
            state_for_physics,
            "state_for_physics",
            expected_channels=self.latent_dim,
        )
        B = v_original.shape[0]
        sigma_embed = self._sigma_embedding(
            sigma,
            batch_size=B,
            device=v_original.device,
            dtype=v_original.dtype,
        )
        sigma_embed = self._require_tensor(sigma_embed, "sigma_embedding")
        sigma_map = sigma_embed.view(B, self.hidden_dim, 1, 1, 1)
        physics_feat_shared = self.physics_encoder_shared(state_for_physics)
        physics_feat_shared = self._require_tensor(
            physics_feat_shared,
            "physics_feat_shared_pre_sigma",
            expected_channels=self.hidden_dim,
        )
        physics_feat_shared = physics_feat_shared + sigma_map
        physics_feat_shared = self._require_tensor(
            physics_feat_shared,
            "physics_feat_shared_post_sigma",
            expected_channels=self.hidden_dim,
        )

        effective_use_moe = bool(self.use_moe and self.core_ablation_mode != "generic_latent_correction")
        requires_explicit_contract = bool(self.interpret_attribute_bank_as_physical)
        if self.strict_physical_state_contract and requires_explicit_contract and (metadata is None or not effective_use_moe):
            reason = "metadata is None" if metadata is None else "MoE routing is disabled"
            raise RuntimeError(
                f"Strict physical-state contract forbids shared fallback because {reason}."
            )

        shared_attribute_bank = self._decode_attribute_bank(
            physics_feat_shared,
            "shared_attribute_bank",
        )

        cond_feat = torch.zeros(
            B,
            self.hidden_dim,
            device=v_original.device,
            dtype=physics_feat_shared.dtype,
        )
        topk_indices = torch.empty(B, 0, device=v_original.device, dtype=torch.long)
        topk_weights = torch.empty(B, 0, device=v_original.device, dtype=physics_feat_shared.dtype)
        route_logits = None
        dominant_expert = None
        label_ids = None
        active_label_ids = None
        using_shared_fallback = metadata is None or not effective_use_moe
        fallback_reason = None
        branch_attribute_updates = None
        offlabel_selected_experts = None

        fused_attribute_delta = torch.zeros_like(shared_attribute_bank)

        if using_shared_fallback:
            fallback_reason = "no_metadata" if metadata is None else "moe_disabled"
        else:
            cond_feat = self._encode_condition(
                metadata,
                B,
                v_original.device,
                physics_feat_shared.dtype,
            )
            cond_feat = self._require_tensor(cond_feat, "cond_feat", expected_channels=self.hidden_dim)
            cond_feat = cond_feat + sigma_embed.to(dtype=cond_feat.dtype)
            cond_feat = self._require_tensor(cond_feat, "cond_feat_post_sigma", expected_channels=self.hidden_dim)
            (
                route_logits,
                topk_indices,
                topk_weights,
                dominant_expert,
                label_ids,
                active_label_ids,
                offlabel_selected_experts,
            ) = self._route_experts(
                cond_feat=cond_feat,
                metadata=metadata,
                batch_size=B,
                device=v_original.device,
            )
            cond_map = cond_feat.view(B, self.hidden_dim, 1, 1, 1).expand(
                -1, -1, *physics_feat_shared.shape[2:]
            )
            operator_input = torch.cat(
                [
                    physics_feat_shared,
                    cond_map,
                    shared_attribute_bank.to(dtype=physics_feat_shared.dtype),
                ],
                dim=1,
            )
            operator_input = self._require_tensor(operator_input, "operator_input")

            branch_attribute_updates = torch.zeros(
                B,
                topk_indices.shape[1],
                self.physics_attr_dim,
                *state_for_physics.shape[2:],
                device=state_for_physics.device,
                dtype=state_for_physics.dtype,
            )

            for sample_idx in range(B):
                sample_input = operator_input[sample_idx:sample_idx + 1]
                sample_input = self._require_tensor(sample_input, "sample_input")
                self.set_debug_context(
                    sample_index=sample_idx,
                    selected_topk_experts=topk_indices[sample_idx].detach(),
                    selected_topk_weights=topk_weights[sample_idx].detach(),
                    active_label_ids=(
                        active_label_ids[sample_idx].detach()
                        if isinstance(active_label_ids, torch.Tensor)
                        and active_label_ids.ndim >= 1
                        and sample_idx < active_label_ids.shape[0]
                        else None
                    ),
                    offlabel_selected_experts=(
                        offlabel_selected_experts[sample_idx].detach()
                        if isinstance(offlabel_selected_experts, torch.Tensor)
                        and offlabel_selected_experts.ndim >= 1
                        and sample_idx < offlabel_selected_experts.shape[0]
                        else None
                    ),
                    dominant_expert=(
                        int(dominant_expert[sample_idx].item())
                        if isinstance(dominant_expert, torch.Tensor)
                        and dominant_expert.ndim >= 1
                        and sample_idx < dominant_expert.shape[0]
                        else None
                    ),
                )
                for slot_idx in range(topk_indices.shape[1]):
                    expert_idx = int(topk_indices[sample_idx, slot_idx].item())
                    expert_weight = topk_weights[sample_idx, slot_idx].to(
                        device=sample_input.device,
                        dtype=sample_input.dtype,
                    )
                    self.set_debug_context(
                        expert_idx=expert_idx,
                        expert_name=self.phenomenon_name_from_index(expert_idx),
                    )
                    operator_module_idx = (
                        expert_idx
                        if self.use_phenomenon_specific_operators
                        else 0
                    )
                    operator_output = self.operator_experts[operator_module_idx](sample_input)
                    operator_output = self._require_tensor(
                        operator_output,
                        f"branch_attribute_update_expert_{expert_idx}_source",
                    )
                    delta_attr = self._decode_operator_attribute_update(
                        operator_output,
                        expert_idx,
                        f"branch_attribute_update_expert_{expert_idx}",
                    )
                    branch_attribute_updates[sample_idx:sample_idx + 1, slot_idx] = delta_attr.to(
                        dtype=branch_attribute_updates.dtype
                    )
                    updated_fused_attribute_delta = (
                        fused_attribute_delta[sample_idx:sample_idx + 1]
                        + expert_weight.to(dtype=delta_attr.dtype) * delta_attr
                    )
                    updated_fused_attribute_delta = self._require_tensor(
                        updated_fused_attribute_delta,
                        "fused_attribute_delta_accumulated",
                    )
                    fused_attribute_delta[sample_idx:sample_idx + 1] = updated_fused_attribute_delta

            with torch.no_grad():
                hist = torch.zeros(self.num_phenomena, device=v_original.device, dtype=torch.float32)
                if topk_indices.numel() > 0:
                    hist.scatter_add_(0, topk_indices.reshape(-1), topk_weights.reshape(-1).float())
                hist = hist / max(float(B), 1.0)
                self.expert_usage_ema.mul_(0.99).add_(0.01 * hist)
            self.set_debug_context(
                sample_index=None,
                expert_idx=None,
                expert_name=None,
                selected_topk_experts=topk_indices.detach(),
                selected_topk_weights=topk_weights.detach(),
                active_label_ids=(
                    active_label_ids.detach()
                    if isinstance(active_label_ids, torch.Tensor)
                    else None
                ),
                offlabel_selected_experts=(
                    offlabel_selected_experts.detach()
                    if isinstance(offlabel_selected_experts, torch.Tensor)
                    else None
                ),
                dominant_expert=dominant_expert.detach() if isinstance(dominant_expert, torch.Tensor) else None,
            )

        fused_attribute_bank = self._safe_physical_tensor(
            shared_attribute_bank + fused_attribute_delta,
            "fused_attribute_bank",
        )
        physical_field_dict_live = None
        physical_field_metrics = {}
        if self._uses_only_u_policy():
            physical_field_dict_live, fused_attribute_bank, physical_field_metrics = self.build_physical_field_dict(
                fused_attribute_bank,
                physics_feat_shared,
                metadata=metadata,
                field_recovery_phase=self.field_recovery_phase,
            )
        cond_map = cond_feat.view(B, self.hidden_dim, 1, 1, 1).expand(
            -1, -1, *physics_feat_shared.shape[2:]
        )
        decoder_input = torch.cat(
            [
                physics_feat_shared,
                cond_map,
                fused_attribute_bank.to(dtype=physics_feat_shared.dtype),
                v_original.to(dtype=physics_feat_shared.dtype),
            ],
            dim=1,
        )
        raw_correction = self.shared_decoder(decoder_input)
        raw_correction = self._safe_correction(raw_correction)
        injected_correction = raw_correction
        if not self.physics_to_flow_injection_enabled:
            injected_correction = raw_correction * 0.0
        v_corrected = v_original + injected_correction
        expert_correction_maps = None
        if self.export_expert_attention and isinstance(branch_attribute_updates, torch.Tensor):
            expert_correction_maps = self._compute_expert_correction_maps(
                v_original=v_original,
                physics_feat_shared=physics_feat_shared,
                cond_feat=cond_feat,
                shared_attribute_bank=shared_attribute_bank,
                branch_attribute_updates=branch_attribute_updates,
                topk_weights=topk_weights,
                apply_router_weight=self.expert_attention_apply_router_weight,
                metadata=metadata,
            )

        raw_correction_norm = raw_correction.detach().float().reshape(B, -1).norm(dim=1)
        injected_correction_norm = injected_correction.detach().float().reshape(B, -1).norm(dim=1)
        sigma_gate = torch.ones(B, device=v_original.device, dtype=v_original.dtype)
        effective_scale = torch.ones(B, device=v_original.device, dtype=v_original.dtype)

        self._cache = {
            "core_ablation_mode": self.core_ablation_mode,
            "interpret_attribute_bank_as_physical": bool(self.interpret_attribute_bank_as_physical),
            "use_phenomenon_specific_operators": bool(self.use_phenomenon_specific_operators),
            "physics_to_flow_injection_enabled": bool(self.physics_to_flow_injection_enabled),
            "force_label_only_routing": bool(self.force_label_only_routing),
            "using_shared_fallback": using_shared_fallback,
            "fallback_reason": fallback_reason,
            "moe_top_k": int(self.moe_top_k),
            "excluded_expert_names": list(self.excluded_expert_names),
            "excluded_expert_indices": sorted(int(idx) for idx in self.excluded_expert_indices),
            "physics_feat": physics_feat_shared.detach(),
            "physics_feat_live": physics_feat_shared,
            "raw_correction": raw_correction.detach(),
            "raw_correction_live": raw_correction,
            "injected_correction": injected_correction.detach(),
            "injected_correction_live": injected_correction,
            "cond_feat": cond_feat.detach(),
            "cond_feat_live": cond_feat,
            "sigma_embedding": sigma_embed.detach(),
            "sigma_embedding_live": sigma_embed,
            "label_ids": label_ids.detach() if isinstance(label_ids, torch.Tensor) else None,
            "active_label_ids": (
                active_label_ids.detach()
                if isinstance(active_label_ids, torch.Tensor) else None
            ),
            "active_expert_indices": topk_indices.detach(),
            "active_expert_weights": topk_weights.detach(),
            "active_expert_weights_live": topk_weights,
            "selected_topk_experts": topk_indices.detach(),
            "selected_topk_weights": topk_weights.detach(),
            "router_topk_weights": topk_weights.detach(),
            "router_topk_weights_live": topk_weights,
            "offlabel_selected_experts": (
                offlabel_selected_experts.detach()
                if isinstance(offlabel_selected_experts, torch.Tensor) else None
            ),
            "shared_attribute_bank": shared_attribute_bank.detach(),
            "shared_attribute_bank_live": shared_attribute_bank,
            "fused_attribute_bank": fused_attribute_bank.detach(),
            "fused_attribute_bank_live": fused_attribute_bank,
            "physical_field_dict": {
                key: value.detach()
                for key, value in (physical_field_dict_live or {}).items()
            },
            "physical_field_dict_live": physical_field_dict_live,
            "physical_field_metrics": {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in physical_field_metrics.items()
            },
            "physical_field_metrics_live": physical_field_metrics,
            "branch_attribute_updates": (
                branch_attribute_updates.detach()
                if isinstance(branch_attribute_updates, torch.Tensor) else None
            ),
            "branch_attribute_updates_live": branch_attribute_updates,
            "expert_correction_maps": expert_correction_maps,
            # Compatibility aliases for downstream tooling migrating from shared-slots v1.
            "shared_x_phys": shared_attribute_bank.detach(),
            "shared_x_phys_live": shared_attribute_bank,
            "shared_v_phys": shared_attribute_bank.detach(),
            "shared_v_phys_live": shared_attribute_bank,
            "fused_x_phys": fused_attribute_bank.detach(),
            "fused_x_phys_live": fused_attribute_bank,
            "fused_v_phys": fused_attribute_bank.detach(),
            "fused_v_phys_live": fused_attribute_bank,
            "branch_delta_x": None,
            "branch_delta_v": None,
            "branch_delta_x_live": None,
            "branch_delta_v_live": None,
            "dominant_expert": dominant_expert.detach() if isinstance(dominant_expert, torch.Tensor) else None,
            "route_logits": route_logits.detach() if isinstance(route_logits, torch.Tensor) else None,
            "sigma_gate": sigma_gate,
            "effective_scale": effective_scale,
            "raw_correction_norm": raw_correction_norm,
            "gated_correction_norm": injected_correction_norm,
            "correction_norm": injected_correction_norm,
        }

        return v_corrected

    def _compute_expert_correction_maps(
        self,
        v_original,
        physics_feat_shared,
        cond_feat,
        shared_attribute_bank,
        branch_attribute_updates,
        topk_weights,
        apply_router_weight=True,
        metadata=None,
    ):
        """
        Decode each active expert's weighted attribute delta into an independent
        spatial correction contribution map: abs(correction_e - correction_base).mean(C).
        The returned tensor is CPU float32 with shape [B, K, T, H, W].
        """
        if branch_attribute_updates.numel() == 0 or topk_weights.numel() == 0:
            return None
        B, topk = branch_attribute_updates.shape[:2]
        maps_per_batch = []
        with torch.no_grad():
            for sample_idx in range(B):
                sample_physics = physics_feat_shared[sample_idx:sample_idx + 1]
                sample_cond = cond_feat[sample_idx:sample_idx + 1]
                sample_cond_map = sample_cond.view(1, self.hidden_dim, 1, 1, 1).expand(
                    -1, -1, *sample_physics.shape[2:]
                )
                sample_v = v_original[sample_idx:sample_idx + 1].to(dtype=sample_physics.dtype)
                base_bank = shared_attribute_bank[sample_idx:sample_idx + 1].to(
                    device=sample_physics.device,
                    dtype=sample_physics.dtype,
                )
                base_bank = self._prepare_attribute_bank_for_decode(
                    base_bank,
                    sample_physics,
                    metadata=metadata,
                    name="expert_attention_base_bank",
                )
                base_input = torch.cat(
                    [sample_physics, sample_cond_map, base_bank, sample_v],
                    dim=1,
                )
                base_correction = self._safe_correction(self.shared_decoder(base_input))

                maps_per_slot = []
                for slot_idx in range(topk):
                    weight = topk_weights[sample_idx, slot_idx].to(
                        device=sample_physics.device,
                        dtype=sample_physics.dtype,
                    )
                    delta_attr = branch_attribute_updates[sample_idx:sample_idx + 1, slot_idx].to(
                        device=sample_physics.device,
                        dtype=sample_physics.dtype,
                    )
                    branch_bank = shared_attribute_bank[sample_idx:sample_idx + 1].to(
                        device=sample_physics.device,
                        dtype=sample_physics.dtype,
                    )
                    delta_scale = weight if apply_router_weight else torch.ones_like(weight)
                    branch_bank = self._safe_physical_tensor(
                        branch_bank + delta_scale * delta_attr,
                        "expert_attention_branch_bank",
                    )
                    branch_bank = self._prepare_attribute_bank_for_decode(
                        branch_bank,
                        sample_physics,
                        metadata=metadata,
                        name="expert_attention_branch_bank_decoded",
                    )
                    branch_input = torch.cat(
                        [sample_physics, sample_cond_map, branch_bank, sample_v],
                        dim=1,
                    )
                    branch_correction = self._safe_correction(self.shared_decoder(branch_input))
                    contrib = (branch_correction - base_correction).abs().mean(dim=1)[0]
                    maps_per_slot.append(contrib.detach().float().cpu())
                maps_per_batch.append(torch.stack(maps_per_slot, dim=0))
        return torch.stack(maps_per_batch, dim=0)

    def _prepare_attribute_bank_for_decode(self, attribute_bank, physics_feat_shared, metadata=None, name="attribute_bank"):
        attribute_bank = self._safe_physical_tensor(attribute_bank, name)
        if self._uses_only_u_policy():
            _, attribute_bank, _ = self.build_physical_field_dict(
                attribute_bank,
                physics_feat_shared,
                metadata=metadata,
                field_recovery_phase=self.field_recovery_phase,
            )
        return attribute_bank.to(dtype=physics_feat_shared.dtype)

    def set_pde_residuals(self, pde_residuals):
        """设置PDE残差模块（用于约束计算）"""
        self.pde_residuals = pde_residuals

    def set_constraint_mode(self, enabled=True, step_size=0.01):
        """设置物理约束应用模式"""
        self.apply_constraints_in_forward = bool(enabled)
        self.constraint_step_size = max(float(step_size), 0.0)

    def set_export_expert_attention(self, enabled=True, apply_router_weight=True):
        """Enable/disable per-active-expert spatial correction attribution export."""
        self.export_expert_attention = bool(enabled)
        self.expert_attention_apply_router_weight = bool(apply_router_weight)

    def set_core_ablation_mode(self, mode="full"):
        """Configure coarse mechanism ablations used by isolated experiment runners."""
        mode = str(mode or "full")
        if mode not in CORE_ABLATION_MODES:
            raise ValueError(
                f"Unsupported core_ablation_mode={mode!r}. "
                f"Expected one of {CORE_ABLATION_MODES}."
            )
        self.core_ablation_mode = mode
        self.interpret_attribute_bank_as_physical = mode not in {
            "generic_latent_correction",
            "wo_explicit_physical_interface",
        }
        self.use_phenomenon_specific_operators = mode not in {
            "generic_latent_correction",
            "wo_phenomenon_specific_operators",
        }
        self.physics_to_flow_injection_enabled = mode != "wo_physics_to_flow_injection"
        self.force_label_only_routing = mode == "wo_learned_expert_routing"

    def set_ablation_modes(self, use_moe=True, label_only_mode=False):
        """设置消融模式开关"""
        self.use_moe = bool(use_moe) and self.core_ablation_mode != "generic_latent_correction"
        self.label_only_mode = bool(label_only_mode) or self.force_label_only_routing

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
            "core_ablation_mode": self.core_ablation_mode,
            "interpret_attribute_bank_as_physical": self.interpret_attribute_bank_as_physical,
            "use_phenomenon_specific_operators": self.use_phenomenon_specific_operators,
            "physics_to_flow_injection_enabled": self.physics_to_flow_injection_enabled,
            "using_shared_fallback": self._cache.get("using_shared_fallback", True),
            "fallback_reason": self._cache.get("fallback_reason", None),
            "use_moe_setting": self.use_moe,
            "num_experts": self.num_phenomena,
            "active_expert_indices": self._cache.get("active_expert_indices", None),
            "active_expert_weights": self._cache.get("active_expert_weights", None),
        }

    def compute_auxiliary_losses(self):
        """返回最小闭环下仍保留的轻量辅助统计。"""
        device = self.scale.device
        expert_balance = torch.mean(
            (self.expert_usage_ema - 1.0 / max(self.num_phenomena, 1)) ** 2
        ).to(device=device)

        return {
            "expert_balance": expert_balance,
            "condition_consistency": torch.zeros((), device=device),
            "policy_rl": torch.zeros((), device=device),
            "policy_entropy": torch.zeros((), device=device),
            "rl_reward_mean": torch.zeros((), device=device),
            "rl_advantage_mean": torch.zeros((), device=device),
        }
