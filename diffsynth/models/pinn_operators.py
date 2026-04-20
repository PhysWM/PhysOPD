"""
Physics-Informed Neural Network Operators
微分算子和物理方程定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .pinn_contracts import (
    EXPERT_FIELD_RECIPES,
    PHYSICS_ATTR_DIM,
    split_attribute_bank,
)


class DifferentialOperators:
    """
    微分算子工具类
    
    所有算子使用有限差分实现，避免 autograd（计算量太大）。
    输入 v: [B, C, T, H, W]，将 C 视为多分量，在空间维度 (H, W) 上做微分。
    
    注意：所有操作只用纯张量运算，不做就地赋值，确保梯度可传播。
    """
    
    @staticmethod
    def compute_divergence(v):
        """
        计算散度近似 ∇·v (有限差分)
        将 v 的 C 通道视为多个分量，分别在 H 和 W 方向做偏导后求和。

        Args:
            v: [B, C, T, H, W]
        Returns:
            div: [B, 1, T, H, W]
        """
        # Clamp input to prevent overflow
        v = torch.clamp(v, min=-10.0, max=10.0)

        components = []

        # dv/dH：沿 H 方向的偏导，所有通道求平均
        # [B, C, T, H-1, W] -> pad -> [B, C, T, H, W] -> mean over C
        dv_dh = v[:, :, :, 1:, :] - v[:, :, :, :-1, :]
        dv_dh = F.pad(dv_dh, (0, 0, 0, 1))  # pad H 方向最后一行
        components.append(dv_dh.mean(dim=1, keepdim=True))

        # dv/dW：沿 W 方向的偏导
        dv_dw = v[:, :, :, :, 1:] - v[:, :, :, :, :-1]
        dv_dw = F.pad(dv_dw, (0, 1))  # pad W 方向最后一列
        components.append(dv_dw.mean(dim=1, keepdim=True))

        div = components[0] + components[1]  # [B, 1, T, H, W]
        div = torch.clamp(div, min=-10.0, max=10.0)
        return div
    
    @staticmethod
    def compute_laplacian(v):
        """
        计算拉普拉斯算子 ∇²v (有限差分，中心差分)

        Args:
            v: [B, C, T, H, W]
        Returns:
            laplacian: [B, C, T, H, W]  (边界处为 0)
        """
        # Clamp input to prevent overflow
        v = torch.clamp(v, min=-10.0, max=10.0)

        parts = []

        # H 方向二阶导数: d²v/dH²
        if v.shape[3] > 2:
            d2v_dh2 = v[:, :, :, 2:, :] - 2 * v[:, :, :, 1:-1, :] + v[:, :, :, :-2, :]
            d2v_dh2 = torch.clamp(d2v_dh2, min=-10.0, max=10.0)
            # pad 回原始 H 尺寸（上下各一行0）
            d2v_dh2 = F.pad(d2v_dh2, (0, 0, 1, 1))  # (W_left, W_right, H_top, H_bottom)
            parts.append(d2v_dh2)

        # W 方向二阶导数: d²v/dW²
        if v.shape[4] > 2:
            d2v_dw2 = v[:, :, :, :, 2:] - 2 * v[:, :, :, :, 1:-1] + v[:, :, :, :, :-2]
            d2v_dw2 = torch.clamp(d2v_dw2, min=-10.0, max=10.0)
            d2v_dw2 = F.pad(d2v_dw2, (1, 1))  # pad W 方向
            parts.append(d2v_dw2)

        if len(parts) == 0:
            return v * 0.0  # 保持 grad_fn

        laplacian = sum(parts)
        laplacian = torch.clamp(laplacian, min=-10.0, max=10.0)
        return laplacian
    
    @staticmethod
    def compute_curl_2d(v):
        """
        计算 2D 涡量（旋度的 z 分量）
        curl_z ≈ dv_w/dH - dv_h/dW（跨通道的近似）

        Args:
            v: [B, C, T, H, W]
        Returns:
            curl: [B, 1, T, H-1, W-1]  (内部区域)
        """
        # Clamp input to prevent overflow
        v = torch.clamp(v, min=-10.0, max=10.0)

        # dv/dH
        dv_dh = v[:, :, :, 1:, :] - v[:, :, :, :-1, :]  # [B, C, T, H-1, W]
        dv_dh = torch.clamp(dv_dh, min=-10.0, max=10.0)

        # dv/dW
        dv_dw = v[:, :, :, :, 1:] - v[:, :, :, :, :-1]  # [B, C, T, H, W-1]
        dv_dw = torch.clamp(dv_dw, min=-10.0, max=10.0)

        # 对齐到相同的空间尺寸：取内部交叉区域
        dv_dh = dv_dh[:, :, :, :, :-1]  # [B, C, T, H-1, W-1]
        dv_dw = dv_dw[:, :, :, :-1, :]  # [B, C, T, H-1, W-1]

        # 跨通道近似：前半通道作为 x 分量，后半通道作为 y 分量
        C = v.shape[1]
        half_C = max(C // 2, 1)

        curl = dv_dw[:, :half_C].mean(dim=1, keepdim=True) - dv_dh[:, half_C:].mean(dim=1, keepdim=True)
        curl = torch.clamp(curl, min=-10.0, max=10.0)
        return curl


class MaterialPDEResiduals(nn.Module):
    """各种材质的PDE残差计算"""
    
    def __init__(self, num_phenomena=10, q_input_dim=64, n_numeric_dim=12, strict_metadata_contract=False):
        super().__init__()
        self.diff_ops = DifferentialOperators()
        self.num_phenomena = num_phenomena
        self.q_input_dim = q_input_dim
        self.n_numeric_dim = n_numeric_dim
        self.enable_conditioning = True
        self.strict_metadata_contract = bool(strict_metadata_contract)
        
        # 物理参数（可学习，但有界）
        self.nu = nn.Parameter(torch.tensor(0.01))  # 粘度
        self.rho = nn.Parameter(torch.tensor(1.0))  # 密度
        self.lambda_lame = nn.Parameter(torch.tensor(1.0))  # 拉梅常数
        self.mu = nn.Parameter(torch.tensor(1.0))  # 剪切模量
        self.friction_coef = nn.Parameter(torch.tensor(0.1))  # 摩擦系数

        # Register hooks to clamp parameters after each update
        for param_name in ['nu', 'rho', 'lambda_lame', 'mu', 'friction_coef']:
            param = getattr(self, param_name)
            param.register_hook(lambda grad, pn=param_name: self._clamp_param_grad(pn, grad))

        # 条件调制参数：初始为零，使模型一开始接近原始残差形式
        self.label_embedding = nn.Embedding(num_phenomena, 4)
        self.q_projector = nn.Linear(q_input_dim, 4, bias=False)
        self.n_projector = nn.Linear(n_numeric_dim, 4, bias=False)
        nn.init.zeros_(self.label_embedding.weight)
        nn.init.zeros_(self.q_projector.weight)
        nn.init.zeros_(self.n_projector.weight)
        self.fail_on_invalid = True

    @staticmethod
    def _zero_loss(v):
        return torch.mean(v ** 2) * 0.0

    def _raise_invalid(self, name, value=None):
        detail = ""
        if isinstance(value, torch.Tensor):
            tensor = value.detach().float()
            if tensor.numel() > 0:
                detail = (
                    f" shape={tuple(tensor.shape)}"
                    f" min={float(torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).min().item()):.6f}"
                    f" max={float(torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).max().item()):.6f}"
                )
        raise FloatingPointError(f"Invalid physics value detected in {name}.{detail}")

    def _safe_loss(self, loss, max_value=100.0, name="loss"):
        """确保 loss 不是 NaN/Inf，并裁剪到合理范围"""
        if loss is None:
            self._raise_invalid(name, value=None)

        # Handle scalar tensors
        if loss.dim() == 0:
            if torch.isnan(loss) or torch.isinf(loss):
                self._raise_invalid(name, loss)
            return torch.clamp(loss, min=-max_value, max=max_value)

        # Handle multi-dimensional tensors
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            self._raise_invalid(name, loss)
        return torch.clamp(loss, min=-max_value, max=max_value)

    @staticmethod
    def _metadata_label_name(metadata):
        if not isinstance(metadata, dict):
            return ""
        return str(metadata.get("label_name", "")).strip().lower()

    def _fit_2d(self, tensor, target_dim, batch_size, device, dtype):
        if target_dim <= 0:
            return torch.zeros(batch_size, 0, device=device, dtype=dtype)
        if tensor is None:
            if self.strict_metadata_contract:
                raise RuntimeError("Residual metadata contract violation: missing required conditioning tensor.")
            return torch.zeros(batch_size, target_dim, device=device, dtype=dtype)
        tensor = tensor.to(device=device, dtype=dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.repeat(batch_size, 1)
        if tensor.shape[0] != batch_size:
            if self.strict_metadata_contract:
                raise RuntimeError(
                    f"Residual metadata contract violation: batch mismatch, expected {batch_size}, got {tensor.shape[0]}."
                )
            tensor = tensor[:1].repeat(batch_size, 1)
        if tensor.shape[1] > target_dim:
            if self.strict_metadata_contract:
                raise RuntimeError(
                    f"Residual metadata contract violation: feature dim mismatch, expected {target_dim}, got {tensor.shape[1]}."
                )
            tensor = tensor[:, :target_dim]
        elif tensor.shape[1] < target_dim:
            if self.strict_metadata_contract:
                raise RuntimeError(
                    f"Residual metadata contract violation: feature dim mismatch, expected {target_dim}, got {tensor.shape[1]}."
                )
            pad = torch.zeros(batch_size, target_dim - tensor.shape[1], device=device, dtype=dtype)
            tensor = torch.cat([tensor, pad], dim=1)
        return tensor

    def _metadata_condition_vector(self, metadata, batch_size, device, dtype):
        if not isinstance(metadata, dict):
            if self.strict_metadata_contract:
                raise RuntimeError("Residual metadata contract violation: metadata must be a dict.")
            return torch.zeros(4, device=device, dtype=dtype), 0.0, 0.0, 0.0

        label_ids = metadata.get("label_id")
        if label_ids is None:
            if self.strict_metadata_contract:
                raise RuntimeError("Residual metadata contract violation: missing label_id.")
            label_ids = torch.zeros(batch_size, device=device, dtype=torch.long)
        elif isinstance(label_ids, int):
            label_ids = torch.tensor([label_ids], device=device, dtype=torch.long)
        else:
            label_ids = label_ids.to(device=device, dtype=torch.long).view(-1)
            if label_ids.numel() == 1 and batch_size > 1:
                label_ids = label_ids.repeat(batch_size)
            if label_ids.numel() != batch_size:
                if self.strict_metadata_contract:
                    raise RuntimeError(
                        f"Residual metadata contract violation: label batch mismatch, expected {batch_size}, got {label_ids.numel()}."
                    )
                label_ids = label_ids[:1].repeat(batch_size)
            label_ids = torch.clamp(label_ids, min=0, max=self.num_phenomena - 1)

        q_vector = self._fit_2d(metadata.get("q_vector"), self.q_input_dim, batch_size, device, dtype)
        n_numeric = self._fit_2d(metadata.get("n_numeric"), self.n_numeric_dim, batch_size, device, dtype)

        cond = (
            self.label_embedding(label_ids)
            + self.q_projector(q_vector)
            + self.n_projector(n_numeric)
        )
        cond = torch.tanh(cond).mean(dim=0).to(dtype=dtype)

        density_mean = float(n_numeric[:, 2].mean().detach().item()) if n_numeric.shape[1] > 2 else 0.0
        time_mean = float(n_numeric[:, 6].mean().detach().item()) if n_numeric.shape[1] > 6 else 0.0
        temp_mean = float(n_numeric[:, 10].mean().detach().item()) if n_numeric.shape[1] > 10 else 0.0
        return cond, density_mean, time_mean, temp_mean

    def _conditioned_scales(self, metadata, v):
        if not self.enable_conditioning:
            one = torch.ones((), device=v.device, dtype=v.dtype)
            return {"s0": one, "s1": one, "s2": one, "s3": one}
        cond, density_mean, time_mean, temp_mean = self._metadata_condition_vector(
            metadata=metadata,
            batch_size=v.shape[0],
            device=v.device,
            dtype=v.dtype,
        )
        # 基于条件向量与 n0/n1/n2 的轻量调制，输出接近 1.0 的乘子
        base = 1.0 + 0.25 * cond
        density_scale = 1.0 + 0.02 * torch.tanh(torch.tensor(density_mean, device=v.device, dtype=v.dtype))
        time_scale = 1.0 + 0.02 * torch.tanh(torch.tensor(time_mean / 10.0, device=v.device, dtype=v.dtype))
        temp_scale = 1.0 + 0.02 * torch.tanh(torch.tensor(temp_mean / 100.0, device=v.device, dtype=v.dtype))
        return {
            "s0": base[0] * density_scale,
            "s1": base[1] * time_scale,
            "s2": base[2] * temp_scale,
            "s3": base[3],
        }

    def _temporal_difference(self, v, order=1):
        diff = v
        for _ in range(order):
            if diff.shape[2] <= 1:
                return None
            diff = diff[:, :, 1:] - diff[:, :, :-1]
        return diff

    @staticmethod
    def _scalar_field(v):
        return v.mean(dim=1, keepdim=True)

    def _resolve_motion_mask(self, metadata, value, ref_tensor):
        if not isinstance(metadata, dict):
            return None
        mask = metadata.get("motion_mask")
        if not isinstance(mask, torch.Tensor):
            return None
        if mask.ndim == 4:
            mask = mask.unsqueeze(1)
        if mask.ndim != 5:
            return None

        mask = mask.to(device=value.device, dtype=value.dtype)
        if mask.shape[0] != value.shape[0]:
            if mask.shape[0] == 1 and value.shape[0] > 1:
                mask = mask.repeat(value.shape[0], 1, 1, 1, 1)
            else:
                mask = mask[:1].repeat(value.shape[0], 1, 1, 1, 1)
        if mask.shape[1] != 1:
            mask = mask.mean(dim=1, keepdim=True)

        target_shape = value.shape[2:]
        if mask.shape[2:] != target_shape:
            mask = F.interpolate(mask, size=target_shape, mode="trilinear", align_corners=False)
        if value.shape[1] != 1:
            mask = mask.expand(-1, value.shape[1], -1, -1, -1)
        return torch.clamp(mask, 0.0, 1.0)

    def _weighted_mean(self, value, metadata=None, ref_tensor=None):
        # Aggressive clamping at input
        value = torch.clamp(value, min=-100.0, max=100.0)

        ref = value if ref_tensor is None else ref_tensor
        mask = self._resolve_motion_mask(metadata, value, ref)

        if mask is None:
            result = torch.mean(value)
        else:
            numer = torch.sum(value * mask)
            denom = torch.sum(mask) + 1e-8  # Increased epsilon
            result = numer / denom

        # Ensure result is finite
        if torch.isnan(result) or torch.isinf(result):
            self._raise_invalid("weighted_mean", result)

        return torch.clamp(result, min=-100.0, max=100.0)

    def _weighted_square_mean(self, value, metadata=None, ref_tensor=None):
        # Clamp value to prevent overflow in squaring
        value_clamped = torch.clamp(value, min=-10.0, max=10.0)
        return self._weighted_mean(value_clamped ** 2, metadata=metadata, ref_tensor=ref_tensor)

    def _spatial_gradient_energy(self, v, metadata=None):
        parts = []
        if v.shape[3] > 1:
            parts.append(self._weighted_square_mean(v[:, :, :, 1:, :] - v[:, :, :, :-1, :], metadata=metadata, ref_tensor=v))
        if v.shape[4] > 1:
            parts.append(self._weighted_square_mean(v[:, :, :, :, 1:] - v[:, :, :, :, :-1], metadata=metadata, ref_tensor=v))
        if len(parts) == 0:
            return self._zero_loss(v)
        return sum(parts) / len(parts)

    def _divergence_field(self, v):
        """Compute divergence field using diff_ops."""
        return self.diff_ops.compute_divergence(v)

    def _laplacian_field(self, v, metadata=None):
        """Compute laplacian field using diff_ops, with optional motion weighting."""
        del metadata  # Reserved for future use
        return self.diff_ops.compute_laplacian(v)

    def _vorticity_field(self, v):
        """Compute vorticity field (curl) using diff_ops."""
        return self.diff_ops.compute_curl_2d(v)

    @staticmethod
    def _grad_height(field):
        if field.shape[-2] <= 1:
            return torch.zeros_like(field)
        diff = field[..., 1:, :] - field[..., :-1, :]
        return F.pad(diff, (0, 0, 0, 1))

    @staticmethod
    def _grad_width(field):
        if field.shape[-1] <= 1:
            return torch.zeros_like(field)
        diff = field[..., :, 1:] - field[..., :, :-1]
        return F.pad(diff, (0, 1, 0, 0))

    @staticmethod
    def _pressure_scalar(p):
        if p.shape[1] == 1:
            return p
        return p.mean(dim=1, keepdim=True)

    def _only_u_fluid_terms(self, u, p, rho, metadata=None):
        if u.shape[1] < 2:
            raise ValueError(f"Only-u residuals expect at least 2 velocity channels, got {u.shape[1]}.")
        u = u[:, :2]
        ux = u[:, 0:1]
        uy = u[:, 1:2]
        p_scalar = self._pressure_scalar(p)
        rho_scalar = rho.mean(dim=1, keepdim=True).clamp_min(1e-4)

        dux_dx = self._grad_width(ux)
        dux_dy = self._grad_height(ux)
        duy_dx = self._grad_width(uy)
        duy_dy = self._grad_height(uy)
        div_u = dux_dx + duy_dy

        rho_t = self._temporal_derivative(rho_scalar, metadata=metadata, order=1)
        mass_residual_field = rho_t + self._grad_width(rho_scalar * ux) + self._grad_height(rho_scalar * uy)
        mass_residual = self._weighted_square_mean(
            mass_residual_field,
            metadata=metadata,
            ref_tensor=rho_scalar,
        )

        u_t = self._temporal_derivative(u, metadata=metadata, order=1)
        conv_x = ux * dux_dx + uy * dux_dy
        conv_y = ux * duy_dx + uy * duy_dy
        grad_px = self._grad_width(p_scalar)
        grad_py = self._grad_height(p_scalar)
        lap_u = self._laplacian_field(u, metadata=metadata)
        viscosity = 1e-3
        momentum_field = torch.cat(
            [
                u_t[:, 0:1] + conv_x + grad_px / rho_scalar - viscosity * lap_u[:, 0:1],
                u_t[:, 1:2] + conv_y + grad_py / rho_scalar - viscosity * lap_u[:, 1:2],
            ],
            dim=1,
        )
        momentum_residual = self._weighted_square_mean(
            momentum_field,
            metadata=metadata,
            ref_tensor=u,
        )

        pressure_smoothness = self._spatial_gradient_energy(p_scalar, metadata=metadata)
        density_smoothness = self._spatial_gradient_energy(rho_scalar, metadata=metadata)
        density_floor = self._weighted_square_mean(
            torch.relu(1e-4 - rho_scalar),
            metadata=metadata,
            ref_tensor=rho_scalar,
        )

        return {
            "u": u,
            "p_scalar": p_scalar,
            "rho_scalar": rho_scalar,
            "div_u": div_u,
            "mass_residual": mass_residual,
            "momentum_residual": momentum_residual,
            "pressure_smoothness": pressure_smoothness,
            "density_smoothness": density_smoothness,
            "density_floor": density_floor,
        }

    def _only_u_fluid_residual(self, u, p, rho, metadata=None, prefix="fluid"):
        scales = self._conditioned_scales(metadata, u)
        terms = self._only_u_fluid_terms(u, p, rho, metadata=metadata)

        mass_term = terms["mass_residual"] * scales["s0"]
        momentum_term = terms["momentum_residual"] * scales["s1"]
        pressure_smoothness = terms["pressure_smoothness"] * scales["s2"]
        density_stability = (terms["density_smoothness"] + terms["density_floor"]) * scales["s3"]
        total_loss = torch.clamp(
            mass_term + momentum_term + pressure_smoothness + density_stability,
            min=0.0,
            max=100.0,
        )
        return total_loss, {
            "mass_residual": float(terms["mass_residual"].detach().item()),
            "momentum_residual": float(terms["momentum_residual"].detach().item()),
            "rho_mean": float(terms["rho_scalar"].detach().mean().item()),
            "rho_min": float(terms["rho_scalar"].detach().min().item()),
            "p_abs_mean": float(terms["p_scalar"].detach().abs().mean().item()),
            "div_u_abs_mean": float(terms["div_u"].detach().abs().mean().item()),
            f"{prefix}_mass_residual": float(mass_term.detach().item()),
            f"{prefix}_momentum_residual": float(momentum_term.detach().item()),
            f"{prefix}_pressure_smoothness": float(pressure_smoothness.detach().item()),
            f"{prefix}_density_stability": float(density_stability.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
            "cond_s3": float(scales["s3"].detach().item()),
            **self._field_family_diagnostics(
                {"u": terms["u"], "p": p, "rho": rho},
                metadata=metadata,
            ),
        }

    def _coerce_field_dict(self, primary, secondary=None):
        if isinstance(primary, dict):
            return primary
        if isinstance(secondary, dict):
            return secondary

        candidate = None
        for tensor in (primary, secondary):
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 5 and tensor.shape[1] == PHYSICS_ATTR_DIM:
                candidate = tensor
                break
        if candidate is None:
            raise RuntimeError(
                "Explicit attribute-bank PDE residuals require a [B, 32, T, H, W] tensor or field dict input."
            )
        return split_attribute_bank(candidate)

    @staticmethod
    def _field_mean(field):
        return field.mean(dim=1, keepdim=True)

    def _default_frame_delta_t(self, ref_tensor):
        time_dim = int(ref_tensor.shape[2]) if ref_tensor.ndim >= 3 else 1
        return 1.0 / max(time_dim - 1, 1)

    def _metadata_frame_delta_t(self, metadata, ref_tensor):
        batch_size = ref_tensor.shape[0]
        device = ref_tensor.device
        dtype = ref_tensor.dtype
        default_delta_t = self._default_frame_delta_t(ref_tensor)

        if isinstance(metadata, dict):
            raw_value = metadata.get("frame_delta_t")
            if isinstance(raw_value, torch.Tensor):
                frame_delta_t = raw_value.to(device=device, dtype=dtype).view(-1)
                if frame_delta_t.numel() == 1 and batch_size > 1:
                    frame_delta_t = frame_delta_t.repeat(batch_size)
                elif frame_delta_t.numel() != batch_size:
                    frame_delta_t = frame_delta_t[:1].repeat(batch_size)
                fallback = torch.full_like(frame_delta_t, float(default_delta_t))
                return torch.where(frame_delta_t > 0, frame_delta_t, fallback)
            if isinstance(raw_value, (int, float)) and float(raw_value) > 0:
                return torch.full(
                    (batch_size,),
                    float(raw_value),
                    device=device,
                    dtype=dtype,
                )

        return torch.full(
            (batch_size,),
            float(default_delta_t),
            device=device,
            dtype=dtype,
        )

    def _metadata_frame_time_grid(self, metadata, ref_tensor):
        batch_size = ref_tensor.shape[0]
        time_dim = int(ref_tensor.shape[2]) if ref_tensor.ndim >= 3 else 1
        device = ref_tensor.device
        dtype = ref_tensor.dtype

        base_grid = torch.linspace(
            0.0,
            1.0,
            steps=max(time_dim, 1),
            device=device,
            dtype=dtype,
        )
        if time_dim <= 1:
            base_grid = torch.zeros(1, device=device, dtype=dtype)

        if isinstance(metadata, dict):
            raw_grid = metadata.get("frame_time_grid")
            if isinstance(raw_grid, torch.Tensor):
                frame_time_grid = raw_grid.to(device=device, dtype=dtype)
                if frame_time_grid.ndim == 1:
                    frame_time_grid = frame_time_grid.unsqueeze(0)
                elif frame_time_grid.ndim > 2:
                    frame_time_grid = frame_time_grid.view(frame_time_grid.shape[0], -1)

                if frame_time_grid.shape[0] == 1 and batch_size > 1:
                    frame_time_grid = frame_time_grid.repeat(batch_size, 1)
                elif frame_time_grid.shape[0] != batch_size:
                    frame_time_grid = frame_time_grid[:1].repeat(batch_size, 1)

                if frame_time_grid.shape[1] == time_dim:
                    return frame_time_grid

        return base_grid.unsqueeze(0).repeat(batch_size, 1)

    def _time_semantics_metrics(self, metadata, ref_tensor):
        frame_delta_t = self._metadata_frame_delta_t(metadata, ref_tensor)
        frame_time_grid = self._metadata_frame_time_grid(metadata, ref_tensor)
        if frame_time_grid.shape[1] > 1:
            frame_time_span = frame_time_grid[:, -1] - frame_time_grid[:, 0]
        else:
            frame_time_span = torch.zeros(
                frame_time_grid.shape[0],
                device=frame_time_grid.device,
                dtype=frame_time_grid.dtype,
            )
        time_source_flag = 1.0 if isinstance(metadata, dict) and metadata.get("physics_time_source") == "video_frames" else 0.0
        return {
            "frame_delta_t": float(frame_delta_t.detach().float().mean().item()),
            "frame_time_span": float(frame_time_span.detach().float().mean().item()),
            "frame_count": float(frame_time_grid.shape[1]),
            "physics_time_source_video_frames": float(time_source_flag),
        }

    def _temporal_derivative(self, field, metadata=None, order=1):
        if order not in (1, 2):
            raise ValueError(f"Unsupported temporal derivative order: {order}")
        if field.ndim != 5:
            raise ValueError(
                f"Temporal derivatives expect [B, C, T, H, W] tensors; got {tuple(field.shape)}."
            )

        field = torch.clamp(field, min=-10.0, max=10.0)
        result = torch.zeros_like(field)
        time_dim = field.shape[2]
        if time_dim <= 1:
            return result

        frame_delta_t = self._metadata_frame_delta_t(metadata, field).view(field.shape[0], 1, 1, 1, 1)
        frame_delta_t = torch.clamp(frame_delta_t, min=1e-6)

        if order == 1:
            if time_dim == 2:
                edge = (field[:, :, 1:2] - field[:, :, 0:1]) / frame_delta_t
                result[:, :, 0:1] = edge
                result[:, :, 1:2] = edge
            else:
                result[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2.0 * frame_delta_t)
                result[:, :, 0:1] = (
                    -3.0 * field[:, :, 0:1]
                    + 4.0 * field[:, :, 1:2]
                    - field[:, :, 2:3]
                ) / (2.0 * frame_delta_t)
                result[:, :, -1:] = (
                    3.0 * field[:, :, -1:]
                    - 4.0 * field[:, :, -2:-1]
                    + field[:, :, -3:-2]
                ) / (2.0 * frame_delta_t)
        else:
            frame_delta_t_sq = frame_delta_t * frame_delta_t
            if time_dim == 2:
                return result
            if time_dim == 3:
                center = (
                    field[:, :, 2:3]
                    - 2.0 * field[:, :, 1:2]
                    + field[:, :, 0:1]
                ) / frame_delta_t_sq
                result[:, :, 0:1] = center
                result[:, :, 1:2] = center
                result[:, :, 2:3] = center
            else:
                result[:, :, 1:-1] = (
                    field[:, :, 2:]
                    - 2.0 * field[:, :, 1:-1]
                    + field[:, :, :-2]
                ) / frame_delta_t_sq
                result[:, :, 0:1] = (
                    2.0 * field[:, :, 0:1]
                    - 5.0 * field[:, :, 1:2]
                    + 4.0 * field[:, :, 2:3]
                    - field[:, :, 3:4]
                ) / frame_delta_t_sq
                result[:, :, -1:] = (
                    2.0 * field[:, :, -1:]
                    - 5.0 * field[:, :, -2:-1]
                    + 4.0 * field[:, :, -3:-2]
                    - field[:, :, -4:-3]
                ) / frame_delta_t_sq

        return torch.clamp(result, min=-100.0, max=100.0)

    def _field_family_diagnostics(self, fields, metadata=None):
        if not fields:
            return {}

        info = {}
        total_temporal_energy = 0.0
        total_spatial_energy = 0.0

        with torch.no_grad():
            for field_name, field in fields.items():
                field_ref = field.detach()
                dt = self._temporal_derivative(field_ref, metadata=metadata, order=1)
                ddt = self._temporal_derivative(field_ref, metadata=metadata, order=2)
                temporal_energy = self._weighted_square_mean(
                    dt,
                    metadata=metadata,
                    ref_tensor=field_ref,
                ) + self._weighted_square_mean(
                    ddt,
                    metadata=metadata,
                    ref_tensor=field_ref,
                )
                spatial_energy = self._spatial_gradient_energy(field_ref, metadata=metadata)
                dt_norm = self._weighted_mean(
                    torch.abs(dt),
                    metadata=metadata,
                    ref_tensor=field_ref,
                )
                ddt_norm = self._weighted_mean(
                    torch.abs(ddt),
                    metadata=metadata,
                    ref_tensor=field_ref,
                )

                safe_name = str(field_name).strip().lower()
                info[f"temporal_dt_norm_{safe_name}"] = float(dt_norm.detach().item())
                info[f"temporal_ddt_norm_{safe_name}"] = float(ddt_norm.detach().item())
                info[f"spatial_energy_{safe_name}"] = float(spatial_energy.detach().item())
                total_temporal_energy += float(temporal_energy.detach().item())
                total_spatial_energy += float(spatial_energy.detach().item())

        info["temporal_energy_total"] = float(total_temporal_energy)
        info["spatial_energy_total"] = float(total_spatial_energy)
        info["temporal_to_spatial_energy_ratio"] = float(
            total_temporal_energy / max(total_spatial_energy, 1e-8)
        )
        info.update(self._time_semantics_metrics(metadata, next(iter(fields.values()))))
        return info

    def _temporal_alignment_loss(self, source, target, metadata=None):
        if target.shape[2] <= 1:
            return self._zero_loss(target)
        source_dt = self._temporal_derivative(source, metadata=metadata, order=1)
        target_ref = target.to(dtype=source_dt.dtype)
        return self._weighted_square_mean(source_dt - target_ref, metadata=metadata, ref_tensor=target_ref)

    def _field_match_loss(self, lhs, rhs, metadata=None):
        lhs_scalar = self._field_mean(lhs)
        rhs_scalar = self._field_mean(rhs)
        return self._weighted_square_mean(lhs_scalar - rhs_scalar, metadata=metadata, ref_tensor=lhs_scalar)

    def _wave_equation_loss(self, field, metadata=None):
        second_dt = self._temporal_derivative(field, metadata=metadata, order=2)
        lap = self._laplacian_field(field, metadata=metadata)
        wave_residual = second_dt + lap
        return self._weighted_square_mean(wave_residual, metadata=metadata, ref_tensor=field)

    def set_conditioning_enabled(self, enabled=True):
        """启用/禁用 metadata 条件化调制"""
        self.enable_conditioning = bool(enabled)

    def _clamp_param_grad(self, param_name, grad):
        """Clamp gradients to prevent parameter explosion"""
        if grad is not None:
            return torch.clamp(grad, min=-0.1, max=0.1)
        return grad

    def _clamp_parameters(self):
        """Clamp physical parameters to valid ranges"""
        with torch.no_grad():
            self.nu.data = torch.clamp(self.nu.data, min=0.001, max=0.1)
            self.rho.data = torch.clamp(self.rho.data, min=0.1, max=10.0)
            self.lambda_lame.data = torch.clamp(self.lambda_lame.data, min=0.1, max=10.0)
            self.mu.data = torch.clamp(self.mu.data, min=0.1, max=10.0)
            self.friction_coef.data = torch.clamp(self.friction_coef.data, min=0.01, max=1.0)

    def _ensure_finite_loss(self, loss, info, method_name="unknown"):
        """Ensure loss and info values are finite"""
        if torch.isnan(loss) or torch.isinf(loss):
            self._raise_invalid(method_name, loss)
        else:
            loss = torch.clamp(loss, min=0.0, max=100.0)

        # Clean up info dict
        clean_info = {}
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v) or torch.isinf(v):
                    clean_info[k] = 0.0
                else:
                    clean_info[k] = float(torch.clamp(v, min=-100.0, max=100.0).detach().item())
            elif isinstance(v, (int, float)):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    clean_info[k] = 0.0
                else:
                    clean_info[k] = float(v)
            else:
                clean_info[k] = v

        return loss, clean_info
    
    def _base_fluid_terms(self, z, v, metadata=None):
        """
        流体基础项：连续性 + 粘性 + 涡量平滑。
        仅供现象级 residual 复用。
        """
        del z
        self._clamp_parameters()  # Ensure parameters are in valid range
        scales = self._conditioned_scales(metadata, v)
        div_v = self.diff_ops.compute_divergence(v)
        loss_continuity = self._weighted_square_mean(div_v, metadata=metadata, ref_tensor=v) * scales["s0"]

        laplacian_v = self.diff_ops.compute_laplacian(v)
        # Clamp nu to prevent numerical issues
        nu_safe = torch.clamp(self.nu, min=0.001, max=0.1)
        loss_viscosity = -self._weighted_mean(nu_safe * laplacian_v * v, metadata=metadata, ref_tensor=v) * scales["s1"]

        curl_v = self.diff_ops.compute_curl_2d(v)
        if curl_v.shape[2] > 1:
            curl_dt = curl_v[:, :, 1:] - curl_v[:, :, :-1]
            loss_vorticity = self._weighted_square_mean(curl_dt, metadata=metadata, ref_tensor=v) * 0.1 * scales["s2"]
        else:
            loss_vorticity = self._zero_loss(v)

        # 安全处理各项损失
        loss_continuity = self._safe_loss(loss_continuity)
        loss_viscosity = self._safe_loss(loss_viscosity)
        loss_vorticity = self._safe_loss(loss_vorticity)

        total_loss = loss_continuity + loss_viscosity * 0.1 + loss_vorticity
        total_loss = self._safe_loss(total_loss)
        return total_loss, {
            "continuity": float(loss_continuity.detach().item()),
            "viscosity": float(loss_viscosity.detach().item()),
            "vorticity": float(loss_vorticity.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
        }

    def _base_rigid_terms(self, z, v, metadata=None):
        """
        刚体基础项：局部不变形 + 动量平滑。
        仅供现象级 residual 复用。
        """
        del z
        scales = self._conditioned_scales(metadata, v)
        _, _, T, H, W = v.shape

        if H > 1 and W > 1:
            dv_dh = v[:, :, :, 1:, :] - v[:, :, :, :-1, :]
            dv_dw = v[:, :, :, :, 1:] - v[:, :, :, :, :-1]
            loss_rigidity = (
                self._weighted_square_mean(dv_dh, metadata=metadata, ref_tensor=v)
                + self._weighted_square_mean(dv_dw, metadata=metadata, ref_tensor=v)
            ) * scales["s0"]
        else:
            loss_rigidity = self._zero_loss(v)

        if T > 1:
            dv_dt = v[:, :, 1:] - v[:, :, :-1]
            loss_momentum = self._weighted_square_mean(dv_dt, metadata=metadata, ref_tensor=v) * 0.1 * scales["s1"]
        else:
            loss_momentum = self._zero_loss(v)

        # 安全处理各项损失
        loss_rigidity = self._safe_loss(loss_rigidity)
        loss_momentum = self._safe_loss(loss_momentum)

        total_loss = loss_rigidity + loss_momentum
        total_loss = self._safe_loss(total_loss)
        return total_loss, {
            "rigidity": float(loss_rigidity.detach().item()),
            "momentum": float(loss_momentum.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
        }

    def _base_elastic_terms(self, z, v, metadata=None):
        """
        弹性基础项：应变能 + 二阶时间加速度。
        仅供现象级 residual 复用。
        """
        del z
        self._clamp_parameters()  # Ensure parameters are in valid range
        scales = self._conditioned_scales(metadata, v)
        laplacian_v = self.diff_ops.compute_laplacian(v)

        # Clamp Lamé parameters to prevent numerical issues
        lambda_safe = torch.clamp(self.lambda_lame, min=0.1, max=10.0)
        mu_safe = torch.clamp(self.mu, min=0.1, max=10.0)

        strain_energy = self._weighted_mean(
            (lambda_safe + 2 * mu_safe) * laplacian_v * v,
            metadata=metadata,
            ref_tensor=v,
        ) * scales["s0"]
        loss_elastic = -strain_energy * 0.01

        if v.shape[2] > 2:
            acceleration = v[:, :, 2:] - 2 * v[:, :, 1:-1] + v[:, :, :-2]
            loss_wave = self._weighted_square_mean(acceleration, metadata=metadata, ref_tensor=v) * 0.1 * scales["s2"]
        else:
            loss_wave = self._zero_loss(v)

        loss_elastic = self._safe_loss(loss_elastic)
        loss_wave = self._safe_loss(loss_wave)

        total_loss = loss_elastic + loss_wave
        return total_loss, {
            "elastic": float(loss_elastic.detach().item()),
            "wave": float(loss_wave.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
        }

    def _base_particle_terms(self, z, v, metadata=None):
        """
        颗粒基础项：接触力 + 摩擦 + 速度突变。
        仅供现象级 residual 复用。
        """
        del z
        self._clamp_parameters()  # Ensure parameters are in valid range
        scales = self._conditioned_scales(metadata, v)
        grad_v = self.diff_ops.compute_laplacian(v)
        contact_force = self._weighted_mean(grad_v * v, metadata=metadata, ref_tensor=v) * 0.01 * scales["s0"]

        # Clamp friction coefficient to prevent numerical issues
        friction_safe = torch.clamp(self.friction_coef, min=0.01, max=1.0)
        friction_force = self._weighted_square_mean(v, metadata=metadata, ref_tensor=v) * friction_safe * scales["s1"]

        if v.shape[2] > 1:
            collision = self._weighted_square_mean(
                (v[:, :, 1:] - v[:, :, :-1]), metadata=metadata, ref_tensor=v
            ) * 0.1 * scales["s2"]
        else:
            collision = self._zero_loss(v)

        contact_force = self._safe_loss(contact_force)
        friction_force = self._safe_loss(friction_force)
        collision = self._safe_loss(collision)

        total_loss = contact_force + friction_force + collision
        return total_loss, {
            "contact": float(contact_force.detach().item()),
            "friction": float(friction_force.detach().item()),
            "collision": float(collision.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
        }

    def _base_material_residual(self, material_type, z, v, metadata=None):
        if material_type == "fluid":
            return self._base_fluid_terms(z, v, metadata=metadata)
        if material_type == "rigid":
            return self._base_rigid_terms(z, v, metadata=metadata)
        if material_type == "elastic":
            return self._base_elastic_terms(z, v, metadata=metadata)
        if material_type == "particle":
            return self._base_particle_terms(z, v, metadata=metadata)
        if material_type == "mixed":
            loss_f, info_f = self._base_fluid_terms(z, v, metadata=metadata)
            loss_r, info_r = self._base_rigid_terms(z, v, metadata=metadata)
            loss = (loss_f + loss_r) * 0.5
            info = {f"fluid_{k}": val for k, val in info_f.items()}
            info.update({f"rigid_{k}": val for k, val in info_r.items()})
            return loss, info
        return self._base_fluid_terms(z, v, metadata=metadata)

    def _gas_motion_terms(self, v, metadata=None):
        scales = self._conditioned_scales(metadata, v)
        div_v = self.diff_ops.compute_divergence(v)
        laplacian_v = self.diff_ops.compute_laplacian(v)
        curl_v = self.diff_ops.compute_curl_2d(v)
        field = self._scalar_field(v)
        field_dt = self._temporal_difference(field, order=1)

        compressibility = self._weighted_square_mean(div_v, metadata=metadata, ref_tensor=v) * 0.2 * scales["s0"]
        diffusion = self._weighted_square_mean(laplacian_v, metadata=metadata, ref_tensor=v) * 0.08 * scales["s1"]
        transport = (
            self._weighted_square_mean(field_dt, metadata=metadata, ref_tensor=v) * 0.1 * scales["s2"]
            if field_dt is not None else self._zero_loss(v)
        )
        if curl_v.shape[2] > 1:
            curl_dt = curl_v[:, :, 1:] - curl_v[:, :, :-1]
            vorticity = self._weighted_square_mean(curl_dt, metadata=metadata, ref_tensor=v) * 0.05 * scales["s3"]
        else:
            vorticity = self._zero_loss(v)

        total_loss = compressibility + diffusion + transport + vorticity
        return total_loss, {
            "compressibility": float(compressibility.detach().item()),
            "gas_diffusion": float(diffusion.detach().item()),
            "gas_transport": float(transport.detach().item()),
            "gas_vorticity": float(vorticity.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
            "cond_s3": float(scales["s3"].detach().item()),
        }

    def _phase_transition_context(self, z, v, metadata=None):
        scales = self._conditioned_scales(metadata, v)
        liquid_loss, _ = self.liquid_motion_residual(z, v, metadata=metadata)
        gas_loss, _ = self.gas_motion_residual(z, v, metadata=metadata)
        particle_loss, _ = self._base_particle_terms(z, v, metadata=metadata)
        field = self._scalar_field(v)
        field_dt = self._temporal_difference(field, order=1)
        lap_field = self.diff_ops.compute_laplacian(field)
        div_v = self.diff_ops.compute_divergence(v)
        temporal_transition = (
            self._weighted_square_mean(field_dt, metadata=metadata, ref_tensor=v) * 0.08 * scales["s2"]
            if field_dt is not None else self._zero_loss(v)
        )
        interface_smoothness = self._weighted_square_mean(lap_field, metadata=metadata, ref_tensor=v) * 0.05 * scales["s3"]
        return {
            "scales": scales,
            "liquid_loss": liquid_loss,
            "gas_loss": gas_loss,
            "particle_loss": particle_loss,
            "temporal_transition": temporal_transition,
            "interface_smoothness": interface_smoothness,
            "div_v": div_v,
        }

    def _optical_context(self, v, metadata=None):
        scales = self._conditioned_scales(metadata, v)
        field = self._scalar_field(v)
        lap_field = self.diff_ops.compute_laplacian(field)
        grad_energy = self._spatial_gradient_energy(field, metadata=metadata)
        field_dt = self._temporal_difference(field, order=1)
        field_ddt = self._temporal_difference(field, order=2)
        propagation = self._weighted_square_mean(lap_field, metadata=metadata, ref_tensor=v) * 0.1 * scales["s0"]
        temporal_coherence = (
            self._weighted_square_mean(field_dt, metadata=metadata, ref_tensor=v) * 0.05 * scales["s1"]
            if field_dt is not None else self._zero_loss(v)
        )
        return {
            "scales": scales,
            "field": field,
            "lap_field": lap_field,
            "grad_energy": grad_energy,
            "field_dt": field_dt,
            "field_ddt": field_ddt,
            "propagation": propagation,
            "temporal_coherence": temporal_coherence,
        }

    def _fallback_material_residual(self, material_type, z, v, metadata=None):
        return self._base_material_residual(material_type, z, v, metadata=metadata)

    def rigid_body_motion_residual(self, z, v, t=None, metadata=None):
        del t
        base_loss, base_info = self._base_rigid_terms(z, v, metadata=metadata)
        scales = self._conditioned_scales(metadata, v)
        spatial_coherence = self._spatial_gradient_energy(v, metadata=metadata) * 0.05 * scales["s0"]
        centered_v = v - v.mean(dim=(3, 4), keepdim=True)
        rigid_coherence = self._weighted_square_mean(centered_v, metadata=metadata, ref_tensor=v) * 0.05 * scales["s1"]
        total_loss = base_loss + spatial_coherence + rigid_coherence

        # Ensure finite
        total_loss = torch.clamp(total_loss, min=0.0, max=100.0)
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            self._raise_invalid("rigid_body_motion_residual.total_loss", total_loss)

        info = dict(base_info)
        info.update({
            "mechanical_spatial": float(torch.clamp(spatial_coherence, min=0.0, max=100.0).detach().item()) if not (torch.isnan(spatial_coherence) or torch.isinf(spatial_coherence)) else 0.0,
            "rigid_coherence": float(torch.clamp(rigid_coherence, min=0.0, max=100.0).detach().item()) if not (torch.isnan(rigid_coherence) or torch.isinf(rigid_coherence)) else 0.0,
        })
        return total_loss, info

    def collision_residual(self, z, v, t=None, metadata=None):
        del t
        base_loss, base_info = self._base_rigid_terms(z, v, metadata=metadata)
        scales = self._conditioned_scales(metadata, v)
        spatial_coherence = self._spatial_gradient_energy(v, metadata=metadata) * 0.05 * scales["s0"]
        second_dt = self._temporal_difference(v, order=2)
        if second_dt is not None:
            collision_impulse = self._weighted_square_mean(second_dt, metadata=metadata, ref_tensor=v) * 0.2 * scales["s2"]
        else:
            collision_impulse = self._zero_loss(v)
        total_loss = base_loss + spatial_coherence + collision_impulse
        info = dict(base_info)
        info.update({
            "mechanical_spatial": float(spatial_coherence.detach().item()),
            "collision_impulse": float(collision_impulse.detach().item()),
        })
        return total_loss, info

    def liquid_motion_residual(self, z, v, t=None, metadata=None):
        del t
        base_loss, base_info = self._base_fluid_terms(z, v, metadata=metadata)
        base_loss = torch.clamp(base_loss, min=0.0, max=100.0)
        if torch.isnan(base_loss) or torch.isinf(base_loss):
            self._raise_invalid("liquid_motion_residual.base_loss", base_loss)

        scales = self._conditioned_scales(metadata, v)
        field = self._scalar_field(v)
        field_dt = self._temporal_difference(field, order=1)

        if field_dt is not None:
            surface_transport = self._weighted_square_mean(field_dt, metadata=metadata, ref_tensor=v) * 0.05 * scales["s2"]
            surface_transport = torch.clamp(surface_transport, min=0.0, max=100.0)
            if torch.isnan(surface_transport) or torch.isinf(surface_transport):
                self._raise_invalid("liquid_motion_residual.surface_transport", surface_transport)
        else:
            surface_transport = self._zero_loss(v)

        total_loss = base_loss + surface_transport
        total_loss = torch.clamp(total_loss, min=0.0, max=100.0)
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            self._raise_invalid("liquid_motion_residual.total_loss", total_loss)

        info = dict(base_info)
        info["surface_transport"] = float(surface_transport.detach().item()) if not (torch.isnan(surface_transport) or torch.isinf(surface_transport)) else 0.0
        return total_loss, info

    def gas_motion_residual(self, z, v, t=None, metadata=None):
        del z, t
        return self._gas_motion_terms(v, metadata=metadata)

    def elastic_motion_residual(self, z, v, t=None, metadata=None):
        del t
        base_loss, base_info = self._base_elastic_terms(z, v, metadata=metadata)
        scales = self._conditioned_scales(metadata, v)
        first_dt = self._temporal_difference(v, order=1)
        if first_dt is not None:
            elastic_wave = self._weighted_square_mean(first_dt, metadata=metadata, ref_tensor=v) * 0.1 * scales["s2"]
        else:
            elastic_wave = self._zero_loss(v)
        total_loss = base_loss + elastic_wave
        info = dict(base_info)
        info["elastic_wave"] = float(elastic_wave.detach().item())
        return total_loss, info

    def deformation_residual(self, z, v, t=None, metadata=None):
        del t
        base_loss, base_info = self._base_elastic_terms(z, v, metadata=metadata)
        scales = self._conditioned_scales(metadata, v)
        strain = self.diff_ops.compute_laplacian(v)
        strain_dt = self._temporal_difference(strain, order=1)
        if strain_dt is not None:
            deformation_flow = self._weighted_square_mean(strain_dt, metadata=metadata, ref_tensor=v) * 0.1 * scales["s2"]
        else:
            deformation_flow = self._zero_loss(v)
        total_loss = base_loss + deformation_flow
        info = dict(base_info)
        info["deformation_flow"] = float(deformation_flow.detach().item())
        return total_loss, info

    def melting_residual(self, z, v, t=None, metadata=None):
        del t
        ctx = self._phase_transition_context(z, v, metadata=metadata)
        base_loss = 0.65 * ctx["liquid_loss"] + 0.35 * ctx["particle_loss"]
        phenomenon_term = ctx["temporal_transition"] + ctx["interface_smoothness"]
        total_loss = base_loss + phenomenon_term
        return total_loss, {
            "phase_base": float(base_loss.detach().item()),
            "phase_transition": float(ctx["temporal_transition"].detach().item()),
            "phase_interface": float(ctx["interface_smoothness"].detach().item()),
            "phase_special": float(phenomenon_term.detach().item()),
            "cond_s2": float(ctx["scales"]["s2"].detach().item()),
            "cond_s3": float(ctx["scales"]["s3"].detach().item()),
        }

    def solidification_residual(self, z, v, t=None, metadata=None):
        del t
        ctx = self._phase_transition_context(z, v, metadata=metadata)
        base_loss = 0.35 * ctx["liquid_loss"] + 0.65 * ctx["particle_loss"]
        phenomenon_term = ctx["interface_smoothness"] * 1.5
        total_loss = base_loss + phenomenon_term
        return total_loss, {
            "phase_base": float(base_loss.detach().item()),
            "phase_transition": float(ctx["temporal_transition"].detach().item()),
            "phase_interface": float(ctx["interface_smoothness"].detach().item()),
            "phase_special": float(phenomenon_term.detach().item()),
            "cond_s2": float(ctx["scales"]["s2"].detach().item()),
            "cond_s3": float(ctx["scales"]["s3"].detach().item()),
        }

    def vaporization_residual(self, z, v, t=None, metadata=None):
        del t
        ctx = self._phase_transition_context(z, v, metadata=metadata)
        base_loss = 0.5 * ctx["liquid_loss"] + 0.5 * ctx["gas_loss"]
        phenomenon_term = (
            ctx["temporal_transition"]
            + self._weighted_square_mean(F.relu(-ctx["div_v"]), metadata=metadata, ref_tensor=v) * 0.15 * ctx["scales"]["s3"]
        )
        total_loss = base_loss + phenomenon_term
        return total_loss, {
            "phase_base": float(base_loss.detach().item()),
            "phase_transition": float(ctx["temporal_transition"].detach().item()),
            "phase_interface": float(ctx["interface_smoothness"].detach().item()),
            "phase_special": float(phenomenon_term.detach().item()),
            "cond_s2": float(ctx["scales"]["s2"].detach().item()),
            "cond_s3": float(ctx["scales"]["s3"].detach().item()),
        }

    def liquefaction_residual(self, z, v, t=None, metadata=None):
        del t
        ctx = self._phase_transition_context(z, v, metadata=metadata)
        base_loss = 0.7 * ctx["liquid_loss"] + 0.3 * ctx["particle_loss"]
        phenomenon_term = (
            ctx["temporal_transition"]
            + self._weighted_square_mean(ctx["div_v"], metadata=metadata, ref_tensor=v) * 0.05 * ctx["scales"]["s3"]
        )
        total_loss = base_loss + phenomenon_term
        return total_loss, {
            "phase_base": float(base_loss.detach().item()),
            "phase_transition": float(ctx["temporal_transition"].detach().item()),
            "phase_interface": float(ctx["interface_smoothness"].detach().item()),
            "phase_special": float(phenomenon_term.detach().item()),
            "cond_s2": float(ctx["scales"]["s2"].detach().item()),
            "cond_s3": float(ctx["scales"]["s3"].detach().item()),
        }

    def combustion_residual(self, z, v, t=None, metadata=None):
        del t
        gas_loss, gas_info = self.gas_motion_residual(z, v, metadata=metadata)
        scales = self._conditioned_scales(metadata, v)
        field = self._scalar_field(v)
        field_dt = self._temporal_difference(field, order=1)
        lap_field = self.diff_ops.compute_laplacian(field)
        if field_dt is not None:
            heat_release = self._weighted_square_mean(F.relu(field_dt), metadata=metadata, ref_tensor=v) * 0.1 * scales["s2"]
        else:
            heat_release = self._zero_loss(v)
        reaction_diffusion = self._weighted_square_mean(lap_field, metadata=metadata, ref_tensor=v) * 0.05 * scales["s3"]
        total_loss = gas_loss + heat_release + reaction_diffusion
        info = dict(gas_info)
        info.update({
            "combustion_heat_release": float(heat_release.detach().item()),
            "combustion_reaction_diffusion": float(reaction_diffusion.detach().item()),
            "cond_s3": float(scales["s3"].detach().item()),
        })
        return total_loss, info

    def explosion_residual(self, z, v, t=None, metadata=None):
        del z, t
        scales = self._conditioned_scales(metadata, v)
        div_v = self.diff_ops.compute_divergence(v)
        laplacian_v = self.diff_ops.compute_laplacian(v)
        field = self._scalar_field(v)
        field_ddt = self._temporal_difference(field, order=2)
        expansion = self._weighted_square_mean(F.relu(-div_v), metadata=metadata, ref_tensor=v) * 0.5 * scales["s0"]
        if field_ddt is not None:
            burst = self._weighted_square_mean(field_ddt, metadata=metadata, ref_tensor=v) * 0.2 * scales["s1"]
        else:
            burst = self._zero_loss(v)
        diffusion = self._weighted_square_mean(laplacian_v, metadata=metadata, ref_tensor=v) * 0.05 * scales["s2"]
        total_loss = expansion + burst + diffusion
        return total_loss, {
            "expansion_bias": float(expansion.detach().item()),
            "burst_response": float(burst.detach().item()),
            "explosion_diffusion": float(diffusion.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
        }

    def reflection_residual(self, z, v, t=None, metadata=None):
        del z, t
        ctx = self._optical_context(v, metadata=metadata)
        reflection_symmetry = self._weighted_square_mean(
            (ctx["field"] - torch.flip(ctx["field"], dims=[4])),
            metadata=metadata,
            ref_tensor=v,
        ) * 0.02 * ctx["scales"]["s2"]
        total_loss = ctx["propagation"] + ctx["temporal_coherence"] + reflection_symmetry
        return total_loss, {
            "optical_propagation": float(ctx["propagation"].detach().item()),
            "optical_temporal": float(ctx["temporal_coherence"].detach().item()),
            "reflection_symmetry": float(reflection_symmetry.detach().item()),
            "cond_s0": float(ctx["scales"]["s0"].detach().item()),
            "cond_s1": float(ctx["scales"]["s1"].detach().item()),
            "cond_s2": float(ctx["scales"]["s2"].detach().item()),
        }

    def refraction_residual(self, z, v, t=None, metadata=None):
        del z, t
        ctx = self._optical_context(v, metadata=metadata)
        refraction_bending = (
            ctx["grad_energy"] + self._weighted_square_mean(ctx["lap_field"], metadata=metadata, ref_tensor=v)
        ) * 0.08 * ctx["scales"]["s2"]
        total_loss = ctx["propagation"] + ctx["temporal_coherence"] + refraction_bending
        return total_loss, {
            "optical_propagation": float(ctx["propagation"].detach().item()),
            "optical_temporal": float(ctx["temporal_coherence"].detach().item()),
            "refraction_bending": float(refraction_bending.detach().item()),
            "cond_s0": float(ctx["scales"]["s0"].detach().item()),
            "cond_s1": float(ctx["scales"]["s1"].detach().item()),
            "cond_s2": float(ctx["scales"]["s2"].detach().item()),
        }

    def scattering_residual(self, z, v, t=None, metadata=None):
        del z, t
        ctx = self._optical_context(v, metadata=metadata)
        scattering_diffusion = ctx["grad_energy"] * 0.2 * ctx["scales"]["s2"]
        total_loss = ctx["propagation"] + ctx["temporal_coherence"] + scattering_diffusion
        return total_loss, {
            "optical_propagation": float(ctx["propagation"].detach().item()),
            "optical_temporal": float(ctx["temporal_coherence"].detach().item()),
            "scattering_diffusion": float(scattering_diffusion.detach().item()),
            "cond_s0": float(ctx["scales"]["s0"].detach().item()),
            "cond_s1": float(ctx["scales"]["s1"].detach().item()),
            "cond_s2": float(ctx["scales"]["s2"].detach().item()),
        }

    def interference_diffraction_residual(self, z, v, t=None, metadata=None):
        del z, t
        ctx = self._optical_context(v, metadata=metadata)
        if ctx["field_ddt"] is not None:
            interference_wave = (
                self._weighted_square_mean(ctx["field_ddt"], metadata=metadata, ref_tensor=v) * 0.1
                + self._weighted_square_mean(ctx["lap_field"], metadata=metadata, ref_tensor=v) * 0.1
            ) * ctx["scales"]["s2"]
        else:
            interference_wave = self._weighted_square_mean(ctx["lap_field"], metadata=metadata, ref_tensor=v) * 0.1 * ctx["scales"]["s2"]
        total_loss = ctx["propagation"] + ctx["temporal_coherence"] + interference_wave
        return total_loss, {
            "optical_propagation": float(ctx["propagation"].detach().item()),
            "optical_temporal": float(ctx["temporal_coherence"].detach().item()),
            "interference_wave": float(interference_wave.detach().item()),
            "cond_s0": float(ctx["scales"]["s0"].detach().item()),
            "cond_s1": float(ctx["scales"]["s1"].detach().item()),
            "cond_s2": float(ctx["scales"]["s2"].detach().item()),
        }

    def unnatural_light_source_residual(self, z, v, t=None, metadata=None):
        del z, t
        ctx = self._optical_context(v, metadata=metadata)
        localized_source = ctx["field"] - ctx["field"].mean(dim=(3, 4), keepdim=True)
        source_localization = self._weighted_square_mean(localized_source, metadata=metadata, ref_tensor=v) * 0.05 * ctx["scales"]["s2"]
        total_loss = ctx["propagation"] + ctx["temporal_coherence"] + source_localization
        return total_loss, {
            "optical_propagation": float(ctx["propagation"].detach().item()),
            "optical_temporal": float(ctx["temporal_coherence"].detach().item()),
            "source_localization": float(source_localization.detach().item()),
            "cond_s0": float(ctx["scales"]["s0"].detach().item()),
            "cond_s1": float(ctx["scales"]["s1"].detach().item()),
            "cond_s2": float(ctx["scales"]["s2"].detach().item()),
        }

    # ========================================================================
    # 10 Physics-Based Phenomenon Residuals (Aligned with Table 1)
    # ========================================================================

    def rigid_body_residual(self, fields_or_bank, v=None, t=None, metadata=None):
        del t
        fields = self._coerce_field_dict(fields_or_bank, v)
        d, u, eps, sigma = (fields[name] for name in EXPERT_FIELD_RECIPES["Rigid Body"])
        scales = self._conditioned_scales(metadata, u)

        kinematic = self._temporal_alignment_loss(d, u, metadata=metadata) * scales["s0"]
        strain_energy = self._weighted_square_mean(eps, metadata=metadata, ref_tensor=eps) * scales["s1"]
        stress_strain = self._field_match_loss(sigma, eps, metadata=metadata) * scales["s2"]
        rigid_velocity = self._spatial_gradient_energy(u, metadata=metadata) * scales["s3"]

        total_loss = torch.clamp(kinematic + strain_energy + stress_strain + rigid_velocity, min=0.0, max=100.0)
        return total_loss, {
            "rigid_kinematic": float(kinematic.detach().item()),
            "rigid_strain_energy": float(strain_energy.detach().item()),
            "rigid_stress_strain": float(stress_strain.detach().item()),
            "rigid_velocity_smoothness": float(rigid_velocity.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
            "cond_s3": float(scales["s3"].detach().item()),
            **self._field_family_diagnostics({"d": d, "u": u}, metadata=metadata),
        }

    def elastic_residual(self, fields_or_bank, v=None, t=None, metadata=None):
        del t
        fields = self._coerce_field_dict(fields_or_bank, v)
        d, u, eps, sigma = (fields[name] for name in EXPERT_FIELD_RECIPES["Elastic"])
        scales = self._conditioned_scales(metadata, u)

        elastic_wave = self._wave_equation_loss(d, metadata=metadata) * scales["s0"]
        velocity_wave = self._wave_equation_loss(u, metadata=metadata) * scales["s1"]
        strain_match = self._field_match_loss(eps, d, metadata=metadata) * scales["s2"]
        stress_strain = self._field_match_loss(sigma, eps, metadata=metadata) * scales["s3"]

        total_loss = torch.clamp(elastic_wave + velocity_wave + strain_match + stress_strain, min=0.0, max=100.0)
        return total_loss, {
            "elastic_displacement_wave": float(elastic_wave.detach().item()),
            "elastic_velocity_wave": float(velocity_wave.detach().item()),
            "elastic_strain_match": float(strain_match.detach().item()),
            "elastic_stress_strain": float(stress_strain.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
            "cond_s3": float(scales["s3"].detach().item()),
            **self._field_family_diagnostics({"d": d, "u": u}, metadata=metadata),
        }

    def fluid_residual(self, fields_or_bank, v=None, t=None, metadata=None):
        del t
        fields = self._coerce_field_dict(fields_or_bank, v)
        u, p, rho = (fields[name] for name in EXPERT_FIELD_RECIPES["Fluid"])
        if u.shape[1] == 2:
            return self._only_u_fluid_residual(u, p, rho, metadata=metadata, prefix="fluid")
        scales = self._conditioned_scales(metadata, u)

        rho_dt = self._temporal_derivative(rho, metadata=metadata, order=1)
        div_u = self._divergence_field(u)
        continuity = self._weighted_square_mean(
            self._field_mean(rho_dt) + div_u,
            metadata=metadata,
            ref_tensor=div_u,
        ) * scales["s0"]
        pressure_smoothness = self._weighted_square_mean(
            self._laplacian_field(p, metadata=metadata), metadata=metadata, ref_tensor=p
        ) * scales["s1"]
        vorticity_term = self._weighted_square_mean(
            self._vorticity_field(u), metadata=metadata, ref_tensor=u
        ) * scales["s2"]
        density_pressure = self._field_match_loss(rho, p, metadata=metadata) * scales["s3"]

        total_loss = torch.clamp(continuity + pressure_smoothness + vorticity_term + density_pressure, min=0.0, max=100.0)
        return total_loss, {
            "fluid_continuity": float(continuity.detach().item()),
            "fluid_pressure_smoothness": float(pressure_smoothness.detach().item()),
            "fluid_vorticity": float(vorticity_term.detach().item()),
            "fluid_density_pressure": float(density_pressure.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
            "cond_s3": float(scales["s3"].detach().item()),
            **self._field_family_diagnostics({"u": u, "p": p, "rho": rho}, metadata=metadata),
        }

    def compressible_flow_residual(self, fields_or_bank, v=None, t=None, metadata=None):
        del t
        fields = self._coerce_field_dict(fields_or_bank, v)
        u, p, rho = (fields[name] for name in EXPERT_FIELD_RECIPES["Compressible Flow"])
        if u.shape[1] == 2:
            return self._only_u_fluid_residual(u, p, rho, metadata=metadata, prefix="compressible")
        scales = self._conditioned_scales(metadata, u)

        div_u = self._divergence_field(u)
        rho_dt = self._temporal_derivative(rho, metadata=metadata, order=1)
        mass_conservation = self._weighted_square_mean(
            self._field_mean(rho_dt) + div_u,
            metadata=metadata,
            ref_tensor=div_u,
        ) * scales["s0"]
        div_dt = self._temporal_derivative(div_u, metadata=metadata, order=1)
        compression_dynamics = self._weighted_square_mean(
            div_dt,
            metadata=metadata,
            ref_tensor=div_u,
        ) * scales["s1"]
        pressure_density = self._field_match_loss(p, rho, metadata=metadata) * scales["s2"]
        grad = self._spatial_gradient_energy(u, metadata=metadata) * scales["s3"]

        total_loss = torch.clamp(mass_conservation + compression_dynamics + pressure_density + grad, min=0.0, max=100.0)
        return total_loss, {
            "compressible_mass": float(mass_conservation.detach().item()),
            "compressible_dynamics": float(compression_dynamics.detach().item()),
            "compressible_pressure_density": float(pressure_density.detach().item()),
            "compressible_gradient": float(grad.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
            "cond_s3": float(scales["s3"].detach().item()),
            **self._field_family_diagnostics({"u": u, "p": p, "rho": rho}, metadata=metadata),
        }

    def phase_change_residual(self, fields_or_bank, v=None, t=None, metadata=None):
        del t
        fields = self._coerce_field_dict(fields_or_bank, v)
        u, rho, T, alpha = (fields[name] for name in EXPERT_FIELD_RECIPES["Phase Change"])
        scales = self._conditioned_scales(metadata, u)

        alpha_dt = self._temporal_derivative(alpha, metadata=metadata, order=1)
        div_u = self._divergence_field(u)
        expansion = self._weighted_square_mean(
            self._field_mean(alpha_dt) + div_u,
            metadata=metadata,
            ref_tensor=div_u,
        ) * scales["s0"]
        temp_dt = self._temporal_derivative(T, metadata=metadata, order=1)
        latent_heat = self._weighted_square_mean(temp_dt, metadata=metadata, ref_tensor=T) * scales["s1"]
        temperature_smoothness = self._weighted_square_mean(
            self._laplacian_field(T, metadata=metadata), metadata=metadata, ref_tensor=T
        ) * scales["s2"]
        density_phase = self._field_match_loss(rho, alpha, metadata=metadata) * scales["s3"]

        total_loss = torch.clamp(expansion + latent_heat + temperature_smoothness + density_phase, min=0.0, max=100.0)
        return total_loss, {
            "phase_expansion": float(expansion.detach().item()),
            "phase_latent_heat": float(latent_heat.detach().item()),
            "phase_temperature_smoothness": float(temperature_smoothness.detach().item()),
            "phase_density_alpha": float(density_phase.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
            "cond_s3": float(scales["s3"].detach().item()),
            **self._field_family_diagnostics({"u": u, "rho": rho, "temperature": T, "alpha": alpha}, metadata=metadata),
        }

    def collision_contact_residual(self, fields_or_bank, v=None, t=None, metadata=None):
        del t
        fields = self._coerce_field_dict(fields_or_bank, v)
        d, u, j = (fields[name] for name in EXPERT_FIELD_RECIPES["Collision/Contact"])
        scales = self._conditioned_scales(metadata, u)

        kinematic = self._temporal_alignment_loss(d, u, metadata=metadata) * scales["s0"]
        impulse_exchange = self._temporal_alignment_loss(u, j, metadata=metadata) * scales["s1"]
        contact_compression = self._weighted_square_mean(
            self._divergence_field(d) + self._field_mean(j),
            metadata=metadata,
            ref_tensor=d,
        ) * scales["s2"]
        impact_propagation = self._weighted_square_mean(
            self._laplacian_field(j, metadata=metadata), metadata=metadata, ref_tensor=j
        ) * scales["s3"]

        total_loss = torch.clamp(kinematic + impulse_exchange + contact_compression + impact_propagation, min=0.0, max=100.0)
        return total_loss, {
            "collision_kinematic": float(kinematic.detach().item()),
            "collision_impulse_exchange": float(impulse_exchange.detach().item()),
            "collision_compression": float(contact_compression.detach().item()),
            "collision_impact": float(impact_propagation.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
            "cond_s3": float(scales["s3"].detach().item()),
            **self._field_family_diagnostics({"d": d, "u": u, "j": j}, metadata=metadata),
        }

    def granular_residual(self, fields_or_bank, v=None, t=None, metadata=None):
        del t
        fields = self._coerce_field_dict(fields_or_bank, v)
        u, rho, alpha, j = (fields[name] for name in EXPERT_FIELD_RECIPES["Granular"])
        scales = self._conditioned_scales(metadata, u)

        grad = self._spatial_gradient_energy(u, metadata=metadata) * scales["s0"]
        rho_dt = self._temporal_derivative(rho, metadata=metadata, order=1)
        div_u = self._divergence_field(u)
        transport = self._weighted_square_mean(
            self._field_mean(rho_dt) + div_u,
            metadata=metadata,
            ref_tensor=div_u,
        ) * scales["s1"]
        packing = self._field_match_loss(rho, alpha, metadata=metadata) * scales["s2"]
        contact_impulse = self._field_match_loss(j, u, metadata=metadata) * scales["s3"]

        total_loss = torch.clamp(grad + transport + packing + contact_impulse, min=0.0, max=100.0)
        return total_loss, {
            "granular_friction": float(grad.detach().item()),
            "granular_transport": float(transport.detach().item()),
            "granular_packing": float(packing.detach().item()),
            "granular_contact_impulse": float(contact_impulse.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
            "cond_s3": float(scales["s3"].detach().item()),
            **self._field_family_diagnostics({"u": u, "rho": rho, "alpha": alpha, "j": j}, metadata=metadata),
        }

    def fracture_residual(self, fields_or_bank, v=None, t=None, metadata=None):
        del t
        fields = self._coerce_field_dict(fields_or_bank, v)
        d, u, eps, sigma, j, D = (fields[name] for name in EXPERT_FIELD_RECIPES["Fracture"])
        scales = self._conditioned_scales(metadata, u)

        kinematic = self._temporal_alignment_loss(d, u, metadata=metadata) * scales["s0"]
        strain_energy = self._weighted_square_mean(eps, metadata=metadata, ref_tensor=eps) * scales["s1"]
        stress_strain = self._field_match_loss(sigma, eps, metadata=metadata) * scales["s2"]
        impact_transfer = self._field_match_loss(j, u, metadata=metadata) * scales["s3"]
        damage_dt = self._temporal_derivative(D, metadata=metadata, order=1)
        j_scalar = torch.abs(self._field_mean(j))
        damage_progress = self._weighted_square_mean(
            self._field_mean(damage_dt) - j_scalar,
            metadata=metadata,
            ref_tensor=j_scalar,
        )

        total_loss = torch.clamp(
            kinematic + strain_energy + stress_strain + impact_transfer + damage_progress,
            min=0.0,
            max=100.0,
        )
        return total_loss, {
            "fracture_kinematic": float(kinematic.detach().item()),
            "fracture_strain_energy": float(strain_energy.detach().item()),
            "fracture_stress_strain": float(stress_strain.detach().item()),
            "fracture_impact_transfer": float(impact_transfer.detach().item()),
            "fracture_damage_progress": float(damage_progress.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
            "cond_s3": float(scales["s3"].detach().item()),
            **self._field_family_diagnostics({"d": d, "u": u, "damage": D, "j": j}, metadata=metadata),
        }

    def thermal_residual(self, fields_or_bank, v=None, t=None, metadata=None):
        del t
        fields = self._coerce_field_dict(fields_or_bank, v)
        u, T = (fields[name] for name in EXPERT_FIELD_RECIPES["Thermal"])
        scales = self._conditioned_scales(metadata, u)

        temp_dt = self._temporal_derivative(T, metadata=metadata, order=1)
        temporal_diffusion = self._weighted_square_mean(temp_dt, metadata=metadata, ref_tensor=T) * scales["s0"]
        spatial_diffusion = self._weighted_square_mean(
            self._laplacian_field(T, metadata=metadata), metadata=metadata, ref_tensor=T
        ) * scales["s1"]
        thermal_advection = self._weighted_square_mean(
            self._divergence_field(u * self._field_mean(T).expand_as(u)),
            metadata=metadata,
            ref_tensor=u,
        ) * scales["s2"]

        total_loss = torch.clamp(temporal_diffusion + spatial_diffusion + thermal_advection, min=0.0, max=100.0)
        return total_loss, {
            "thermal_temporal": float(temporal_diffusion.detach().item()),
            "thermal_spatial": float(spatial_diffusion.detach().item()),
            "thermal_advection": float(thermal_advection.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
            **self._field_family_diagnostics({"u": u, "temperature": T}, metadata=metadata),
        }

    def optical_residual(self, fields_or_bank, v=None, t=None, metadata=None):
        del t
        fields = self._coerce_field_dict(fields_or_bank, v)
        psi, alpha = (fields[name] for name in EXPERT_FIELD_RECIPES["Optical"])
        scales = self._conditioned_scales(metadata, psi)

        wave_propagation = self._wave_equation_loss(psi, metadata=metadata) * scales["s0"]
        alpha_smoothness = self._weighted_square_mean(
            self._laplacian_field(alpha, metadata=metadata), metadata=metadata, ref_tensor=alpha
        ) * scales["s1"]
        interference = self._field_match_loss(psi, alpha, metadata=metadata) * scales["s2"]

        total_loss = torch.clamp(wave_propagation + alpha_smoothness + interference, min=0.0, max=100.0)
        return total_loss, {
            "optical_wave": float(wave_propagation.detach().item()),
            "optical_alpha_smoothness": float(alpha_smoothness.detach().item()),
            "optical_interference": float(interference.detach().item()),
            "cond_s0": float(scales["s0"].detach().item()),
            "cond_s1": float(scales["s1"].detach().item()),
            "cond_s2": float(scales["s2"].detach().item()),
            **self._field_family_diagnostics({"psi": psi, "alpha": alpha}, metadata=metadata),
        }


class MaterialClassifier(nn.Module):
    """从文本提示词识别材质类型"""
    
    def __init__(self):
        super().__init__()
        # 材质关键词
        self.material_keywords = {
            'fluid': ['water', 'liquid', 'fluid', 'flow', 'smoke', 'cloud', 'splash', 'pour'],
            'rigid': ['ball', 'box', 'cube', 'stone', 'rock', 'metal', 'rigid', 'fall', 'drop'],
            'elastic': ['cloth', 'rubber', 'elastic', 'bounce', 'deform', 'bend', 'stretch'],
            'particle': ['sand', 'dust', 'particle', 'grain', 'powder', 'debris']
        }
    
    def classify(self, text):
        """
        根据文本分类材质
        Returns:
            material_type: str, one of ['fluid', 'rigid', 'elastic', 'particle', 'mixed']
        """
        text_lower = text.lower()
        
        scores = {}
        for material, keywords in self.material_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[material] = score
        
        # 找到最高分
        max_score = max(scores.values())
        if max_score == 0:
            return 'fluid'  # 默认流体
        
        # 找到所有最高分的材质
        top_materials = [m for m, s in scores.items() if s == max_score]
        
        if len(top_materials) > 1:
            return 'mixed'
        else:
            return top_materials[0]
    
    def classify_batch(self, texts):
        """批量分类"""
        return [self.classify(text) for text in texts]
