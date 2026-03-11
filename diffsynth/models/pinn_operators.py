"""
Physics-Informed Neural Network Operators
微分算子和物理方程定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        parts = []
        
        # H 方向二阶导数: d²v/dH²
        if v.shape[3] > 2:
            d2v_dh2 = v[:, :, :, 2:, :] - 2 * v[:, :, :, 1:-1, :] + v[:, :, :, :-2, :]
            # pad 回原始 H 尺寸（上下各一行0）
            d2v_dh2 = F.pad(d2v_dh2, (0, 0, 1, 1))  # (W_left, W_right, H_top, H_bottom)
            parts.append(d2v_dh2)
        
        # W 方向二阶导数: d²v/dW²
        if v.shape[4] > 2:
            d2v_dw2 = v[:, :, :, :, 2:] - 2 * v[:, :, :, :, 1:-1] + v[:, :, :, :, :-2]
            d2v_dw2 = F.pad(d2v_dw2, (1, 1))  # pad W 方向
            parts.append(d2v_dw2)
        
        if len(parts) == 0:
            return v * 0.0  # 保持 grad_fn
        
        laplacian = sum(parts)
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
        # dv/dH
        dv_dh = v[:, :, :, 1:, :] - v[:, :, :, :-1, :]  # [B, C, T, H-1, W]
        # dv/dW
        dv_dw = v[:, :, :, :, 1:] - v[:, :, :, :, :-1]  # [B, C, T, H, W-1]
        
        # 对齐到相同的空间尺寸：取内部交叉区域
        dv_dh = dv_dh[:, :, :, :, :-1]  # [B, C, T, H-1, W-1]
        dv_dw = dv_dw[:, :, :, :-1, :]  # [B, C, T, H-1, W-1]
        
        # 跨通道近似：前半通道作为 x 分量，后半通道作为 y 分量
        C = v.shape[1]
        half_C = max(C // 2, 1)
        
        curl = dv_dw[:, :half_C].mean(dim=1, keepdim=True) - dv_dh[:, half_C:].mean(dim=1, keepdim=True)
        return curl


class MaterialPDEResiduals(nn.Module):
    """各种材质的PDE残差计算"""
    
    def __init__(self, num_phenomena=17, q_input_dim=64, n_numeric_dim=12):
        super().__init__()
        self.diff_ops = DifferentialOperators()
        self.num_phenomena = num_phenomena
        self.q_input_dim = q_input_dim
        self.n_numeric_dim = n_numeric_dim
        self.enable_conditioning = True
        
        # 物理参数（可学习）
        self.nu = nn.Parameter(torch.tensor(0.01))  # 粘度
        self.rho = nn.Parameter(torch.tensor(1.0))  # 密度
        self.lambda_lame = nn.Parameter(torch.tensor(1.0))  # 拉梅常数
        self.mu = nn.Parameter(torch.tensor(1.0))  # 剪切模量
        self.friction_coef = nn.Parameter(torch.tensor(0.1))  # 摩擦系数

        # 条件调制参数：初始为零，使模型一开始接近原始残差形式
        self.label_embedding = nn.Embedding(num_phenomena, 4)
        self.q_projector = nn.Linear(q_input_dim, 4, bias=False)
        self.n_projector = nn.Linear(n_numeric_dim, 4, bias=False)
        nn.init.zeros_(self.label_embedding.weight)
        nn.init.zeros_(self.q_projector.weight)
        nn.init.zeros_(self.n_projector.weight)

    @staticmethod
    def _zero_loss(v):
        return torch.mean(v ** 2) * 0.0

    @staticmethod
    def _metadata_label_name(metadata):
        if not isinstance(metadata, dict):
            return ""
        return str(metadata.get("label_name", "")).strip().lower()

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
        if tensor.shape[1] > target_dim:
            tensor = tensor[:, :target_dim]
        elif tensor.shape[1] < target_dim:
            pad = torch.zeros(batch_size, target_dim - tensor.shape[1], device=device, dtype=dtype)
            tensor = torch.cat([tensor, pad], dim=1)
        return tensor

    def _metadata_condition_vector(self, metadata, batch_size, device, dtype):
        if not isinstance(metadata, dict):
            return torch.zeros(4, device=device, dtype=dtype), 0.0, 0.0, 0.0

        label_ids = metadata.get("label_id")
        if label_ids is None:
            label_ids = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            label_ids = label_ids.to(device=device, dtype=torch.long).view(-1)
            if label_ids.numel() == 1 and batch_size > 1:
                label_ids = label_ids.repeat(batch_size)
            if label_ids.numel() != batch_size:
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
        ref = value if ref_tensor is None else ref_tensor
        mask = self._resolve_motion_mask(metadata, value, ref)
        if mask is None:
            return torch.mean(value)
        numer = torch.sum(value * mask)
        denom = torch.sum(mask) + 1e-6
        return numer / denom

    def _weighted_square_mean(self, value, metadata=None, ref_tensor=None):
        return self._weighted_mean(value ** 2, metadata=metadata, ref_tensor=ref_tensor)

    def _spatial_gradient_energy(self, v, metadata=None):
        parts = []
        if v.shape[3] > 1:
            parts.append(self._weighted_square_mean(v[:, :, :, 1:, :] - v[:, :, :, :-1, :], metadata=metadata, ref_tensor=v))
        if v.shape[4] > 1:
            parts.append(self._weighted_square_mean(v[:, :, :, :, 1:] - v[:, :, :, :, :-1], metadata=metadata, ref_tensor=v))
        if len(parts) == 0:
            return self._zero_loss(v)
        return sum(parts) / len(parts)

    def set_conditioning_enabled(self, enabled=True):
        """启用/禁用 metadata 条件化调制"""
        self.enable_conditioning = bool(enabled)
    
    def _base_fluid_terms(self, z, v, metadata=None):
        """
        流体基础项：连续性 + 粘性 + 涡量平滑。
        仅供现象级 residual 复用。
        """
        del z
        scales = self._conditioned_scales(metadata, v)
        div_v = self.diff_ops.compute_divergence(v)
        loss_continuity = self._weighted_square_mean(div_v, metadata=metadata, ref_tensor=v) * scales["s0"]

        laplacian_v = self.diff_ops.compute_laplacian(v)
        loss_viscosity = -self._weighted_mean(self.nu * laplacian_v * v, metadata=metadata, ref_tensor=v) * scales["s1"]

        curl_v = self.diff_ops.compute_curl_2d(v)
        if curl_v.shape[2] > 1:
            curl_dt = curl_v[:, :, 1:] - curl_v[:, :, :-1]
            loss_vorticity = self._weighted_square_mean(curl_dt, metadata=metadata, ref_tensor=v) * 0.1 * scales["s2"]
        else:
            loss_vorticity = self._zero_loss(v)

        total_loss = loss_continuity + loss_viscosity * 0.1 + loss_vorticity
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

        total_loss = loss_rigidity + loss_momentum
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
        scales = self._conditioned_scales(metadata, v)
        laplacian_v = self.diff_ops.compute_laplacian(v)
        strain_energy = self._weighted_mean(
            (self.lambda_lame + 2 * self.mu) * laplacian_v * v,
            metadata=metadata,
            ref_tensor=v,
        ) * scales["s0"]
        loss_elastic = -strain_energy * 0.01

        if v.shape[2] > 2:
            acceleration = v[:, :, 2:] - 2 * v[:, :, 1:-1] + v[:, :, :-2]
            loss_wave = self._weighted_square_mean(acceleration, metadata=metadata, ref_tensor=v) * 0.1 * scales["s2"]
        else:
            loss_wave = self._zero_loss(v)

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
        scales = self._conditioned_scales(metadata, v)
        grad_v = self.diff_ops.compute_laplacian(v)
        contact_force = self._weighted_mean(grad_v * v, metadata=metadata, ref_tensor=v) * 0.01 * scales["s0"]
        friction_force = self._weighted_square_mean(v, metadata=metadata, ref_tensor=v) * self.friction_coef * scales["s1"]

        if v.shape[2] > 1:
            collision = self._weighted_square_mean(
                (v[:, :, 1:] - v[:, :, :-1]), metadata=metadata, ref_tensor=v
            ) * 0.1 * scales["s2"]
        else:
            collision = self._zero_loss(v)

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
        info = dict(base_info)
        info.update({
            "mechanical_spatial": float(spatial_coherence.detach().item()),
            "rigid_coherence": float(rigid_coherence.detach().item()),
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
        scales = self._conditioned_scales(metadata, v)
        field = self._scalar_field(v)
        field_dt = self._temporal_difference(field, order=1)
        surface_transport = (
            self._weighted_square_mean(field_dt, metadata=metadata, ref_tensor=v) * 0.05 * scales["s2"]
            if field_dt is not None else self._zero_loss(v)
        )
        total_loss = base_loss + surface_transport
        info = dict(base_info)
        info["surface_transport"] = float(surface_transport.detach().item())
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
