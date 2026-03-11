"""
Physics-Informed Video Generation Training Script
物理约束视频生成训练脚本

使用方法:
    accelerate launch --config_file examples/wanvideo/pinn_training/accelerate_config_pinn.yaml \
        examples/wanvideo/pinn_training/train_pinn.py \
        --dataset_base_path data/example_video_dataset \
        --dataset_metadata_path data/example_video_dataset/metadata.csv \
        --height 480 --width 832 --num_frames 49 \
        --model_id_with_origin_paths "Wan-AI/Wan2.2-T2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
        --learning_rate 1e-5 \
        --num_epochs 2 \
        --output_path "./models/train/pinn_plugin" \
        --physics_weight 0.1 \
        --material_type auto
"""
import torch
import torch.nn.functional as F
import os
import json
import re
import hashlib
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.pinn_operators import MaterialPDEResiduals, MaterialClassifier
from diffsynth.models.pinn_adapter import PhysicsAdapter
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, ModelLogger, launch_training_task, wan_parser

os.environ["TOKENIZERS_PARALLELISM"] = "false"


PHENOMENON_LABELS = [
    "rigid body motion",
    "collision",
    "liquid motion",
    "gas motion",
    "elastic motion",
    "deformation",
    "melting",
    "solidification",
    "vaporization",
    "liquefaction",
    "combustion",
    "explosion",
    "reflection",
    "refraction",
    "scattering",
    "interference and diffraction",
    "unnatural light source",
]
PHENOMENON_TO_ID = {name: idx for idx, name in enumerate(PHENOMENON_LABELS)}
PHENOMENON_ALIAS = {
    "liquid_motion": "liquid motion",
    "rigid_body_motion": "rigid body motion",
    "elastic_motion": "elastic motion",
    "gas_motion": "gas motion",
    "interference_and_diffraction": "interference and diffraction",
    "unnatural_light_source": "unnatural light source",
}
PHENOMENON_TO_MATERIAL = {
    "rigid body motion": "rigid",
    "collision": "rigid",
    "liquid motion": "fluid",
    "gas motion": "fluid",
    "elastic motion": "elastic",
    "deformation": "elastic",
    "melting": "particle",
    "solidification": "particle",
    "vaporization": "fluid",
    "liquefaction": "fluid",
    "combustion": "fluid",
    "explosion": "particle",
    "reflection": "rigid",
    "refraction": "fluid",
    "scattering": "particle",
    "interference and diffraction": "rigid",
    "unnatural light source": "fluid",
}
PHENOMENON_TO_RESIDUAL_METHOD = {
    "rigid body motion": "rigid_body_motion_residual",
    "collision": "collision_residual",
    "liquid motion": "liquid_motion_residual",
    "gas motion": "gas_motion_residual",
    "elastic motion": "elastic_motion_residual",
    "deformation": "deformation_residual",
    "melting": "melting_residual",
    "solidification": "solidification_residual",
    "vaporization": "vaporization_residual",
    "liquefaction": "liquefaction_residual",
    "combustion": "combustion_residual",
    "explosion": "explosion_residual",
    "reflection": "reflection_residual",
    "refraction": "refraction_residual",
    "scattering": "scattering_residual",
    "interference and diffraction": "interference_diffraction_residual",
    "unnatural light source": "unnatural_light_source_residual",
}



class WanPINNTrainingModule(DiffusionTrainingModule):
    """
    PINN 插件训练模块
    
    核心设计：
    - 冻结原始 Wan 模型（DiT, VAE, TextEncoder）
    - 只训练 PINN 插件组件（PhysicsAdapter + MaterialPDEResiduals）
    - 使用 accelerate + DeepSpeed 多卡并行
    """
    
    def __init__(
        self,
        # 原始模型参数（与 train.py 保持一致）
        model_paths=None,
        model_id_with_origin_paths=None,
        trainable_models=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        frozen_model_gradient_checkpointing=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        # PINN 专用参数
        physics_weight=0.1,
        physics_warmup_steps=500,
        conditioned_physics_warmup_steps=1000,
        material_type="auto",
        adapter_hidden_dim=64,
        pinn_checkpoint=None,
        expert_balance_weight=1e-3,
        condition_consistency_weight=1e-2,
        moe_top_k=4,
        ablate_disable_moe=False,
        ablate_disable_conditioned_pde=False,
        ablate_disable_aux_losses=False,
        ablate_label_only_router=False,
        diagnostic_metrics_interval=10,
        moe_fast_mode=True,
        moe_pde_branches_per_sample=1,
        moe_weight_threshold=0.05,
        motion_mask_floor=0.08,
        motion_mask_quantile=0.9,
        motion_mask_warmup_steps=300,
        # LoRA（兼容原框架，但 PINN 模式下通常不用）
        lora_base_model=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        lora_checkpoint=None,
    ):
        super().__init__()
        
        # ================================================================
        # 1. 加载原始 Wan 模型
        # ================================================================
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [
                ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1])
                for i in model_id_with_origin_paths
            ]
        
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
        )
        
        # ================================================================
        # 2. 冻结原始模型（关键！保持原始性能）
        # ================================================================
        # 冻结所有模型参数
        self.pipe.freeze_except([])
        print("=" * 60)
        print("PLUGIN MODE: All original model parameters are FROZEN")
        frozen_params = sum(p.numel() for p in self.pipe.parameters())
        print(f"  Frozen parameters: {frozen_params:,}")
        print("=" * 60)
        
        # ================================================================
        # 3. 初始化 PINN 插件组件（可训练）
        # ================================================================
        # 获取 latent 维度
        latent_dim = 16  # Wan 的默认 latent 维度
        if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
            if hasattr(self.pipe.vae, 'model') and hasattr(self.pipe.vae.model, 'z_dim'):
                latent_dim = self.pipe.vae.model.z_dim
        
        # 物理适配器（核心插件）
        self.physics_adapter = PhysicsAdapter(
            latent_dim=latent_dim,
            hidden_dim=adapter_hidden_dim,
            material_types=4,
            num_phenomena=len(PHENOMENON_LABELS),
            moe_top_k=moe_top_k,
        )
        self.physics_adapter.train()
        self.physics_adapter.requires_grad_(True)
        
        # PDE 残差计算器
        self.pde_residuals = MaterialPDEResiduals(
            num_phenomena=len(PHENOMENON_LABELS),
            q_input_dim=self.physics_adapter.q_input_dim,
            n_numeric_dim=self.physics_adapter.n_numeric_dim,
        )
        self.pde_residuals.train()
        self.pde_residuals.requires_grad_(True)
        
        # 材质分类器（不需要训练）
        self.material_classifier = MaterialClassifier()
        
        # 加载 PINN checkpoint（如果有）
        if pinn_checkpoint is not None:
            checkpoint = torch.load(pinn_checkpoint, map_location="cpu")
            if 'physics_adapter_state_dict' in checkpoint:
                load_result = self.physics_adapter.load_state_dict(
                    checkpoint['physics_adapter_state_dict'], strict=False
                )
                print(f"Loaded PhysicsAdapter from {pinn_checkpoint}")
                if len(load_result.missing_keys) > 0:
                    print(f"PhysicsAdapter missing keys: {load_result.missing_keys[:5]}...")
                if len(load_result.unexpected_keys) > 0:
                    print(f"PhysicsAdapter unexpected keys: {load_result.unexpected_keys[:5]}...")
            if 'pde_residuals_state_dict' in checkpoint:
                load_result = self.pde_residuals.load_state_dict(
                    checkpoint['pde_residuals_state_dict'], strict=False
                )
                print(f"Loaded PDE Residuals from {pinn_checkpoint}")
                if len(load_result.missing_keys) > 0:
                    print(f"PDE missing keys: {load_result.missing_keys[:5]}...")
                if len(load_result.unexpected_keys) > 0:
                    print(f"PDE unexpected keys: {load_result.unexpected_keys[:5]}...")
        
        # 统计可训练参数
        trainable_params = sum(
            p.numel() for p in self.physics_adapter.parameters() if p.requires_grad
        ) + sum(
            p.numel() for p in self.pde_residuals.parameters() if p.requires_grad
        )
        print(f"  Trainable parameters (PINN plugin): {trainable_params:,}")
        print(f"  Ratio: {trainable_params / max(frozen_params, 1) * 100:.4f}% of original model")
        
        # 可选：给原模型加 LoRA
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected: {load_result[1]}")
            setattr(self.pipe, lora_base_model, model)
        
        # ================================================================
        # 4. 存储训练配置
        # ================================================================
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.frozen_model_gradient_checkpointing = frozen_model_gradient_checkpointing
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
        # PINN 配置
        self.physics_weight = physics_weight
        self.physics_warmup_steps = physics_warmup_steps
        self.conditioned_physics_warmup_steps = conditioned_physics_warmup_steps
        self.material_type = material_type
        self.expert_balance_weight = expert_balance_weight
        self.condition_consistency_weight = condition_consistency_weight
        self.moe_top_k = moe_top_k
        self.ablate_disable_moe = ablate_disable_moe
        self.ablate_disable_conditioned_pde = ablate_disable_conditioned_pde
        self.ablate_disable_aux_losses = ablate_disable_aux_losses
        self.ablate_label_only_router = ablate_label_only_router
        self.diagnostic_metrics_interval = max(int(diagnostic_metrics_interval), 1)
        self.moe_fast_mode = bool(moe_fast_mode)
        self.moe_pde_branches_per_sample = max(int(moe_pde_branches_per_sample), 1)
        self.moe_weight_threshold = max(float(moe_weight_threshold), 0.0)
        self.motion_mask_floor = min(max(float(motion_mask_floor), 0.0), 0.5)
        self.motion_mask_quantile = min(max(float(motion_mask_quantile), 0.5), 0.995)
        self.motion_mask_warmup_steps = max(int(motion_mask_warmup_steps), 0)
        self.current_step = 0
        self._last_metrics = {}

        # 将消融配置同步到底层模块，保持调用层逻辑清晰
        self.physics_adapter.set_ablation_modes(
            use_moe=not self.ablate_disable_moe,
            label_only_mode=self.ablate_label_only_router,
        )
        self.physics_adapter.set_cache_mode(lightweight=self.moe_fast_mode)
        self.pde_residuals.set_conditioning_enabled(
            enabled=not self.ablate_disable_conditioned_pde
        )
        print("Ablation flags:")
        print(f"  disable_moe={self.ablate_disable_moe}")
        print(f"  disable_conditioned_pde={self.ablate_disable_conditioned_pde}")
        print(f"  disable_aux_losses={self.ablate_disable_aux_losses}")
        print(f"  label_only_router={self.ablate_label_only_router}")
        print(f"  moe_fast_mode={self.moe_fast_mode}")
        print(f"  moe_pde_branches_per_sample={self.moe_pde_branches_per_sample}")
        print(f"  moe_weight_threshold={self.moe_weight_threshold}")
        print(f"  motion_mask_floor={self.motion_mask_floor}")
        print(f"  motion_mask_quantile={self.motion_mask_quantile}")
        print(f"  motion_mask_warmup_steps={self.motion_mask_warmup_steps}")
    
    
    def get_physics_weight(self):
        """获取当前物理损失权重（带预热）"""
        if self.current_step < self.physics_warmup_steps:
            alpha = self.current_step / max(self.physics_warmup_steps, 1)
            return self.physics_weight * alpha
        return self.physics_weight

    def get_conditioned_alpha(self):
        """条件化约束强度预热"""
        if self.current_step < self.conditioned_physics_warmup_steps:
            return self.current_step / max(self.conditioned_physics_warmup_steps, 1)
        return 1.0

    @staticmethod
    def _safe_text(value):
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _normalize_label(label):
        clean = WanPINNTrainingModule._safe_text(label).lower().replace("_", " ")
        clean = re.sub(r"\s+", " ", clean)
        return PHENOMENON_ALIAS.get(clean.replace(" ", "_"), clean)

    @staticmethod
    def _hash_to_id(text, modulo):
        if modulo <= 1:
            return 0
        text = WanPINNTrainingModule._safe_text(text).lower()
        if text == "":
            return 0
        digest = hashlib.sha1(text.encode("utf-8")).digest()
        stable_hash = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return (stable_hash % (modulo - 1)) + 1

    @staticmethod
    def _parse_numeric_range(text):
        text = WanPINNTrainingModule._safe_text(text).lower()
        matches = re.findall(r"-?\d+(?:\.\d+)?", text)
        if len(matches) == 0:
            return 0.0, 0.0, 0.0, 0.0
        values = [float(x) for x in matches]
        min_val = min(values)
        max_val = max(values)
        mean_val = sum(values) / max(len(values), 1)
        return min_val, max_val, mean_val, 1.0

    @staticmethod
    def _encode_q_field(text, dim):
        vec = torch.zeros(dim, dtype=torch.float32)
        text = WanPINNTrainingModule._safe_text(text).lower()
        if text == "" or dim == 0:
            return vec
        tokens = re.split(r"[,;|/]| and |\.", text)
        tokens = [re.sub(r"\s+", " ", t).strip() for t in tokens if t.strip()]
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            stable_hash = int.from_bytes(digest[:8], byteorder="big", signed=False)
            idx = stable_hash % dim
            vec[idx] = 1.0
        return vec

    @staticmethod
    def _metric_key(name):
        key = WanPINNTrainingModule._safe_text(name).lower()
        key = re.sub(r"[^a-z0-9]+", "_", key)
        return key.strip("_") or "unknown"

    def _build_motion_mask(self, v_original, z_t):
        """
        基于速度场时序变化 + 空间梯度，构建自监督 soft motion mask: [B,1,T,H,W]。
        """
        B, _, T, H, W = v_original.shape
        device = v_original.device
        dtype = v_original.dtype
        zero = torch.zeros((B, 1, T, H, W), device=device, dtype=dtype)

        if T > 1:
            v_dt = torch.mean(torch.abs(v_original[:, :, 1:] - v_original[:, :, :-1]), dim=1, keepdim=True)
            v_dt = F.pad(v_dt, (0, 0, 0, 0, 0, 1))
            z_dt = torch.mean(torch.abs(z_t[:, :, 1:] - z_t[:, :, :-1]), dim=1, keepdim=True)
            z_dt = F.pad(z_dt, (0, 0, 0, 0, 0, 1))
        else:
            v_dt = zero
            z_dt = zero

        if H > 1:
            v_dh = torch.mean(torch.abs(v_original[:, :, :, 1:, :] - v_original[:, :, :, :-1, :]), dim=1, keepdim=True)
            v_dh = F.pad(v_dh, (0, 0, 0, 1, 0, 0))
        else:
            v_dh = zero

        if W > 1:
            v_dw = torch.mean(torch.abs(v_original[:, :, :, :, 1:] - v_original[:, :, :, :, :-1]), dim=1, keepdim=True)
            v_dw = F.pad(v_dw, (0, 1, 0, 0, 0, 0))
        else:
            v_dw = zero

        spatial_energy = 0.5 * (v_dh + v_dw)
        energy = 0.55 * v_dt + 0.25 * z_dt + 0.20 * spatial_energy

        # 每个样本独立稳健归一化，避免全局尺度偏置到背景纹理。
        energy_flat = energy.float().reshape(B, -1)
        denom = torch.quantile(
            energy_flat, q=self.motion_mask_quantile, dim=1, keepdim=True
        ).to(dtype=dtype).view(B, 1, 1, 1, 1)
        energy_norm = torch.clamp(energy / (denom + 1e-6), 0.0, 1.0)
        mask = self.motion_mask_floor + (1.0 - self.motion_mask_floor) * energy_norm

        if self.motion_mask_warmup_steps > 0 and self.current_step < self.motion_mask_warmup_steps:
            alpha = float(self.current_step) / float(max(self.motion_mask_warmup_steps, 1))
            mask = (1.0 - alpha) + alpha * mask

        return torch.clamp(mask, self.motion_mask_floor, 1.0)

    def _collect_motion_mask_metrics(self, motion_mask):
        if motion_mask is None:
            return {}
        floor = float(self.motion_mask_floor)
        active_threshold = floor + 0.25 * (1.0 - floor)
        active = (motion_mask > active_threshold).float()
        return {
            "motion_mask_mean": float(motion_mask.detach().mean().item()),
            "motion_mask_sparsity": float((1.0 - active.mean()).detach().item()),
        }

    def extract_physics_metadata(self, data, batch_size, device, dtype):
        """从 CSV metadata 提取并编码 PINN 条件输入，缺失字段时自动回退"""
        if not isinstance(data, dict):
            return None

        label_name = self._normalize_label(data.get("label", ""))
        label_id = PHENOMENON_TO_ID.get(label_name, PHENOMENON_TO_ID["liquid motion"])
        label_ids = torch.full((batch_size,), label_id, dtype=torch.long, device=device)

        n_numeric_list = []
        valid_count = 0.0
        for key in ("n0", "n1", "n2"):
            n_min, n_max, n_mean, n_valid = self._parse_numeric_range(data.get(key, ""))
            n_numeric_list.extend([n_min, n_max, n_mean, n_valid])
            valid_count += n_valid
        n_numeric = torch.tensor(n_numeric_list, dtype=torch.float32, device=device).unsqueeze(0)
        n_numeric = n_numeric.repeat(batch_size, 1)

        n_text_ids = [
            self._hash_to_id(data.get("n0", ""), self.physics_adapter.n_text_vocab_size),
            self._hash_to_id(data.get("n1", ""), self.physics_adapter.n_text_vocab_size),
            self._hash_to_id(data.get("n2", ""), self.physics_adapter.n_text_vocab_size),
        ]
        n_text_ids = torch.tensor(n_text_ids, dtype=torch.long, device=device).unsqueeze(0)
        n_text_ids = n_text_ids.repeat(batch_size, 1)

        q_dim = self.physics_adapter.q_input_dim
        q_vector = torch.zeros(q_dim, dtype=torch.float32)
        q_vector = q_vector + self._encode_q_field(data.get("q0", ""), q_dim)
        q_vector = q_vector + self._encode_q_field(data.get("q1", ""), q_dim)
        q_vector = q_vector + self._encode_q_field(data.get("q2", ""), q_dim)
        q_vector = q_vector + self._encode_q_field(data.get("q4", ""), q_dim)
        q3 = self._safe_text(data.get("q3", "")).lower()
        if q_dim > 0:
            if q3 in {"yes", "true", "1"}:
                q_vector[0] = 1.0
            elif q3 in {"no", "false", "0"} and q_dim > 1:
                q_vector[1] = 1.0
        q_vector = torch.clamp(q_vector, 0.0, 1.0).to(device=device).unsqueeze(0)
        q_vector = q_vector.repeat(batch_size, 1)

        parse_success_ratio = torch.tensor(
            valid_count / 3.0, dtype=torch.float32, device=device
        )

        return {
            "label_name": label_name,
            "label_id": label_ids,
            "n_numeric": n_numeric.to(dtype=dtype),
            "n_text_ids": n_text_ids,
            "q_vector": q_vector.to(dtype=dtype),
            "parse_success_ratio": parse_success_ratio.to(dtype=dtype),
        }

    @staticmethod
    def _metadata_without_motion_mask(metadata):
        if not isinstance(metadata, dict):
            return metadata
        copied = dict(metadata)
        copied.pop("motion_mask", None)
        return copied

    @staticmethod
    def _motion_mask_stats_from_metadata(metadata):
        if not isinstance(metadata, dict):
            return {}
        motion_mask = metadata.get("motion_mask")
        if not isinstance(motion_mask, torch.Tensor):
            return {}
        if motion_mask.numel() == 0:
            return {}
        motion_mask = motion_mask.detach().float()
        active = (motion_mask > 0.25).float()
        return {
            "motion_mask_mean": float(motion_mask.mean().item()),
            "motion_mask_sparsity": float((1.0 - active.mean()).item()),
        }

    def compute_physics_loss(self, v_pred, z_t, material_type, metadata=None):
        """计算 PDE 残差损失"""
        pde_metadata = None if self.ablate_disable_conditioned_pde else metadata
        phenomenon_name = ""
        if isinstance(metadata, dict):
            phenomenon_name = self._safe_text(metadata.get("label_name", "")).lower()
        residual_method_name = PHENOMENON_TO_RESIDUAL_METHOD.get(phenomenon_name)
        if residual_method_name is not None:
            residual_method = getattr(self.pde_residuals, residual_method_name)
            loss, info = residual_method(z_t, v_pred, metadata=pde_metadata)
        else:
            loss, info = self.pde_residuals._fallback_material_residual(
                material_type, z_t, v_pred, metadata=pde_metadata
            )

        info = dict(info)
        info.update(self._motion_mask_stats_from_metadata(pde_metadata))
        info["motion_mask_enabled"] = float(
            isinstance(pde_metadata, dict) and isinstance(pde_metadata.get("motion_mask"), torch.Tensor)
        )
        if (
            isinstance(pde_metadata, dict)
            and isinstance(pde_metadata.get("motion_mask"), torch.Tensor)
            and residual_method_name is not None
        ):
            with torch.no_grad():
                unmasked_metadata = self._metadata_without_motion_mask(pde_metadata)
                unmasked_loss, _ = residual_method(
                    z_t.detach(),
                    v_pred.detach(),
                    metadata=unmasked_metadata,
                )
            info["masked_vs_unmasked_residual_ratio"] = float(
                (loss.detach() / (unmasked_loss.detach() + 1e-8)).item()
            )
        if phenomenon_name:
            info["phenomenon_name"] = phenomenon_name
        info["base_material"] = material_type
        return loss, info

    def _metadata_for_sample_and_expert(self, metadata, sample_idx, phenomenon_name, device, dtype):
        if not isinstance(metadata, dict):
            return {
                "label_name": phenomenon_name,
                "label_id": torch.tensor(
                    [PHENOMENON_TO_ID.get(phenomenon_name, 0)], device=device, dtype=torch.long
                ),
            }

        expert_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, torch.Tensor):
                tensor = value
                if tensor.ndim > 0 and tensor.shape[0] > sample_idx:
                    expert_metadata[key] = tensor[sample_idx:sample_idx + 1].to(device=device)
                else:
                    expert_metadata[key] = tensor.to(device=device)
            else:
                expert_metadata[key] = value

        expert_metadata["label_name"] = phenomenon_name
        expert_metadata["label_id"] = torch.tensor(
            [PHENOMENON_TO_ID.get(phenomenon_name, 0)], device=device, dtype=torch.long
        )
        if "q_vector" in expert_metadata and isinstance(expert_metadata["q_vector"], torch.Tensor):
            expert_metadata["q_vector"] = expert_metadata["q_vector"].to(dtype=dtype)
        if "n_numeric" in expert_metadata and isinstance(expert_metadata["n_numeric"], torch.Tensor):
            expert_metadata["n_numeric"] = expert_metadata["n_numeric"].to(dtype=dtype)
        return expert_metadata

    def compute_multi_expert_physics_loss(self, z_t, metadata=None, collect_diagnostics=True):
        cache = getattr(self.physics_adapter, "_cache", {})
        branch_outputs = cache.get("branch_v_corrected_live")
        branch_corrections = cache.get("branch_raw_corrections_live")
        branch_physics_features = cache.get("branch_physics_features_live")
        branch_indices = cache.get("active_expert_indices")
        branch_weights = cache.get("active_expert_weights")
        if branch_outputs is None or branch_indices is None or branch_weights is None:
            raise RuntimeError("Multi-expert latent-space branch outputs are unavailable in PhysicsAdapter cache.")
        if branch_outputs.ndim != 6:
            raise RuntimeError(
                "Expected branch_v_corrected_live to have shape [B, K, C, T, H, W]."
            )
        if branch_outputs.shape[2] != z_t.shape[1]:
            raise RuntimeError(
                "Decoded branch velocity channel dimension does not match latent velocity space."
            )

        total_loss = torch.zeros((), device=branch_outputs.device, dtype=branch_outputs.dtype)
        batch_size = max(int(branch_outputs.shape[0]), 1)
        expert_stats = {}
        selected_branches_total = 0.0

        for sample_idx in range(branch_outputs.shape[0]):
            z_sample = z_t[sample_idx:sample_idx + 1]
            sample_weights = branch_weights[sample_idx]
            if self.moe_fast_mode:
                selected = torch.nonzero(
                    sample_weights > self.moe_weight_threshold, as_tuple=False
                ).view(-1)
                if selected.numel() == 0:
                    selected = torch.argmax(sample_weights).view(1)
                selected_weights = sample_weights[selected]
                order = torch.argsort(selected_weights, descending=True)
                branch_cap = min(self.moe_pde_branches_per_sample, int(selected.numel()))
                selected = selected[order[:branch_cap]]
                selected_weight_sum = torch.clamp(
                    sample_weights[selected].sum(), min=1e-6
                )
            else:
                selected = torch.arange(
                    branch_outputs.shape[1], device=branch_outputs.device, dtype=torch.long
                )
                selected_weight_sum = None

            selected_branches_total += float(selected.numel())

            for branch_idx_t in selected:
                branch_idx = int(branch_idx_t.item())
                expert_idx = int(branch_indices[sample_idx, branch_idx].item())
                expert_weight = branch_weights[sample_idx, branch_idx].to(dtype=branch_outputs.dtype)
                if self.moe_fast_mode:
                    expert_weight = expert_weight / selected_weight_sum.to(dtype=branch_outputs.dtype)
                if float(expert_weight.detach().item()) <= 0.0:
                    continue

                phenomenon_name = PHENOMENON_LABELS[expert_idx]
                residual_method_name = PHENOMENON_TO_RESIDUAL_METHOD.get(phenomenon_name)
                if residual_method_name is None:
                    continue

                # PDE residuals must act on the decoded latent velocity branch,
                # not on hidden physics features.
                branch_v = branch_outputs[sample_idx, branch_idx].unsqueeze(0)
                branch_metadata = self._metadata_for_sample_and_expert(
                    metadata=metadata,
                    sample_idx=sample_idx,
                    phenomenon_name=phenomenon_name,
                    device=branch_v.device,
                    dtype=branch_v.dtype,
                )
                residual_method = getattr(self.pde_residuals, residual_method_name)
                expert_loss, expert_info = residual_method(z_sample, branch_v, metadata=branch_metadata)
                total_loss = total_loss + expert_weight * expert_loss
                if collect_diagnostics:
                    expert_info = dict(expert_info)
                    expert_info.update(self._motion_mask_stats_from_metadata(branch_metadata))
                    if isinstance(branch_metadata.get("motion_mask"), torch.Tensor):
                        unmasked_metadata = self._metadata_without_motion_mask(branch_metadata)
                        with torch.no_grad():
                            unmasked_expert_loss, _ = residual_method(
                                z_sample.detach(),
                                branch_v.detach(),
                                metadata=unmasked_metadata,
                            )
                        expert_info["masked_vs_unmasked_residual_ratio"] = float(
                            (expert_loss.detach() / (unmasked_expert_loss.detach() + 1e-8)).item()
                        )
                    if branch_corrections is not None:
                        branch_delta = branch_corrections[sample_idx, branch_idx].unsqueeze(0)
                        expert_info["branch_residual_norm"] = float(
                            torch.mean(branch_delta.detach().float() ** 2).item()
                        )
                    expert_info["branch_corrected_norm"] = float(
                        torch.mean(branch_v.detach().float() ** 2).item()
                    )
                    if branch_physics_features is not None:
                        branch_physics = branch_physics_features[sample_idx, branch_idx].unsqueeze(0)
                        expert_info["branch_physics_feature_norm"] = float(
                            torch.mean(branch_physics.detach().float() ** 2).item()
                        )

                    if phenomenon_name not in expert_stats:
                        expert_stats[phenomenon_name] = {
                            "count": 0,
                            "weight_sum": 0.0,
                            "residual_sum": 0.0,
                            "weighted_residual_sum": 0.0,
                            "terms": {},
                        }
                    stats = expert_stats[phenomenon_name]
                    residual_value = float(expert_loss.detach().item())
                    weight_value = float(expert_weight.detach().item())
                    stats["count"] += 1
                    stats["weight_sum"] += weight_value
                    stats["residual_sum"] += residual_value
                    stats["weighted_residual_sum"] += weight_value * residual_value
                    for key, value in expert_info.items():
                        if isinstance(value, (int, float)):
                            term_stats = stats["terms"].setdefault(
                                key,
                                {"sum": 0.0, "weighted_sum": 0.0, "count": 0},
                            )
                            term_stats["sum"] += float(value)
                            term_stats["weighted_sum"] += weight_value * float(value)
                            term_stats["count"] += 1

        total_loss = total_loss / float(batch_size)
        info = {
            "physics_mode": "multi_expert",
            "physics_constraint_space": "latent_decoded",
            "physics_feature_space": float(branch_physics_features.shape[2])
            if branch_physics_features is not None else 0.0,
            "decoded_velocity_space": float(branch_outputs.shape[2]),
            "active_expert_branches": float(branch_weights.shape[1]),
            "multi_expert_unique": float(len(expert_stats)),
            "pde_branches_per_sample": selected_branches_total / float(batch_size),
            "moe_fast_mode": float(self.moe_fast_mode),
        }
        info.update(self._motion_mask_stats_from_metadata(metadata))
        info["motion_mask_enabled"] = float(
            isinstance(metadata, dict) and isinstance(metadata.get("motion_mask"), torch.Tensor)
        )
        return total_loss, info, expert_stats

    def _collect_expert_residual_metrics(self, expert_stats, conditioned_scale=1.0, batch_size=1):
        if not isinstance(expert_stats, dict) or len(expert_stats) == 0:
            return {}

        metrics = {}
        batch_size = max(int(batch_size), 1)
        sorted_items = sorted(
            expert_stats.items(),
            key=lambda item: item[1]["weight_sum"],
            reverse=True,
        )[: self.moe_top_k]

        for phenomenon_name, stats in sorted_items:
            key = self._metric_key(phenomenon_name)
            count = max(int(stats["count"]), 1)
            residual_mean = stats["residual_sum"] / float(count)
            weighted_residual = (
                stats["weighted_residual_sum"] / float(batch_size)
            ) * float(conditioned_scale)
            metrics[f"moe_residual_{key}"] = float(residual_mean)
            metrics[f"moe_residual_weighted_{key}"] = float(weighted_residual)
            metrics[f"moe_residual_activation_count_{key}"] = float(stats["count"])
            metrics[f"moe_residual_weight_sum_{key}"] = float(stats["weight_sum"])

            for term_name, term_stats in stats["terms"].items():
                term_key = self._metric_key(term_name)
                term_mean = term_stats["sum"] / float(max(int(term_stats["count"]), 1))
                term_weighted = (
                    term_stats["weighted_sum"] / float(batch_size)
                ) * float(conditioned_scale)
                if term_name == "branch_residual_norm":
                    metrics[f"moe_branch_residual_norm_{key}"] = float(term_mean)
                    metrics[f"moe_branch_residual_norm_weighted_{key}"] = float(term_weighted)
                    continue
                if term_name == "branch_corrected_norm":
                    metrics[f"moe_branch_corrected_norm_{key}"] = float(term_mean)
                    metrics[f"moe_branch_corrected_norm_weighted_{key}"] = float(term_weighted)
                    continue
                term_count = max(int(term_stats["count"]), 1)
                metrics[f"moe_residual_term_{key}_{term_key}"] = float(
                    term_stats["sum"] / float(term_count)
                )
                metrics[f"moe_residual_term_weighted_{key}_{term_key}"] = float(
                    (term_stats["weighted_sum"] / float(batch_size)) * float(conditioned_scale)
                )

        return metrics

    def _collect_decoded_branch_consistency_metrics(self, v_corrected):
        cache = getattr(self.physics_adapter, "_cache", {})
        if not isinstance(cache, dict):
            return {}
        branch_corrections = cache.get("branch_raw_corrections")
        branch_weights = cache.get("active_expert_weights")
        fused_correction = cache.get("raw_correction")
        if branch_corrections is None or branch_weights is None or fused_correction is None:
            return {}
        if branch_corrections.numel() == 0 or branch_weights.numel() == 0:
            return {}

        weight_shape = (branch_weights.shape[0], branch_weights.shape[1], 1, 1, 1, 1)
        weighted_branch_correction = torch.sum(
            branch_corrections.float() * branch_weights.float().view(*weight_shape), dim=1
        )
        fused_correction = fused_correction.float()
        v_corrected = v_corrected.float()

        return {
            "moe_decoded_branch_fused_l1": float(
                torch.mean(torch.abs(weighted_branch_correction - fused_correction)).item()
            ),
            "moe_decoded_branch_fused_l2": float(
                torch.mean((weighted_branch_correction - fused_correction) ** 2).item()
            ),
            "moe_decoded_branch_output_norm": float(
                torch.mean(weighted_branch_correction ** 2).item()
            ),
            "moe_fused_output_norm": float(
                torch.mean(fused_correction ** 2).item()
            ),
            "moe_v_corrected_norm": float(
                torch.mean(v_corrected ** 2).item()
            ),
        }

    def _collect_router_metrics(self, loss_physics_value):
        cache = getattr(self.physics_adapter, "_cache", {})
        if not isinstance(cache, dict):
            return {}
        if "active_expert_indices" not in cache or "active_expert_weights" not in cache:
            return {}

        active_indices = cache["active_expert_indices"]
        active_weights = cache["active_expert_weights"].float()
        if active_indices.numel() == 0 or active_weights.numel() == 0:
            return {}

        batch_size = max(int(active_indices.shape[0]), 1)
        mean_weight = torch.zeros(len(PHENOMENON_LABELS), device=active_weights.device, dtype=torch.float32)
        mean_weight.scatter_add_(0, active_indices.reshape(-1), active_weights.reshape(-1))
        mean_weight = mean_weight / float(batch_size)

        ema = self.physics_adapter.expert_usage_ema.detach().to(device=active_weights.device, dtype=torch.float32)
        active_mask = mean_weight > 1e-6
        sorted_ids = torch.argsort(mean_weight, descending=True)
        dominant_expert = cache.get("dominant_expert")
        dominant_id = -1
        if dominant_expert is not None and dominant_expert.numel() > 0:
            dominant_id = int(torch.mode(dominant_expert.view(-1)).values.item())

        top_ids = [int(idx) for idx in sorted_ids.tolist() if mean_weight[idx].item() > 1e-6][: self.moe_top_k]
        metrics = {
            "moe_top_k": float(active_indices.shape[1]),
            "moe_batch_unique_experts": float(active_mask.sum().item()),
            "moe_active_coverage_ratio": float(active_mask.float().mean().item()),
            "moe_dominant_expert_id": float(dominant_id),
            "moe_dominant_expert_weight": float(active_weights[:, 0].mean().item()) if active_weights.shape[1] > 0 else 0.0,
        }
        if len(top_ids) > 0:
            metrics["moe_active_experts"] = ",".join(PHENOMENON_LABELS[idx] for idx in top_ids)
            metrics["moe_active_expert_weights"] = ",".join(
                f"{mean_weight[idx].item():.4f}" for idx in top_ids
            )
            metrics["moe_expert_usage_topk"] = ",".join(
                f"{PHENOMENON_LABELS[idx]}:{ema[idx].item():.4f}" for idx in top_ids
            )
        if active_indices.shape[0] > 0:
            sample0_indices = active_indices[0].tolist()
            sample0_weights = active_weights[0].tolist()
            metrics["moe_sample0_active_experts"] = ",".join(
                PHENOMENON_LABELS[int(idx)] for idx in sample0_indices
            )
            metrics["moe_sample0_active_weights"] = ",".join(
                f"{float(weight):.4f}" for weight in sample0_weights
            )

        for idx, label in enumerate(PHENOMENON_LABELS):
            key = self._metric_key(label)
            weight_val = float(mean_weight[idx].item())
            metrics[f"moe_mean_weight_{key}"] = weight_val
            metrics[f"moe_usage_ema_{key}"] = float(ema[idx].item())
            metrics[f"moe_physics_proxy_{key}"] = float(loss_physics_value * weight_val)

        return metrics

    @staticmethod
    def _label_only_metadata(metadata):
        if not isinstance(metadata, dict):
            return None
        filtered = {}
        if "label_id" in metadata:
            filtered["label_id"] = metadata["label_id"]
        if "label_name" in metadata:
            filtered["label_name"] = metadata["label_name"]
        # 保留 parse ratio 用于稳定训练时的缩放
        if "parse_success_ratio" in metadata:
            filtered["parse_success_ratio"] = metadata["parse_success_ratio"]
        # 保留 motion mask，确保 label-only 消融不影响动态区域约束
        if "motion_mask" in metadata:
            filtered["motion_mask"] = metadata["motion_mask"]
        return filtered


    def forward_preprocess(self, data):
        """数据预处理（与原版保持一致）"""
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input in ("reference_image", "vace_reference_image"):
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units 预处理
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )
        
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        """
        前向传播：物理PDE损失 + 适配器正则化
        
        由于原模型完全冻结，loss_fm 没有 grad_fn，
        所以训练目标完全来自 PINN 插件（PhysicsAdapter + PDE Residuals）。
        loss_fm 仅用于日志监控。
        
        Loss = λ(t) * Loss_Physics + Loss_Reg
        """
        if inputs is None:
            inputs = self.forward_preprocess(data)
        
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        
        # ============================================================
        # 1. 获取原始模型的速度场预测（frozen, no grad）
        # ============================================================
        max_boundary = int(inputs.get("max_timestep_boundary", 1) * self.pipe.scheduler.num_train_timesteps)
        min_boundary = int(inputs.get("min_timestep_boundary", 0) * self.pipe.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_boundary, max_boundary, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype, device=inputs["latents"].device
        )
        
        input_latents = inputs.get("input_latents", inputs["latents"])
        noise = inputs.get("noise", torch.randn_like(inputs["latents"]))
        z_t = self.pipe.scheduler.add_noise(input_latents, noise, timestep)
        
        # FM 的训练目标（速度场真值）
        v_target = self.pipe.scheduler.training_target(input_latents, noise, timestep)
        
        with torch.no_grad():
            v_original = self.pipe.model_fn(
                **models,
                latents=z_t,
                timestep=timestep,
                context=inputs.get("context"),
                clip_feature=inputs.get("clip_feature"),
                y=inputs.get("y"),
                use_gradient_checkpointing=self.frozen_model_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            )

        metadata = self.extract_physics_metadata(
            data=data,
            batch_size=v_original.shape[0],
            device=v_original.device,
            dtype=v_original.dtype,
        )
        motion_mask = self._build_motion_mask(v_original.detach(), z_t.detach())
        if isinstance(metadata, dict):
            metadata = dict(metadata)
        else:
            metadata = {}
        metadata["motion_mask"] = motion_mask
        metadata["motion_mask_source"] = "self_supervised"
        motion_mask_metrics = self._collect_motion_mask_metrics(motion_mask)
        adapter_metadata = metadata
        if self.ablate_disable_moe:
            adapter_metadata = None
        elif self.ablate_label_only_router:
            adapter_metadata = self._label_only_metadata(metadata)
        
        # ============================================================
        # 2. PhysicsAdapter 施加物理校正（trainable, has grad）
        # ============================================================
        prompt = data.get("prompt", "") if isinstance(data, dict) else ""
        if self.material_type == "auto":
            label_name = metadata.get("label_name", "") if isinstance(metadata, dict) else ""
            material_type = PHENOMENON_TO_MATERIAL.get(
                label_name, self.material_classifier.classify(prompt)
            )
        else:
            material_type = self.material_type
        
        v_corrected = self.physics_adapter(v_original, z_t, metadata=adapter_metadata)
        
        # ============================================================
        # 3. 计算损失（全部来自可训练参数，确保有 grad_fn）
        # ============================================================
        physics_weight = self.get_physics_weight()
        
        # (a) 物理 PDE 残差损失
        pde_metadata = metadata
        if self.ablate_disable_conditioned_pde:
            pde_metadata = None
        expert_stats = {}
        # Skip step 0 diagnostics to avoid very heavy first-step overhead.
        collect_diagnostics = (
            self.current_step > 0
            and self.current_step % self.diagnostic_metrics_interval == 0
        )
        cache = getattr(self.physics_adapter, "_cache", {})
        if (
            not self.ablate_disable_moe
            and isinstance(cache, dict)
            and "branch_v_corrected_live" in cache
        ):
            loss_physics, physics_info, expert_stats = self.compute_multi_expert_physics_loss(
                z_t=z_t,
                metadata=pde_metadata,
                collect_diagnostics=collect_diagnostics,
            )
        else:
            loss_physics, physics_info = self.compute_physics_loss(
                v_corrected, z_t, material_type, metadata=pde_metadata
            )
        cond_alpha = self.get_conditioned_alpha()
        parse_ratio = 1.0
        if isinstance(metadata, dict) and "parse_success_ratio" in metadata:
            parse_ratio = float(metadata["parse_success_ratio"].detach().item())
        conditioned_scale = 0.5 + 0.5 * cond_alpha * parse_ratio
        loss_physics = loss_physics * conditioned_scale
        loss_physics_value = float(loss_physics.detach().item())
        
        # (b) 适配器校正不应偏离原始输出太多（正则化）
        correction = v_corrected - v_original
        loss_reg = torch.mean(correction ** 2)
        
        # (c) 校正后的速度场应仍然接近 FM 训练目标（保持生成质量）
        loss_fm_adapter = torch.nn.functional.mse_loss(
            v_corrected.float(), v_target.float()
        )

        aux_losses = self.physics_adapter.compute_auxiliary_losses()
        loss_expert_balance = aux_losses["expert_balance"]
        loss_condition_consistency = aux_losses["condition_consistency"]
        if self.ablate_disable_aux_losses:
            loss_expert_balance = loss_expert_balance * 0.0
            loss_condition_consistency = loss_condition_consistency * 0.0
        
        # 总损失：全部通过 physics_adapter，有完整的计算图
        total_loss = (
            loss_fm_adapter
            + physics_weight * loss_physics
            + 0.01 * loss_reg
            + self.expert_balance_weight * loss_expert_balance
            + self.condition_consistency_weight * loss_condition_consistency
        )
        router_metrics, expert_residual_metrics, decoded_branch_metrics = {}, {}, {}
        if collect_diagnostics:
            router_metrics = self._collect_router_metrics(loss_physics_value)
            expert_residual_metrics = self._collect_expert_residual_metrics(
                expert_stats=expert_stats,
                conditioned_scale=conditioned_scale,
                batch_size=v_corrected.shape[0],
            )
            decoded_branch_metrics = self._collect_decoded_branch_consistency_metrics(
                v_corrected=v_corrected
            )
        self._last_metrics = {
            "material_type": material_type,
            "ablate_disable_moe": self.ablate_disable_moe,
            "ablate_disable_conditioned_pde": self.ablate_disable_conditioned_pde,
            "ablate_disable_aux_losses": self.ablate_disable_aux_losses,
            "ablate_label_only_router": self.ablate_label_only_router,
            "moe_fast_mode": self.moe_fast_mode,
            "moe_pde_branches_per_sample": self.moe_pde_branches_per_sample,
            "moe_weight_threshold": self.moe_weight_threshold,
            "conditioned_scale": conditioned_scale,
            "physics_total": loss_physics_value,
            "expert_balance": float(loss_expert_balance.detach().item()),
            "condition_consistency": float(loss_condition_consistency.detach().item()),
            "parse_success_ratio": parse_ratio,
            **motion_mask_metrics,
            **{f"physics_{k}": v for k, v in physics_info.items()},
            **router_metrics,
            **expert_residual_metrics,
            **decoded_branch_metrics,
        }
        
        self.current_step += 1
        
        return total_loss


class PINNModelLogger(ModelLogger):
    """
    PINN 专用的模型保存器
    只保存 PINN 插件参数（不保存原模型参数）
    """
    
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x):
        super().__init__(output_path, remove_prefix_in_ckpt, state_dict_converter)
    
    def save_model(self, accelerator, model, file_name):
        """只保存 PINN 插件参数"""
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)

            physics_adapter_state_dict = {}
            pde_residuals_state_dict = {}
            if hasattr(unwrapped_model, 'physics_adapter'):
                physics_adapter_state_dict = {
                    k: v.detach().cpu()
                    for k, v in unwrapped_model.physics_adapter.state_dict().items()
                }
            if hasattr(unwrapped_model, 'pde_residuals'):
                pde_residuals_state_dict = {
                    k: v.detach().cpu()
                    for k, v in unwrapped_model.pde_residuals.state_dict().items()
                }
            
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            
            # 保存为 .pt 格式（包含额外元信息）
            pt_path = path.replace(".safetensors", ".pt")
            torch.save({
                'physics_adapter_state_dict': physics_adapter_state_dict,
                'pde_residuals_state_dict': pde_residuals_state_dict,
                'config': {
                    'checkpoint_format_version': 2,
                    'physics_weight': unwrapped_model.physics_weight if hasattr(unwrapped_model, 'physics_weight') else None,
                    'material_type': unwrapped_model.material_type if hasattr(unwrapped_model, 'material_type') else None,
                    'conditioned_physics_warmup_steps': (
                        unwrapped_model.conditioned_physics_warmup_steps
                        if hasattr(unwrapped_model, 'conditioned_physics_warmup_steps') else None
                    ),
                    'expert_balance_weight': (
                        unwrapped_model.expert_balance_weight
                        if hasattr(unwrapped_model, 'expert_balance_weight') else None
                    ),
                    'condition_consistency_weight': (
                        unwrapped_model.condition_consistency_weight
                        if hasattr(unwrapped_model, 'condition_consistency_weight') else None
                    ),
                    'moe_top_k': (
                        unwrapped_model.moe_top_k
                        if hasattr(unwrapped_model, 'moe_top_k') else None
                    ),
                    'moe_fast_mode': (
                        unwrapped_model.moe_fast_mode
                        if hasattr(unwrapped_model, 'moe_fast_mode') else True
                    ),
                    'moe_pde_branches_per_sample': (
                        unwrapped_model.moe_pde_branches_per_sample
                        if hasattr(unwrapped_model, 'moe_pde_branches_per_sample') else 1
                    ),
                    'moe_weight_threshold': (
                        unwrapped_model.moe_weight_threshold
                        if hasattr(unwrapped_model, 'moe_weight_threshold') else 0.05
                    ),
                    'ablation_flags': {
                        'ablate_disable_moe': (
                            unwrapped_model.ablate_disable_moe
                            if hasattr(unwrapped_model, 'ablate_disable_moe') else False
                        ),
                        'ablate_disable_conditioned_pde': (
                            unwrapped_model.ablate_disable_conditioned_pde
                            if hasattr(unwrapped_model, 'ablate_disable_conditioned_pde') else False
                        ),
                        'ablate_disable_aux_losses': (
                            unwrapped_model.ablate_disable_aux_losses
                            if hasattr(unwrapped_model, 'ablate_disable_aux_losses') else False
                        ),
                        'ablate_label_only_router': (
                            unwrapped_model.ablate_label_only_router
                            if hasattr(unwrapped_model, 'ablate_label_only_router') else False
                        ),
                    },
                },
            }, pt_path)
            print(f"PINN plugin saved to: {pt_path}")
            print(f"  Total keys: {len(physics_adapter_state_dict) + len(pde_residuals_state_dict)}")
    
    def on_epoch_end(self, accelerator, model, epoch_id):
        """每个 epoch 结束保存"""
        self.save_model(accelerator, model, f"pinn_plugin_epoch-{epoch_id}.pt")
    
    def on_training_end(self, accelerator, model, save_steps=None):
        """训练结束保存最终模型"""
        self.save_model(accelerator, model, "pinn_plugin_final.pt")


def pinn_parser():
    """扩展 wan_parser，增加 PINN 参数"""
    parser = wan_parser()
    
    # PINN 专用参数
    parser.add_argument(
        "--physics_weight", type=float, default=0.1,
        help="Weight for physics PDE loss (default: 0.1)"
    )
    parser.add_argument(
        "--physics_warmup_steps", type=int, default=500,
        help="Number of warmup steps for physics loss (default: 500)"
    )
    parser.add_argument(
        "--conditioned_physics_warmup_steps", type=int, default=1000,
        help="Warmup steps for metadata-conditioned constraints (default: 1000)"
    )
    parser.add_argument(
        "--material_type", type=str, default="auto",
        choices=["auto", "fluid", "rigid", "elastic", "particle", "mixed"],
        help="Material type for physics constraints (default: auto)"
    )
    parser.add_argument(
        "--adapter_hidden_dim", type=int, default=64,
        help="Hidden dimension for PhysicsAdapter (default: 64)"
    )
    parser.add_argument(
        "--pinn_checkpoint", type=str, default=None,
        help="Path to existing PINN plugin checkpoint to resume training"
    )
    parser.add_argument(
        "--expert_balance_weight", type=float, default=1e-3,
        help="Weight of expert usage balance regularization (default: 1e-3)"
    )
    parser.add_argument(
        "--condition_consistency_weight", type=float, default=1e-2,
        help="Weight of condition consistency regularization (default: 1e-2)"
    )
    parser.add_argument(
        "--moe_top_k", type=int, default=4,
        help="Top-k experts activated per sample in the MoE adapter (default: 4)"
    )
    parser.add_argument(
        "--ablate_disable_moe", action="store_true",
        help="Ablation: disable MoE experts and use fallback adapter path"
    )
    parser.add_argument(
        "--ablate_disable_conditioned_pde", action="store_true",
        help="Ablation: disable metadata-conditioned PDE modulation"
    )
    parser.add_argument(
        "--ablate_disable_aux_losses", action="store_true",
        help="Ablation: disable expert_balance and condition_consistency losses"
    )
    parser.add_argument(
        "--ablate_label_only_router", action="store_true",
        help="Ablation: only keep label routing, mask n/q conditional inputs"
    )
    parser.add_argument(
        "--frozen_model_gradient_checkpointing", action="store_true",
        help="Enable gradient checkpointing in frozen no-grad model forward (normally disable for speed)."
    )
    parser.add_argument(
        "--diagnostic_metrics_interval", type=int, default=10,
        help="Collect heavy MoE diagnostic metrics every N steps (default: 10)."
    )
    parser.add_argument(
        "--moe_fast_mode", dest="moe_fast_mode", action="store_true",
        help="Enable fast-mode branch pruning for MoE PDE residuals."
    )
    parser.add_argument(
        "--no_moe_fast_mode", dest="moe_fast_mode", action="store_false",
        help="Disable fast mode and use full multi-branch PDE residual computation."
    )
    parser.set_defaults(moe_fast_mode=True)
    parser.add_argument(
        "--moe_pde_branches_per_sample", type=int, default=1,
        help="Number of MoE branches per sample used for PDE residual in fast mode (default: 1)."
    )
    parser.add_argument(
        "--moe_weight_threshold", type=float, default=0.05,
        help="Skip branches with weights below this threshold in fast mode (default: 0.05)."
    )
    parser.add_argument(
        "--motion_mask_floor", type=float, default=0.08,
        help="Lower bound of self-supervised motion mask (default: 0.08)."
    )
    parser.add_argument(
        "--motion_mask_quantile", type=float, default=0.9,
        help="Per-sample quantile used to normalize motion energy (default: 0.9)."
    )
    parser.add_argument(
        "--motion_mask_warmup_steps", type=int, default=300,
        help="Warmup steps to blend from uniform mask to motion mask (default: 300)."
    )
    
    return parser


if __name__ == "__main__":
    parser = pinn_parser()
    args = parser.parse_args()
    
    # 数据集
    dataset = VideoDataset(args=args)
    
    # PINN 训练模块
    model = WanPINNTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        frozen_model_gradient_checkpointing=args.frozen_model_gradient_checkpointing,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        # PINN 参数
        physics_weight=args.physics_weight,
        physics_warmup_steps=args.physics_warmup_steps,
        conditioned_physics_warmup_steps=args.conditioned_physics_warmup_steps,
        material_type=args.material_type,
        adapter_hidden_dim=args.adapter_hidden_dim,
        pinn_checkpoint=args.pinn_checkpoint,
        expert_balance_weight=args.expert_balance_weight,
        condition_consistency_weight=args.condition_consistency_weight,
        moe_top_k=args.moe_top_k,
        ablate_disable_moe=args.ablate_disable_moe,
        ablate_disable_conditioned_pde=args.ablate_disable_conditioned_pde,
        ablate_disable_aux_losses=args.ablate_disable_aux_losses,
        ablate_label_only_router=args.ablate_label_only_router,
        diagnostic_metrics_interval=args.diagnostic_metrics_interval,
        moe_fast_mode=args.moe_fast_mode,
        moe_pde_branches_per_sample=args.moe_pde_branches_per_sample,
        moe_weight_threshold=args.moe_weight_threshold,
        motion_mask_floor=args.motion_mask_floor,
        motion_mask_quantile=args.motion_mask_quantile,
        motion_mask_warmup_steps=args.motion_mask_warmup_steps,
        # LoRA 参数（可选）
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
    )
    
    # 使用 PINN 专用 Logger（只保存插件参数）
    model_logger = PINNModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    
    # 优化器：只优化 PINN 可训练参数
    optimizer = torch.optim.AdamW(
        model.trainable_modules(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    # 启动多卡训练
    tensorboard_dir = args.tensorboard_dir or os.path.join(args.output_path, "tb")
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        find_unused_parameters=args.find_unused_parameters,
        num_workers=args.dataset_num_workers,
        ddp_timeout_seconds=args.ddp_timeout_seconds,
        tensorboard_dir=tensorboard_dir,
        tensorboard_log_steps=args.tensorboard_log_steps,
        heartbeat_log_steps=args.heartbeat_log_steps,
    )
