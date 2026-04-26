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
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import re
import hashlib
import html
import math
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.pinn_operators import MaterialPDEResiduals, MaterialClassifier
from diffsynth.models.pinn_adapter import (
    CORE_ABLATION_MODES,
    PhysicsAdapter,
    ONLY_U_RECOVERY_PHASES,
)
from diffsynth.models.pinn_contracts import (
    EXPERT_FIELD_RECIPE_VERSION,
    EXPERT_FIELD_RECIPES,
    FIELD_CONTRACT_VERSION,
    PHENOMENON_LABELS,
    PHYSICS_ATTR_DIM,
    split_attribute_bank,
)
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, ModelLogger, launch_training_task, wan_parser

os.environ["TOKENIZERS_PARALLELISM"] = "false"


PHENOMENON_TO_ID = {name: idx for idx, name in enumerate(PHENOMENON_LABELS)}
PHENOMENON_NAME_LOOKUP = {name.lower(): name for name in PHENOMENON_LABELS}
PHENOMENON_ALIAS = {}
PHENOMENON_TO_RESIDUAL_METHOD = {
    "Rigid Body": "rigid_body_residual",
    "Elastic": "elastic_residual",
    "Fluid": "fluid_residual",
    "Compressible Flow": "compressible_flow_residual",
    "Phase Change": "phase_change_residual",
    "Collision/Contact": "collision_contact_residual",
    "Granular": "granular_residual",
    "Fracture": "fracture_residual",
    "Thermal": "thermal_residual",
    "Optical": "optical_residual",
}

PHYSICAL_MASK_RECIPE_VERSION = "active_fused_field_v1"
PHYSICAL_MASK_ELIGIBLE_FIELDS = ("d", "u", "rho", "T", "alpha", "psi", "j", "D")
PHYSICAL_MASK_EVENT_FIELDS = ("j", "D")
PHYSICAL_MASK_BOUNDARY_FIELDS = ("rho", "alpha", "T")
PHYSICAL_MASK_PHENOMENON_WEIGHTS = {
    "Thermal": {
        "evolution": {"T": 2.0, "u": 0.5},
        "boundary": {"T": 2.0},
    },
    "Phase Change": {
        "evolution": {"T": 1.5, "rho": 1.5, "alpha": 1.5, "u": 0.75},
        "boundary": {"T": 1.5, "rho": 1.5, "alpha": 1.5},
    },
    "Optical": {
        "evolution": {"psi": 1.5, "alpha": 1.25},
        "boundary": {"alpha": 1.25},
    },
    "Collision/Contact": {
        "event": {"j": 1.5},
    },
    "Fracture": {
        "evolution": {"D": 1.25},
        "event": {"D": 1.5, "j": 1.25},
    },
}
OBSERVABLE_PROXY_RECIPE_VERSION = "dense_flow_jacobian_region_v1"
STAGE1_ENCODER_STATE_PREFIXES = (
    "physics_encoder_shared.",
    "shared_attribute_head.",
    "u_head.",
    "d_head.",
)
ABLATION_PRESET_DEFAULTS = {
    "legacy_direct_bank": {
        "observable_target_mode": "flow_plus_deformation",
        "secondary_field_strategy": "legacy_direct_bank",
        "active_field_set": "legacy",
        "field_enable_schedule": "legacy",
    },
    "u_only_direct_prho": {
        "observable_target_mode": "flow_only",
        "secondary_field_strategy": "direct_bank",
        "active_field_set": "u,p,rho",
        "field_enable_schedule": "fixed_only_u_recovery",
    },
    "u_only_ufirst_prho": {
        "observable_target_mode": "flow_only",
        "secondary_field_strategy": "u_first_constructor",
        "active_field_set": "u,p,rho",
        "field_enable_schedule": "fixed_only_u_recovery",
    },
    "u_only_ufirst_prho_detach": {
        "observable_target_mode": "flow_only",
        "secondary_field_strategy": "u_first_constructor_detach",
        "active_field_set": "u,p,rho",
        "field_enable_schedule": "fixed_only_u_recovery",
    },
}
FIELD_RECOVERY_PHASE_TO_INDEX = {
    phase_name: phase_idx
    for phase_idx, phase_name in enumerate(ONLY_U_RECOVERY_PHASES)
}


def resolve_ablation_policy(
    ablation_preset,
    observable_target_mode,
    secondary_field_strategy,
    active_field_set,
    field_enable_schedule,
):
    preset = str(ablation_preset or "legacy_direct_bank")
    if preset not in ABLATION_PRESET_DEFAULTS:
        raise ValueError(
            f"Unsupported ablation_preset={preset!r}. "
            f"Expected one of {sorted(ABLATION_PRESET_DEFAULTS.keys())}."
        )
    defaults = ABLATION_PRESET_DEFAULTS[preset]

    def _resolve(value, key):
        if value in (None, "", "auto"):
            return defaults[key]
        return str(value)

    return {
        "ablation_preset": preset,
        "observable_target_mode": _resolve(observable_target_mode, "observable_target_mode"),
        "secondary_field_strategy": _resolve(secondary_field_strategy, "secondary_field_strategy"),
        "active_field_set": _resolve(active_field_set, "active_field_set"),
        "field_enable_schedule": _resolve(field_enable_schedule, "field_enable_schedule"),
    }


class FrozenDenseFlowTeacher(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        hidden_dim = max(int(hidden_dim), 8)
        self.pair_backbone = nn.Sequential(
            nn.Conv2d(6, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1),
        )
        self.has_learned_weights = False
        self.eval()
        self.requires_grad_(False)

    @staticmethod
    def _grad_x(image):
        return F.pad(image[..., 1:] - image[..., :-1], (0, 1, 0, 0))

    @staticmethod
    def _grad_y(image):
        return F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in (
                "flow_backbone_state_dict",
                "state_dict",
                "model_state_dict",
                "model",
            ):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    state_dict = value
                    break
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"Invalid flow backbone checkpoint at {checkpoint_path}: expected dict payload."
            )
        cleaned_state_dict = {}
        known_prefixes = (
            "module.",
            "flow_teacher.",
            "observable_proxy_extractor.flow_teacher.",
        )
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            clean_key = key
            changed = True
            while changed:
                changed = False
                for prefix in known_prefixes:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        changed = True
            if clean_key.startswith("pair_backbone."):
                cleaned_state_dict[clean_key] = value
        load_result = self.load_state_dict(cleaned_state_dict, strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys in flow backbone checkpoint: {load_result.unexpected_keys[:20]}"
            )
        missing = [
            key for key in load_result.missing_keys
            if not key.endswith("num_batches_tracked")
        ]
        if missing:
            raise RuntimeError(
                f"Missing keys in flow backbone checkpoint: {missing[:20]}"
            )
        self.has_learned_weights = True
        self.eval()
        self.requires_grad_(False)

    def _analytic_flow(self, video):
        batch_size, _, num_frames, height, width = video.shape
        if num_frames <= 1:
            return torch.zeros(
                batch_size,
                2,
                1,
                height,
                width,
                device=video.device,
                dtype=video.dtype,
            )
        gray = video.float().mean(dim=1, keepdim=True)
        current = gray[:, :, :-1].permute(0, 2, 1, 3, 4).reshape(-1, 1, height, width)
        future = gray[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(-1, 1, height, width)
        grad_x = self._grad_x(current)
        grad_y = self._grad_y(current)
        grad_t = future - current
        denom = grad_x.square() + grad_y.square() + 1e-4
        flow_x = (-grad_t * grad_x / denom).clamp(-1.0, 1.0)
        flow_y = (-grad_t * grad_y / denom).clamp(-1.0, 1.0)
        pair_flow = torch.cat([flow_x, flow_y], dim=1)
        return pair_flow.view(batch_size, num_frames - 1, 2, height, width).permute(0, 2, 1, 3, 4)

    def _learned_flow(self, video):
        batch_size, channels, num_frames, height, width = video.shape
        if num_frames <= 1:
            return torch.zeros(
                batch_size,
                2,
                1,
                height,
                width,
                device=video.device,
                dtype=video.dtype,
            )
        current = video[:, :, :-1].permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
        future = video[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
        pair_input = torch.cat([current, future], dim=1).to(dtype=video.dtype)
        pair_flow = self.pair_backbone(pair_input)
        return pair_flow.view(batch_size, num_frames - 1, 2, height, width).permute(0, 2, 1, 3, 4)

    def forward(self, video):
        if video.ndim != 5:
            raise RuntimeError(f"Flow teacher expects 5D BCHWT video tensor, got {tuple(video.shape)}.")
        if self.has_learned_weights:
            pair_flow = self._learned_flow(video)
        else:
            pair_flow = self._analytic_flow(video)
        if pair_flow.shape[2] == 0:
            return torch.zeros(
                video.shape[0],
                2,
                video.shape[2],
                video.shape[3],
                video.shape[4],
                device=video.device,
                dtype=video.dtype,
            )
        last_flow = pair_flow[:, :, -1:]
        full_flow = torch.cat([pair_flow, last_flow], dim=2)
        return torch.nan_to_num(full_flow.to(dtype=video.dtype), nan=0.0, posinf=0.0, neginf=0.0)


class ObservableProxyExtractor(nn.Module):
    def __init__(self, flow_teacher):
        super().__init__()
        self.flow_teacher = flow_teacher
        self.eval()
        self.requires_grad_(False)

    @staticmethod
    def _grad_x(tensor):
        return F.pad(tensor[..., 1:] - tensor[..., :-1], (0, 1, 0, 0))

    @staticmethod
    def _grad_y(tensor):
        return F.pad(tensor[..., 1:, :] - tensor[..., :-1, :], (0, 0, 0, 1))

    def forward(self, rgb_video):
        if rgb_video.ndim != 5 or rgb_video.shape[1] != 3:
            raise RuntimeError(
                f"Observable proxy extractor expects [B,3,T,H,W], got {tuple(rgb_video.shape)}."
            )
        rgb_video = torch.nan_to_num(rgb_video.float(), nan=0.0, posinf=0.0, neginf=0.0)
        flow_proxy = self.flow_teacher(rgb_video).detach()
        flow_x = flow_proxy[:, 0:1]
        flow_y = flow_proxy[:, 1:2]
        du_dx = self._grad_x(flow_x)
        du_dy = self._grad_y(flow_x)
        dv_dx = self._grad_x(flow_y)
        dv_dy = self._grad_y(flow_y)
        deformation_proxy = torch.cat([du_dx, du_dy, dv_dx, dv_dy], dim=1)

        divergence = du_dx + dv_dy
        curl = dv_dx - du_dy
        motion_mag = torch.sqrt(flow_proxy.square().sum(dim=1, keepdim=True) + 1e-6)
        deformation_mag = torch.sqrt(deformation_proxy.square().sum(dim=1, keepdim=True) + 1e-6)
        affine_flow = F.avg_pool3d(flow_proxy, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        affine_residual = (flow_proxy - affine_flow).abs().mean(dim=1, keepdim=True)

        gray_video = rgb_video.mean(dim=1, keepdim=True)
        gray_grad_x = self._grad_x(gray_video)
        gray_grad_y = self._grad_y(gray_video)
        texture_mag = torch.sqrt(gray_grad_x.square() + gray_grad_y.square() + 1e-6)
        proxy_conf = torch.sigmoid(2.0 * (texture_mag + 0.5 * motion_mag - 0.15)).clamp(0.05, 1.0)

        return {
            "flow_proxy": flow_proxy.to(dtype=rgb_video.dtype).detach(),
            "deformation_proxy": deformation_proxy.to(dtype=rgb_video.dtype).detach(),
            "proxy_conf": proxy_conf.to(dtype=rgb_video.dtype).detach(),
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
        use_dual_noise_experts=None,
        dual_noise_expert_boundary=0.417,
        # PINN 专用参数
        physics_weight=0.1,
        physics_warmup_steps=500,
        conditioned_physics_warmup_steps=1000,
        adapter_hidden_dim=64,
        physics_attr_dim=PHYSICS_ATTR_DIM,
        expert_pde_sigma_threshold=0.40,
        expert_pde_sigma_threshold_target=1.00,
        training_stage="full_pinn",
        pinn_checkpoint=None,
        stage1_pretrained_encoder=None,
        flow_backbone_ckpt=None,
        encoder_freeze_steps=1000,
        encoder_lr_scale=0.3,
        ablation_preset="legacy_direct_bank",
        observable_target_mode="auto",
        secondary_field_strategy="auto",
        active_field_set="auto",
        field_enable_schedule="auto",
        field_recovery_phase="core",
        field_recovery_step_schedule="",
        field_recovery_loss_ramp_steps=100,
        run_full_pinn_after_recovery=False,
        freeze_u_encoder_during_recovery=True,
        expert_balance_weight=1e-3,
        condition_consistency_weight=1e-2,
        moe_top_k=4,
        ablate_disable_moe=False,
        ablate_disable_conditioned_pde=False,
        ablate_disable_aux_losses=False,
        ablate_label_only_router=False,
        core_ablation_mode="full",
        allow_ablation_checkpoint_mismatch=False,
        diagnostic_metrics_interval=10,
        motion_mask_floor=0.08,
        motion_mask_quantile=0.9,
        motion_mask_warmup_steps=300,
        physical_mask_transition_steps=1000,
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
        rl_policy_weight=1e-2,
        rl_entropy_weight=1e-3,
        rl_reward_decay=0.95,
        rl_reward_quality_weight=0.5,
        rl_reward_stability_weight=0.1,
        rl_warmup_steps=500,
        rl_hidden_dim=None,
        state_align_warmup_steps=1000,
        state_align_x_weight=0.0,
        state_align_v_weight=0.05,
        curriculum_transition_start_step=1000,
        curriculum_transition_steps=1000,
        physics_weight_target=None,
        output_physics_weight=1.0,
        state_align_v_weight_target=None,
        decoded_branch_consistency_weight=1e-2,
        enable_explainability_reports=True,
        explainability_top_experts=6,
        debug_fixed_timestep_fraction=None,
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
            physics_attr_dim=physics_attr_dim,
            num_phenomena=len(PHENOMENON_LABELS),
            moe_top_k=moe_top_k,
            pde_residuals=None,  # 稍后设置，因为还需要创建 pde_residuals
            physics_state_mode=physics_state_mode,
            use_sigma_gate=use_sigma_gate,
            sigma_gate_curve=sigma_gate_curve,
            use_sigma_conditioning=use_sigma_conditioning,
            sigma_conditioning_dim=(
                adapter_hidden_dim if sigma_conditioning_dim is None else sigma_conditioning_dim
            ),
            sigma_gate_floor=sigma_gate_floor,
            use_adaptive_condition_injection=use_adaptive_condition_injection,
            adaptive_conditioning_dim=(
                adapter_hidden_dim if adaptive_conditioning_dim is None else adaptive_conditioning_dim
            ),
            adaptive_conditioning_strength=adaptive_conditioning_strength,
            adaptive_conditioning_gate_floor=adaptive_conditioning_gate_floor,
            enable_rl_expert_optimization=enable_rl_expert_optimization,
            rl_hidden_dim=adapter_hidden_dim if rl_hidden_dim is None else rl_hidden_dim,
            rl_reward_decay=rl_reward_decay,
            strict_physical_state_contract=True,
            core_ablation_mode=core_ablation_mode,
        )
        self.physics_adapter.train()
        self.physics_adapter.requires_grad_(True)
        
        # PDE 残差计算器
        self.pde_residuals = MaterialPDEResiduals(
            num_phenomena=len(PHENOMENON_LABELS),
            q_input_dim=self.physics_adapter.q_input_dim,
            n_numeric_dim=self.physics_adapter.n_numeric_dim,
            strict_metadata_contract=True,
        )
        self.pde_residuals.eval()
        self.pde_residuals.requires_grad_(False)
        self.training_stage = training_stage
        self.stage1_pretrained_encoder = stage1_pretrained_encoder
        self.flow_backbone_ckpt = flow_backbone_ckpt
        self.encoder_freeze_steps = max(int(encoder_freeze_steps), 0)
        self.encoder_lr_scale = max(float(encoder_lr_scale), 0.0)
        self.effective_encoder_freeze_steps = int(self.encoder_freeze_steps)
        self.effective_encoder_lr_scale = float(self.encoder_lr_scale)
        self._resume_stability_mode = False
        self._resume_stability_lr_cap = 2e-6
        self._resume_stability_extra_freeze_steps = 1000
        self._resume_stability_encoder_lr_scale = 0.1
        self._resume_checkpoint_step = 0
        self._resume_checkpoint_epoch = 0
        self._stage2_encoder_frozen = None
        self._encoder_lr_scale_hooks = []
        self.observable_proxy_extractor = ObservableProxyExtractor(
            FrozenDenseFlowTeacher(hidden_dim=max(adapter_hidden_dim // 2, 16))
        )
        if self.flow_backbone_ckpt is not None:
            self.observable_proxy_extractor.flow_teacher.load_checkpoint(self.flow_backbone_ckpt)
        self.observable_proxy_extractor.eval()
        self.observable_proxy_extractor.requires_grad_(False)
        self.observable_diagnostics_enabled = self.training_stage in {"observable_pretrain", "encoder_completion"}

        # Checkpoint runtime validation compares against these fields, so they
        # must exist before any checkpoint compatibility checks run.
        self._initialize_runtime_training_config(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            frozen_model_gradient_checkpointing=frozen_model_gradient_checkpointing,
            extra_inputs=extra_inputs,
            max_timestep_boundary=max_timestep_boundary,
            min_timestep_boundary=min_timestep_boundary,
            use_dual_noise_experts=use_dual_noise_experts,
            dual_noise_expert_boundary=dual_noise_expert_boundary,
            physics_weight=physics_weight,
            physics_warmup_steps=physics_warmup_steps,
            conditioned_physics_warmup_steps=conditioned_physics_warmup_steps,
            adapter_hidden_dim=adapter_hidden_dim,
            physics_attr_dim=physics_attr_dim,
            expert_pde_sigma_threshold=expert_pde_sigma_threshold,
            expert_pde_sigma_threshold_target=expert_pde_sigma_threshold_target,
            expert_balance_weight=expert_balance_weight,
            condition_consistency_weight=condition_consistency_weight,
            ablate_disable_moe=ablate_disable_moe,
            ablate_disable_conditioned_pde=ablate_disable_conditioned_pde,
            ablate_disable_aux_losses=ablate_disable_aux_losses,
            ablate_label_only_router=ablate_label_only_router,
            core_ablation_mode=core_ablation_mode,
            allow_ablation_checkpoint_mismatch=allow_ablation_checkpoint_mismatch,
            diagnostic_metrics_interval=diagnostic_metrics_interval,
            motion_mask_floor=motion_mask_floor,
            motion_mask_quantile=motion_mask_quantile,
            motion_mask_warmup_steps=motion_mask_warmup_steps,
            physical_mask_transition_steps=physical_mask_transition_steps,
            physics_state_mode=physics_state_mode,
            use_sigma_gate=use_sigma_gate,
            sigma_gate_curve=sigma_gate_curve,
            use_sigma_conditioning=use_sigma_conditioning,
            use_adaptive_condition_injection=use_adaptive_condition_injection,
            enable_rl_expert_optimization=enable_rl_expert_optimization,
            rl_policy_weight=rl_policy_weight,
            rl_entropy_weight=rl_entropy_weight,
            rl_reward_decay=rl_reward_decay,
            rl_reward_quality_weight=rl_reward_quality_weight,
            rl_reward_stability_weight=rl_reward_stability_weight,
            rl_warmup_steps=rl_warmup_steps,
            state_align_warmup_steps=state_align_warmup_steps,
            state_align_x_weight=state_align_x_weight,
            state_align_v_weight=state_align_v_weight,
            curriculum_transition_start_step=curriculum_transition_start_step,
            curriculum_transition_steps=curriculum_transition_steps,
            physics_weight_target=physics_weight_target,
            output_physics_weight=output_physics_weight,
            state_align_v_weight_target=state_align_v_weight_target,
            decoded_branch_consistency_weight=decoded_branch_consistency_weight,
            enable_explainability_reports=enable_explainability_reports,
            explainability_top_experts=explainability_top_experts,
            training_stage=training_stage,
            stage1_pretrained_encoder=stage1_pretrained_encoder,
            flow_backbone_ckpt=flow_backbone_ckpt,
            encoder_freeze_steps=encoder_freeze_steps,
            encoder_lr_scale=encoder_lr_scale,
            ablation_preset=ablation_preset,
            observable_target_mode=observable_target_mode,
            secondary_field_strategy=secondary_field_strategy,
            active_field_set=active_field_set,
            field_enable_schedule=field_enable_schedule,
            field_recovery_phase=field_recovery_phase,
            field_recovery_step_schedule=field_recovery_step_schedule,
            field_recovery_loss_ramp_steps=field_recovery_loss_ramp_steps,
            run_full_pinn_after_recovery=run_full_pinn_after_recovery,
            freeze_u_encoder_during_recovery=freeze_u_encoder_during_recovery,
        )

        # 加载 PINN checkpoint（如果有）
        self._checkpoint_training_state = None  # 存储训练状态用于后续恢复
        if pinn_checkpoint is not None:
            checkpoint = torch.load(pinn_checkpoint, map_location="cpu", weights_only=False)
            checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
            checkpoint_format_version = int(checkpoint_config.get("checkpoint_format_version", 0) or 0)
            if checkpoint_format_version < 14:
                raise RuntimeError(
                    "Checkpoint is incompatible with the hierarchical-refine-removed explicit-attribute-bank v2 architecture. "
                    "Resume from a format_version >= 14 checkpoint or train a new adapter from scratch."
                )
            self._validate_checkpoint_runtime_config(checkpoint_config)
            if 'physics_adapter_state_dict' in checkpoint:
                load_result = self.physics_adapter.load_state_dict(
                    checkpoint['physics_adapter_state_dict'], strict=False
                )
                self._validate_state_dict_load(
                    "physics_adapter",
                    load_result,
                    checkpoint_format_version=checkpoint_format_version,
                )
                print(f"Loaded PhysicsAdapter from {pinn_checkpoint}")
            if 'pde_residuals_state_dict' in checkpoint:
                load_result = self.pde_residuals.load_state_dict(
                    checkpoint['pde_residuals_state_dict'], strict=False
                )
                self._validate_state_dict_load(
                    "pde_residuals",
                    load_result,
                    checkpoint_format_version=checkpoint_format_version,
                )
                print(f"Loaded PDE Residuals from {pinn_checkpoint}")
                self.pde_residuals.eval()
                self.pde_residuals.requires_grad_(False)
            # 加载训练状态（如果有）
            if 'training_state' in checkpoint and not self.allow_ablation_checkpoint_mismatch:
                self._checkpoint_training_state = checkpoint['training_state']
                print(f"Found training state in checkpoint: step={self._checkpoint_training_state.get('current_step', 0)}, epoch={self._checkpoint_training_state.get('current_epoch', 0)}")
            elif 'training_state' in checkpoint:
                print("Ignoring checkpoint training_state for ablation warm-start.")
            if checkpoint_config.get("training_stage") in {"observable_pretrain", "encoder_completion"}:
                self.observable_diagnostics_enabled = True
            encoder_stage_state_dict = checkpoint.get("encoder_stage_state_dict")
            if isinstance(encoder_stage_state_dict, dict) and any(
                key.startswith(("u_head.", "d_head."))
                for key in encoder_stage_state_dict.keys()
            ):
                self.observable_diagnostics_enabled = True

        if (
            self.training_stage in {"encoder_completion", "full_pinn"}
            and pinn_checkpoint is None
            and stage1_pretrained_encoder is not None
        ):
            self._load_stage1_pretrained_encoder(stage1_pretrained_encoder)
            self.observable_diagnostics_enabled = True

        self.current_step = 0
        self._configure_resume_stability_mode()
        self._register_encoder_lr_scale_hooks()
        self._configure_training_stage_trainability()

        # 统计可训练参数
        
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
        self.debug_fixed_timestep_fraction = None
        if debug_fixed_timestep_fraction is not None:
            self.debug_fixed_timestep_fraction = float(
                min(max(debug_fixed_timestep_fraction, 0.0), 1.0)
            )
        self._validate_multi_expert_training_config()
        self.current_step = 0
        self._last_metrics = {}
        self._latest_explainability_snapshot = None

        # 将消融配置同步到底层模块，保持调用层逻辑清晰
        self.physics_adapter.set_ablation_modes(
            use_moe=(
                not self.ablate_disable_moe
                and self.core_ablation_mode != "generic_latent_correction"
            ),
            label_only_mode=(
                self.ablate_label_only_router
                or self.core_ablation_mode == "wo_learned_expert_routing"
            ),
        )
        self.pde_residuals.set_conditioning_enabled(
            enabled=(
                not self.ablate_disable_conditioned_pde
                and self.core_ablation_mode not in {
                    "generic_latent_correction",
                    "wo_explicit_physical_interface",
                    "wo_pde_residuals",
                }
            )
        )
        print("Ablation flags:")
        print(f"  core_ablation_mode={self.core_ablation_mode}")
        print(f"  allow_ablation_checkpoint_mismatch={self.allow_ablation_checkpoint_mismatch}")
        print(f"  disable_moe={self.ablate_disable_moe}")
        print(f"  disable_conditioned_pde={self.ablate_disable_conditioned_pde}")
        print(f"  disable_aux_losses={self.ablate_disable_aux_losses}")
        print(f"  label_only_router={self.ablate_label_only_router}")
        print(f"  moe_top_k={self.moe_top_k}")
        print(f"  motion_mask_floor={self.motion_mask_floor}")
        print(f"  motion_mask_quantile={self.motion_mask_quantile}")
        print(f"  motion_mask_warmup_steps={self.motion_mask_warmup_steps}")
        print(f"  physical_mask_transition_steps={self.physical_mask_transition_steps}")
        print(f"  physics_state_mode={self.physics_state_mode}")
        print(f"  use_sigma_gate={self.use_sigma_gate}")
        print(f"  sigma_gate_curve={self.sigma_gate_curve}")
        print(f"  use_sigma_conditioning={self.use_sigma_conditioning}")
        print(f"  sigma_conditioning_dim={self.sigma_conditioning_dim}")
        print(f"  sigma_gate_floor={self.sigma_gate_floor}")
        print(f"  use_dual_noise_experts={self.use_dual_noise_experts}")
        print(f"  dual_noise_expert_boundary={self.dual_noise_expert_boundary}")
        print(f"  has_second_dit_expert={self.pipe.dit2 is not None}")
        print("  dual_noise_expert_order=dit->high_noise, dit2->low_noise")
        print(f"  use_adaptive_condition_injection={self.use_adaptive_condition_injection}")
        print(f"  adaptive_conditioning_dim={self.adaptive_conditioning_dim}")
        print(f"  adaptive_conditioning_strength={self.adaptive_conditioning_strength}")
        print(f"  adaptive_conditioning_gate_floor={self.adaptive_conditioning_gate_floor}")
        print(f"  enable_rl_expert_optimization={self.enable_rl_expert_optimization}")
        print(f"  rl_policy_weight={self.rl_policy_weight}")
        print(f"  rl_entropy_weight={self.rl_entropy_weight}")
        print(f"  rl_reward_decay={self.rl_reward_decay}")
        print(f"  rl_reward_quality_weight={self.rl_reward_quality_weight}")
        print(f"  rl_reward_stability_weight={self.rl_reward_stability_weight}")
        print(f"  rl_warmup_steps={self.rl_warmup_steps}")
        print(f"  rl_hidden_dim={self.rl_hidden_dim}")
        print(f"  state_align_warmup_steps={self.state_align_warmup_steps}")
        print(f"  state_align_x_weight={self.state_align_x_weight}")
        print(f"  state_align_v_weight={self.state_align_v_weight}")
        print(f"  curriculum_transition_start_step={self.curriculum_transition_start_step}")
        print(f"  curriculum_transition_steps={self.curriculum_transition_steps}")
        print(f"  expert_pde_sigma_threshold={self.expert_pde_sigma_threshold}")
        print(f"  expert_pde_sigma_threshold_target={self.expert_pde_sigma_threshold_target}")
        print(f"  physics_weight_target={self.physics_weight_target}")
        print(f"  output_physics_weight={self.output_physics_weight}")
        print(f"  state_align_v_weight_target={self.state_align_v_weight_target}")
        print(f"  enable_explainability_reports={self.enable_explainability_reports}")
        print(f"  explainability_top_experts={self.explainability_top_experts}")
        print(f"  debug_fixed_timestep_fraction={self.debug_fixed_timestep_fraction}")
        print(f"  effective_encoder_freeze_steps={self.effective_encoder_freeze_steps}")
        print(f"  effective_encoder_lr_scale={self.effective_encoder_lr_scale}")
        print(f"  resume_stability_mode={self._resume_stability_mode}")
    
    
    def get_physics_weight(self):
        """获取当前物理损失权重（带预热）"""
        scheduled_weight = self._get_curriculum_weight(
            self.physics_weight,
            self.physics_weight_target,
        )
        if self.current_step < self.physics_warmup_steps:
            alpha = self.current_step / max(self.physics_warmup_steps, 1)
            return scheduled_weight * alpha
        return scheduled_weight

    def get_expert_pde_sigma_threshold(self):
        """训练前期只约束低噪声，随后逐步放宽到目标 sigma 范围。"""
        if self.physics_warmup_steps <= 0:
            return float(self.expert_pde_sigma_threshold_target)
        alpha = min(max(self.current_step / max(self.physics_warmup_steps, 1), 0.0), 1.0)
        return (
            float(self.expert_pde_sigma_threshold)
            + (
                float(self.expert_pde_sigma_threshold_target)
                - float(self.expert_pde_sigma_threshold)
            ) * alpha
        )

    def _get_curriculum_progress(self):
        """两阶段课程学习进度: 0 表示 phase A, 1 表示完全进入 phase B。"""
        if self.curriculum_transition_steps <= 0:
            return 1.0
        if self.current_step <= self.curriculum_transition_start_step:
            return 0.0
        progress = (
            (self.current_step - self.curriculum_transition_start_step)
            / max(self.curriculum_transition_steps, 1)
        )
        return min(max(progress, 0.0), 1.0)

    def _get_curriculum_weight(self, base_weight, target_weight):
        progress = self._get_curriculum_progress()
        return float(base_weight) + (float(target_weight) - float(base_weight)) * progress

    def _validate_multi_expert_training_config(self):
        if self.use_dual_noise_experts and self.pipe.dit2 is None:
            raise ValueError(
                "Dual-noise-expert training is enabled, but only one Wan DiT expert was loaded. "
                "Load both high_noise_model and low_noise_model, or disable --use_dual_noise_experts."
            )
        if self.ablate_disable_moe:
            return
        if self.moe_top_k < 2:
            raise ValueError(
                "Invalid multi-expert training configuration: moe_top_k < 2 degenerates the setup "
                "into single-expert routing, and cannot support RL or cooperative MoE training. "
                "Use --ablate_disable_moe for a true single-path ablation, or set --moe_top_k >= 2."
            )

    def get_conditioned_alpha(self):
        """条件化约束强度预热"""
        if self.current_step < self.conditioned_physics_warmup_steps:
            return self.current_step / max(self.conditioned_physics_warmup_steps, 1)
        return 1.0

    def get_rl_alpha(self):
        """RL 策略损失预热"""
        if not self.enable_rl_expert_optimization:
            return 0.0
        if self.rl_warmup_steps <= 0:
            return 1.0
        if self.current_step < self.rl_warmup_steps:
            return self.current_step / max(self.rl_warmup_steps, 1)
        return 1.0

    def get_state_align_alpha(self):
        """显式物理状态对齐预热，避免冷启动阶段压过主任务。"""
        if self.state_align_warmup_steps <= 0:
            return 1.0
        if self.current_step < self.state_align_warmup_steps:
            return self.current_step / max(self.state_align_warmup_steps, 1)
        return 1.0

    def get_state_align_v_weight(self):
        """v-状态对齐只作为桥接约束，phase B 后逐步弱化。"""
        return self._get_curriculum_weight(
            self.state_align_v_weight,
            self.state_align_v_weight_target,
        )

    @staticmethod
    def _safe_text(value):
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _normalize_label(label):
        """Normalize label: strip whitespace and standardize spacing."""
        clean = WanPINNTrainingModule._safe_text(label)
        # Remove extra whitespace but preserve original case
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

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

    @staticmethod
    def _tensor_debug_summary(value):
        if not isinstance(value, torch.Tensor):
            return "value=<non-tensor>"
        tensor = value.detach().float()
        finite_mask = torch.isfinite(tensor)
        finite_count = int(finite_mask.sum().item())
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
        if finite_count > 0:
            finite_values = tensor[finite_mask]
            min_value = float(finite_values.min().item())
            max_value = float(finite_values.max().item())
            mean_value = float(finite_values.mean().item())
        else:
            min_value = float("nan")
            max_value = float("nan")
            mean_value = float("nan")
        return (
            f"shape={tuple(tensor.shape)} "
            f"min={min_value:.6f} "
            f"max={max_value:.6f} "
            f"mean={mean_value:.6f} "
            f"finite_count={finite_count} "
            f"nan_count={nan_count} "
            f"inf_count={inf_count}"
        )

    @staticmethod
    def _debug_value_summary(value):
        if isinstance(value, torch.Tensor):
            detached = value.detach().cpu()
            if detached.numel() == 1:
                return f"{float(detached.reshape(-1)[0].item()):.6f}"
            if detached.ndim <= 1 and detached.numel() <= 8:
                return str([float(x) for x in detached.reshape(-1).tolist()])
            return WanPINNTrainingModule._tensor_debug_summary(detached)
        if isinstance(value, (list, tuple)):
            if len(value) <= 8:
                return str(list(value))
            return str(list(value[:8]) + ["..."])
        return str(value)

    @staticmethod
    def _metadata_debug_summary(metadata):
        if not isinstance(metadata, dict):
            return "metadata=<none>"
        fields = []
        for key in (
            "label_name",
            "label_names",
            "label_id",
            "label_ids",
            "noise_regime",
            "active_dit_expert_index",
            "motion_mask_source",
        ):
            if key not in metadata:
                continue
            fields.append(f"{key}={WanPINNTrainingModule._debug_value_summary(metadata[key])}")
        return " ".join(fields) if len(fields) > 0 else "metadata=<empty>"

    def _set_forward_debug_context(self, **kwargs):
        context = getattr(self, "_forward_debug_context", {})
        if not isinstance(context, dict):
            context = {}
        for key, value in kwargs.items():
            if value is None:
                context.pop(key, None)
            else:
                context[key] = value
        self._forward_debug_context = context

    def _forward_debug_context_suffix(self):
        context = getattr(self, "_forward_debug_context", None)
        if not isinstance(context, dict) or len(context) == 0:
            return ""
        ordered_keys = (
            "training_stage",
            "timestep_id",
            "timestep",
            "sigma",
            "phenomenon",
            "metadata",
        )
        parts = []
        seen = set()
        for key in ordered_keys:
            if key in context:
                parts.append(f"{key}={self._debug_value_summary(context[key])}")
                seen.add(key)
        for key in sorted(context.keys()):
            if key in seen:
                continue
            parts.append(f"{key}={self._debug_value_summary(context[key])}")
        return " context: " + " ".join(parts)

    def _raise_invalid_tensor(self, name, value):
        raise FloatingPointError(
            f"Invalid tensor detected for {name} at step={self.current_step}. "
            f"{self._tensor_debug_summary(value)}"
            f"{self._forward_debug_context_suffix()}"
        )

    def _ensure_finite_tensor(self, name, value):
        if isinstance(value, torch.Tensor) and (torch.isnan(value).any() or torch.isinf(value).any()):
            self._raise_invalid_tensor(name, value)
        return value

    def _clear_adapter_debug_context(self):
        if hasattr(self.physics_adapter, "clear_debug_context"):
            self.physics_adapter.clear_debug_context()

    def _prepare_physics_adapter_debug(self, phase, phenomenon, timestep_id, sigma, metadata):
        metadata_summary = self._metadata_debug_summary(metadata)
        sigma_branch = "sigma_condition_proj" if phase == "full_pinn" else None
        self._set_forward_debug_context(
            phase=phase,
            phenomenon=phenomenon,
            sigma=sigma,
            metadata=metadata_summary,
            sigma_branch=sigma_branch,
        )
        if hasattr(self.physics_adapter, "set_debug_context"):
            self.physics_adapter.set_debug_context(
                step=int(self.current_step),
                training_stage=self.training_stage,
                phase=phase,
                phenomenon=phenomenon,
                timestep_id=int(timestep_id),
                sigma=sigma,
                metadata=metadata_summary,
                sigma_branch=sigma_branch,
            )
        self._maybe_validate_adapter_parameter_finiteness()

    def _maybe_validate_adapter_parameter_finiteness(self):
        interval = max(int(self.diagnostic_metrics_interval), 1)
        if (self.current_step % interval) != 0:
            return
        if hasattr(self.physics_adapter, "validate_monitored_parameters_finite"):
            self.physics_adapter.validate_monitored_parameters_finite(
                prefixes=self._stage2_monitored_parameter_prefixes()
            )

    def recommended_optimizer_learning_rate(self, base_learning_rate):
        learning_rate = float(base_learning_rate)
        if not getattr(self, "_resume_stability_mode", False):
            return learning_rate
        return min(learning_rate, float(self._resume_stability_lr_cap))

    def _validate_state_dict_load(self, component_name, load_result, checkpoint_format_version, allow_missing_buffers=None):
        if allow_missing_buffers is None:
            allow_missing_buffers = set()
        allowed_prefixes = set()
        allowed_missing = set(allow_missing_buffers)

        if component_name == "physics_adapter":
            allowed_missing.add("expert_usage_ema")
            allowed_prefixes.add("obs_dynamics_head.")
            allowed_prefixes.add("alpha_head.")
            if checkpoint_format_version < 5 or not self.enable_rl_expert_optimization:
                allowed_missing.add("rl_reward_ema")
                allowed_prefixes.update({
                    "rl_expert_embedding.",
                    "rl_state_proj.",
                    "rl_policy_head.",
                })
            if checkpoint_format_version < 6 or not self.use_adaptive_condition_injection:
                allowed_prefixes.update({
                    "adaptive_condition_expert_embedding.",
                    "adaptive_condition_state_proj.",
                    "adaptive_condition_modulator.",
                    "shared_adaptive_condition_modulator.",
                })
            if checkpoint_format_version < 15:
                allowed_prefixes.update({
                    "u_head.",
                    "d_head.",
                })
            if checkpoint_format_version < 18:
                allowed_prefixes.add("prho_constructor.")
            if checkpoint_format_version < 19:
                allowed_prefixes.update({
                    "alpha_head.",
                    "T_head.",
                    "j_head.",
                    "D_head.",
                    "psi_head.",
                })
        missing_keys = []
        for key in load_result.missing_keys:
            if key in allowed_missing:
                continue
            if any(key.startswith(prefix) for prefix in allowed_prefixes):
                continue
            missing_keys.append(key)
        unexpected_keys = [
            key for key in load_result.unexpected_keys
            if not any(key.startswith(prefix) for prefix in allowed_prefixes)
        ]
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Incompatible {component_name} checkpoint load detected. "
                f"missing_keys={missing_keys[:20]}, unexpected_keys={unexpected_keys[:20]}"
            )

    def _initialize_runtime_training_config(
        self,
        *,
        use_gradient_checkpointing,
        use_gradient_checkpointing_offload,
        frozen_model_gradient_checkpointing,
        extra_inputs,
        max_timestep_boundary,
        min_timestep_boundary,
        use_dual_noise_experts,
        dual_noise_expert_boundary,
        physics_weight,
        physics_warmup_steps,
        conditioned_physics_warmup_steps,
        adapter_hidden_dim,
        physics_attr_dim,
        expert_pde_sigma_threshold,
        expert_pde_sigma_threshold_target,
        expert_balance_weight,
        condition_consistency_weight,
        ablate_disable_moe,
        ablate_disable_conditioned_pde,
        ablate_disable_aux_losses,
        ablate_label_only_router,
        core_ablation_mode,
        allow_ablation_checkpoint_mismatch,
        diagnostic_metrics_interval,
        motion_mask_floor,
        motion_mask_quantile,
        motion_mask_warmup_steps,
        physical_mask_transition_steps,
        physics_state_mode,
        use_sigma_gate,
        sigma_gate_curve,
        use_sigma_conditioning,
        use_adaptive_condition_injection,
        enable_rl_expert_optimization,
        rl_policy_weight,
        rl_entropy_weight,
        rl_reward_decay,
        rl_reward_quality_weight,
        rl_reward_stability_weight,
        rl_warmup_steps,
        state_align_warmup_steps,
        state_align_x_weight,
        state_align_v_weight,
        curriculum_transition_start_step,
        curriculum_transition_steps,
        physics_weight_target,
        output_physics_weight,
        state_align_v_weight_target,
        decoded_branch_consistency_weight,
        enable_explainability_reports,
        explainability_top_experts,
        training_stage,
        stage1_pretrained_encoder,
        flow_backbone_ckpt,
        encoder_freeze_steps,
        encoder_lr_scale,
        ablation_preset,
        observable_target_mode,
        secondary_field_strategy,
        active_field_set,
        field_enable_schedule,
        field_recovery_phase,
        field_recovery_step_schedule,
        field_recovery_loss_ramp_steps,
        run_full_pinn_after_recovery,
        freeze_u_encoder_during_recovery,
    ):
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.frozen_model_gradient_checkpointing = frozen_model_gradient_checkpointing
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.physics_attr_dim = int(physics_attr_dim)
        self.expert_pde_sigma_threshold = max(float(expert_pde_sigma_threshold), 0.0)
        self.expert_pde_sigma_threshold_target = max(
            float(expert_pde_sigma_threshold_target), 0.0
        )
        self.use_dual_noise_experts = (
            self.pipe.dit2 is not None
            if use_dual_noise_experts is None
            else bool(use_dual_noise_experts)
        )
        self.dual_noise_expert_boundary = min(max(float(dual_noise_expert_boundary), 0.0), 1.0)

        self.physics_weight = max(float(physics_weight), 0.0)
        self.curriculum_transition_start_step = max(int(curriculum_transition_start_step), 0)
        self.curriculum_transition_steps = max(int(curriculum_transition_steps), 0)
        self.physics_weight_target = (
            self.physics_weight
            if physics_weight_target is None
            else max(float(physics_weight_target), 0.0)
        )
        self.output_physics_weight = max(float(output_physics_weight), 0.0)
        self.physics_warmup_steps = physics_warmup_steps
        self.conditioned_physics_warmup_steps = conditioned_physics_warmup_steps
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.expert_balance_weight = expert_balance_weight
        self.condition_consistency_weight = condition_consistency_weight
        self.moe_top_k = int(self.physics_adapter.moe_top_k)
        self.ablate_disable_moe = ablate_disable_moe
        self.ablate_disable_conditioned_pde = ablate_disable_conditioned_pde
        self.ablate_disable_aux_losses = ablate_disable_aux_losses
        self.ablate_label_only_router = ablate_label_only_router
        self.core_ablation_mode = str(core_ablation_mode or "full")
        if self.core_ablation_mode not in CORE_ABLATION_MODES:
            raise ValueError(
                f"Unsupported core_ablation_mode={self.core_ablation_mode!r}; "
                f"expected one of {CORE_ABLATION_MODES}."
            )
        self.allow_ablation_checkpoint_mismatch = bool(allow_ablation_checkpoint_mismatch)
        self.physics_adapter.set_core_ablation_mode(self.core_ablation_mode)
        self.diagnostic_metrics_interval = max(int(diagnostic_metrics_interval), 1)
        self.motion_mask_floor = min(max(float(motion_mask_floor), 0.0), 0.5)
        self.motion_mask_quantile = min(max(float(motion_mask_quantile), 0.5), 0.995)
        self.motion_mask_warmup_steps = max(int(motion_mask_warmup_steps), 0)
        self.physical_mask_transition_steps = max(int(physical_mask_transition_steps), 0)
        self.physics_state_mode = physics_state_mode
        self.use_sigma_gate = bool(use_sigma_gate)
        self.sigma_gate_curve = sigma_gate_curve
        self.use_sigma_conditioning = bool(use_sigma_conditioning)
        self.sigma_conditioning_dim = int(self.physics_adapter.sigma_conditioning_dim)
        self.sigma_gate_floor = float(self.physics_adapter.sigma_gate_floor)
        self.use_adaptive_condition_injection = bool(
            self.physics_adapter.use_adaptive_condition_injection
        )
        self.adaptive_conditioning_dim = int(self.physics_adapter.adaptive_conditioning_dim)
        self.adaptive_conditioning_strength = float(self.physics_adapter.adaptive_conditioning_strength)
        self.adaptive_conditioning_gate_floor = float(self.physics_adapter.adaptive_conditioning_gate_floor)
        self.enable_rl_expert_optimization = bool(
            self.physics_adapter.enable_rl_expert_optimization
        )
        self.rl_policy_weight = max(float(rl_policy_weight), 0.0)
        self.rl_entropy_weight = max(float(rl_entropy_weight), 0.0)
        self.rl_reward_decay = min(max(float(rl_reward_decay), 0.0), 0.999)
        self.rl_reward_quality_weight = max(float(rl_reward_quality_weight), 0.0)
        self.rl_reward_stability_weight = max(float(rl_reward_stability_weight), 0.0)
        self.rl_warmup_steps = max(int(rl_warmup_steps), 0)
        self.rl_hidden_dim = int(self.physics_adapter.rl_hidden_dim)
        self.state_align_warmup_steps = max(int(state_align_warmup_steps), 0)
        self.state_align_x_weight = max(float(state_align_x_weight), 0.0)
        self.state_align_v_weight = max(float(state_align_v_weight), 0.0)
        self.state_align_v_weight_target = (
            self.state_align_v_weight
            if state_align_v_weight_target is None
            else max(float(state_align_v_weight_target), 0.0)
        )
        self.decoded_branch_consistency_weight = max(float(decoded_branch_consistency_weight), 0.0)
        self.enable_explainability_reports = bool(enable_explainability_reports)
        self.explainability_top_experts = max(int(explainability_top_experts), 1)
        if training_stage not in {"observable_pretrain", "encoder_completion", "full_pinn"}:
            raise ValueError(
                "Unsupported training_stage="
                f"{training_stage!r}; expected observable_pretrain, encoder_completion, or full_pinn."
            )
        self.training_stage = training_stage
        self.stage1_pretrained_encoder = stage1_pretrained_encoder
        self.flow_backbone_ckpt = flow_backbone_ckpt
        self.encoder_freeze_steps = max(int(encoder_freeze_steps), 0)
        self.encoder_lr_scale = max(float(encoder_lr_scale), 0.0)
        self.field_recovery_phase = str(field_recovery_phase or "core")
        if self.field_recovery_phase not in ONLY_U_RECOVERY_PHASES:
            raise ValueError(
                f"Unsupported field_recovery_phase={self.field_recovery_phase!r}; "
                f"expected one of {ONLY_U_RECOVERY_PHASES}."
            )
        self.field_recovery_step_schedule = str(field_recovery_step_schedule or "").strip()
        self.field_recovery_loss_ramp_steps = max(int(field_recovery_loss_ramp_steps), 0)
        (
            self._field_recovery_schedule_entries,
            self._field_recovery_step_starts,
        ) = self._parse_field_recovery_step_schedule(self.field_recovery_step_schedule)
        self.run_full_pinn_after_recovery = bool(run_full_pinn_after_recovery)
        self.freeze_u_encoder_during_recovery = bool(freeze_u_encoder_during_recovery)
        field_policy = resolve_ablation_policy(
            ablation_preset=ablation_preset,
            observable_target_mode=observable_target_mode,
            secondary_field_strategy=secondary_field_strategy,
            active_field_set=active_field_set,
            field_enable_schedule=field_enable_schedule,
        )
        self.ablation_preset = field_policy["ablation_preset"]
        self.observable_target_mode = field_policy["observable_target_mode"]
        self.secondary_field_strategy = field_policy["secondary_field_strategy"]
        self.active_field_set = field_policy["active_field_set"]
        self.field_enable_schedule = field_policy["field_enable_schedule"]
        self.physics_adapter.set_field_policy(
            ablation_preset=self.ablation_preset,
            secondary_field_strategy=self.secondary_field_strategy,
            active_field_set=self.active_field_set,
            field_enable_schedule=self.field_enable_schedule,
            field_recovery_phase=self.field_recovery_phase,
        )

    def _parse_field_recovery_step_schedule(self, schedule_spec):
        spec = str(schedule_spec or "").strip()
        if not spec:
            return tuple(), {}

        phase_starts = {}
        max_allowed_idx = FIELD_RECOVERY_PHASE_TO_INDEX[self.field_recovery_phase]
        for raw_entry in spec.split(","):
            entry = raw_entry.strip()
            if not entry:
                continue
            if ":" not in entry:
                raise ValueError(
                    "field_recovery_step_schedule entries must use phase:step format; "
                    f"got {entry!r}."
                )
            phase_chunk, step_chunk = entry.split(":", 1)
            phase_names = [token.strip() for token in re.split(r"[+/|]", phase_chunk) if token.strip()]
            if not phase_names:
                raise ValueError(
                    "field_recovery_step_schedule contains an empty phase group; "
                    f"got {entry!r}."
                )
            try:
                start_step = int(step_chunk.strip())
            except ValueError as exc:
                raise ValueError(
                    "field_recovery_step_schedule step values must be integers; "
                    f"got {step_chunk!r} in {entry!r}."
                ) from exc
            if start_step < 0:
                raise ValueError(
                    "field_recovery_step_schedule step values must be >= 0; "
                    f"got {start_step} in {entry!r}."
                )
            for phase_name in phase_names:
                if phase_name not in ONLY_U_RECOVERY_PHASES:
                    raise ValueError(
                        f"Unsupported recovery phase {phase_name!r} in field_recovery_step_schedule; "
                        f"expected one of {ONLY_U_RECOVERY_PHASES}."
                    )
                phase_idx = FIELD_RECOVERY_PHASE_TO_INDEX[phase_name]
                if phase_idx > max_allowed_idx:
                    raise ValueError(
                        "field_recovery_step_schedule requests phase "
                        f"{phase_name!r} beyond configured field_recovery_phase={self.field_recovery_phase!r}."
                    )
                existing_start = phase_starts.get(phase_name)
                if existing_start is not None and existing_start != start_step:
                    raise ValueError(
                        f"Recovery phase {phase_name!r} is assigned multiple start steps "
                        f"({existing_start} and {start_step})."
                    )
                phase_starts[phase_name] = start_step

        phase_starts.setdefault("core", 0)
        max_phase_idx = max(FIELD_RECOVERY_PHASE_TO_INDEX[name] for name in phase_starts)
        for required_idx in range(max_phase_idx + 1):
            required_phase = ONLY_U_RECOVERY_PHASES[required_idx]
            if required_phase not in phase_starts:
                raise ValueError(
                    "field_recovery_step_schedule must define every phase up to the last active phase; "
                    f"missing {required_phase!r}."
                )

        ordered_entries = tuple(
            sorted(
                phase_starts.items(),
                key=lambda item: (item[1], FIELD_RECOVERY_PHASE_TO_INDEX[item[0]]),
            )
        )
        prev_phase_idx = -1
        prev_start = -1
        for phase_name, start_step in ordered_entries:
            phase_idx = FIELD_RECOVERY_PHASE_TO_INDEX[phase_name]
            if start_step < prev_start or phase_idx < prev_phase_idx:
                raise ValueError(
                    "field_recovery_step_schedule must follow ONLY_U_RECOVERY_PHASES order; "
                    f"got {schedule_spec!r}."
                )
            prev_start = start_step
            prev_phase_idx = phase_idx
        return ordered_entries, dict(phase_starts)

    def _has_field_recovery_schedule(self):
        return len(self._field_recovery_schedule_entries) > 0

    def _active_field_recovery_phase(self):
        if not self._has_field_recovery_schedule():
            return str(self.field_recovery_phase or "core")
        active_phase = "core"
        current_step = max(int(self.current_step), 0)
        for phase_name, start_step in self._field_recovery_schedule_entries:
            if current_step >= int(start_step):
                active_phase = phase_name
            else:
                break
        return active_phase

    def _field_recovery_phase_start_step(self, phase_name):
        if not self._has_field_recovery_schedule():
            return 0
        return int(self._field_recovery_step_starts.get(str(phase_name), 0))

    def _field_recovery_phase_ramp(self, phase_name):
        start_step = self._field_recovery_phase_start_step(phase_name)
        if start_step <= 0 or self.field_recovery_loss_ramp_steps <= 0:
            return 1.0
        if self.current_step < start_step:
            return 0.0
        return min(
            float(self.current_step - start_step + 1) / float(self.field_recovery_loss_ramp_steps),
            1.0,
        )

    def _validate_checkpoint_runtime_config(self, checkpoint_config):
        if not isinstance(checkpoint_config, dict) or len(checkpoint_config) == 0:
            return
        adapter_architecture = checkpoint_config.get("adapter_architecture")
        if adapter_architecture != "explicit_attribute_bank_v2":
            raise RuntimeError(
                f"Checkpoint architecture mismatch: expected explicit_attribute_bank_v2, got {adapter_architecture!r}."
            )
        current = {
            "field_contract_version": FIELD_CONTRACT_VERSION,
            "expert_field_recipe_version": EXPERT_FIELD_RECIPE_VERSION,
            "adapter_hidden_dim": int(self.physics_adapter.hidden_dim),
            "physics_attr_dim": int(self.physics_attr_dim),
            "num_phenomena": int(self.physics_adapter.num_phenomena),
            "n_numeric_dim": int(self.physics_adapter.n_numeric_dim),
            "q_input_dim": int(self.physics_adapter.q_input_dim),
            "n_text_vocab_size": int(self.physics_adapter.n_text_vocab_size),
            "moe_top_k": int(self.physics_adapter.moe_top_k),
            "physics_state_mode": self.physics_adapter.physics_state_mode,
            "use_sigma_gate": bool(self.physics_adapter.use_sigma_gate),
            "sigma_gate_curve": self.physics_adapter.sigma_gate_curve,
            "use_sigma_conditioning": bool(self.physics_adapter.use_sigma_conditioning),
            "sigma_conditioning_dim": int(self.physics_adapter.sigma_conditioning_dim),
            "sigma_gate_floor": float(self.physics_adapter.sigma_gate_floor),
            "use_dual_noise_experts": bool(self.use_dual_noise_experts),
            "dual_noise_expert_boundary": float(self.dual_noise_expert_boundary),
            "use_adaptive_condition_injection": bool(self.physics_adapter.use_adaptive_condition_injection),
            "adaptive_conditioning_dim": int(self.physics_adapter.adaptive_conditioning_dim),
            "adaptive_conditioning_strength": float(self.physics_adapter.adaptive_conditioning_strength),
            "adaptive_conditioning_gate_floor": float(self.physics_adapter.adaptive_conditioning_gate_floor),
            "enable_rl_expert_optimization": bool(self.physics_adapter.enable_rl_expert_optimization),
            "rl_hidden_dim": int(self.physics_adapter.rl_hidden_dim),
            "rl_reward_decay": float(self.physics_adapter.rl_reward_decay),
            "state_align_warmup_steps": int(self.state_align_warmup_steps),
            "state_align_x_weight": float(self.state_align_x_weight),
            "state_align_v_weight": float(self.state_align_v_weight),
            "curriculum_transition_start_step": int(self.curriculum_transition_start_step),
            "curriculum_transition_steps": int(self.curriculum_transition_steps),
            "physics_weight_target": float(self.physics_weight_target),
            "output_physics_weight": float(self.output_physics_weight),
            "state_align_v_weight_target": float(self.state_align_v_weight_target),
            "decoded_branch_consistency_weight": float(self.decoded_branch_consistency_weight),
            "expert_pde_sigma_threshold": float(self.expert_pde_sigma_threshold),
            "expert_pde_sigma_threshold_target": float(self.expert_pde_sigma_threshold_target),
            "training_stage": self.training_stage,
            "encoder_freeze_steps": int(self.encoder_freeze_steps),
            "encoder_lr_scale": float(self.encoder_lr_scale),
            "ablation_preset": self.ablation_preset,
            "observable_target_mode": self.observable_target_mode,
            "secondary_field_strategy": self.secondary_field_strategy,
            "active_field_set": self.active_field_set,
            "field_enable_schedule": self.field_enable_schedule,
            "field_recovery_phase": self.field_recovery_phase,
            "field_recovery_step_schedule": self.field_recovery_step_schedule,
            "field_recovery_loss_ramp_steps": int(self.field_recovery_loss_ramp_steps),
            "run_full_pinn_after_recovery": bool(self.run_full_pinn_after_recovery),
            "freeze_u_encoder_during_recovery": bool(self.freeze_u_encoder_during_recovery),
        }
        mismatches = []
        for key, current_value in current.items():
            if key not in checkpoint_config or checkpoint_config[key] is None:
                continue
            expected_value = checkpoint_config[key]
            if key == "field_recovery_phase":
                checkpoint_stage = checkpoint_config.get("training_stage")
                if (
                    self.training_stage == "encoder_completion"
                    and checkpoint_stage == "encoder_completion"
                    and str(expected_value) in FIELD_RECOVERY_PHASE_TO_INDEX
                    and current_value in FIELD_RECOVERY_PHASE_TO_INDEX
                    and FIELD_RECOVERY_PHASE_TO_INDEX[str(expected_value)]
                    <= FIELD_RECOVERY_PHASE_TO_INDEX[current_value]
                ):
                    continue
            if key == "training_stage":
                checkpoint_stage = str(expected_value)
                checkpoint_phase = str(checkpoint_config.get("field_recovery_phase") or "core")
                if (
                    self.training_stage == "full_pinn"
                    and checkpoint_stage == "encoder_completion"
                    and checkpoint_phase == "psi"
                ):
                    continue
            if key == "encoder_freeze_steps":
                checkpoint_stage = str(checkpoint_config.get("training_stage") or "")
                if (
                    self.training_stage == "full_pinn"
                    and checkpoint_stage == "full_pinn"
                    and int(current_value) >= int(expected_value)
                ):
                    continue
            if key == "encoder_lr_scale":
                checkpoint_stage = str(checkpoint_config.get("training_stage") or "")
                if (
                    self.training_stage == "full_pinn"
                    and checkpoint_stage == "full_pinn"
                    and float(current_value) <= float(expected_value)
                ):
                    continue
            if key == "run_full_pinn_after_recovery":
                continue
            if isinstance(current_value, float):
                if abs(float(expected_value) - current_value) > 1e-6:
                    mismatches.append((key, expected_value, current_value))
            else:
                if expected_value != current_value:
                    mismatches.append((key, expected_value, current_value))
        if mismatches:
            if getattr(self, "allow_ablation_checkpoint_mismatch", False):
                ablation_relaxed_keys = {
                    "physics_weight_target",
                    "output_physics_weight",
                    "state_align_x_weight",
                    "state_align_v_weight",
                    "state_align_v_weight_target",
                    "decoded_branch_consistency_weight",
                    "expert_pde_sigma_threshold",
                    "expert_pde_sigma_threshold_target",
                    "encoder_freeze_steps",
                    "encoder_lr_scale",
                }
                mismatches = [
                    mismatch for mismatch in mismatches
                    if mismatch[0] not in ablation_relaxed_keys
                ]
        if mismatches:
            details = ", ".join(
                f"{key}: checkpoint={expected} current={current}"
                for key, expected, current in mismatches[:20]
            )
            hint = ""
            mismatch_keys = {key for key, _, _ in mismatches}
            if {"use_dual_noise_experts", "dual_noise_expert_boundary"} & mismatch_keys:
                hint = (
                    " This usually means the checkpoint was created for a different Wan expert layout "
                    "(for example, resuming a Wan2.2 dual-expert checkpoint in a Wan2.1 single-expert run)."
                )
            raise RuntimeError(f"Checkpoint runtime config mismatch detected. {details}{hint}")

    @staticmethod
    def _tensor_stats(value):
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return None
        tensor = value.detach().float()
        return {
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std(unbiased=False).item()),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
        }

    @staticmethod
    def _tensor_channel_means(value):
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return []
        tensor = value.detach().float().abs()
        if tensor.ndim < 2:
            return [float(tensor.mean().item())]
        reduce_dims = tuple(dim for dim in range(tensor.ndim) if dim != 1)
        channel_means = tensor.mean(dim=reduce_dims)
        return [float(x) for x in channel_means.cpu().tolist()]

    @staticmethod
    def _cache_mean_or_nan(cache, key):
        if not isinstance(cache, dict):
            return float("nan")
        value = cache.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return float("nan")
        return float(value.detach().float().mean().item())

    @staticmethod
    def _cache_scalar_mean_or_nan(cache, key):
        return WanPINNTrainingModule._cache_mean_or_nan(cache, key)

    @staticmethod
    def _cache_norm_mean_or_nan(cache, key):
        if not isinstance(cache, dict):
            return float("nan")
        value = cache.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0 or value.ndim < 2:
            return float("nan")
        flat = value.detach().float().reshape(value.shape[0], -1)
        return float(flat.norm(dim=1).mean().item())

    @staticmethod
    def _set_module_trainability(module, requires_grad, training=None):
        if module is None:
            return
        module.requires_grad_(requires_grad)
        if training is None:
            training = bool(requires_grad)
        module.train(training)

    def _observable_stage_modules(self):
        modules = [
            self.physics_adapter.physics_encoder_shared,
            self.physics_adapter.shared_attribute_head,
            self.physics_adapter.u_head,
        ]
        if self.observable_target_mode != "flow_only":
            modules.append(self.physics_adapter.d_head)
        return modules

    def _observable_head_modules(self):
        modules = [self.physics_adapter.u_head]
        if self.observable_target_mode != "flow_only":
            modules.append(self.physics_adapter.d_head)
        return modules

    def _stage2_monitored_parameter_prefixes(self):
        return (
            "physics_encoder_shared.",
            "shared_attribute_head.",
            "sigma_condition_proj.",
            "n_numeric_proj.",
            "n_text_embedding.",
            "q_proj.",
            "condition_fuse.",
            "expert_router.",
            "operator_experts.",
        )

    def _stage2_protected_conditioning_modules(self):
        return [
            self.physics_adapter.physics_encoder_shared,
            self.physics_adapter.shared_attribute_head,
            self.physics_adapter.sigma_condition_proj,
            self.physics_adapter.n_numeric_proj,
            self.physics_adapter.n_text_embedding,
            self.physics_adapter.q_proj,
            self.physics_adapter.condition_fuse,
            self.physics_adapter.expert_router,
            self.physics_adapter.operator_experts,
        ]

    def _encoder_completion_modules(self):
        phase_modules = [self.physics_adapter.prho_constructor]
        if not self.freeze_u_encoder_during_recovery:
            phase_modules[:0] = [
                self.physics_adapter.physics_encoder_shared,
                self.physics_adapter.shared_attribute_head,
                self.physics_adapter.u_head,
            ]
        if self._has_field_recovery_schedule():
            phase_modules.extend(
                [
                    self.physics_adapter.alpha_head,
                    self.physics_adapter.T_head,
                    self.physics_adapter.j_head,
                    self.physics_adapter.D_head,
                    self.physics_adapter.psi_head,
                ]
            )
            return phase_modules
        phase = str(self.field_recovery_phase or "core")
        if FIELD_RECOVERY_PHASE_TO_INDEX.get(phase, -1) >= FIELD_RECOVERY_PHASE_TO_INDEX["alpha"]:
            phase_modules.append(self.physics_adapter.alpha_head)
        if FIELD_RECOVERY_PHASE_TO_INDEX.get(phase, -1) >= FIELD_RECOVERY_PHASE_TO_INDEX["T"]:
            phase_modules.append(self.physics_adapter.T_head)
        if FIELD_RECOVERY_PHASE_TO_INDEX.get(phase, -1) >= FIELD_RECOVERY_PHASE_TO_INDEX["j"]:
            phase_modules.append(self.physics_adapter.j_head)
        if FIELD_RECOVERY_PHASE_TO_INDEX.get(phase, -1) >= FIELD_RECOVERY_PHASE_TO_INDEX["D"]:
            phase_modules.append(self.physics_adapter.D_head)
        if FIELD_RECOVERY_PHASE_TO_INDEX.get(phase, -1) >= FIELD_RECOVERY_PHASE_TO_INDEX["psi"]:
            phase_modules.append(self.physics_adapter.psi_head)
        return phase_modules

    def _configure_training_stage_trainability(self):
        self.physics_adapter.train()
        self.pde_residuals.eval()
        self.pde_residuals.requires_grad_(False)
        if self.training_stage == "observable_pretrain":
            self.physics_adapter.requires_grad_(False)
            for module in self._observable_stage_modules():
                self._set_module_trainability(module, True, training=True)
            self._stage2_encoder_frozen = None
            return
        if self.training_stage == "encoder_completion":
            self.physics_adapter.requires_grad_(False)
            for module in self._encoder_completion_modules():
                self._set_module_trainability(module, True, training=True)
            if self.freeze_u_encoder_during_recovery:
                self._set_module_trainability(
                    self.physics_adapter.physics_encoder_shared,
                    False,
                    training=False,
                )
                self._set_module_trainability(
                    self.physics_adapter.shared_attribute_head,
                    False,
                    training=False,
                )
                self._set_module_trainability(
                    self.physics_adapter.u_head,
                    False,
                    training=False,
                )
            self._set_module_trainability(self.physics_adapter.d_head, False, training=False)
            self._stage2_encoder_frozen = None
            return
        self.physics_adapter.requires_grad_(True)
        for module in self._observable_head_modules():
            self._set_module_trainability(module, False, training=False)
        self._apply_stage2_encoder_schedule(force=True)

    def _configure_resume_stability_mode(self):
        self._resume_stability_mode = False
        self.effective_encoder_freeze_steps = int(self.encoder_freeze_steps)
        self.effective_encoder_lr_scale = float(self.encoder_lr_scale)
        training_state = self._checkpoint_training_state
        if self.training_stage != "full_pinn" or not isinstance(training_state, dict):
            return
        resume_step = max(int(training_state.get("current_step", 0) or 0), 0)
        resume_epoch = max(int(training_state.get("current_epoch", 0) or 0), 0)
        if resume_step <= 0:
            return
        self._resume_checkpoint_step = resume_step
        self._resume_checkpoint_epoch = resume_epoch
        self._resume_stability_mode = True
        self.effective_encoder_freeze_steps = max(
            int(self.encoder_freeze_steps),
            resume_step + int(self._resume_stability_extra_freeze_steps),
        )
        self.effective_encoder_lr_scale = min(
            float(self.encoder_lr_scale),
            float(self._resume_stability_encoder_lr_scale),
        )
        print("Resume stability mode enabled:")
        print(f"  resume_step={resume_step}")
        print(f"  resume_epoch={resume_epoch}")
        print(f"  encoder_freeze_steps: base={self.encoder_freeze_steps} effective={self.effective_encoder_freeze_steps}")
        print(f"  encoder_lr_scale: base={self.encoder_lr_scale} effective={self.effective_encoder_lr_scale}")
        print(f"  optimizer_lr_cap={self._resume_stability_lr_cap}")

    def _apply_stage2_encoder_schedule(self, force=False):
        if self.training_stage != "full_pinn":
            return
        should_freeze = self.current_step < self.effective_encoder_freeze_steps
        if not force and self._stage2_encoder_frozen is should_freeze:
            return
        for module in self._stage2_protected_conditioning_modules():
            self._set_module_trainability(
                module,
                not should_freeze,
                training=not should_freeze,
            )
        for module in self._observable_head_modules():
            self._set_module_trainability(module, False, training=False)
        self._stage2_encoder_frozen = should_freeze

    def _register_encoder_lr_scale_hooks(self):
        for hook in self._encoder_lr_scale_hooks:
            hook.remove()
        self._encoder_lr_scale_hooks = []
        if self.training_stage != "full_pinn":
            return
        if abs(self.effective_encoder_lr_scale - 1.0) < 1e-6:
            return
        if self.effective_encoder_lr_scale < 0.0:
            raise ValueError(
                f"encoder_lr_scale must be non-negative, got {self.effective_encoder_lr_scale}."
            )
        for module in self._stage2_protected_conditioning_modules():
            for param in module.parameters():
                self._encoder_lr_scale_hooks.append(
                    param.register_hook(
                        lambda grad, scale=self.effective_encoder_lr_scale: grad * scale
                    )
                )

    @staticmethod
    def _build_encoder_stage_state_dict_from_adapter_state(physics_adapter_state_dict):
        return {
            key: value
            for key, value in physics_adapter_state_dict.items()
            if key.startswith(STAGE1_ENCODER_STATE_PREFIXES)
        }

    @staticmethod
    def _strip_known_state_dict_prefixes(state_dict):
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            clean_key = key
            changed = True
            while changed:
                changed = False
                for prefix in ("module.", "physics_adapter."):
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        changed = True
            cleaned_state_dict[clean_key] = value
        return cleaned_state_dict

    def _load_stage1_pretrained_encoder(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint
        checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        if isinstance(checkpoint, dict):
            if isinstance(checkpoint.get("encoder_stage_state_dict"), dict):
                state_dict = checkpoint["encoder_stage_state_dict"]
            elif isinstance(checkpoint.get("physics_adapter_state_dict"), dict):
                state_dict = checkpoint["physics_adapter_state_dict"]
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"Invalid stage1 encoder checkpoint at {checkpoint_path}: expected state dict payload."
            )
        state_dict = self._strip_known_state_dict_prefixes(state_dict)
        loaded_modules = []
        for prefix, module, required in (
            ("physics_encoder_shared.", self.physics_adapter.physics_encoder_shared, True),
            ("shared_attribute_head.", self.physics_adapter.shared_attribute_head, True),
            ("u_head.", self.physics_adapter.u_head, False),
            ("d_head.", self.physics_adapter.d_head, False),
        ):
            module_state_dict = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if not module_state_dict:
                if required:
                    raise RuntimeError(
                        f"Stage1 checkpoint {checkpoint_path} is missing required module prefix {prefix!r}."
                    )
                continue
            load_result = module.load_state_dict(module_state_dict, strict=False)
            if load_result.missing_keys or load_result.unexpected_keys:
                raise RuntimeError(
                    f"Incompatible stage1 module load for prefix {prefix!r}: "
                    f"missing_keys={load_result.missing_keys[:20]}, "
                    f"unexpected_keys={load_result.unexpected_keys[:20]}"
                )
            loaded_modules.append(prefix.rstrip("."))
        if not loaded_modules:
            raise RuntimeError(f"No stage1 encoder weights were loaded from {checkpoint_path}.")
        print(f"Loaded stage1 encoder scaffold from {checkpoint_path}: {', '.join(loaded_modules)}")
        if checkpoint_config.get("training_stage") in {"observable_pretrain", "encoder_completion"}:
            self.observable_diagnostics_enabled = True

    def _decode_observable_rgb_frames(self, physics_state):
        target_shape = physics_state.shape[2:]
        decoded_video = None
        vae = getattr(self.pipe, "vae", None)
        decode_device = getattr(self.pipe, "device", physics_state.device)
        if vae is not None:
            try:
                with torch.no_grad():
                    if hasattr(self.pipe, "load_models_to_device"):
                        self.pipe.load_models_to_device(["vae"])
                    try:
                        vae_dtype = next(vae.parameters()).dtype
                    except StopIteration:
                        vae_dtype = physics_state.dtype
                    decoded_video = vae.decode(
                        physics_state.detach().to(device=decode_device, dtype=vae_dtype),
                        device=decode_device,
                        tiled=False,
                    )
            except Exception as exc:
                print(f"Warning: observable VAE decode failed, falling back to latent RGB proxy. error={exc}")
                decoded_video = None
            finally:
                if hasattr(self.pipe, "load_models_to_device"):
                    self.pipe.load_models_to_device([])
        if not isinstance(decoded_video, torch.Tensor) or decoded_video.ndim != 5:
            decoded_video = physics_state.detach()
            if decoded_video.shape[1] < 3:
                decoded_video = decoded_video.repeat(1, math.ceil(3 / max(decoded_video.shape[1], 1)), 1, 1, 1)
            decoded_video = decoded_video[:, :3]
        decoded_video = decoded_video.to(device=physics_state.device, dtype=physics_state.dtype)
        if tuple(decoded_video.shape[2:]) != tuple(target_shape):
            decoded_video = F.interpolate(
                decoded_video.float(),
                size=target_shape,
                mode="trilinear",
                align_corners=False,
            ).to(dtype=physics_state.dtype)
        return torch.clamp(decoded_video, min=-1.0, max=1.0)

    def _build_observable_proxy_targets(self, physics_state):
        with torch.no_grad():
            rgb_frames = self._decode_observable_rgb_frames(physics_state.detach())
            proxy_targets = self.observable_proxy_extractor(rgb_frames)
        return proxy_targets

    @staticmethod
    def _observable_alignment_terms(predicted, target, proxy_conf, target_mode="flow_plus_deformation"):
        proxy_conf = proxy_conf.to(dtype=predicted["flow"].dtype)
        conf_norm = proxy_conf.sum().clamp_min(1.0)
        flow_map = F.smooth_l1_loss(predicted["flow"], target["flow_proxy"], reduction="none").mean(dim=1, keepdim=True)
        flow_error = (flow_map * proxy_conf).sum() / conf_norm
        deformation_error = torch.zeros_like(flow_error)
        if str(target_mode) != "flow_only":
            deformation_map = F.smooth_l1_loss(
                predicted["deformation"],
                target["deformation_proxy"],
                reduction="none",
            ).mean(dim=1, keepdim=True)
            deformation_error = (deformation_map * proxy_conf).sum() / conf_norm
        total_loss = flow_error + deformation_error
        return total_loss, {
            "flow_error": flow_error,
            "deformation_error": deformation_error,
        }

    def _forward_observable_pretrain(self, phenomenon, physics_state_original, sigma):
        proxy_targets = self._build_observable_proxy_targets(physics_state_original)
        stage1_outputs = self.physics_adapter.forward_observable_pretrain(
            physics_state_original,
            sigma=sigma,
        )
        loss_obs, obs_errors = self._observable_alignment_terms(
            stage1_outputs["observable_outputs"],
            proxy_targets,
            proxy_targets["proxy_conf"],
            self.observable_target_mode,
        )
        total_loss = loss_obs
        self._last_metrics = {
            "phenomenon": phenomenon,
            "training_stage": self.training_stage,
            "loss_obs": float(loss_obs.detach().item()),
            "obs_flow_error": float(obs_errors["flow_error"].detach().item()),
            "obs_deformation_error": float(obs_errors["deformation_error"].detach().item()),
            "proxy_conf_mean": float(proxy_targets["proxy_conf"].detach().mean().item()),
            "sigma_mean": float(sigma.detach().float().mean().item()),
            "encoder_frozen": float(self.freeze_u_encoder_during_recovery),
            "ablation_preset": self.ablation_preset,
            "observable_target_mode": self.observable_target_mode,
        }
        self._latest_explainability_snapshot = None
        self.current_step += 1
        return total_loss

    def _phase_at_least(self, target_phase):
        current_idx = FIELD_RECOVERY_PHASE_TO_INDEX.get(self._active_field_recovery_phase(), -1)
        target_idx = FIELD_RECOVERY_PHASE_TO_INDEX.get(str(target_phase), -1)
        return current_idx >= target_idx >= 0

    def _encoder_completion_loss_weight(self, owner_phase):
        if not self._phase_at_least(owner_phase):
            return 0.0
        if not self._has_field_recovery_schedule():
            if str(self.field_recovery_phase or "core") == str(owner_phase):
                return 1.0
            return 0.25

        active_phase = self._active_field_recovery_phase()
        active_block_start = self._field_recovery_phase_start_step(active_phase)
        owner_start = self._field_recovery_phase_start_step(owner_phase)
        base_weight = 1.0 if owner_start == active_block_start else 0.25
        ramp = self._field_recovery_phase_ramp(owner_phase)
        if owner_start == 0:
            return base_weight
        if owner_start == active_block_start:
            return base_weight * ramp
        return base_weight

    def _build_encoder_completion_physical_mask(self, fused_attribute_bank, metadata):
        if not isinstance(fused_attribute_bank, torch.Tensor) or fused_attribute_bank.ndim != 5:
            raise ValueError(
                "encoder_completion physical mask requires fused_attribute_bank with shape [B, C, T, H, W]."
            )
        fallback_mask = torch.ones(
            fused_attribute_bank.shape[0],
            1,
            fused_attribute_bank.shape[2],
            fused_attribute_bank.shape[3],
            fused_attribute_bank.shape[4],
            device=fused_attribute_bank.device,
            dtype=fused_attribute_bank.dtype,
        )
        if not isinstance(metadata, dict):
            return fallback_mask, {
                "active_phenomena": [[] for _ in range(fused_attribute_bank.shape[0])],
                "active_fields": [[] for _ in range(fused_attribute_bank.shape[0])],
                "active_field_count": torch.zeros(
                    fused_attribute_bank.shape[0],
                    device=fused_attribute_bank.device,
                    dtype=fused_attribute_bank.dtype,
                ),
                "source": "encoder_completion_fallback",
                "recipe_version": PHYSICAL_MASK_RECIPE_VERSION,
            }

        try:
            active_label_ids = []
            max_label_count = 1
            for sample_idx in range(int(fused_attribute_bank.shape[0])):
                sample_ids = list(self._resolve_active_label_ids_for_sample(metadata, sample_idx))
                if len(sample_ids) == 0:
                    sample_ids = list(self._resolve_active_label_ids(metadata))
                if len(sample_ids) == 0:
                    sample_ids = [PHENOMENON_TO_ID.get("Fluid", 0)]
                active_label_ids.append(sample_ids)
                max_label_count = max(max_label_count, len(sample_ids))

            active_expert_indices = torch.zeros(
                fused_attribute_bank.shape[0],
                max_label_count,
                device=fused_attribute_bank.device,
                dtype=torch.long,
            )
            for sample_idx, sample_ids in enumerate(active_label_ids):
                for label_pos, expert_idx in enumerate(sample_ids[:max_label_count]):
                    active_expert_indices[sample_idx, label_pos] = int(expert_idx)

            return self._build_active_physical_mask(
                fused_attribute_bank,
                metadata=metadata,
                cache={"active_expert_indices": active_expert_indices},
            )
        except Exception:
            return fallback_mask, {
                "active_phenomena": [[] for _ in range(fused_attribute_bank.shape[0])],
                "active_fields": [[] for _ in range(fused_attribute_bank.shape[0])],
                "active_field_count": torch.zeros(
                    fused_attribute_bank.shape[0],
                    device=fused_attribute_bank.device,
                    dtype=fused_attribute_bank.dtype,
                ),
                "source": "encoder_completion_fallback",
                "recipe_version": PHYSICAL_MASK_RECIPE_VERSION,
            }

    def _only_u_momentum_residual_vector(self, u, p, rho, metadata=None):
        pde = self.pde_residuals
        u = u[:, :2]
        ux = u[:, 0:1]
        uy = u[:, 1:2]
        p_scalar = p.mean(dim=1, keepdim=True)
        rho_scalar = rho.mean(dim=1, keepdim=True).clamp_min(1e-4)
        dux_dx = pde._grad_width(ux)
        dux_dy = pde._grad_height(ux)
        duy_dx = pde._grad_width(uy)
        duy_dy = pde._grad_height(uy)
        u_t = pde._temporal_derivative(u, metadata=metadata, order=1)
        conv_x = ux * dux_dx + uy * dux_dy
        conv_y = ux * duy_dx + uy * duy_dy
        grad_px = pde._grad_width(p_scalar)
        grad_py = pde._grad_height(p_scalar)
        lap_u = pde._laplacian_field(u, metadata=metadata)
        viscosity = 1e-3
        return torch.cat(
            [
                u_t[:, 0:1] + conv_x + grad_px / rho_scalar - viscosity * lap_u[:, 0:1],
                u_t[:, 1:2] + conv_y + grad_py / rho_scalar - viscosity * lap_u[:, 1:2],
            ],
            dim=1,
        )

    def _encoder_completion_local_losses(self, field_dict, final_bank, metadata, physical_mask):
        pde = self.pde_residuals
        phase = self._active_field_recovery_phase()
        metrics = {}
        objectives = {}

        only_u_terms = pde._only_u_fluid_terms(
            field_dict["u"],
            field_dict["p"],
            field_dict["rho"],
            metadata=metadata,
        )
        d_phys = field_dict["d_phys"]
        if d_phys.shape[2] > 1:
            time_grid = metadata.get("frame_time_grid") if isinstance(metadata, dict) else None
            if not isinstance(time_grid, torch.Tensor) or time_grid.shape[-1] != d_phys.shape[2]:
                time_grid = torch.linspace(
                    0.0,
                    1.0,
                    steps=d_phys.shape[2],
                    device=d_phys.device,
                    dtype=d_phys.dtype,
                ).unsqueeze(0).repeat(d_phys.shape[0], 1)
            else:
                time_grid = time_grid.to(device=d_phys.device, dtype=d_phys.dtype)
            dt = (time_grid[:, 1:] - time_grid[:, :-1]).view(d_phys.shape[0], 1, d_phys.shape[2] - 1, 1, 1)
            d_step = d_phys[:, :, 1:] - d_phys[:, :, :-1]
            u_mid = 0.5 * (field_dict["u"][:, :, 1:] + field_dict["u"][:, :, :-1])
            loss_d_integral = torch.mean((d_step - dt * u_mid) ** 2)
        else:
            loss_d_integral = pde._zero_loss(d_phys)
        loss_d_kinematic = pde._temporal_alignment_loss(d_phys, field_dict["u"], metadata=metadata)
        objectives["core"] = (
            only_u_terms["mass_residual"]
            + only_u_terms["momentum_residual"]
            + 0.25 * only_u_terms["pressure_smoothness"]
            + 0.25 * (only_u_terms["density_smoothness"] + only_u_terms["density_floor"])
            + 0.5 * loss_d_integral
            + 0.5 * loss_d_kinematic
        )
        metrics.update({
            "mass_residual": float(only_u_terms["mass_residual"].detach().item()),
            "momentum_residual": float(only_u_terms["momentum_residual"].detach().item()),
            "rho_mean": float(only_u_terms["rho_scalar"].detach().mean().item()),
            "rho_min": float(only_u_terms["rho_scalar"].detach().min().item()),
            "p_abs_mean": float(only_u_terms["p_scalar"].detach().abs().mean().item()),
            "div_u_abs_mean": float(only_u_terms["div_u"].detach().abs().mean().item()),
            "d_alignment_error": float(loss_d_kinematic.detach().item()),
            "d_integral_consistency": float(loss_d_integral.detach().item()),
        })

        if self._phase_at_least("alpha"):
            alpha_scalar = field_dict["alpha_scalar"]
            alpha_t = pde._temporal_derivative(alpha_scalar, metadata=metadata, order=1)
            alpha_adv = (
                field_dict["u"][:, 0:1] * pde._grad_width(alpha_scalar)
                + field_dict["u"][:, 1:2] * pde._grad_height(alpha_scalar)
            )
            alpha_lap = pde._laplacian_field(alpha_scalar, metadata=metadata)
            loss_alpha_transport = pde._weighted_square_mean(
                alpha_t + alpha_adv - 1e-3 * alpha_lap,
                metadata=metadata,
                ref_tensor=alpha_scalar,
            )
            loss_alpha_range = pde._weighted_square_mean(
                torch.relu(-alpha_scalar) + torch.relu(alpha_scalar - 1.0),
                metadata=metadata,
                ref_tensor=alpha_scalar,
            )
            loss_alpha_smooth = pde._spatial_gradient_energy(alpha_scalar, metadata=metadata)
            objectives["alpha"] = loss_alpha_transport + 0.1 * loss_alpha_range + 0.1 * loss_alpha_smooth
            metrics.update({
                "alpha_mean": float(alpha_scalar.detach().mean().item()),
                "alpha_transport_residual": float(loss_alpha_transport.detach().item()),
            })

        if self._phase_at_least("T"):
            temperature = field_dict["T_scalar"]
            temp_t = pde._temporal_derivative(temperature, metadata=metadata, order=1)
            temp_adv = (
                field_dict["u"][:, 0:1] * pde._grad_width(temperature)
                + field_dict["u"][:, 1:2] * pde._grad_height(temperature)
            )
            temp_lap = pde._laplacian_field(temperature, metadata=metadata)
            loss_T_transport = pde._weighted_square_mean(
                temp_t + temp_adv - 1e-3 * temp_lap,
                metadata=metadata,
                ref_tensor=temperature,
            )
            loss_T_smooth = pde._spatial_gradient_energy(temperature, metadata=metadata)
            objectives["T"] = loss_T_transport + 0.1 * loss_T_smooth
            metrics.update({
                "T_mean": float(temperature.detach().mean().item()),
                "T_transport_residual": float(loss_T_transport.detach().item()),
            })

        if self._phase_at_least("j"):
            j_phys = field_dict["j_phys"]
            momentum_residual_vector = self._only_u_momentum_residual_vector(
                field_dict["u"],
                field_dict["p"],
                field_dict["rho"],
                metadata=metadata,
            ).detach()
            mask = physical_mask.to(device=j_phys.device, dtype=j_phys.dtype)
            loss_j_contact = pde._weighted_square_mean(
                mask * (j_phys + momentum_residual_vector),
                metadata=metadata,
                ref_tensor=j_phys,
            )
            loss_j_sparse = pde._weighted_square_mean(
                (1.0 - mask) * j_phys,
                metadata=metadata,
                ref_tensor=j_phys,
            )
            objectives["j"] = loss_j_contact + 0.1 * loss_j_sparse
            metrics.update({
                "j_mask_mean": float(mask.detach().mean().item()),
                "j_residual_fit": float(loss_j_contact.detach().item()),
            })

        if self._phase_at_least("D"):
            damage = field_dict["D_scalar"]
            damage_drive = torch.relu(
                torch.linalg.vector_norm(field_dict["j_phys"], dim=1, keepdim=True)
                + field_dict["eps"].abs().mean(dim=1, keepdim=True)
                - 0.1
            )
            loss_D_accumulate = pde._weighted_square_mean(
                pde._temporal_derivative(damage, metadata=metadata, order=1) - damage_drive,
                metadata=metadata,
                ref_tensor=damage,
            )
            if damage.shape[2] > 1:
                monotonic_violation = torch.relu(damage[:, :, :-1] - damage[:, :, 1:])
                loss_D_monotonic = torch.mean(monotonic_violation ** 2)
            else:
                monotonic_violation = damage * 0.0
                loss_D_monotonic = pde._zero_loss(damage)
            loss_D_range = pde._weighted_square_mean(
                torch.relu(-damage) + torch.relu(damage - 1.0),
                metadata=metadata,
                ref_tensor=damage,
            )
            objectives["D"] = loss_D_accumulate + 0.25 * loss_D_monotonic + 0.1 * loss_D_range
            metrics.update({
                "D_mean": float(damage.detach().mean().item()),
                "D_monotonic_violation": float(monotonic_violation.detach().mean().item()),
            })

        if self._phase_at_least("psi"):
            psi = field_dict["psi_scalar"]
            alpha_ref = field_dict["alpha_scalar"]
            loss_psi_wave = pde._wave_equation_loss(psi, metadata=metadata)
            loss_psi_interference = pde._field_match_loss(psi, alpha_ref, metadata=metadata)
            loss_psi_smooth = pde._spatial_gradient_energy(psi, metadata=metadata)
            objectives["psi"] = loss_psi_wave + 0.2 * loss_psi_interference + 0.1 * loss_psi_smooth
            metrics.update({
                "psi_mean": float(psi.detach().mean().item()),
                "psi_wave_residual": float(loss_psi_wave.detach().item()),
            })

        total_local_loss = pde._zero_loss(final_bank)
        for owner_phase, objective in objectives.items():
            total_local_loss = total_local_loss + self._encoder_completion_loss_weight(owner_phase) * objective
        metrics["encoder_completion_local_loss"] = float(total_local_loss.detach().item())
        return total_local_loss, metrics

    def _forward_encoder_completion(self, phenomenon, physics_state_original, sigma, metadata):
        if not isinstance(metadata, dict):
            raise RuntimeError("encoder_completion requires metadata dict.")
        proxy_targets = self._build_observable_proxy_targets(physics_state_original)
        stage_outputs = self.physics_adapter.forward_observable_pretrain(
            physics_state_original,
            sigma=sigma,
        )
        loss_obs_raw, obs_errors = self._observable_alignment_terms(
            stage_outputs["observable_outputs"],
            proxy_targets,
            proxy_targets["proxy_conf"],
            self.observable_target_mode,
        )
        cache = getattr(self.physics_adapter, "_cache", {})
        if not isinstance(cache, dict):
            raise RuntimeError("PhysicsAdapter cache is required for encoder_completion.")
        shared_attribute_bank = cache.get("shared_attribute_bank_live")
        physics_feat_live = cache.get("physics_feat_live")
        if shared_attribute_bank is None or physics_feat_live is None:
            raise RuntimeError("encoder_completion requires shared_attribute_bank_live and physics_feat_live.")
        field_dict, final_bank, field_metrics = self.physics_adapter.build_physical_field_dict(
            shared_attribute_bank,
            physics_feat_live,
            metadata=metadata,
            field_recovery_phase=self._active_field_recovery_phase(),
        )
        cache["fused_attribute_bank_live"] = final_bank
        cache["fused_attribute_bank"] = final_bank.detach()
        cache["physical_field_metrics_live"] = field_metrics
        cache["physical_field_metrics"] = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in field_metrics.items()
        }
        physical_mask, physical_mask_info = self._build_encoder_completion_physical_mask(final_bank, metadata)
        local_loss, local_metrics = self._encoder_completion_local_losses(
            field_dict,
            final_bank,
            metadata,
            physical_mask,
        )
        loss_obs = 0.1 * loss_obs_raw
        total_loss = local_loss + loss_obs
        self._last_metrics = {
            "phenomenon": phenomenon,
            "training_stage": self.training_stage,
            "field_recovery_phase": self.field_recovery_phase,
            "active_field_recovery_phase": self._active_field_recovery_phase(),
            "field_recovery_step_schedule": self.field_recovery_step_schedule,
            "loss_obs": float(loss_obs.detach().item()),
            "obs_flow_error": float(obs_errors["flow_error"].detach().item()),
            "obs_deformation_error": float(obs_errors["deformation_error"].detach().item()),
            "proxy_conf_mean": float(proxy_targets["proxy_conf"].detach().mean().item()),
            "sigma_mean": float(sigma.detach().float().mean().item()),
            "encoder_frozen": 0.0,
            "ablation_preset": self.ablation_preset,
            "observable_target_mode": self.observable_target_mode,
            "physical_mask_mean": float(physical_mask.detach().mean().item()),
            "physical_mask_active_field_count": float(
                physical_mask_info["active_field_count"].detach().float().mean().item()
            ),
            **{
                key: self._scalar_metric_or_nan(value)
                for key, value in field_metrics.items()
            },
            **local_metrics,
        }
        self._latest_explainability_snapshot = None
        self.current_step += 1
        return total_loss

    def _collect_observable_diagnostics(self, shared_attribute_bank, physics_state_original):
        if not self.observable_diagnostics_enabled:
            return {
                "obs_flow_error": float("nan"),
                "obs_deformation_error": float("nan"),
            }
        with torch.no_grad():
            proxy_targets = self._build_observable_proxy_targets(physics_state_original)
            _, observable_outputs = self.physics_adapter.predict_observable_proxies(
                shared_attribute_bank.detach()
            )
            _, obs_errors = self._observable_alignment_terms(
                observable_outputs,
                proxy_targets,
                proxy_targets["proxy_conf"],
                self.observable_target_mode,
            )
        return {
            "obs_flow_error": float(obs_errors["flow_error"].detach().item()),
            "obs_deformation_error": float(obs_errors["deformation_error"].detach().item()),
        }

    @staticmethod
    def _summarize_data_for_report(data):
        if not isinstance(data, dict):
            return {}
        summary = {}
        prompt = data.get("prompt")
        if prompt is not None:
            summary["prompt"] = WanPINNTrainingModule._safe_text(prompt)[:240]
        for key, value in data.items():
            if key in ("video", "prompt"):
                continue
            if isinstance(value, (str, int, float, bool)):
                summary[key] = value
            elif isinstance(value, list) and len(value) > 0 and all(isinstance(v, str) for v in value[:4]):
                summary[key] = value[:4]
            elif value is not None:
                summary[key] = WanPINNTrainingModule._safe_text(value)[:160]
        return summary

    @staticmethod
    def _select_mask_keyframe_indices(time_dim, max_frames=5):
        time_dim = max(int(time_dim), 0)
        if time_dim <= 0:
            return []
        if time_dim <= max_frames:
            return list(range(time_dim))
        raw = torch.linspace(0, time_dim - 1, steps=max_frames)
        indices = []
        seen = set()
        for value in raw.tolist():
            index = int(round(float(value)))
            index = min(max(index, 0), time_dim - 1)
            if index not in seen:
                indices.append(index)
                seen.add(index)
        if indices[-1] != time_dim - 1:
            indices[-1] = time_dim - 1
        return indices

    @staticmethod
    def _mask_heatmap_payload(mask, target_size=24, max_frames=5):
        if not isinstance(mask, torch.Tensor) or mask.numel() == 0:
            return None
        mask = mask.detach().float()
        if mask.ndim == 5:
            sample_mask = mask[0:1].clamp(0.0, 1.0)
            frame_indices = WanPINNTrainingModule._select_mask_keyframe_indices(
                sample_mask.shape[2],
                max_frames=max_frames,
            )
            frames = []
            height = max(1, min(int(sample_mask.shape[-2]), int(target_size)))
            width = max(1, min(int(sample_mask.shape[-1]), int(target_size)))
            for frame_index in frame_indices:
                frame = sample_mask[:, :, frame_index]
                resized = F.interpolate(
                    frame,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                frames.append(
                    {
                        "frame_index": int(frame_index),
                        "grid": resized[0, 0].cpu().tolist(),
                        "mean": float(frame.mean().item()),
                        "max": float(frame.max().item()),
                    }
                )
            return {
                "frames": frames,
                "height": int(height),
                "width": int(width),
                "mean": float(sample_mask.mean().item()),
                "max": float(sample_mask.max().item()),
                "time_dim": int(sample_mask.shape[2]),
            }
        elif mask.ndim == 4:
            projected = mask[0:1].clamp(0.0, 1.0)
        else:
            return None
        projected = projected.clamp(0.0, 1.0)
        height = max(1, min(int(projected.shape[-2]), int(target_size)))
        width = max(1, min(int(projected.shape[-1]), int(target_size)))
        resized = F.interpolate(projected, size=(height, width), mode="bilinear", align_corners=False)
        grid = resized[0, 0].cpu().tolist()
        return {
            "grid": grid,
            "height": int(height),
            "width": int(width),
            "mean": float(projected.mean().item()),
            "max": float(projected.max().item()),
        }

    def _build_explainability_snapshot(
        self,
        data,
        metadata,
        cache,
        sigma,
        bootstrap_motion_mask,
        physical_mask,
        effective_motion_mask,
        loss_physics_value,
        loss_output_physics,
        output_divergence_before,
        output_divergence_after,
        output_divergence_reduction,
        raw_scale_estimate,
        correction_ratio,
        phenomenon,
    ):
        if not self.enable_explainability_reports:
            return None
        if not isinstance(cache, dict):
            cache = {}

        batch_size = 0
        active_indices = cache.get("active_expert_indices")
        active_weights = cache.get("active_expert_weights")
        router_topk_weights = cache.get("router_topk_weights")
        route_logits = cache.get("route_logits")
        usage_ema = self.physics_adapter.expert_usage_ema.detach().float().cpu()
        expert_rows = []
        sample0_rows = []
        if isinstance(active_indices, torch.Tensor) and isinstance(active_weights, torch.Tensor):
            active_indices_cpu = active_indices.detach().cpu()
            active_weights_cpu = active_weights.detach().float().cpu()
            batch_size = max(int(active_indices_cpu.shape[0]), 1)
            mean_policy_weight = torch.zeros(len(PHENOMENON_LABELS), dtype=torch.float32)
            mean_policy_weight.scatter_add_(0, active_indices_cpu.reshape(-1), active_weights_cpu.reshape(-1))
            mean_policy_weight = mean_policy_weight / float(batch_size)

            mean_router_weight = torch.zeros(len(PHENOMENON_LABELS), dtype=torch.float32)
            if isinstance(router_topk_weights, torch.Tensor) and tuple(router_topk_weights.shape) == tuple(active_weights.shape):
                router_topk_cpu = router_topk_weights.detach().float().cpu()
                mean_router_weight.scatter_add_(0, active_indices_cpu.reshape(-1), router_topk_cpu.reshape(-1))
                mean_router_weight = mean_router_weight / float(batch_size)
            else:
                router_topk_cpu = None

            mean_route_logits = torch.zeros(len(PHENOMENON_LABELS), dtype=torch.float32)
            if isinstance(route_logits, torch.Tensor) and route_logits.ndim == 2:
                mean_route_logits = route_logits.detach().float().cpu().mean(dim=0)

            active_mask = mean_policy_weight > 1e-6
            sorted_ids = torch.argsort(mean_policy_weight, descending=True)
            top_ids = [int(idx) for idx in sorted_ids.tolist() if active_mask[idx].item()][: self.explainability_top_experts]
            for expert_id in top_ids:
                expert_rows.append(
                    {
                        "expert_id": expert_id,
                        "label": PHENOMENON_LABELS[expert_id],
                        "mean_policy_weight": float(mean_policy_weight[expert_id].item()),
                        "mean_router_weight": float(mean_router_weight[expert_id].item()),
                        "mean_route_logit": float(mean_route_logits[expert_id].item()),
                        "usage_ema": float(usage_ema[expert_id].item()),
                    }
                )

            if active_indices_cpu.shape[0] > 0:
                sample0_indices = active_indices_cpu[0].tolist()
                sample0_weights = active_weights_cpu[0].tolist()
                router_sample0 = (
                    router_topk_cpu[0].tolist()
                    if isinstance(router_topk_weights, torch.Tensor) and tuple(router_topk_weights.shape) == tuple(active_weights.shape)
                    else [0.0] * len(sample0_indices)
                )
                route_logits_sample0 = (
                    route_logits.detach().float().cpu()[0]
                    if isinstance(route_logits, torch.Tensor) and route_logits.ndim == 2 and route_logits.shape[0] > 0
                    else None
                )
                for slot, expert_id in enumerate(sample0_indices):
                    expert_id = int(expert_id)
                    logit_value = 0.0
                    if route_logits_sample0 is not None and expert_id < route_logits_sample0.shape[0]:
                        logit_value = float(route_logits_sample0[expert_id].item())
                    sample0_rows.append(
                        {
                            "slot": int(slot),
                            "expert_id": expert_id,
                            "label": PHENOMENON_LABELS[expert_id],
                            "policy_weight": float(sample0_weights[slot]),
                            "router_weight": float(router_sample0[slot]),
                            "route_logit": logit_value,
                        }
                    )

        summary = {
            "step": int(self.current_step + 1),
            "phenomenon": WanPINNTrainingModule._safe_text(phenomenon),
            "batch_size": int(batch_size if batch_size > 0 else 1),
            "sigma_mean": float(sigma.detach().float().mean().item()) if isinstance(sigma, torch.Tensor) and sigma.numel() > 0 else 0.0,
            "physics_total": float(loss_physics_value),
            "output_physics_total": float(loss_output_physics.detach().item()),
            "output_divergence_before": float(output_divergence_before),
            "output_divergence_after": float(output_divergence_after),
            "output_divergence_reduction": float(output_divergence_reduction),
            "raw_scale_estimate": float(raw_scale_estimate),
            "correction_ratio": float(correction_ratio),
            "raw_correction_norm_mean": self._cache_mean_or_nan(cache, "raw_correction_norm"),
            "adaptive_condition_gate_mean": self._cache_scalar_mean_or_nan(cache, "adaptive_condition_gate"),
            "adaptive_correction_gate_mean": self._cache_scalar_mean_or_nan(cache, "adaptive_correction_gate"),
            "shared_condition_gate_mean": self._cache_scalar_mean_or_nan(cache, "shared_condition_gate"),
            "shared_correction_gate_mean": self._cache_scalar_mean_or_nan(cache, "shared_correction_gate"),
            "motion_mask_mean": float(effective_motion_mask.detach().float().mean().item()) if isinstance(effective_motion_mask, torch.Tensor) else 0.0,
            "motion_mask_max": float(effective_motion_mask.detach().float().max().item()) if isinstance(effective_motion_mask, torch.Tensor) and effective_motion_mask.numel() > 0 else 0.0,
            "bootstrap_motion_mask_mean": float(bootstrap_motion_mask.detach().float().mean().item()) if isinstance(bootstrap_motion_mask, torch.Tensor) else 0.0,
            "physical_mask_mean": float(physical_mask.detach().float().mean().item()) if isinstance(physical_mask, torch.Tensor) else 0.0,
            "physical_mask_blend_alpha": (
                float(metadata["physical_mask_blend_alpha"].detach().float().mean().item())
                if isinstance(metadata, dict) and isinstance(metadata.get("physical_mask_blend_alpha"), torch.Tensor)
                else 0.0
            ),
            "physical_mask_active_field_count": (
                float(metadata["physical_mask_active_field_count"].detach().float().mean().item())
                if isinstance(metadata, dict) and isinstance(metadata.get("physical_mask_active_field_count"), torch.Tensor)
                else 0.0
            ),
            "heuristic_physical_mask_overlap": (
                float(metadata["heuristic_physical_mask_overlap"].detach().float().mean().item())
                if isinstance(metadata, dict) and isinstance(metadata.get("heuristic_physical_mask_overlap"), torch.Tensor)
                else 0.0
            ),
        }
        if len(sample0_rows) > 0:
            summary["sample0_entropy"] = float(
                -sum(
                    row["policy_weight"] * max(math.log(max(row["policy_weight"], 1e-6)), -20.0)
                    for row in sample0_rows
                )
            )
        if len(sample0_rows) > 0:
            summary["sample0_rl_weight_shift"] = float(
                sum(abs(row["policy_weight"] - row["router_weight"]) for row in sample0_rows) / float(len(sample0_rows))
            )
        else:
            summary["sample0_entropy"] = 0.0
            summary["sample0_rl_weight_shift"] = 0.0

        return {
            "summary": summary,
            "data": self._summarize_data_for_report(data),
            "metadata": {
                "label_name": WanPINNTrainingModule._safe_text(metadata.get("label_name")) if isinstance(metadata, dict) else "",
                "label_names": [
                    WanPINNTrainingModule._safe_text(v) for v in metadata.get("label_names", [])
                ] if isinstance(metadata, dict) and isinstance(metadata.get("label_names", []), list) else [],
                "parse_success_ratio": float(metadata["parse_success_ratio"].detach().item()) if isinstance(metadata, dict) and isinstance(metadata.get("parse_success_ratio"), torch.Tensor) else None,
                "motion_mask_source": metadata.get("motion_mask_source") if isinstance(metadata, dict) else None,
                "bootstrap_motion_mask_source": metadata.get("bootstrap_motion_mask_source") if isinstance(metadata, dict) else None,
                "physical_mask_source": metadata.get("physical_mask_source") if isinstance(metadata, dict) else None,
                "physical_mask_recipe_version": metadata.get("physical_mask_recipe_version") if isinstance(metadata, dict) else None,
                "physical_mask_active_phenomena": metadata.get("physical_mask_active_phenomena") if isinstance(metadata, dict) else None,
                "physical_mask_active_fields": metadata.get("physical_mask_active_fields") if isinstance(metadata, dict) else None,
            },
            "experts": expert_rows,
            "sample0_experts": sample0_rows,
            "tensor_stats": {
                "final_correction": self._tensor_stats(cache.get("raw_correction")),
                "bootstrap_motion_mask": self._tensor_stats(bootstrap_motion_mask),
                "physical_mask": self._tensor_stats(physical_mask),
                "motion_mask": self._tensor_stats(effective_motion_mask),
            },
            "mask_heatmaps": {
                "bootstrap_motion_mask": self._mask_heatmap_payload(bootstrap_motion_mask),
                "physical_mask": self._mask_heatmap_payload(physical_mask),
                "effective_motion_mask": self._mask_heatmap_payload(effective_motion_mask),
            },
        }

    def _scheduler_sigma(self, timestep, device, dtype):
        sigma = self.pipe.scheduler.sigma_from_timestep(
            timestep,
            device=device,
            dtype=dtype,
        )
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        return sigma

    @staticmethod
    def _expand_sigma_for_like(sigma, ref_tensor):
        sigma_expanded = sigma.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        while sigma_expanded.ndim < ref_tensor.ndim:
            sigma_expanded = sigma_expanded.unsqueeze(-1)
        return sigma_expanded

    def _physics_state_from_prediction(self, latents, v_pred, sigma):
        if self.physics_state_mode == "x0_hat":
            sigma_expanded = self._expand_sigma_for_like(sigma, latents)
            return latents - sigma_expanded * v_pred.to(dtype=latents.dtype)
        return latents

    @staticmethod
    def _sigma_bucket_metrics(sigma):
        if not isinstance(sigma, torch.Tensor) or sigma.numel() == 0:
            return {}
        sigma_mean = float(sigma.detach().float().mean().item())
        return {
            "sigma_bucket_low": float(sigma_mean < 0.33),
            "sigma_bucket_mid": float(0.33 <= sigma_mean < 0.66),
            "sigma_bucket_high": float(sigma_mean >= 0.66),
        }

    @staticmethod
    def _sigma_bucket_name_from_value(sigma_value):
        sigma_value = float(sigma_value)
        if sigma_value < 0.33:
            return "low"
        if sigma_value < 0.66:
            return "mid"
        return "high"

    def _build_motion_mask(self, v_original, state_for_mask):
        """
        基于速度场时序变化 + 空间梯度，构建 bootstrapping heuristic motion mask: [B,1,T,H,W]。
        """
        B, _, T, H, W = v_original.shape
        device = v_original.device
        dtype = v_original.dtype
        zero = torch.zeros((B, 1, T, H, W), device=device, dtype=dtype)

        if T > 1:
            v_dt = torch.mean(torch.abs(v_original[:, :, 1:] - v_original[:, :, :-1]), dim=1, keepdim=True)
            v_dt = F.pad(v_dt, (0, 0, 0, 0, 0, 1))
            state_dt = torch.mean(
                torch.abs(state_for_mask[:, :, 1:] - state_for_mask[:, :, :-1]),
                dim=1,
                keepdim=True,
            )
            state_dt = F.pad(state_dt, (0, 0, 0, 0, 0, 1))
        else:
            v_dt = zero
            state_dt = zero

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
        energy = 0.55 * v_dt + 0.25 * state_dt + 0.20 * spatial_energy

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

    def _physical_mask_blend_alpha(self):
        if self.current_step < self.motion_mask_warmup_steps:
            return 0.0
        if self.physical_mask_transition_steps <= 0:
            return 1.0
        progress = (
            float(self.current_step - self.motion_mask_warmup_steps)
            / float(max(self.physical_mask_transition_steps, 1))
        )
        return float(min(max(progress, 0.0), 1.0))

    def _normalize_mask_energy(self, energy):
        if not isinstance(energy, torch.Tensor) or energy.ndim != 5:
            raise ValueError("Mask energy must be a [B, 1, T, H, W] tensor.")
        batch_size = int(energy.shape[0])
        energy = torch.clamp(energy, min=0.0)
        flat_energy = energy.float().reshape(batch_size, -1)
        denom = torch.quantile(
            flat_energy,
            q=self.motion_mask_quantile,
            dim=1,
            keepdim=True,
        ).to(dtype=energy.dtype).view(batch_size, 1, 1, 1, 1)
        energy_norm = torch.clamp(energy / (denom + 1e-6), 0.0, 1.0)
        mask = self.motion_mask_floor + (1.0 - self.motion_mask_floor) * energy_norm
        return torch.clamp(mask, self.motion_mask_floor, 1.0)

    @staticmethod
    def _field_abs_activity_map(field):
        return torch.mean(torch.abs(field), dim=1, keepdim=True)

    def _field_temporal_activity_map(self, field, metadata):
        field_dt = self.pde_residuals._temporal_derivative(field, metadata=metadata, order=1)
        return torch.mean(torch.abs(field_dt), dim=1, keepdim=True)

    def _field_boundary_activity_map(self, field):
        grad_h = torch.abs(self._first_difference_along_axis(field, axis=3))
        grad_w = torch.abs(self._first_difference_along_axis(field, axis=4))
        return 0.5 * (
            torch.mean(grad_h, dim=1, keepdim=True)
            + torch.mean(grad_w, dim=1, keepdim=True)
        )

    @staticmethod
    def _mean_activity_maps(activity_maps, ref_tensor):
        if len(activity_maps) == 0:
            return torch.zeros(
                ref_tensor.shape[0],
                1,
                ref_tensor.shape[2],
                ref_tensor.shape[3],
                ref_tensor.shape[4],
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )
        return torch.stack(activity_maps, dim=0).mean(dim=0)

    def _resolve_active_label_ids_for_sample(self, metadata, sample_idx):
        if not isinstance(metadata, dict):
            return []

        label_ids = metadata.get("label_ids")
        ids = []
        if isinstance(label_ids, torch.Tensor):
            if label_ids.ndim > 1 and label_ids.shape[0] > sample_idx:
                ids = label_ids[sample_idx].detach().view(-1).tolist()
            else:
                ids = label_ids.detach().view(-1).tolist()
        elif isinstance(label_ids, (list, tuple)):
            if len(label_ids) > 0 and isinstance(label_ids[0], (list, tuple)):
                nested = label_ids[sample_idx] if sample_idx < len(label_ids) else label_ids[0]
                ids = list(nested)
            else:
                ids = list(label_ids)

        active_ids = []
        for label_id in ids:
            try:
                idx = int(label_id)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(PHENOMENON_LABELS):
                active_ids.append(idx)
        if active_ids:
            return list(dict.fromkeys(active_ids))

        sample_metadata = self._metadata_for_sample(metadata, sample_idx, device="cpu")
        return self._resolve_active_label_ids(sample_metadata)

    def _resolve_active_physical_phenomena(self, metadata, cache, sample_idx):
        label_ids = set(self._resolve_active_label_ids_for_sample(metadata, sample_idx))
        if len(label_ids) == 0:
            return []

        active_expert_indices = cache.get("active_expert_indices") if isinstance(cache, dict) else None
        if not isinstance(active_expert_indices, torch.Tensor):
            return []
        if active_expert_indices.ndim < 2 or sample_idx >= active_expert_indices.shape[0]:
            return []

        active_names = []
        seen = set()
        for expert_idx in active_expert_indices[sample_idx].detach().view(-1).tolist():
            expert_idx = int(expert_idx)
            if expert_idx not in label_ids:
                continue
            phenomenon_name = PHENOMENON_LABELS[expert_idx]
            if phenomenon_name in seen:
                continue
            seen.add(phenomenon_name)
            active_names.append(phenomenon_name)
        return active_names

    @staticmethod
    def _active_fields_for_phenomena(active_phenomena):
        active_fields = set()
        for phenomenon_name in active_phenomena:
            for field_name in EXPERT_FIELD_RECIPES.get(phenomenon_name, ()):
                if field_name in PHYSICAL_MASK_ELIGIBLE_FIELDS:
                    active_fields.add(field_name)
        return [field_name for field_name in PHYSICAL_MASK_ELIGIBLE_FIELDS if field_name in active_fields]

    @staticmethod
    def _physical_mask_term_weights(active_phenomena, active_fields):
        evolution = {field_name: 1.0 for field_name in active_fields}
        event = {field_name: 1.0 for field_name in active_fields}
        boundary = {field_name: 1.0 for field_name in active_fields}

        for phenomenon_name in active_phenomena:
            config = PHYSICAL_MASK_PHENOMENON_WEIGHTS.get(phenomenon_name, {})
            for term_name, weights in (
                ("evolution", evolution),
                ("event", event),
                ("boundary", boundary),
            ):
                for field_name, field_weight in config.get(term_name, {}).items():
                    if field_name not in weights:
                        continue
                    field_weight = float(field_weight)
                    if field_weight >= 1.0:
                        weights[field_name] = max(weights[field_name], field_weight)
                    else:
                        weights[field_name] = min(weights[field_name], field_weight)
        return {
            "evolution": evolution,
            "event": event,
            "boundary": boundary,
        }

    def _build_active_physical_mask(self, fused_attribute_bank, metadata, cache):
        if not isinstance(fused_attribute_bank, torch.Tensor) or fused_attribute_bank.ndim != 5:
            raise ValueError("Active physical mask requires fused_attribute_bank with shape [B, C, T, H, W].")

        fused_attribute_bank = fused_attribute_bank.detach()
        field_dict = split_attribute_bank(fused_attribute_bank)
        batch_size, _, time_dim, height, width = fused_attribute_bank.shape
        device = fused_attribute_bank.device
        dtype = fused_attribute_bank.dtype
        physical_mask = torch.full(
            (batch_size, 1, time_dim, height, width),
            float(self.motion_mask_floor),
            device=device,
            dtype=dtype,
        )
        active_field_count = torch.zeros(batch_size, device=device, dtype=dtype)
        active_phenomena = []
        active_fields = []

        for sample_idx in range(batch_size):
            sample_phenomena = self._resolve_active_physical_phenomena(
                metadata,
                cache,
                sample_idx,
            )
            active_phenomena.append(sample_phenomena)
            sample_fields = self._active_fields_for_phenomena(sample_phenomena)
            active_fields.append(sample_fields)
            active_field_count[sample_idx] = float(len(sample_fields))

            if len(sample_fields) == 0:
                continue

            sample_metadata = self._metadata_for_sample(metadata, sample_idx, device=device)
            sample_field_dict = {
                field_name: field_dict[field_name][sample_idx:sample_idx + 1]
                for field_name in sample_fields
            }
            term_weights = self._physical_mask_term_weights(sample_phenomena, sample_fields)

            evolution_maps = []
            if "d" in sample_field_dict:
                evolution_maps.append(
                    term_weights["evolution"].get("d", 1.0)
                    * self._field_temporal_activity_map(sample_field_dict["d"], sample_metadata)
                )
            if "u" in sample_field_dict:
                evolution_maps.append(
                    term_weights["evolution"].get("u", 1.0)
                    * self._field_abs_activity_map(sample_field_dict["u"])
                )
            for field_name in ("rho", "T", "alpha", "psi", "D"):
                if field_name in sample_field_dict:
                    evolution_maps.append(
                        term_weights["evolution"].get(field_name, 1.0)
                        * self._field_temporal_activity_map(sample_field_dict[field_name], sample_metadata)
                    )

            event_maps = []
            for field_name in PHYSICAL_MASK_EVENT_FIELDS:
                if field_name in sample_field_dict:
                    event_maps.append(
                        term_weights["event"].get(field_name, 1.0)
                        * self._field_abs_activity_map(sample_field_dict[field_name])
                    )

            boundary_maps = []
            for field_name in PHYSICAL_MASK_BOUNDARY_FIELDS:
                if field_name in sample_field_dict:
                    boundary_maps.append(
                        term_weights["boundary"].get(field_name, 1.0)
                        * self._field_boundary_activity_map(sample_field_dict[field_name])
                    )

            reference_field = next(iter(sample_field_dict.values()))
            evolution_term = self._mean_activity_maps(evolution_maps, reference_field)
            event_term = self._mean_activity_maps(event_maps, reference_field)
            boundary_term = self._mean_activity_maps(boundary_maps, reference_field)
            energy = (
                0.65 * evolution_term
                + 0.25 * event_term
                + 0.10 * boundary_term
            )
            physical_mask[sample_idx:sample_idx + 1] = self._normalize_mask_energy(energy)

        return torch.clamp(physical_mask, self.motion_mask_floor, 1.0), {
            "active_phenomena": active_phenomena,
            "active_fields": active_fields,
            "active_field_count": active_field_count,
            "source": "fused_attribute_bank_live",
            "recipe_version": PHYSICAL_MASK_RECIPE_VERSION,
        }

    @staticmethod
    def _soft_mask_overlap(mask_a, mask_b):
        if not isinstance(mask_a, torch.Tensor) or not isinstance(mask_b, torch.Tensor):
            return torch.zeros(1, dtype=torch.float32)
        if mask_a.shape != mask_b.shape:
            raise ValueError("Soft mask overlap expects identical tensor shapes.")
        mask_a = mask_a.detach().float()
        mask_b = mask_b.detach().float()
        flat_a = mask_a.reshape(mask_a.shape[0], -1)
        flat_b = mask_b.reshape(mask_b.shape[0], -1)
        numer = torch.minimum(flat_a, flat_b).sum(dim=1)
        denom = torch.maximum(flat_a, flat_b).sum(dim=1) + 1e-8
        return numer / denom

    def _blend_motion_masks(self, bootstrap_motion_mask, physical_mask):
        blend_alpha = self._physical_mask_blend_alpha()
        if not isinstance(bootstrap_motion_mask, torch.Tensor):
            return physical_mask, blend_alpha
        if not isinstance(physical_mask, torch.Tensor):
            return bootstrap_motion_mask, blend_alpha
        effective_mask = (
            (1.0 - blend_alpha) * bootstrap_motion_mask
            + blend_alpha * physical_mask
        )
        return torch.clamp(effective_mask, self.motion_mask_floor, 1.0), blend_alpha

    def _collect_mask_metrics(self, motion_mask, prefix="motion_mask"):
        if motion_mask is None:
            return {}
        floor = float(self.motion_mask_floor)
        active_threshold = floor + 0.25 * (1.0 - floor)
        active = (motion_mask > active_threshold).float()
        return {
            f"{prefix}_mean": float(motion_mask.detach().mean().item()),
            f"{prefix}_sparsity": float((1.0 - active.mean()).detach().item()),
        }

    def _collect_motion_mask_metrics(self, motion_mask):
        return self._collect_mask_metrics(motion_mask, prefix="motion_mask")

    def extract_physics_metadata(self, data, batch_size, device, dtype):
        """从 CSV metadata 提取并编码 PINN 条件输入，缺失字段时自动回退，支持多标签"""
        if not isinstance(data, dict):
            raise RuntimeError("Training requires physics metadata dict; received non-dict sample.")

        # 支持多标签解析（逗号分隔）
        label_name = self._normalize_label(data.get("label", ""))
        label_ids_list = []
        for part in label_name.split(","):
            part = part.strip()
            canonical_part = PHENOMENON_NAME_LOOKUP.get(part.lower())
            if canonical_part in PHENOMENON_TO_ID:
                label_ids_list.append(PHENOMENON_TO_ID[canonical_part])

        if not label_ids_list:
            if label_name != "":
                raise ValueError(f"Unknown training label metadata: {label_name!r}")
            raise RuntimeError("Training sample is missing `label`; strict physical-state mode forbids fallback labels.")

        primary_label_id = label_ids_list[0]  # 主标签（向后兼容）

        # 主标签用于单个标签的场景
        label_ids_tensor = torch.full((batch_size,), primary_label_id, dtype=torch.long, device=device)

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
            "label_id": label_ids_tensor,  # 主标签（向后兼容）
            "label_ids": label_ids_list,   # 多标签列表（用于多标签路由）
            "n_numeric": n_numeric.to(dtype=dtype),
            "n_text_ids": n_text_ids,
            "q_vector": q_vector.to(dtype=dtype),
            "parse_success_ratio": parse_success_ratio.to(dtype=dtype),
        }

    def _select_training_dit_expert(self, timestep_id, models):
        if not self.use_dual_noise_experts or self.pipe.dit2 is None:
            return dict(models), "single", 0
        threshold = int(self.dual_noise_expert_boundary * self.pipe.scheduler.num_train_timesteps)
        threshold = min(max(threshold, 0), self.pipe.scheduler.num_train_timesteps)
        selected_models = dict(models)
        if int(timestep_id.item()) >= threshold:
            selected_models["dit"] = self.pipe.dit2
            return selected_models, "low_noise", 1
        return selected_models, "high_noise", 0

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

    def _resolve_active_label_names(self, metadata):
        if not isinstance(metadata, dict):
            raise RuntimeError("Training metadata contract violation: expected metadata dict.")

        active_names = []
        label_ids = metadata.get("label_ids")
        if isinstance(label_ids, torch.Tensor):
            label_ids_iter = label_ids.detach().view(-1).tolist()
        elif isinstance(label_ids, (list, tuple)):
            label_ids_iter = list(label_ids)
        else:
            label_ids_iter = []
        if label_ids_iter:
            for label_id in label_ids_iter:
                try:
                    idx = int(label_id)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(PHENOMENON_LABELS):
                    active_names.append(PHENOMENON_LABELS[idx])

        if not active_names:
            label_name = metadata.get("label_name", "")
            if isinstance(label_name, str):
                for part in label_name.split(","):
                    normalized = self._normalize_label(part)
                    canonical_name = PHENOMENON_NAME_LOOKUP.get(normalized.lower())
                    if canonical_name in PHENOMENON_TO_ID:
                        active_names.append(canonical_name)

        if not active_names:
            label_id = metadata.get("label_id")
            if isinstance(label_id, torch.Tensor) and label_id.numel() > 0:
                label_id = int(label_id.view(-1)[0].item())
            try:
                label_id = int(label_id)
            except (TypeError, ValueError):
                raise RuntimeError("Training metadata contract violation: missing label_id / label_name / label_ids.")
            if 0 <= label_id < len(PHENOMENON_LABELS):
                active_names.append(PHENOMENON_LABELS[label_id])

        if not active_names:
            raise RuntimeError("Training metadata contract violation: no valid active phenomenon labels found.")
        return list(dict.fromkeys(active_names))

    def _resolve_active_label_ids(self, metadata):
        return [
            PHENOMENON_TO_ID[name]
            for name in self._resolve_active_label_names(metadata)
            if name in PHENOMENON_TO_ID
        ]

    @staticmethod
    def _build_video_time_metadata(batch_size, num_frames, device, dtype):
        num_frames = max(int(num_frames), 1)
        frame_delta_t = 1.0 / max(num_frames - 1, 1)
        frame_time_grid = torch.linspace(
            0.0,
            1.0,
            steps=num_frames,
            device=device,
            dtype=dtype,
        )
        if num_frames <= 1:
            frame_time_grid = torch.zeros(1, device=device, dtype=dtype)
        return {
            "frame_count": torch.full(
                (batch_size,),
                float(num_frames),
                device=device,
                dtype=dtype,
            ),
            "frame_delta_t": torch.full(
                (batch_size,),
                float(frame_delta_t),
                device=device,
                dtype=dtype,
            ),
            "frame_time_grid": frame_time_grid.unsqueeze(0).repeat(batch_size, 1),
            "physics_time_source": "video_frames",
        }

    @staticmethod
    def _physics_time_metrics_from_metadata(metadata):
        if not isinstance(metadata, dict):
            return {}

        info = {}
        frame_delta_t = metadata.get("frame_delta_t")
        if isinstance(frame_delta_t, torch.Tensor) and frame_delta_t.numel() > 0:
            info["frame_delta_t"] = float(frame_delta_t.detach().float().mean().item())
        elif isinstance(frame_delta_t, (int, float)):
            info["frame_delta_t"] = float(frame_delta_t)

        frame_time_grid = metadata.get("frame_time_grid")
        if isinstance(frame_time_grid, torch.Tensor) and frame_time_grid.numel() > 0:
            grid = frame_time_grid.detach().float()
            if grid.ndim == 1:
                grid = grid.unsqueeze(0)
            elif grid.ndim > 2:
                grid = grid.view(grid.shape[0], -1)
            if grid.shape[1] > 1:
                info["frame_time_span"] = float((grid[:, -1] - grid[:, 0]).mean().item())
            else:
                info["frame_time_span"] = 0.0
            info["frame_count"] = float(grid.shape[1])

        info["physics_time_source_video_frames"] = float(
            metadata.get("physics_time_source") == "video_frames"
        )
        return info

    def _metadata_for_sample(self, metadata, sample_idx, device):
        if not isinstance(metadata, dict):
            return {}

        sample_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, torch.Tensor):
                tensor = value
                if tensor.ndim > 0 and tensor.shape[0] > sample_idx:
                    sample_metadata[key] = tensor[sample_idx:sample_idx + 1].to(device=device)
                else:
                    sample_metadata[key] = tensor.to(device=device)
            elif isinstance(value, (list, tuple)) and len(value) > sample_idx:
                sample_metadata[key] = value[sample_idx]
            else:
                sample_metadata[key] = value
        return sample_metadata

    @staticmethod
    def _first_difference_along_axis(tensor, axis):
        diff = torch.zeros_like(tensor)
        if tensor.shape[axis] <= 1:
            return diff
        if axis == 2:
            diff[:, :, :-1] = tensor[:, :, 1:] - tensor[:, :, :-1]
        elif axis == 3:
            diff[:, :, :, :-1] = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        elif axis == 4:
            diff[:, :, :, :, :-1] = tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1]
        return diff

    @staticmethod
    def _second_difference_along_axis(tensor, axis):
        second = torch.zeros_like(tensor)
        if tensor.shape[axis] <= 2:
            return second
        if axis == 2:
            second[:, :, 1:-1] = tensor[:, :, 2:] - 2.0 * tensor[:, :, 1:-1] + tensor[:, :, :-2]
        elif axis == 3:
            second[:, :, :, 1:-1] = tensor[:, :, :, 2:] - 2.0 * tensor[:, :, :, 1:-1] + tensor[:, :, :, :-2]
        elif axis == 4:
            second[:, :, :, :, 1:-1] = tensor[:, :, :, :, 2:] - 2.0 * tensor[:, :, :, :, 1:-1] + tensor[:, :, :, :, :-2]
        return second

    def _build_hierarchical_descriptor_maps(self, field):
        temporal_acceleration = self._second_difference_along_axis(field, axis=2)
        grad_h = self._first_difference_along_axis(field, axis=3)
        grad_w = self._first_difference_along_axis(field, axis=4)
        divergence_proxy = torch.mean(grad_h + grad_w, dim=1, keepdim=True)
        vorticity_proxy = torch.mean(torch.abs(grad_h - grad_w), dim=1, keepdim=True)
        acceleration_proxy = torch.mean(torch.abs(temporal_acceleration), dim=1, keepdim=True)
        return {
            "divergence": divergence_proxy,
            "vorticity": vorticity_proxy,
            "acceleration": acceleration_proxy,
        }

    def compute_conditioned_physics_loss(self, x_phys, v_phys, metadata=None, stage_name="fused"):
        active_names = self._resolve_active_label_names(metadata)
        losses = []
        infos = []
        for phenomenon_name in active_names:
            loss_i, info_i = self.compute_physics_loss(
                x_phys,
                v_phys,
                phenomenon_name,
                metadata=metadata,
            )
            losses.append(loss_i)
            infos.append(dict(info_i))

        if len(losses) == 1:
            info = infos[0]
            info["physics_stage"] = stage_name
            info["active_label_count"] = 1.0
            info["active_label_names"] = active_names[0]
            return losses[0], info

        stacked_losses = torch.stack(losses)
        merged_info = {
            "physics_mode": f"{stage_name}_multi_label",
            "physics_stage": stage_name,
            "active_label_count": float(len(active_names)),
            "active_label_names": ",".join(active_names),
        }
        numeric_keys = set()
        for info in infos:
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    numeric_keys.add(key)
        for key in numeric_keys:
            values = [float(info[key]) for info in infos if isinstance(info.get(key), (int, float))]
            if values:
                merged_info[key] = float(sum(values) / max(len(values), 1))
        return stacked_losses.mean(), merged_info

    def _mean_abs_divergence(self, v_field):
        if not isinstance(v_field, torch.Tensor) or v_field.numel() == 0:
            return torch.zeros((), device=self.physics_adapter.scale.device, dtype=torch.float32)
        divergence = self.pde_residuals.diff_ops.compute_divergence(v_field.float())
        return torch.mean(torch.abs(divergence))

    @staticmethod
    def _scalar_metric_or_nan(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return float("nan")
            return float(value.detach().float().mean().item())
        if isinstance(value, (int, float)):
            return float(value)
        return float("nan")

    def _collect_only_u_metrics(self, cache, physics_info=None):
        metrics = {
            "mass_residual": float("nan"),
            "momentum_residual": float("nan"),
            "rho_mean": float("nan"),
            "rho_min": float("nan"),
            "p_abs_mean": float("nan"),
            "div_u_abs_mean": float("nan"),
        }
        if isinstance(cache, dict):
            field_metrics = cache.get("physical_field_metrics_live")
            if field_metrics is None:
                field_metrics = cache.get("physical_field_metrics")
            if isinstance(field_metrics, dict):
                for key in ("rho_mean", "rho_min", "p_abs_mean", "div_u_abs_mean"):
                    if key in field_metrics:
                        metrics[key] = self._scalar_metric_or_nan(field_metrics[key])
        if isinstance(physics_info, dict):
            for key in ("mass_residual", "momentum_residual"):
                if key in physics_info:
                    metrics[key] = self._scalar_metric_or_nan(physics_info[key])
                elif f"physics_{key}" in physics_info:
                    metrics[key] = self._scalar_metric_or_nan(physics_info[f"physics_{key}"])
        return metrics

    def compute_physics_loss(self, x_phys, v_phys=None, label_name=None, metadata=None):
        """计算 expert-specific PDE 残差损失，输入为显式共享属性库。"""
        if label_name is None and isinstance(v_phys, str):
            attribute_bank = x_phys
            phenomenon_name = v_phys
        else:
            attribute_bank = x_phys
            phenomenon_name = label_name

        residual_input = attribute_bank
        if isinstance(attribute_bank, dict):
            residual_input = attribute_bank
        elif torch.is_tensor(attribute_bank):
            if attribute_bank.ndim != 5:
                raise ValueError(
                    "Expert PDE residuals expect attribute_bank with shape [B, C, T, H, W]; "
                    f"got {tuple(attribute_bank.shape)}."
                )
            if attribute_bank.shape[1] != PHYSICS_ATTR_DIM:
                raise ValueError(
                    f"Expert PDE residuals expect attribute_bank channel dim == {PHYSICS_ATTR_DIM}; "
                    f"got {attribute_bank.shape[1]}."
                )
            residual_input = torch.clamp(attribute_bank, min=-10.0, max=10.0)
        else:
            raise TypeError(
                "Expert PDE residuals expect a tensor attribute bank or physical field dict; "
                f"got {type(attribute_bank).__name__}."
            )

        pde_metadata = None if self.ablate_disable_conditioned_pde else metadata
        phenomenon_name = self._normalize_label(phenomenon_name) if isinstance(phenomenon_name, str) else ""
        phenomenon_name = PHENOMENON_NAME_LOOKUP.get(phenomenon_name.lower(), phenomenon_name)
        residual_method_name = PHENOMENON_TO_RESIDUAL_METHOD.get(phenomenon_name)
        if residual_method_name is not None:
            residual_method = getattr(self.pde_residuals, residual_method_name)
            loss, info = residual_method(residual_input, metadata=pde_metadata)
        else:
            raise KeyError(f"Unknown phenomenon label for PDE residual routing: {phenomenon_name!r}")

        self._ensure_finite_tensor(f"physics_loss.{phenomenon_name}", loss)
        loss = torch.clamp(loss, min=0.0, max=100.0)

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
                if isinstance(residual_input, dict):
                    detached_input = {
                        key: value.detach() if torch.is_tensor(value) else value
                        for key, value in residual_input.items()
                    }
                else:
                    detached_input = residual_input.detach()
                unmasked_loss, _ = residual_method(
                    detached_input,
                    metadata=unmasked_metadata,
                )
                if torch.isnan(unmasked_loss) or torch.isinf(unmasked_loss):
                    unmasked_loss = torch.tensor(1e-6, device=loss.device, dtype=loss.dtype)
                else:
                    unmasked_loss = torch.clamp(unmasked_loss, min=1e-8, max=100.0)
            info["masked_vs_unmasked_residual_ratio"] = float(
                (loss.detach() / (unmasked_loss.detach() + 1e-8)).item()
            )
        if phenomenon_name:
            info["phenomenon_name"] = phenomenon_name
        info["phenomenon"] = phenomenon_name
        return loss, info

    def compute_targeted_expert_physics_loss(self, shared_attribute_bank, sigma=None, metadata=None):
        cache = getattr(self.physics_adapter, "_cache", {})
        branch_attribute_updates = cache.get("branch_attribute_updates_live")
        branch_indices = cache.get("active_expert_indices")
        branch_weights = cache.get("active_expert_weights_live")
        physics_feat_live = cache.get("physics_feat_live")
        if physics_feat_live is None:
            physics_feat_live = cache.get("physics_feat")
        if branch_weights is None:
            branch_weights = cache.get("active_expert_weights")
        if (
            shared_attribute_bank is None
            or branch_attribute_updates is None
            or branch_indices is None
            or branch_weights is None
        ):
            raise RuntimeError(
                "Explicit attribute-bank PDE loss requires shared_attribute_bank_live, "
                "branch_attribute_updates_live, active_expert_indices, and active_expert_weights."
            )

        batch_size = int(shared_attribute_bank.shape[0])
        effective_sigma_threshold = self.get_expert_pde_sigma_threshold()
        total_loss = torch.zeros((), device=shared_attribute_bank.device, dtype=shared_attribute_bank.dtype)
        enabled_samples = 0
        sigma_threshold_pass_samples = 0
        sigma_enabled_bucket_counts = {"low": 0.0, "mid": 0.0, "high": 0.0}
        total_target_experts = 0.0
        selected_weight_mass = 0.0
        expert_usage = torch.zeros(len(PHENOMENON_LABELS), device=shared_attribute_bank.device, dtype=torch.float32)
        weighted_expert_loss = torch.zeros_like(expert_usage)
        aggregated_metric_sum = {}
        aggregated_metric_count = {}

        for sample_idx in range(int(shared_attribute_bank.shape[0])):
            if isinstance(sigma, torch.Tensor):
                sigma_sample = sigma[sample_idx:sample_idx + 1].detach().float().view(-1)
                sigma_value = float(sigma_sample.mean().item()) if sigma_sample.numel() > 0 else float("inf")
            else:
                sigma_value = float(sigma) if sigma is not None else 0.0
            sigma_bucket_name = self._sigma_bucket_name_from_value(sigma_value)
            if sigma_value > effective_sigma_threshold:
                continue

            sigma_threshold_pass_samples += 1
            try:
                active_label_ids = set(self._resolve_active_label_ids_for_sample(metadata, sample_idx))
            except RuntimeError:
                continue

            sample_indices = branch_indices[sample_idx].detach().view(-1)
            sample_weights = branch_weights[sample_idx].view(-1)
            target_positions = [
                pos for pos, expert_idx in enumerate(sample_indices.tolist())
                if expert_idx in active_label_ids
            ]
            if not target_positions:
                continue

            sample_shared_bank = shared_attribute_bank[sample_idx:sample_idx + 1]
            sample_loss = torch.zeros((), device=total_loss.device, dtype=total_loss.dtype)
            sample_weight_sum = torch.zeros((), device=total_loss.device, dtype=total_loss.dtype)

            for pos in target_positions:
                expert_idx = int(sample_indices[pos].item())
                weight = torch.clamp(sample_weights[pos], min=0.0)
                if float(weight.detach().item()) <= 0.0:
                    continue

                phenomenon_name = PHENOMENON_LABELS[expert_idx]
                expert_bank = sample_shared_bank + branch_attribute_updates[sample_idx:sample_idx + 1, pos]
                expert_metadata = self._metadata_for_sample_and_expert(
                    metadata,
                    sample_idx,
                    phenomenon_name,
                    device=shared_attribute_bank.device,
                    dtype=shared_attribute_bank.dtype,
                )
                expert_source = expert_bank
                if self.ablation_preset.startswith("u_only_"):
                    if not isinstance(physics_feat_live, torch.Tensor):
                        raise RuntimeError(
                            "Only-u PDE routing requires physics_feat_live in PhysicsAdapter cache."
                        )
                    expert_fields, _, _ = self.physics_adapter.build_physical_field_dict(
                        expert_bank,
                        physics_feat_live[sample_idx:sample_idx + 1],
                        metadata=expert_metadata,
                        field_recovery_phase=self.field_recovery_phase,
                    )
                    expert_source = expert_fields
                loss_i, info_i = self.compute_physics_loss(
                    expert_source,
                    phenomenon_name,
                    metadata=expert_metadata,
                )
                weight = weight.to(device=loss_i.device, dtype=loss_i.dtype)
                sample_loss = sample_loss + weight * loss_i
                sample_weight_sum = sample_weight_sum + weight
                expert_usage[expert_idx] += 1.0
                weighted_expert_loss[expert_idx] += float((weight.detach() * loss_i.detach()).item())
                selected_weight_mass += float(weight.detach().item())
                total_target_experts += 1.0
                for key, value in info_i.items():
                    if isinstance(value, (int, float)):
                        aggregated_metric_sum[key] = aggregated_metric_sum.get(key, 0.0) + float(value)
                        aggregated_metric_count[key] = aggregated_metric_count.get(key, 0.0) + 1.0

            if float(sample_weight_sum.detach().item()) <= 0.0:
                continue

            total_loss = total_loss + sample_loss / sample_weight_sum
            enabled_samples += 1
            sigma_enabled_bucket_counts[sigma_bucket_name] += 1.0

        if enabled_samples == 0:
            zero = torch.zeros((), device=shared_attribute_bank.device, dtype=shared_attribute_bank.dtype)
            return zero, {
                "physics_mode": "explicit_attribute_bank_v2_expert_disabled",
                "physics_stage": "expert_targeted",
                "sigma_threshold": float(effective_sigma_threshold),
                "sigma_threshold_effective": float(effective_sigma_threshold),
                "sigma_threshold_start": float(self.expert_pde_sigma_threshold),
                "sigma_threshold_target": float(self.expert_pde_sigma_threshold_target),
                "sigma_threshold_pass_samples": float(sigma_threshold_pass_samples),
                "sigma_enabled_samples": 0.0,
                "sigma_enabled_ratio": 0.0,
                "sigma_enabled_low": 0.0,
                "sigma_enabled_mid": 0.0,
                "sigma_enabled_high": 0.0,
                "physics_enabled_samples": 0.0,
                "target_expert_count": 0.0,
                **self._physics_time_metrics_from_metadata(metadata),
            }

        total_loss = total_loss / float(enabled_samples)
        info = {
            "physics_mode": "explicit_attribute_bank_v2_expert",
            "physics_stage": "expert_targeted",
            "sigma_threshold": float(effective_sigma_threshold),
            "sigma_threshold_effective": float(effective_sigma_threshold),
            "sigma_threshold_start": float(self.expert_pde_sigma_threshold),
            "sigma_threshold_target": float(self.expert_pde_sigma_threshold_target),
            "sigma_threshold_pass_samples": float(sigma_threshold_pass_samples),
            "sigma_enabled_samples": float(enabled_samples),
            "sigma_enabled_ratio": float(enabled_samples / max(batch_size, 1)),
            "sigma_enabled_low": float(sigma_enabled_bucket_counts["low"] / max(batch_size, 1)),
            "sigma_enabled_mid": float(sigma_enabled_bucket_counts["mid"] / max(batch_size, 1)),
            "sigma_enabled_high": float(sigma_enabled_bucket_counts["high"] / max(batch_size, 1)),
            "physics_enabled_samples": float(enabled_samples),
            "target_expert_count": float(total_target_experts / max(enabled_samples, 1)),
            "target_weight_mass": float(selected_weight_mass / max(enabled_samples, 1)),
            **self._physics_time_metrics_from_metadata(metadata),
        }
        for expert_idx, label in enumerate(PHENOMENON_LABELS):
            if expert_usage[expert_idx].item() <= 0:
                continue
            info[f"pde_target_usage/{label}"] = float(expert_usage[expert_idx].item())
            info[f"pde_target_loss/{label}"] = float(
                weighted_expert_loss[expert_idx].item() / max(expert_usage[expert_idx].item(), 1.0)
            )
        for key, total in aggregated_metric_sum.items():
            count = max(aggregated_metric_count.get(key, 0.0), 1.0)
            info[key] = float(total / count)
        return total_loss, info

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

    def _compute_decoded_branch_consistency_loss(self):
        if self.ablate_disable_moe:
            return None

        cache = getattr(self.physics_adapter, "_cache", {})
        if not isinstance(cache, dict):
            raise RuntimeError("Decoded branch consistency requires a valid adapter cache.")

        branch_corrections = cache.get("branch_raw_corrections")
        branch_weights = cache.get("active_expert_weights")
        fused_correction = cache.get("raw_correction")
        if branch_corrections is None or branch_weights is None or fused_correction is None:
            raise RuntimeError(
                "Decoded branch consistency contract violation: missing branch_raw_corrections, "
                "active_expert_weights, or fused correction tensor."
            )
        if branch_corrections.numel() == 0 or branch_weights.numel() == 0:
            raise RuntimeError("Decoded branch consistency contract violation: empty branch tensors.")

        weight_shape = (branch_weights.shape[0], branch_weights.shape[1], 1, 1, 1, 1)
        weighted_branch_correction = torch.sum(
            branch_corrections.float() * branch_weights.float().view(*weight_shape), dim=1
        )
        fused_correction = fused_correction.float()
        l1 = torch.mean(torch.abs(weighted_branch_correction - fused_correction))
        l2 = torch.mean((weighted_branch_correction - fused_correction) ** 2)
        loss = torch.clamp(l1 + l2, min=0.0, max=100.0)
        self._ensure_finite_tensor("loss_decoded_branch_consistency", loss)
        return loss

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
        base_weights = cache.get("router_topk_weights")
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
            "moe_rl_policy_entropy": float(
                (-(active_weights * torch.log(active_weights.clamp_min(1e-6)))).sum(dim=-1).mean().item()
            ),
        }
        offlabel_selected = cache.get("offlabel_selected_experts")
        if (
            isinstance(offlabel_selected, torch.Tensor)
            and tuple(offlabel_selected.shape) == tuple(active_indices.shape)
        ):
            offlabel_mask = offlabel_selected.ge(0)
            metrics["moe_offlabel_selected_count_mean"] = float(
                offlabel_mask.float().sum(dim=1).mean().item()
            )
            metrics["moe_offlabel_selected_weight_mean"] = float(
                (active_weights * offlabel_mask.float()).sum(dim=1).mean().item()
            )
            metrics["moe_offlabel_selected_ratio"] = float(offlabel_mask.float().mean().item())
        adaptive_condition_gate = cache.get("adaptive_condition_gate")
        adaptive_correction_gate = cache.get("adaptive_correction_gate")
        shared_condition_gate = cache.get("shared_condition_gate")
        shared_correction_gate = cache.get("shared_correction_gate")
        if adaptive_condition_gate is not None:
            metrics["moe_adaptive_condition_gate_mean"] = float(
                adaptive_condition_gate.float().mean().item()
            )
        if adaptive_correction_gate is not None:
            metrics["moe_adaptive_correction_gate_mean"] = float(
                adaptive_correction_gate.float().mean().item()
            )
        if shared_condition_gate is not None:
            metrics["moe_shared_condition_gate_mean"] = float(
                shared_condition_gate.float().mean().item()
            )
        if shared_correction_gate is not None:
            metrics["moe_shared_correction_gate_mean"] = float(
                shared_correction_gate.float().mean().item()
            )
        if base_weights is not None and tuple(base_weights.shape) == tuple(active_weights.shape):
            metrics["moe_rl_weight_shift"] = float(
                torch.mean(torch.abs(active_weights - base_weights.float())).item()
            )
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
        if "label_ids" in metadata:
            filtered["label_ids"] = metadata["label_ids"]
        if "label_name" in metadata:
            filtered["label_name"] = metadata["label_name"]
        # 保留 parse ratio 用于稳定训练时的缩放
        if "parse_success_ratio" in metadata:
            filtered["parse_success_ratio"] = metadata["parse_success_ratio"]
        # 保留 motion mask，确保 label-only 消融不影响动态区域约束
        if "motion_mask" in metadata:
            filtered["motion_mask"] = metadata["motion_mask"]
        if "bootstrap_motion_mask" in metadata:
            filtered["bootstrap_motion_mask"] = metadata["bootstrap_motion_mask"]
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
        self._apply_stage2_encoder_schedule()
        self._set_forward_debug_context(
            training_stage=self.training_stage,
            current_step=int(self.current_step),
        )

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        
        # ============================================================
        # 1. 获取原始模型的速度场预测（frozen, no grad）
        # ============================================================
        if self.debug_fixed_timestep_fraction is not None:
            timestep_index = int(round(
                self.debug_fixed_timestep_fraction
                * max(self.pipe.scheduler.num_train_timesteps - 1, 0)
            ))
            timestep_id = torch.tensor([timestep_index], dtype=torch.long)
        else:
            max_boundary = int(inputs.get("max_timestep_boundary", 1) * self.pipe.scheduler.num_train_timesteps)
            min_boundary = int(inputs.get("min_timestep_boundary", 0) * self.pipe.scheduler.num_train_timesteps)
            timestep_id = torch.randint(min_boundary, max_boundary, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype, device=inputs["latents"].device
        )
        models, active_noise_regime, active_dit_expert_index = self._select_training_dit_expert(
            timestep_id, models
        )
        self._set_forward_debug_context(
            timestep_id=int(timestep_id.item()),
            timestep=float(timestep.detach().float().reshape(-1)[0].item()),
            noise_regime=active_noise_regime,
        )
        
        input_latents = inputs.get("input_latents", inputs["latents"])
        noise = inputs.get("noise", torch.randn_like(inputs["latents"]))
        z_t = self.pipe.scheduler.add_noise(input_latents, noise, timestep)
        self._ensure_finite_tensor("z_t", z_t)
        
        # FM 的训练目标（速度场真值）
        v_target = self.pipe.scheduler.training_target(input_latents, noise, timestep)

        # Clamp v_target to prevent numerical issues
        v_target = torch.clamp(v_target, min=-10.0, max=10.0)
        self._ensure_finite_tensor("v_target", v_target)

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

        # Aggressive clamping immediately after v_original computation
        v_original = torch.clamp(v_original, min=-10.0, max=10.0)
        self._ensure_finite_tensor("v_original", v_original)

        metadata = self.extract_physics_metadata(
            data=data,
            batch_size=v_original.shape[0],
            device=v_original.device,
            dtype=v_original.dtype,
        )
        metadata["noise_regime"] = active_noise_regime
        metadata["active_dit_expert_index"] = active_dit_expert_index
        metadata["dual_noise_expert_boundary"] = float(self.dual_noise_expert_boundary)
        metadata["timestep_index_fraction"] = torch.full(
            (v_original.shape[0],),
            float(timestep_id.item()) / max(float(self.pipe.scheduler.num_train_timesteps), 1.0),
            device=v_original.device,
            dtype=v_original.dtype,
        )
        sigma = self._scheduler_sigma(
            timestep,
            device=v_original.device,
            dtype=v_original.dtype,
        )
        self._ensure_finite_tensor("sigma", sigma)
        self._set_forward_debug_context(
            sigma=sigma,
            metadata=self._metadata_debug_summary(metadata),
        )
        physics_state_target = self._physics_state_from_prediction(
            z_t,
            v_target,
            sigma,
        )
        self._ensure_finite_tensor("physics_state_target", physics_state_target)
        physics_state_original = self._physics_state_from_prediction(
            z_t,
            v_original,
            sigma,
        )
        self._ensure_finite_tensor("physics_state_original", physics_state_original)
        phenomenon = metadata.get("label_name")
        if not isinstance(phenomenon, str) or not phenomenon.strip():
            active_names = self._resolve_active_label_names(metadata)
            phenomenon = ",".join(active_names)
        self._set_forward_debug_context(phenomenon=phenomenon)
        if self.training_stage == "observable_pretrain":
            self._prepare_physics_adapter_debug(
                phase="observable_pretrain",
                phenomenon=phenomenon,
                timestep_id=int(timestep_id.item()),
                sigma=sigma,
                metadata=metadata,
            )
            return self._forward_observable_pretrain(
                phenomenon=phenomenon,
                physics_state_original=physics_state_original,
                sigma=sigma,
            )
        if self.training_stage == "encoder_completion":
            if isinstance(metadata, dict):
                metadata = dict(metadata)
            else:
                raise RuntimeError("encoder_completion requires extracted metadata dict.")
            metadata.update(
                self._build_video_time_metadata(
                    batch_size=v_original.shape[0],
                    num_frames=physics_state_original.shape[2],
                    device=v_original.device,
                    dtype=v_original.dtype,
                )
            )
            self._set_forward_debug_context(metadata=self._metadata_debug_summary(metadata))
            self._prepare_physics_adapter_debug(
                phase="encoder_completion",
                phenomenon=phenomenon,
                timestep_id=int(timestep_id.item()),
                sigma=sigma,
                metadata=metadata,
            )
            return self._forward_encoder_completion(
                phenomenon=phenomenon,
                physics_state_original=physics_state_original,
                sigma=sigma,
                metadata=metadata,
            )
        bootstrap_motion_mask = self._build_motion_mask(
            v_original.detach(),
            physics_state_original.detach(),
        )
        if isinstance(metadata, dict):
            metadata = dict(metadata)
        else:
            raise RuntimeError("Physics training requires extracted metadata dict.")
        metadata.update(
            self._build_video_time_metadata(
                batch_size=v_original.shape[0],
                num_frames=physics_state_original.shape[2],
                device=v_original.device,
                dtype=v_original.dtype,
            )
        )
        metadata["bootstrap_motion_mask"] = bootstrap_motion_mask
        metadata["motion_mask"] = bootstrap_motion_mask
        metadata["motion_mask_source"] = "bootstrap_self_supervised"
        metadata["bootstrap_motion_mask_source"] = "self_supervised"

        # Extract phenomenon label for logging
        if not isinstance(metadata, dict):
            raise RuntimeError("Physics training requires extracted metadata dict for phenomenon logging.")
        adapter_metadata = metadata
        if self.ablate_disable_moe:
            adapter_metadata = None
        elif self.ablate_label_only_router:
            adapter_metadata = self._label_only_metadata(metadata)
        self._prepare_physics_adapter_debug(
            phase="full_pinn",
            phenomenon=phenomenon,
            timestep_id=int(timestep_id.item()),
            sigma=sigma,
            metadata=adapter_metadata if isinstance(adapter_metadata, dict) else metadata,
        )
        
        # ============================================================
        # 2. PhysicsAdapter 施加物理校正（trainable, has grad）
        # ============================================================
        v_corrected = self.physics_adapter(
            v_original,
            physics_state_original,
            sigma=sigma,
            metadata=adapter_metadata,
        )

        # Aggressive clamping immediately after v_corrected computation
        # v_corrected = torch.clamp(v_corrected, min=-10.0, max=10.0)
        self._ensure_finite_tensor("v_corrected", v_corrected)

        
        # ============================================================
        # 3. 计算损失（全部来自可训练参数，确保有 grad_fn）
        # ============================================================
        physics_weight = self.get_physics_weight()
        pde_metadata = metadata
        if self.ablate_disable_conditioned_pde:
            pde_metadata = None
        expert_stats = {}
        collect_diagnostics = (
            self.current_step > 0
            and self.current_step % self.diagnostic_metrics_interval == 0
        )
        cache = getattr(self.physics_adapter, "_cache", {})
        if not isinstance(cache, dict):
            raise RuntimeError("PhysicsAdapter must expose cache dict in training mode.")
        shared_attribute_bank = cache.get("shared_attribute_bank_live")
        if shared_attribute_bank is None:
            raise RuntimeError("Shared attribute bank is missing from PhysicsAdapter cache.")
        fused_attribute_bank = cache.get("fused_attribute_bank_live")
        if fused_attribute_bank is None:
            raise RuntimeError("Fused attribute bank is missing from PhysicsAdapter cache.")
        observable_diagnostics = self._collect_observable_diagnostics(
            shared_attribute_bank,
            physics_state_original.detach(),
        )

        zero_loss = torch.zeros((), device=v_corrected.device, dtype=v_corrected.dtype)
        explicit_physical_interface = bool(
            getattr(self.physics_adapter, "interpret_attribute_bank_as_physical", True)
        )
        if explicit_physical_interface:
            physical_mask, physical_mask_info = self._build_active_physical_mask(
                fused_attribute_bank,
                metadata=metadata,
                cache=cache,
            )
        else:
            physical_mask = torch.clamp(
                bootstrap_motion_mask.detach(),
                self.motion_mask_floor,
                1.0,
            )
            physical_mask_info = {
                "active_phenomena": [[] for _ in range(v_original.shape[0])],
                "active_fields": [[] for _ in range(v_original.shape[0])],
                "active_field_count": torch.zeros(
                    v_original.shape[0],
                    device=v_original.device,
                    dtype=v_original.dtype,
                ),
                "source": f"{self.core_ablation_mode}_bootstrap_motion_mask",
                "recipe_version": "no_explicit_physical_interface",
            }
        effective_motion_mask, physical_mask_blend_alpha = self._blend_motion_masks(
            bootstrap_motion_mask.detach(),
            physical_mask.detach(),
        )
        heuristic_physical_mask_overlap = self._soft_mask_overlap(
            bootstrap_motion_mask,
            physical_mask,
        ).to(device=v_original.device, dtype=v_original.dtype)

        metadata["bootstrap_motion_mask"] = bootstrap_motion_mask.detach()
        metadata["physical_mask"] = physical_mask.detach()
        metadata["effective_motion_mask"] = effective_motion_mask.detach()
        metadata["motion_mask"] = effective_motion_mask.detach()
        metadata["motion_mask_source"] = (
            "active_physical_blend"
            if physical_mask_blend_alpha < 1.0 else "fused_active_physical"
        )
        metadata["physical_mask_source"] = physical_mask_info["source"]
        metadata["physical_mask_recipe_version"] = physical_mask_info["recipe_version"]
        metadata["physical_mask_blend_alpha"] = torch.full(
            (v_original.shape[0],),
            float(physical_mask_blend_alpha),
            device=v_original.device,
            dtype=v_original.dtype,
        )
        metadata["heuristic_physical_mask_overlap"] = heuristic_physical_mask_overlap
        metadata["physical_mask_active_field_count"] = physical_mask_info["active_field_count"].to(
            device=v_original.device,
            dtype=v_original.dtype,
        )
        metadata["physical_mask_active_phenomena"] = [
            ",".join(names) for names in physical_mask_info["active_phenomena"]
        ]
        metadata["physical_mask_active_fields"] = [
            ",".join(fields) for fields in physical_mask_info["active_fields"]
        ]

        motion_mask_metrics = {
            **self._collect_mask_metrics(effective_motion_mask, prefix="motion_mask"),
            **self._collect_mask_metrics(bootstrap_motion_mask, prefix="bootstrap_motion_mask"),
            **self._collect_mask_metrics(physical_mask, prefix="physical_mask"),
            "physical_mask_blend_alpha": float(physical_mask_blend_alpha),
            "physical_mask_source_fused_field": 1.0,
            "physical_mask_active_field_count": float(
                metadata["physical_mask_active_field_count"].detach().float().mean().item()
            ),
            "heuristic_physical_mask_overlap": float(
                heuristic_physical_mask_overlap.detach().float().mean().item()
            ),
            "active_dit_expert_index": float(active_dit_expert_index),
        }
        if active_noise_regime != "single":
            motion_mask_metrics[f"noise_regime_{active_noise_regime}"] = 1.0

        pde_disabled_by_core_mode = self.core_ablation_mode in {
            "generic_latent_correction",
            "wo_explicit_physical_interface",
            "wo_pde_residuals",
        }
        pde_disabled_by_weight = float(physics_weight) <= 0.0
        if pde_disabled_by_core_mode or pde_disabled_by_weight:
            loss_physics_shared = zero_loss
            physics_info = {
                "physics_mode": f"{self.core_ablation_mode}_pde_disabled",
                "physics_stage": "disabled",
                "pde_disabled_by_core_mode": float(pde_disabled_by_core_mode),
                "pde_disabled_by_weight": float(pde_disabled_by_weight),
                "active_label_count": float(len(self._resolve_active_label_names(metadata))),
            }
        else:
            loss_physics_shared, physics_info = self.compute_targeted_expert_physics_loss(
                shared_attribute_bank,
                sigma=sigma,
                metadata=pde_metadata,
            )
        physics_info = {
            **physics_info,
            "physics_mode": (
                str(physics_info.get("physics_mode"))
                if pde_disabled_by_core_mode or pde_disabled_by_weight
                else
                "only_u_physical_fields_expert"
                if self.ablation_preset.startswith("u_only_")
                else "explicit_attribute_bank_v2_expert"
            ),
        }
        loss_physics = loss_physics_shared
        loss_physics_value = float(loss_physics.detach().item())

        parse_ratio = 1.0
        if isinstance(metadata, dict) and "parse_success_ratio" in metadata:
            parse_ratio = float(metadata["parse_success_ratio"].detach().item())
        conditioned_scale = 1.0

        self._ensure_finite_tensor("loss_physics", loss_physics)
        loss_output_physics = zero_loss
        output_physics_info = {
            "physics_stage": "output_disabled",
            "active_label_count": float(len(self._resolve_active_label_names(metadata))),
        }
        self._ensure_finite_tensor("loss_output_physics", loss_output_physics)
        output_divergence_before = float(
            self._mean_abs_divergence(v_original.detach()).detach().item()
        )
        output_divergence_after = float(
            self._mean_abs_divergence(v_corrected.detach()).detach().item()
        )
        output_divergence_reduction = float(
            (output_divergence_before - output_divergence_after)
            / max(output_divergence_before, 1e-8)
        )

        state_align_alpha = 0.0
        state_align_v_weight = 0.0
        loss_state_align_x = zero_loss
        loss_state_align_v = zero_loss
        loss_state_align_x_weighted = zero_loss
        loss_state_align_v_weighted = zero_loss
        
        correction = v_corrected - v_original
        loss_reg = torch.mean(correction ** 2)
        correction_ratio = float(
            correction.detach().abs().mean().item() /
            (v_original.detach().abs().mean().item() + 1e-10)
        )
        raw_scale_estimate = float("nan")
        raw_correction_norm = cache.get("raw_correction_norm")
        gated_correction_norm = cache.get("gated_correction_norm")
        if (
            isinstance(raw_correction_norm, torch.Tensor)
            and isinstance(gated_correction_norm, torch.Tensor)
            and raw_correction_norm.numel() > 0
            and gated_correction_norm.numel() > 0
        ):
            raw_scale_estimate = float(
                (
                    gated_correction_norm.detach().float()
                    / (raw_correction_norm.detach().float() + 1e-8)
                ).mean().item()
            )
        loss_reg = torch.clamp(loss_reg, min=0.0, max=100.0)
        self._ensure_finite_tensor("loss_reg", loss_reg)

        loss_fm_adapter = torch.nn.functional.mse_loss(
            v_corrected.float(), v_target.float()
        )
        loss_fm_adapter = torch.clamp(loss_fm_adapter, min=0.0, max=100.0)
        self._ensure_finite_tensor("loss_fm_adapter", loss_fm_adapter)

        aux_losses = self.physics_adapter.compute_auxiliary_losses()
        loss_expert_balance = aux_losses["expert_balance"]
        loss_condition_consistency = zero_loss
        loss_policy_rl = zero_loss
        policy_entropy = zero_loss
        rl_reward_mean = zero_loss
        rl_advantage_mean = zero_loss
        rl_reward_metrics = {}

        loss_expert_balance = torch.clamp(loss_expert_balance, min=0.0, max=100.0)
        self._ensure_finite_tensor("loss_expert_balance", loss_expert_balance)
        loss_decoded_branch_consistency = None
        branch_consistency_weighted = None

        if self.ablate_disable_aux_losses:
            loss_expert_balance = loss_expert_balance * 0.0
        rl_alpha = 0.0

        total_loss = loss_fm_adapter + physics_weight * loss_physics + 0.01 * loss_reg

        self._ensure_finite_tensor("total_loss", total_loss)
        router_metrics, expert_residual_metrics, decoded_branch_metrics = {}, {}, {}
        if collect_diagnostics:
            router_metrics = self._collect_router_metrics(loss_physics_value)
            expert_residual_metrics = {}
            decoded_branch_metrics = {}
        only_u_metrics = self._collect_only_u_metrics(cache, physics_info=physics_info)
        self._last_metrics = {
            "phenomenon": phenomenon,
            "core_ablation_mode": self.core_ablation_mode,
            "explicit_physical_interface": float(explicit_physical_interface),
            "physics_to_flow_injection_enabled": float(
                getattr(self.physics_adapter, "physics_to_flow_injection_enabled", True)
            ),
            "phenomenon_specific_operators": float(
                getattr(self.physics_adapter, "use_phenomenon_specific_operators", True)
            ),
            "ablation_preset": self.ablation_preset,
            "observable_target_mode": self.observable_target_mode,
            "secondary_field_strategy": self.secondary_field_strategy,
            "active_field_set": self.active_field_set,
            "field_enable_schedule": self.field_enable_schedule,
            "ablate_disable_moe": self.ablate_disable_moe,
            "ablate_disable_conditioned_pde": self.ablate_disable_conditioned_pde,
            "ablate_disable_aux_losses": self.ablate_disable_aux_losses,
            "ablate_label_only_router": self.ablate_label_only_router,
            "conditioned_scale": conditioned_scale,
            "loss_fm_adapter": float(loss_fm_adapter.detach().item()),
            "loss_reg": float(loss_reg.detach().item()),
            "physics_weight_effective": float(physics_weight),
            "curriculum_progress": float(self._get_curriculum_progress()),
            "output_physics_weight": float(self.output_physics_weight),
            "state_align_v_weight_effective": float(state_align_alpha * state_align_v_weight),
            "loss_physics_weighted": float(
                (physics_weight * loss_physics).detach().item()
            ),
            "loss_state_align_x_weighted": float(
                loss_state_align_x_weighted.detach().item()
            ),
            "loss_state_align_v_weighted": float(
                loss_state_align_v_weighted.detach().item()
            ),
            "loss_decoded_branch_consistency_weighted": (
                float(branch_consistency_weighted.detach().item())
                if branch_consistency_weighted is not None else float("nan")
            ),
            "loss_expert_balance_weighted": float(
                0.0
            ),
            "loss_condition_consistency_weighted": float(
                0.0
            ),
            "loss_policy_rl_weighted": float(
                0.0
            ),
            "loss_policy_entropy_weighted": float(
                0.0
            ),
            "physics_total": loss_physics_value,
            "physics_shared": loss_physics_value,
            "output_physics_total": float(loss_output_physics.detach().item()),
            "output_divergence_before": output_divergence_before,
            "output_divergence_after": output_divergence_after,
            "output_divergence_reduction": output_divergence_reduction,
            "raw_scale_estimate": raw_scale_estimate,
            "state_align_x": float(loss_state_align_x.detach().item()),
            "state_align_v": float(loss_state_align_v.detach().item()),
            "decoded_branch_consistency": (
                float(loss_decoded_branch_consistency.detach().item())
                if loss_decoded_branch_consistency is not None else float("nan")
            ),
            "expert_balance": float(loss_expert_balance.detach().item()),
            "condition_consistency": float(loss_condition_consistency.detach().item()),
            "policy_rl": float(loss_policy_rl.detach().item()),
            "policy_entropy": float(policy_entropy.detach().item()),
            "rl_reward_mean": float(rl_reward_mean.detach().item()),
            "rl_advantage_mean": float(rl_advantage_mean.detach().item()),
            "rl_alpha": float(rl_alpha),
            "state_align_alpha": float(state_align_alpha),
            "parse_success_ratio": parse_ratio,
            "sigma_mean": float(sigma.detach().float().mean().item()),
            "training_stage": self.training_stage,
            "encoder_frozen": float(self._stage2_encoder_frozen is True),
            "sigma_gate_mean": self._cache_scalar_mean_or_nan(cache, "sigma_gate"),
            "effective_scale_mean": self._cache_scalar_mean_or_nan(cache, "effective_scale"),
            "adaptive_condition_gate_mean": self._cache_scalar_mean_or_nan(cache, "adaptive_condition_gate"),
            "adaptive_correction_gate_mean": self._cache_scalar_mean_or_nan(cache, "adaptive_correction_gate"),
            "shared_condition_gate_mean": self._cache_scalar_mean_or_nan(cache, "shared_condition_gate"),
            "shared_correction_gate_mean": self._cache_scalar_mean_or_nan(cache, "shared_correction_gate"),
            "raw_correction_norm_mean": self._cache_mean_or_nan(cache, "raw_correction_norm"),
            "gated_correction_norm_mean": self._cache_mean_or_nan(cache, "gated_correction_norm"),
            "fused_x_phys_norm_mean": self._cache_norm_mean_or_nan(cache, "fused_x_phys"),
            "fused_v_phys_norm_mean": self._cache_norm_mean_or_nan(cache, "fused_v_phys"),
            "correction_ratio": correction_ratio,
            **motion_mask_metrics,
            **rl_reward_metrics,
            **only_u_metrics,
            **observable_diagnostics,
            **self._sigma_bucket_metrics(sigma),
            **{f"physics_{k}": v for k, v in physics_info.items()},
            **{
                f"output_physics_{k}": v
                for k, v in output_physics_info.items()
                if isinstance(v, (int, float))
            },
            **router_metrics,
            **expert_residual_metrics,
            **decoded_branch_metrics,
        }

        self._latest_explainability_snapshot = self._build_explainability_snapshot(
            data=data,
            metadata=metadata,
            cache=cache,
            sigma=sigma,
            bootstrap_motion_mask=bootstrap_motion_mask,
            physical_mask=physical_mask,
            effective_motion_mask=effective_motion_mask,
            loss_physics_value=loss_physics_value,
            loss_output_physics=loss_output_physics,
            output_divergence_before=output_divergence_before,
            output_divergence_after=output_divergence_after,
            output_divergence_reduction=output_divergence_reduction,
            raw_scale_estimate=raw_scale_estimate,
            correction_ratio=correction_ratio,
            phenomenon=phenomenon,
        )

        self.current_step += 1
        
        return total_loss


class PINNModelLogger(ModelLogger):
    """
    PINN 专用的模型保存器
    只保存 PINN 插件参数（不保存原模型参数）
    支持保存和恢复优化器、调度器状态
    """

    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x: x,
        enable_explainability_reports=False,
        explainability_report_interval=100,
        explainability_max_history=50,
        explainability_output_dir=None,
    ):
        super().__init__(output_path, remove_prefix_in_ckpt, state_dict_converter)
        self.optimizer_state = None
        self.scheduler_state = None
        self.current_step = 0
        self.current_epoch = 0
        self.enable_explainability_reports = bool(enable_explainability_reports)
        self.explainability_report_interval = max(int(explainability_report_interval), 1)
        self.explainability_max_history = max(int(explainability_max_history), 1)
        self.explainability_output_dir = (
            explainability_output_dir
            if explainability_output_dir is not None
            else os.path.join(output_path, "explainability")
        )
        self._explainability_history = []

    @staticmethod
    def _to_float(value, default=0.0):
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return float(default)

    @staticmethod
    def _format_scalar(value, precision=4):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        return str(value)

    @staticmethod
    def _html_escape(value):
        return html.escape(str(value), quote=True)

    @staticmethod
    def _build_table(headers, rows):
        if len(rows) == 0:
            return "<p class='empty'>No data.</p>"
        head_html = "".join(f"<th>{PINNModelLogger._html_escape(header)}</th>" for header in headers)
        body_rows = []
        for row in rows:
            body_rows.append(
                "<tr>" + "".join(f"<td>{PINNModelLogger._html_escape(row.get(header, '-'))}</td>" for header in headers) + "</tr>"
            )
        return (
            "<table><thead><tr>"
            + head_html
            + "</tr></thead><tbody>"
            + "".join(body_rows)
            + "</tbody></table>"
        )

    @staticmethod
    def _build_bar_chart(title, items, width=640, row_height=28):
        if len(items) == 0:
            return f"<section><h3>{PINNModelLogger._html_escape(title)}</h3><p class='empty'>No data.</p></section>"
        max_value = max(max(float(item.get("value", 0.0)), 0.0) for item in items)
        max_value = max(max_value, 1e-6)
        height = 40 + row_height * len(items)
        label_x = 12
        bar_x = 190
        bar_width = width - bar_x - 110
        svg_rows = []
        for idx, item in enumerate(items):
            y = 24 + idx * row_height
            value = max(float(item.get("value", 0.0)), 0.0)
            bar_len = (value / max_value) * bar_width
            label = PINNModelLogger._html_escape(item.get("label", f"item_{idx}"))
            value_text = PINNModelLogger._html_escape(PINNModelLogger._format_scalar(value))
            svg_rows.append(
                f"<text x='{label_x}' y='{y + 13}' class='svg-label'>{label}</text>"
                f"<rect x='{bar_x}' y='{y}' width='{bar_width:.1f}' height='14' rx='7' fill='#ebeff5'></rect>"
                f"<rect x='{bar_x}' y='{y}' width='{bar_len:.1f}' height='14' rx='7' fill='#1769aa'></rect>"
                f"<text x='{bar_x + bar_width + 12:.1f}' y='{y + 13}' class='svg-value'>{value_text}</text>"
            )
        return (
            f"<section><h3>{PINNModelLogger._html_escape(title)}</h3>"
            f"<svg viewBox='0 0 {width} {height}' class='chart'>{''.join(svg_rows)}</svg></section>"
        )

    @staticmethod
    def _build_line_chart(title, history, series, width=760, height=240):
        if len(history) < 2:
            return f"<section><h3>{PINNModelLogger._html_escape(title)}</h3><p class='empty'>Need at least two reports.</p></section>"
        margin_left = 52
        margin_right = 18
        margin_top = 18
        margin_bottom = 28
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom
        values = []
        for row in history:
            for entry in series:
                values.append(float(row.get(entry["key"], 0.0)))
        y_min = min(values)
        y_max = max(values)
        if abs(y_max - y_min) < 1e-6:
            y_max = y_min + 1.0

        def point_x(idx):
            if len(history) == 1:
                return margin_left
            return margin_left + (float(idx) / float(len(history) - 1)) * plot_width

        def point_y(value):
            ratio = (float(value) - y_min) / (y_max - y_min)
            return margin_top + (1.0 - ratio) * plot_height

        paths = []
        legends = []
        for legend_id, entry in enumerate(series):
            points = []
            for idx, row in enumerate(history):
                points.append(f"{point_x(idx):.1f},{point_y(row.get(entry['key'], 0.0)):.1f}")
            color = entry["color"]
            label = PINNModelLogger._html_escape(entry["label"])
            paths.append(
                f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{' '.join(points)}'></polyline>"
            )
            legends.append(
                f"<span class='legend-item'><span class='legend-swatch' style='background:{color}'></span>{label}</span>"
            )
        axis = (
            f"<line x1='{margin_left}' y1='{margin_top + plot_height}' x2='{width - margin_right}' y2='{margin_top + plot_height}' class='axis'></line>"
            f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + plot_height}' class='axis'></line>"
            f"<text x='{margin_left - 6}' y='{margin_top + 10}' class='svg-value'>{PINNModelLogger._html_escape(PINNModelLogger._format_scalar(y_max))}</text>"
            f"<text x='{margin_left - 6}' y='{margin_top + plot_height}' class='svg-value'>{PINNModelLogger._html_escape(PINNModelLogger._format_scalar(y_min))}</text>"
            f"<text x='{margin_left}' y='{height - 8}' class='svg-value'>{PINNModelLogger._html_escape(str(history[0].get('step', 0)))}</text>"
            f"<text x='{width - margin_right - 24}' y='{height - 8}' class='svg-value'>{PINNModelLogger._html_escape(str(history[-1].get('step', 0)))}</text>"
        )
        return (
            f"<section><h3>{PINNModelLogger._html_escape(title)}</h3>"
            f"<div class='legend'>{''.join(legends)}</div>"
            f"<svg viewBox='0 0 {width} {height}' class='chart'>{axis}{''.join(paths)}</svg></section>"
        )

    def _push_history_row(self, snapshot):
        summary = snapshot.get("summary", {}) if isinstance(snapshot, dict) else {}
        row = {
            "step": int(summary.get("step", self.num_steps)),
            "physics_total": self._to_float(summary.get("physics_total")),
            "sample0_rl_weight_shift": self._to_float(summary.get("sample0_rl_weight_shift")),
            "raw_correction_norm_mean": self._to_float(summary.get("raw_correction_norm_mean")),
            "correction_ratio": self._to_float(summary.get("correction_ratio")),
            "physical_mask_mean": self._to_float(summary.get("physical_mask_mean")),
            "physical_mask_blend_alpha": self._to_float(summary.get("physical_mask_blend_alpha")),
            "heuristic_physical_mask_overlap": self._to_float(summary.get("heuristic_physical_mask_overlap")),
        }
        if len(self._explainability_history) > 0 and self._explainability_history[-1].get("step") == row["step"]:
            self._explainability_history[-1] = row
            return
        self._explainability_history.append(row)
        if len(self._explainability_history) > self.explainability_max_history:
            self._explainability_history = self._explainability_history[-self.explainability_max_history:]

    @staticmethod
    def _mask_heat_color(value):
        value = min(max(float(value), 0.0), 1.0)
        hue = 220.0 - 210.0 * value
        lightness = 96.0 - 46.0 * value
        return f"hsl({hue:.1f}, 82%, {lightness:.1f}%)"

    @classmethod
    def _build_mask_heatmap(cls, title, payload):
        if not isinstance(payload, dict):
            return f"<section><h3>{cls._html_escape(title)}</h3><p class='empty'>No mask heatmap available.</p></section>"

        def render_grid(grid):
            if not isinstance(grid, list) or len(grid) == 0 or not isinstance(grid[0], list):
                return ""
            height = max(len(grid), 1)
            width = max(len(grid[0]), 1)
            cell_size = 14
            svg_width = width * cell_size
            svg_height = height * cell_size
            rects = []
            for row_idx, row in enumerate(grid):
                for col_idx, value in enumerate(row):
                    rects.append(
                        f"<rect x='{col_idx * cell_size}' y='{row_idx * cell_size}' width='{cell_size}' height='{cell_size}' "
                        f"fill='{cls._mask_heat_color(value)}'></rect>"
                    )
            return f"<svg viewBox='0 0 {svg_width} {svg_height}' class='chart'>{''.join(rects)}</svg>"

        stats = (
            f"<p class='muted'>mean={cls._html_escape(cls._format_scalar(payload.get('mean')))} | "
            f"max={cls._html_escape(cls._format_scalar(payload.get('max')))}</p>"
        )
        if isinstance(payload.get("frames"), list) and len(payload["frames"]) > 0:
            time_dim = payload.get("time_dim")
            frame_html = []
            for frame in payload["frames"]:
                frame_index = int(frame.get("frame_index", 0))
                frame_label = f"frame {frame_index}"
                if isinstance(time_dim, int) and time_dim > 1:
                    frame_label = f"frame {frame_index}/{time_dim - 1}"
                frame_stats = (
                    f"<p class='muted'>mean={cls._html_escape(cls._format_scalar(frame.get('mean')))} | "
                    f"max={cls._html_escape(cls._format_scalar(frame.get('max')))}</p>"
                )
                frame_html.append(
                    "<div class='mask-strip-item'>"
                    f"<h4>{cls._html_escape(frame_label)}</h4>"
                    f"{frame_stats}"
                    f"{render_grid(frame.get('grid'))}"
                    "</div>"
                )
            return (
                f"<section><h3>{cls._html_escape(title)}</h3>{stats}"
                f"<div class='mask-strip'>{''.join(frame_html)}</div></section>"
            )

        if not isinstance(payload.get("grid"), list) or len(payload["grid"]) == 0:
            return f"<section><h3>{cls._html_escape(title)}</h3><p class='empty'>No mask heatmap available.</p></section>"
        return (
            f"<section><h3>{cls._html_escape(title)}</h3>{stats}"
            f"{render_grid(payload['grid'])}</section>"
        )

    def _build_explainability_html(self, snapshot):
        summary = snapshot.get("summary", {})
        metadata_rows = []
        for source_name in ("data", "metadata"):
            source = snapshot.get(source_name, {})
            if not isinstance(source, dict):
                continue
            for key, value in source.items():
                metadata_rows.append(
                    {
                        "Key": f"{source_name}.{key}",
                        "Value": json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value,
                    }
                )

        summary_cards = [
            ("step", summary.get("step")),
            ("phenomenon", summary.get("phenomenon")),
            ("physics_total", self._format_scalar(summary.get("physics_total"))),
            ("phys_mask_mean", self._format_scalar(summary.get("physical_mask_mean"))),
            ("mask_alpha", self._format_scalar(summary.get("physical_mask_blend_alpha"))),
            ("raw_corr_norm", self._format_scalar(summary.get("raw_correction_norm_mean"))),
            ("corr_ratio", self._format_scalar(summary.get("correction_ratio"))),
        ]
        card_html = "".join(
            f"<div class='card'><div class='card-key'>{self._html_escape(key)}</div><div class='card-value'>{self._html_escape(value)}</div></div>"
            for key, value in summary_cards
        )
        expert_table = []
        for row in snapshot.get("experts", []):
            expert_table.append(
                {
                    "Expert": row.get("label"),
                    "PolicyWeight": self._format_scalar(row.get("mean_policy_weight")),
                    "RouterWeight": self._format_scalar(row.get("mean_router_weight")),
                    "RouteLogit": self._format_scalar(row.get("mean_route_logit")),
                    "UsageEMA": self._format_scalar(row.get("usage_ema")),
                }
            )
        sample0_table = []
        for row in snapshot.get("sample0_experts", []):
            sample0_table.append(
                {
                    "Slot": row.get("slot"),
                    "Expert": row.get("label"),
                    "PolicyWeight": self._format_scalar(row.get("policy_weight")),
                    "RouterWeight": self._format_scalar(row.get("router_weight")),
                    "RouteLogit": self._format_scalar(row.get("route_logit")),
                }
            )
        tensor_stats_rows = []
        tensor_stats = snapshot.get("tensor_stats", {})
        if isinstance(tensor_stats, dict):
            for key, stats in tensor_stats.items():
                if not isinstance(stats, dict):
                    continue
                tensor_stats_rows.append(
                    {
                        "Tensor": key,
                        "Mean": self._format_scalar(stats.get("mean")),
                        "Std": self._format_scalar(stats.get("std")),
                        "Min": self._format_scalar(stats.get("min")),
                        "Max": self._format_scalar(stats.get("max")),
                    }
                )

        expert_bar_items = [
            {"label": row.get("label", "unknown"), "value": row.get("mean_policy_weight", 0.0)}
            for row in snapshot.get("experts", [])
        ]
        history_html = self._build_line_chart(
            title="Physics / Correction Trend",
            history=self._explainability_history,
            series=[
                {"key": "physics_total", "label": "physics_total", "color": "#1769aa"},
                {"key": "raw_correction_norm_mean", "label": "raw_corr_norm", "color": "#d17b0f"},
                {"key": "correction_ratio", "label": "corr_ratio", "color": "#218739"},
            ],
        )
        shift_history_html = self._build_line_chart(
            title="Expert Shift Trend",
            history=self._explainability_history,
            series=[
                {"key": "sample0_rl_weight_shift", "label": "sample0_rl_shift", "color": "#9c2f2f"},
                {"key": "raw_correction_norm_mean", "label": "raw_corr_norm", "color": "#6a4fb3"},
            ],
        )
        mask_history_html = self._build_line_chart(
            title="Mask Transition Trend",
            history=self._explainability_history,
            series=[
                {"key": "physical_mask_mean", "label": "physical_mask_mean", "color": "#1f7a8c"},
                {"key": "physical_mask_blend_alpha", "label": "blend_alpha", "color": "#e67e22"},
                {"key": "heuristic_physical_mask_overlap", "label": "mask_overlap", "color": "#2e8b57"},
            ],
        )
        mask_heatmaps = snapshot.get("mask_heatmaps", {})
        mask_heatmap_html = "".join(
            [
                self._build_mask_heatmap("Bootstrap Mask", mask_heatmaps.get("bootstrap_motion_mask")),
                self._build_mask_heatmap("Physical Mask", mask_heatmaps.get("physical_mask")),
                self._build_mask_heatmap("Effective Mask", mask_heatmaps.get("effective_motion_mask")),
            ]
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PINN Explainability Report</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: linear-gradient(180deg, #f5f7fb 0%, #eef2f7 100%);
      color: #15202b;
    }}
    h1, h2, h3 {{ margin: 0 0 12px 0; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 20px; margin-top: 28px; }}
    h3 {{ font-size: 16px; margin-top: 0; }}
    p {{ margin: 0; line-height: 1.5; }}
    .muted {{ color: #5b6875; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 18px 0 24px 0;
    }}
    .card, section {{
      background: #ffffff;
      border: 1px solid #d9e1ea;
      border-radius: 14px;
      padding: 14px 16px;
      box-shadow: 0 8px 20px rgba(23, 36, 56, 0.05);
    }}
    .card-key {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #62707c;
      margin-bottom: 8px;
    }}
    .card-value {{
      font-size: 20px;
      font-weight: 600;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid #ecf0f3;
      padding: 8px 6px;
      vertical-align: top;
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #62707c;
    }}
    .chart {{
      width: 100%;
      height: auto;
      overflow: visible;
    }}
    .svg-label {{
      font-size: 12px;
      fill: #364152;
    }}
    .svg-value {{
      font-size: 11px;
      fill: #62707c;
    }}
    .axis {{
      stroke: #c8d3df;
      stroke-width: 1;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: #556371;
    }}
    .legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    .mask-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .mask-strip-item {{
      border: 1px solid #ecf0f3;
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
    }}
    .mask-strip-item h4 {{
      margin: 0 0 6px 0;
      font-size: 13px;
      color: #364152;
    }}
    .empty {{
      color: #6d7986;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <h1>PINN Explainability Report</h1>
  <p class="muted">Step {self._html_escape(summary.get("step", "-"))} | phenomenon={self._html_escape(summary.get("phenomenon", "-"))}</p>
  <div class="grid">{card_html}</div>
  <div class="two-col">
    {self._build_bar_chart("Top Expert Policy Weights", expert_bar_items)}
    {history_html}
  </div>
  <div class="two-col">
    {shift_history_html}
    {mask_history_html}
  </div>
  <div class="two-col">
    {mask_heatmap_html}
  </div>
  <div class="two-col">
    <section>
      <h3>Sample Metadata</h3>
      {self._build_table(["Key", "Value"], metadata_rows)}
    </section>
  </div>
  <div class="two-col">
    <section>
      <h3>Batch Expert Summary</h3>
      {self._build_table(["Expert", "PolicyWeight", "RouterWeight", "RouteLogit", "UsageEMA"], expert_table)}
    </section>
    <section>
      <h3>Sample0 Routed Experts</h3>
      {self._build_table(["Slot", "Expert", "PolicyWeight", "RouterWeight", "RouteLogit"], sample0_table)}
    </section>
  </div>
  <div class="two-col">
    <section>
      <h3>Tensor Statistics</h3>
      {self._build_table(["Tensor", "Mean", "Std", "Min", "Max"], tensor_stats_rows)}
    </section>
  </div>
</body>
</html>"""

    def _export_explainability_report(self, accelerator, model, force=False):
        if not self.enable_explainability_reports:
            return
        if not accelerator.is_main_process:
            return
        if (not force) and (self.num_steps % self.explainability_report_interval != 0):
            return
        snapshot = getattr(accelerator.unwrap_model(model), "_latest_explainability_snapshot", None)
        if not isinstance(snapshot, dict):
            return

        self._push_history_row(snapshot)
        os.makedirs(self.explainability_output_dir, exist_ok=True)
        step = int(snapshot.get("summary", {}).get("step", self.num_steps))
        json_path = os.path.join(self.explainability_output_dir, f"step-{step:08d}.json")
        html_path = os.path.join(self.explainability_output_dir, f"step-{step:08d}.html")
        latest_json_path = os.path.join(self.explainability_output_dir, "latest.json")
        latest_html_path = os.path.join(self.explainability_output_dir, "latest.html")
        index_path = os.path.join(self.explainability_output_dir, "index.jsonl")

        snapshot_to_save = dict(snapshot)
        snapshot_to_save["history"] = list(self._explainability_history)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(snapshot_to_save, f, ensure_ascii=False, indent=2)
        with open(latest_json_path, "w", encoding="utf-8") as f:
            json.dump(snapshot_to_save, f, ensure_ascii=False, indent=2)
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot.get("summary", {}), ensure_ascii=False) + "\n")

        html_text = self._build_explainability_html(snapshot_to_save)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_text)
        with open(latest_html_path, "w", encoding="utf-8") as f:
            f.write(html_text)

    def save_training_state(self, optimizer, scheduler, step, epoch):
        """保存优化器和调度器状态（在保存checkpoint前调用）"""
        self.optimizer_state = optimizer.state_dict() if optimizer is not None else None
        self.scheduler_state = scheduler.state_dict() if scheduler is not None else None
        self.current_step = step
        self.current_epoch = epoch

    def save_model(self, accelerator, model, file_name, save_training_state=False):
        """只保存 PINN 插件参数，可选保存训练状态"""
        accelerator.wait_for_everyone()
        # Use accelerator.get_state_dict to properly gather from all processes
        # This ensures DDP-synchronized weights are saved correctly
        state_dict = accelerator.get_state_dict(model)

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)

            physics_adapter_state_dict = {}
            pde_residuals_state_dict = {}
            if hasattr(unwrapped_model, 'physics_adapter'):
                # Use the gathered state_dict instead of directly accessing the model
                prefix = 'physics_adapter.'
                physics_adapter_state_dict = {
                    k[len(prefix):]: v.detach().cpu().float()  # Convert to float32 for stability
                    for k, v in state_dict.items()
                    if k.startswith(prefix)
                }
            encoder_stage_state_dict = WanPINNTrainingModule._build_encoder_stage_state_dict_from_adapter_state(
                physics_adapter_state_dict
            )
            if hasattr(unwrapped_model, 'pde_residuals'):
                prefix = 'pde_residuals.'
                pde_residuals_state_dict = {
                    k[len(prefix):]: v.detach().cpu().float()  # Convert to float32 for stability
                    for k, v in state_dict.items()
                    if k.startswith(prefix)
                }

            # Check for NaN values in the state dict
            has_nan = any(
                torch.isnan(v).any().item()
                for v in list(physics_adapter_state_dict.values()) + list(pde_residuals_state_dict.values())
                if isinstance(v, torch.Tensor)
            )
            if has_nan:
                print(f"  WARNING: NaN detected in checkpoint! Skipping save.")
                return

            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)

            # 保存为 .pt 格式（包含额外元信息）
            pt_path = path.replace(".safetensors", ".pt")
            checkpoint = {
                'physics_adapter_state_dict': physics_adapter_state_dict,
                'pde_residuals_state_dict': pde_residuals_state_dict,
                'encoder_stage_state_dict': encoder_stage_state_dict,
                'config': {
                    'checkpoint_format_version': 19,
                    'adapter_architecture': 'explicit_attribute_bank_v2',
                    'core_ablation_mode': (
                        unwrapped_model.core_ablation_mode
                        if hasattr(unwrapped_model, 'core_ablation_mode') else "full"
                    ),
                    'field_contract_version': FIELD_CONTRACT_VERSION,
                    'expert_field_recipe_version': EXPERT_FIELD_RECIPE_VERSION,
                    'observable_proxy_recipe_version': OBSERVABLE_PROXY_RECIPE_VERSION,
                    'training_stage': (
                        unwrapped_model.training_stage
                        if hasattr(unwrapped_model, 'training_stage') else "full_pinn"
                    ),
                    'encoder_freeze_steps': (
                        unwrapped_model.encoder_freeze_steps
                        if hasattr(unwrapped_model, 'encoder_freeze_steps') else 1000
                    ),
                    'encoder_lr_scale': (
                        unwrapped_model.encoder_lr_scale
                        if hasattr(unwrapped_model, 'encoder_lr_scale') else 0.3
                    ),
                    'ablation_preset': (
                        unwrapped_model.ablation_preset
                        if hasattr(unwrapped_model, 'ablation_preset') else "legacy_direct_bank"
                    ),
                    'observable_target_mode': (
                        unwrapped_model.observable_target_mode
                        if hasattr(unwrapped_model, 'observable_target_mode') else "flow_plus_deformation"
                    ),
                    'secondary_field_strategy': (
                        unwrapped_model.secondary_field_strategy
                        if hasattr(unwrapped_model, 'secondary_field_strategy') else "legacy_direct_bank"
                    ),
                    'active_field_set': (
                        unwrapped_model.active_field_set
                        if hasattr(unwrapped_model, 'active_field_set') else "legacy"
                    ),
                    'field_enable_schedule': (
                        unwrapped_model.field_enable_schedule
                        if hasattr(unwrapped_model, 'field_enable_schedule') else "legacy"
                    ),
                    'field_recovery_phase': (
                        unwrapped_model.field_recovery_phase
                        if hasattr(unwrapped_model, 'field_recovery_phase') else "core"
                    ),
                    'field_recovery_step_schedule': (
                        unwrapped_model.field_recovery_step_schedule
                        if hasattr(unwrapped_model, 'field_recovery_step_schedule') else ""
                    ),
                    'field_recovery_loss_ramp_steps': (
                        int(unwrapped_model.field_recovery_loss_ramp_steps)
                        if hasattr(unwrapped_model, 'field_recovery_loss_ramp_steps') else 0
                    ),
                    'run_full_pinn_after_recovery': (
                        bool(unwrapped_model.run_full_pinn_after_recovery)
                        if hasattr(unwrapped_model, 'run_full_pinn_after_recovery') else False
                    ),
                    'freeze_u_encoder_during_recovery': (
                        bool(unwrapped_model.freeze_u_encoder_during_recovery)
                        if hasattr(unwrapped_model, 'freeze_u_encoder_during_recovery') else None
                    ),
                    'flow_backbone_ckpt': (
                        unwrapped_model.flow_backbone_ckpt
                        if hasattr(unwrapped_model, 'flow_backbone_ckpt') else None
                    ),
                    'physics_weight': unwrapped_model.physics_weight if hasattr(unwrapped_model, 'physics_weight') else None,
                    'physics_weight_target': (
                        unwrapped_model.physics_weight_target
                        if hasattr(unwrapped_model, 'physics_weight_target') else None
                    ),
                    'output_physics_weight': (
                        unwrapped_model.output_physics_weight
                        if hasattr(unwrapped_model, 'output_physics_weight') else None
                    ),
                    'adapter_hidden_dim': (
                        unwrapped_model.adapter_hidden_dim
                        if hasattr(unwrapped_model, 'adapter_hidden_dim') else None
                    ),
                    'physics_attr_dim': (
                        unwrapped_model.physics_attr_dim
                        if hasattr(unwrapped_model, 'physics_attr_dim') else PHYSICS_ATTR_DIM
                    ),
                    'num_phenomena': len(PHENOMENON_LABELS),
                    'n_numeric_dim': (
                        unwrapped_model.physics_adapter.n_numeric_dim
                        if hasattr(unwrapped_model, 'physics_adapter') else 12
                    ),
                    'q_input_dim': (
                        unwrapped_model.physics_adapter.q_input_dim
                        if hasattr(unwrapped_model, 'physics_adapter') else 64
                    ),
                    'n_text_vocab_size': (
                        unwrapped_model.physics_adapter.n_text_vocab_size
                        if hasattr(unwrapped_model, 'physics_adapter') else 2048
                    ),
                    'conditioned_physics_warmup_steps': (
                        unwrapped_model.conditioned_physics_warmup_steps
                        if hasattr(unwrapped_model, 'conditioned_physics_warmup_steps') else None
                    ),
                    'state_align_warmup_steps': (
                        unwrapped_model.state_align_warmup_steps
                        if hasattr(unwrapped_model, 'state_align_warmup_steps') else None
                    ),
                    'expert_balance_weight': (
                        unwrapped_model.expert_balance_weight
                        if hasattr(unwrapped_model, 'expert_balance_weight') else None
                    ),
                    'condition_consistency_weight': (
                        unwrapped_model.condition_consistency_weight
                        if hasattr(unwrapped_model, 'condition_consistency_weight') else None
                    ),
                    'state_align_x_weight': (
                        unwrapped_model.state_align_x_weight
                        if hasattr(unwrapped_model, 'state_align_x_weight') else None
                    ),
                    'state_align_v_weight': (
                        unwrapped_model.state_align_v_weight
                        if hasattr(unwrapped_model, 'state_align_v_weight') else None
                    ),
                    'state_align_v_weight_target': (
                        unwrapped_model.state_align_v_weight_target
                        if hasattr(unwrapped_model, 'state_align_v_weight_target') else None
                    ),
                    'decoded_branch_consistency_weight': (
                        unwrapped_model.decoded_branch_consistency_weight
                        if hasattr(unwrapped_model, 'decoded_branch_consistency_weight') else None
                    ),
                    'expert_pde_sigma_threshold': (
                        unwrapped_model.expert_pde_sigma_threshold
                        if hasattr(unwrapped_model, 'expert_pde_sigma_threshold') else None
                    ),
                    'expert_pde_sigma_threshold_target': (
                        unwrapped_model.expert_pde_sigma_threshold_target
                        if hasattr(unwrapped_model, 'expert_pde_sigma_threshold_target') else None
                    ),
                    'moe_top_k': (
                        int(unwrapped_model.physics_adapter.moe_top_k)
                        if hasattr(unwrapped_model, 'physics_adapter') else None
                    ),
                    'physics_state_mode': (
                        unwrapped_model.physics_state_mode
                        if hasattr(unwrapped_model, 'physics_state_mode') else "x0_hat"
                    ),
                    'motion_mask_floor': (
                        unwrapped_model.motion_mask_floor
                        if hasattr(unwrapped_model, 'motion_mask_floor') else 0.08
                    ),
                    'motion_mask_quantile': (
                        unwrapped_model.motion_mask_quantile
                        if hasattr(unwrapped_model, 'motion_mask_quantile') else 0.9
                    ),
                    'motion_mask_warmup_steps': (
                        unwrapped_model.motion_mask_warmup_steps
                        if hasattr(unwrapped_model, 'motion_mask_warmup_steps') else 300
                    ),
                    'physical_mask_transition_steps': (
                        unwrapped_model.physical_mask_transition_steps
                        if hasattr(unwrapped_model, 'physical_mask_transition_steps') else 1000
                    ),
                    'physical_mask_recipe_version': PHYSICAL_MASK_RECIPE_VERSION,
                    'use_sigma_gate': (
                        unwrapped_model.use_sigma_gate
                        if hasattr(unwrapped_model, 'use_sigma_gate') else True
                    ),
                    'sigma_gate_curve': (
                        unwrapped_model.sigma_gate_curve
                        if hasattr(unwrapped_model, 'sigma_gate_curve') else "quadratic"
                    ),
                    'use_sigma_conditioning': (
                        unwrapped_model.use_sigma_conditioning
                        if hasattr(unwrapped_model, 'use_sigma_conditioning') else True
                    ),
                    'sigma_conditioning_dim': (
                        unwrapped_model.sigma_conditioning_dim
                        if hasattr(unwrapped_model, 'sigma_conditioning_dim') else None
                    ),
                    'sigma_gate_floor': (
                        unwrapped_model.sigma_gate_floor
                        if hasattr(unwrapped_model, 'sigma_gate_floor') else 0.05
                    ),
                    'use_dual_noise_experts': (
                        unwrapped_model.use_dual_noise_experts
                        if hasattr(unwrapped_model, 'use_dual_noise_experts') else True
                    ),
                    'dual_noise_expert_boundary': (
                        unwrapped_model.dual_noise_expert_boundary
                        if hasattr(unwrapped_model, 'dual_noise_expert_boundary') else 0.417
                    ),
                    'use_adaptive_condition_injection': (
                        unwrapped_model.use_adaptive_condition_injection
                        if hasattr(unwrapped_model, 'use_adaptive_condition_injection') else True
                    ),
                    'adaptive_conditioning_dim': (
                        unwrapped_model.adaptive_conditioning_dim
                        if hasattr(unwrapped_model, 'adaptive_conditioning_dim') else None
                    ),
                    'adaptive_conditioning_strength': (
                        unwrapped_model.adaptive_conditioning_strength
                        if hasattr(unwrapped_model, 'adaptive_conditioning_strength') else 0.5
                    ),
                    'adaptive_conditioning_gate_floor': (
                        unwrapped_model.adaptive_conditioning_gate_floor
                        if hasattr(unwrapped_model, 'adaptive_conditioning_gate_floor') else 0.05
                    ),
                    'enable_rl_expert_optimization': (
                        unwrapped_model.enable_rl_expert_optimization
                        if hasattr(unwrapped_model, 'enable_rl_expert_optimization') else True
                    ),
                    'rl_policy_weight': (
                        unwrapped_model.rl_policy_weight
                        if hasattr(unwrapped_model, 'rl_policy_weight') else 1e-2
                    ),
                    'rl_entropy_weight': (
                        unwrapped_model.rl_entropy_weight
                        if hasattr(unwrapped_model, 'rl_entropy_weight') else 1e-3
                    ),
                    'rl_reward_decay': (
                        unwrapped_model.rl_reward_decay
                        if hasattr(unwrapped_model, 'rl_reward_decay') else 0.95
                    ),
                    'rl_reward_quality_weight': (
                        unwrapped_model.rl_reward_quality_weight
                        if hasattr(unwrapped_model, 'rl_reward_quality_weight') else 0.5
                    ),
                    'rl_reward_stability_weight': (
                        unwrapped_model.rl_reward_stability_weight
                        if hasattr(unwrapped_model, 'rl_reward_stability_weight') else 0.1
                    ),
                    'rl_warmup_steps': (
                        unwrapped_model.rl_warmup_steps
                        if hasattr(unwrapped_model, 'rl_warmup_steps') else 500
                    ),
                    'rl_hidden_dim': (
                        unwrapped_model.rl_hidden_dim
                        if hasattr(unwrapped_model, 'rl_hidden_dim') else None
                    ),
                    'enable_explainability_reports': (
                        unwrapped_model.enable_explainability_reports
                        if hasattr(unwrapped_model, 'enable_explainability_reports') else True
                    ),
                    'explainability_top_experts': (
                        unwrapped_model.explainability_top_experts
                        if hasattr(unwrapped_model, 'explainability_top_experts') else 6
                    ),
                    'ablation_flags': {
                        'core_ablation_mode': (
                            unwrapped_model.core_ablation_mode
                            if hasattr(unwrapped_model, 'core_ablation_mode') else "full"
                        ),
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
            }

            # 可选保存训练状态
            if save_training_state:
                checkpoint['training_state'] = {
                    'optimizer_state_dict': self.optimizer_state,
                    'scheduler_state_dict': self.scheduler_state,
                    'current_step': self.current_step,
                    'current_epoch': self.current_epoch,
                }

            torch.save(checkpoint, pt_path)
            print(f"PINN plugin saved to: {pt_path}")
            print(f"  Total keys: {len(physics_adapter_state_dict) + len(pde_residuals_state_dict)}")
            if save_training_state:
                print(f"  Training state: step={self.current_step}, epoch={self.current_epoch}")

    def on_step_end(self, accelerator, model, save_steps=None):
        """每个 step 结束，支持 save_steps 保存中间 checkpoint（包含训练状态）"""
        self.num_steps += 1
        self._export_explainability_report(accelerator, model, force=False)
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.pt", save_training_state=True)

    def on_epoch_end(self, accelerator, model, epoch_id):
        """每个 epoch 结束保存（包含训练状态）"""
        self._export_explainability_report(accelerator, model, force=True)
        self.save_model(accelerator, model, f"pinn_plugin_epoch-{epoch_id}.pt", save_training_state=True)

    def on_training_end(self, accelerator, model, save_steps=None):
        """训练结束保存最终模型"""
        self._export_explainability_report(accelerator, model, force=True)
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
        "--adapter_hidden_dim", type=int, default=64,
        help="Hidden dimension for PhysicsAdapter (default: 64)"
    )
    parser.add_argument(
        "--physics_attr_dim", type=int, default=PHYSICS_ATTR_DIM,
        help="Explicit shared physics attribute-bank dimension (default: 32)"
    )
    parser.add_argument(
        "--expert_pde_sigma_threshold", type=float, default=0.40,
        help="Enable expert PDE only when sigma <= threshold (default: 0.40)"
    )
    parser.add_argument(
        "--expert_pde_sigma_threshold_target", type=float, default=1.00,
        help="Target sigma threshold reached after physics warmup; 1.00 enables full-noise PDE coverage."
    )
    parser.add_argument(
        "--pinn_checkpoint", type=str, default=None,
        help="Path to existing PINN plugin checkpoint to resume training"
    )
    parser.add_argument(
        "--training_stage", type=str, default="full_pinn",
        choices=["observable_pretrain", "encoder_completion", "full_pinn"],
        help="Training stage: observable pretrain, encoder completion, or full PINN."
    )
    parser.add_argument(
        "--stage1_pretrained_encoder", type=str, default=None,
        help="Path to a stage1 observable_pretrain checkpoint used to initialize the full PINN encoder."
    )
    parser.add_argument(
        "--flow_backbone_ckpt", type=str, default=None,
        help="Optional checkpoint for the frozen dense-flow teacher used by observable proxies."
    )
    parser.add_argument(
        "--debug_overfit_num_samples", type=int, default=None,
        help="Debug only: keep only the first N metadata samples to test whether stage1 can overfit a tiny subset."
    )
    parser.add_argument(
        "--debug_overfit_dataset_repeat", type=int, default=None,
        help="Debug only: override dataset_repeat after subsetting so the tiny subset cycles many times per epoch."
    )
    parser.add_argument(
        "--debug_fixed_timestep_fraction", type=float, default=None,
        help="Debug only: fix timestep sampling to a single scheduler fraction in [0,1]."
    )
    parser.add_argument(
        "--encoder_freeze_steps", type=int, default=1000,
        help="Freeze shared encoder and attribute head for the first N full_pinn steps."
    )
    parser.add_argument(
        "--encoder_lr_scale", type=float, default=0.3,
        help="Gradient scale applied to shared encoder and attribute head after unfreezing in full_pinn."
    )
    parser.add_argument(
        "--ablation_preset", type=str, default="legacy_direct_bank",
        choices=sorted(ABLATION_PRESET_DEFAULTS.keys()),
        help="Preset bundle for only-u and legacy ablations."
    )
    parser.add_argument(
        "--observable_target_mode", type=str, default="auto",
        choices=["auto", "flow_plus_deformation", "flow_only"],
        help="Observable pretrain target mode."
    )
    parser.add_argument(
        "--secondary_field_strategy", type=str, default="auto",
        choices=[
            "auto",
            "legacy_direct_bank",
            "direct_bank",
            "u_first_constructor",
            "u_first_constructor_detach",
        ],
        help="How p/rho are constructed during only-u runs."
    )
    parser.add_argument(
        "--active_field_set", type=str, default="auto",
        help="Comma-separated active field set override; use auto to follow the preset."
    )
    parser.add_argument(
        "--field_enable_schedule", type=str, default="auto",
        help="Field recovery schedule override; use auto to follow the preset."
    )
    parser.add_argument(
        "--field_recovery_phase", type=str, default="core",
        choices=list(ONLY_U_RECOVERY_PHASES),
        help="encoder_completion phase controlling which fields are recovered."
    )
    parser.add_argument(
        "--field_recovery_step_schedule", type=str, default="",
        help="Optional encoder_completion recovery schedule, e.g. core:0,alpha+T:800,j+D:1500,psi:2100."
    )
    parser.add_argument(
        "--field_recovery_loss_ramp_steps", type=int, default=100,
        help="Ramp steps used when a new recovery block is activated under field_recovery_step_schedule."
    )
    parser.add_argument(
        "--run_full_pinn_after_recovery", action="store_true",
        help="Runner hint: mark that full_pinn should follow encoder recovery."
    )
    parser.add_argument(
        "--freeze_u_encoder_during_recovery",
        dest="freeze_u_encoder_during_recovery",
        action="store_true",
        default=True,
        help="Freeze physics_encoder_shared/shared_attribute_head/u_head during encoder_completion."
    )
    parser.add_argument(
        "--no_freeze_u_encoder_during_recovery",
        dest="freeze_u_encoder_during_recovery",
        action="store_false",
        help="Allow encoder_completion to continue updating the u encoder branch."
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
        "--core_ablation_mode", type=str, default="full",
        choices=list(CORE_ABLATION_MODES),
        help="Coarse PhysFM mechanism ablation mode used by isolated core-ablation runners."
    )
    parser.add_argument(
        "--allow_ablation_checkpoint_mismatch", action="store_true",
        help="Allow selected runtime-config mismatches when warm-starting an ablation from a full checkpoint."
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
    parser.add_argument(
        "--physical_mask_transition_steps", type=int, default=1000,
        help="Steps used to linearly replace heuristic mask with active fused physical mask."
    )
    parser.add_argument(
        "--physics_state_mode", type=str, default="x0_hat", choices=["x0_hat", "z_t"],
        help="State fed into the physics branch: x0_hat or z_t (default: x0_hat)."
    )
    parser.add_argument(
        "--use_sigma_gate", action="store_true",
        help="Enable sigma-based correction gating in the physics adapter."
    )
    parser.add_argument(
        "--no_use_sigma_gate", dest="use_sigma_gate", action="store_false",
        help="Disable sigma-based correction gating."
    )
    parser.set_defaults(use_sigma_gate=True)
    parser.add_argument(
        "--sigma_gate_curve", type=str, default="quadratic", choices=["quadratic", "linear", "hard"],
        help="Curve used for sigma correction gate (default: quadratic)."
    )
    parser.add_argument(
        "--use_sigma_conditioning", action="store_true",
        help="Enable explicit sigma conditioning inside the physics adapter."
    )
    parser.add_argument(
        "--no_use_sigma_conditioning", dest="use_sigma_conditioning", action="store_false",
        help="Disable explicit sigma conditioning inside the physics adapter."
    )
    parser.set_defaults(use_sigma_conditioning=True)
    parser.add_argument(
        "--sigma_conditioning_dim", type=int, default=None,
        help="Hidden dimension of the sigma conditioning MLP (default: adapter_hidden_dim)."
    )
    parser.add_argument(
        "--sigma_gate_floor", type=float, default=0.05,
        help="Minimum floor for sigma gate strength (default: 0.05)."
    )
    parser.add_argument(
        "--use_dual_noise_experts", action="store_true",
        help="Route between Wan2.2 high-noise and low-noise DiT experts during PINN training."
    )
    parser.add_argument(
        "--no_use_dual_noise_experts", dest="use_dual_noise_experts", action="store_false",
        help="Disable dual-noise-expert routing and always use the primary DiT."
    )
    parser.set_defaults(use_dual_noise_experts=None)
    parser.add_argument(
        "--dual_noise_expert_boundary", type=float, default=0.417,
        help="Index-space timestep boundary for switching from high-noise expert (dit) to low-noise expert (dit2)."
    )
    parser.add_argument(
        "--use_adaptive_condition_injection", action="store_true",
        help="Enable adaptive state-aware physical condition injection inside experts."
    )
    parser.add_argument(
        "--no_use_adaptive_condition_injection", dest="use_adaptive_condition_injection", action="store_false",
        help="Disable adaptive physical condition injection."
    )
    parser.add_argument(
        "--disable_adaptive_condition_injection", dest="use_adaptive_condition_injection", action="store_false",
        help="Alias for --no_use_adaptive_condition_injection."
    )
    parser.set_defaults(use_adaptive_condition_injection=True)
    parser.add_argument(
        "--adaptive_conditioning_dim", type=int, default=None,
        help="Hidden dimension of adaptive condition injection heads (default: adapter_hidden_dim)."
    )
    parser.add_argument(
        "--adaptive_conditioning_strength", type=float, default=0.5,
        help="Maximum modulation amplitude of adaptive physical condition injection."
    )
    parser.add_argument(
        "--adaptive_conditioning_gate_floor", type=float, default=0.05,
        help="Minimum gate floor for adaptive condition injection."
    )
    parser.add_argument(
        "--enable_rl_expert_optimization", action="store_true",
        help="Enable bandit-style RL reweighting over routed experts."
    )
    parser.add_argument(
        "--disable_rl_expert_optimization", dest="enable_rl_expert_optimization", action="store_false",
        help="Disable RL-based expert contribution optimization."
    )
    parser.set_defaults(enable_rl_expert_optimization=True)
    parser.add_argument(
        "--rl_policy_weight", type=float, default=1e-2,
        help="Weight of the RL policy-gradient loss for expert contribution optimization."
    )
    parser.add_argument(
        "--rl_entropy_weight", type=float, default=1e-3,
        help="Entropy bonus weight for RL expert policy exploration."
    )
    parser.add_argument(
        "--rl_reward_decay", type=float, default=0.95,
        help="EMA decay for per-expert RL reward baseline."
    )
    parser.add_argument(
        "--rl_reward_quality_weight", type=float, default=0.5,
        help="Weight of FM quality term inside expert RL reward."
    )
    parser.add_argument(
        "--rl_reward_stability_weight", type=float, default=0.1,
        help="Weight of correction stability term inside expert RL reward."
    )
    parser.add_argument(
        "--rl_warmup_steps", type=int, default=500,
        help="Warmup steps for scaling in RL expert policy loss."
    )
    parser.add_argument(
        "--rl_hidden_dim", type=int, default=None,
        help="Hidden dimension of RL expert policy network (default: adapter_hidden_dim)."
    )
    parser.add_argument(
        "--state_align_warmup_steps", type=int, default=1000,
        help="Warmup steps before explicit state-alignment losses reach full weight."
    )
    parser.add_argument(
        "--state_align_x_weight", type=float, default=0.0,
        help="Weight of explicit shared-state alignment loss for x_phys."
    )
    parser.add_argument(
        "--state_align_v_weight", type=float, default=0.05,
        help="Weight of explicit shared-state alignment loss for v_phys."
    )
    parser.add_argument(
        "--curriculum_transition_start_step", type=int, default=1000,
        help="Step to start shifting from FM bridging to stronger physics shaping."
    )
    parser.add_argument(
        "--curriculum_transition_steps", type=int, default=1000,
        help="Number of steps used to linearly interpolate curriculum-controlled weights."
    )
    parser.add_argument(
        "--physics_weight_target", type=float, default=None,
        help="Optional phase-B target for physics_weight. Defaults to physics_weight when omitted."
    )
    parser.add_argument(
        "--output_physics_weight", type=float, default=1.0,
        help="Weight for output-space physics loss applied on v_corrected reconstruction."
    )
    parser.add_argument(
        "--state_align_v_weight_target", type=float, default=None,
        help="Optional phase-B target for state_align_v_weight. Defaults to state_align_v_weight when omitted."
    )
    parser.add_argument(
        "--decoded_branch_consistency_weight", type=float, default=1e-2,
        help="Weight of decoded branch-vs-fused correction consistency loss."
    )
    parser.add_argument(
        "--enable_explainability_reports", action="store_true",
        help="Enable physics explainability report export during training."
    )
    parser.add_argument(
        "--disable_explainability_reports", dest="enable_explainability_reports", action="store_false",
        help="Disable physics explainability report export."
    )
    parser.set_defaults(enable_explainability_reports=True)
    parser.add_argument(
        "--explainability_top_experts", type=int, default=6,
        help="Number of experts shown in each explainability report."
    )
    parser.add_argument(
        "--explainability_report_interval", type=int, default=100,
        help="Export explainability HTML/JSON every N optimization steps."
    )
    parser.add_argument(
        "--explainability_max_history", type=int, default=50,
        help="How many recent report points to keep in each explainability history chart."
    )
    parser.add_argument(
        "--explainability_output_dir", type=str, default=None,
        help="Optional custom output directory for explainability reports."
    )
    
    return parser


if __name__ == "__main__":
    parser = pinn_parser()
    args = parser.parse_args()
    
    # 数据集
    dataset = VideoDataset(args=args)
    if args.debug_overfit_num_samples is not None:
        unique_samples = len(dataset.data)
        keep_samples = max(int(args.debug_overfit_num_samples), 1)
        dataset.data = dataset.data[: min(keep_samples, unique_samples)]
        if args.debug_overfit_dataset_repeat is not None:
            dataset.repeat = max(int(args.debug_overfit_dataset_repeat), 1)
        print("Debug overfit mode enabled:")
        print(f"  unique_samples_before={unique_samples}")
        print(f"  unique_samples_after={len(dataset.data)}")
        print(f"  dataset_repeat={dataset.repeat}")
        print(f"  effective_len={len(dataset)}")
    
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
        use_dual_noise_experts=args.use_dual_noise_experts,
        dual_noise_expert_boundary=args.dual_noise_expert_boundary,
        # PINN 参数
        physics_weight=args.physics_weight,
        physics_warmup_steps=args.physics_warmup_steps,
        conditioned_physics_warmup_steps=args.conditioned_physics_warmup_steps,
        adapter_hidden_dim=args.adapter_hidden_dim,
        physics_attr_dim=args.physics_attr_dim,
        expert_pde_sigma_threshold=args.expert_pde_sigma_threshold,
        expert_pde_sigma_threshold_target=args.expert_pde_sigma_threshold_target,
        training_stage=args.training_stage,
        pinn_checkpoint=args.pinn_checkpoint,
        stage1_pretrained_encoder=args.stage1_pretrained_encoder,
        flow_backbone_ckpt=args.flow_backbone_ckpt,
        encoder_freeze_steps=args.encoder_freeze_steps,
        encoder_lr_scale=args.encoder_lr_scale,
        ablation_preset=args.ablation_preset,
        observable_target_mode=args.observable_target_mode,
        secondary_field_strategy=args.secondary_field_strategy,
        active_field_set=args.active_field_set,
        field_enable_schedule=args.field_enable_schedule,
        field_recovery_phase=args.field_recovery_phase,
        field_recovery_step_schedule=args.field_recovery_step_schedule,
        field_recovery_loss_ramp_steps=args.field_recovery_loss_ramp_steps,
        run_full_pinn_after_recovery=args.run_full_pinn_after_recovery,
        freeze_u_encoder_during_recovery=args.freeze_u_encoder_during_recovery,
        expert_balance_weight=args.expert_balance_weight,
        condition_consistency_weight=args.condition_consistency_weight,
        moe_top_k=args.moe_top_k,
        ablate_disable_moe=args.ablate_disable_moe,
        ablate_disable_conditioned_pde=args.ablate_disable_conditioned_pde,
        ablate_disable_aux_losses=args.ablate_disable_aux_losses,
        ablate_label_only_router=args.ablate_label_only_router,
        core_ablation_mode=args.core_ablation_mode,
        allow_ablation_checkpoint_mismatch=args.allow_ablation_checkpoint_mismatch,
        diagnostic_metrics_interval=args.diagnostic_metrics_interval,
        motion_mask_floor=args.motion_mask_floor,
        motion_mask_quantile=args.motion_mask_quantile,
        motion_mask_warmup_steps=args.motion_mask_warmup_steps,
        physical_mask_transition_steps=args.physical_mask_transition_steps,
        physics_state_mode=args.physics_state_mode,
        use_sigma_gate=args.use_sigma_gate,
        sigma_gate_curve=args.sigma_gate_curve,
        use_sigma_conditioning=args.use_sigma_conditioning,
        sigma_conditioning_dim=args.sigma_conditioning_dim,
        sigma_gate_floor=args.sigma_gate_floor,
        use_adaptive_condition_injection=args.use_adaptive_condition_injection,
        adaptive_conditioning_dim=args.adaptive_conditioning_dim,
        adaptive_conditioning_strength=args.adaptive_conditioning_strength,
        adaptive_conditioning_gate_floor=args.adaptive_conditioning_gate_floor,
        enable_rl_expert_optimization=args.enable_rl_expert_optimization,
        rl_policy_weight=args.rl_policy_weight,
        rl_entropy_weight=args.rl_entropy_weight,
        rl_reward_decay=args.rl_reward_decay,
        rl_reward_quality_weight=args.rl_reward_quality_weight,
        rl_reward_stability_weight=args.rl_reward_stability_weight,
        rl_warmup_steps=args.rl_warmup_steps,
        rl_hidden_dim=args.rl_hidden_dim,
        state_align_warmup_steps=args.state_align_warmup_steps,
        state_align_x_weight=args.state_align_x_weight,
        state_align_v_weight=args.state_align_v_weight,
        curriculum_transition_start_step=args.curriculum_transition_start_step,
        curriculum_transition_steps=args.curriculum_transition_steps,
        physics_weight_target=args.physics_weight_target,
        output_physics_weight=args.output_physics_weight,
        state_align_v_weight_target=args.state_align_v_weight_target,
        decoded_branch_consistency_weight=args.decoded_branch_consistency_weight,
        enable_explainability_reports=args.enable_explainability_reports,
        explainability_top_experts=args.explainability_top_experts,
        debug_fixed_timestep_fraction=args.debug_fixed_timestep_fraction,
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
        enable_explainability_reports=args.enable_explainability_reports,
        explainability_report_interval=args.explainability_report_interval,
        explainability_max_history=args.explainability_max_history,
        explainability_output_dir=args.explainability_output_dir,
    )

    # 优化器：只优化 PINN 可训练参数
    optimizer_learning_rate = (
        model.recommended_optimizer_learning_rate(args.learning_rate)
        if hasattr(model, "recommended_optimizer_learning_rate")
        else float(args.learning_rate)
    )
    if abs(float(optimizer_learning_rate) - float(args.learning_rate)) > 1e-12:
        print(
            "Applying resume stability learning-rate cap: "
            f"requested={args.learning_rate:.6e} effective={optimizer_learning_rate:.6e}"
        )
    optimizer = torch.optim.AdamW(
        model.trainable_modules(),
        lr=optimizer_learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # 从 checkpoint 恢复训练状态（如果有）
    resume_step = 0
    resume_epoch = 0
    checkpoint_training_state = None
    if hasattr(model, '_checkpoint_training_state') and model._checkpoint_training_state is not None:
        checkpoint_training_state = model._checkpoint_training_state
        resume_step = checkpoint_training_state.get('current_step', 0)
        resume_epoch = checkpoint_training_state.get('current_epoch', 0)
        print(f"Will resume training from step={resume_step}, epoch={resume_epoch}")
        # 更新 model 的 current_step
        model.current_step = resume_step

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
        max_steps=args.max_steps,
        resume_step=resume_step,
        resume_epoch=resume_epoch,
        checkpoint_training_state=checkpoint_training_state,
    )
