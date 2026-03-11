"""
Physics-Informed Flow Matching for Video Generation
基于物理约束的视频生成 Pipeline
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional
from tqdm import tqdm

from .wan_video_new import WanVideoPipeline, model_fn_wan_video, ModelConfig
from ..models.pinn_operators import MaterialPDEResiduals, MaterialClassifier
from ..models.pinn_adapter import PhysicsAdapter
from ..models.model_manager import ModelManager
from typing import Union


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


class PhysicsInformedWanVideoPipeline(WanVideoPipeline):
    """
    Physics-Informed Video Generation Pipeline
    继承自 WanVideoPipeline，添加物理约束
    """
    
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype, tokenizer_path=tokenizer_path)
        
        # PINN 组件
        self.pde_residuals = MaterialPDEResiduals().to(device)
        self.material_classifier = MaterialClassifier()
        self.physics_adapter = None  # 延迟初始化，在加载模型后
        
        # 物理损失权重（可调节）
        self.lambda_physics = 0.1  # 物理损失权重
        self.physics_warmup_steps = 1000  # 物理损失预热步数
        self.current_step = 0
        
        # 物理约束开关
        self.enable_physics_constraint = True
        self.use_physics_adapter = True  # 是否使用适配器（插件模式）
        
        # 推理时的物理场实时追踪记录（每次推理自动填充）
        self.physics_tracking = None
        # 推理结束后保存 final latent（可用于后续分析/调试）
        self._final_latent = None

    @staticmethod
    def _fit_metadata_2d(value, target_dim, batch_size, device, dtype):
        if target_dim <= 0:
            return torch.zeros(batch_size, 0, device=device, dtype=dtype)
        if value is None:
            return torch.zeros(batch_size, target_dim, device=device, dtype=dtype)
        if isinstance(value, str):
            parts = [it.strip() for it in value.split(",") if it.strip() != ""]
            value = [float(it) for it in parts] if len(parts) > 0 else None
        if value is None:
            return torch.zeros(batch_size, target_dim, device=device, dtype=dtype)
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
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

    def _prepare_adapter_metadata(self, raw_metadata: Any, batch_size: int, device, dtype):
        if not isinstance(raw_metadata, dict):
            return None

        label_id = raw_metadata.get("label_id")
        label_name = str(raw_metadata.get("label_name", "")).strip().lower()
        if label_id is None and label_name != "":
            label_id = PHENOMENON_TO_ID.get(label_name, 0)
        if isinstance(label_id, str):
            label_id = int(label_id.strip()) if label_id.strip() != "" else None

        if label_id is None:
            label_ids = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            label_ids = label_id if isinstance(label_id, torch.Tensor) else torch.tensor(label_id)
            label_ids = label_ids.to(device=device, dtype=torch.long).view(-1)
            if label_ids.numel() == 1:
                label_ids = label_ids.repeat(batch_size)
            elif label_ids.numel() != batch_size:
                label_ids = label_ids[:1].repeat(batch_size)
        max_label = max(int(getattr(self.physics_adapter, "num_phenomena", 17)) - 1, 0)
        label_ids = torch.clamp(label_ids, min=0, max=max_label)

        n_numeric_dim = int(getattr(self.physics_adapter, "n_numeric_dim", 0))
        q_input_dim = int(getattr(self.physics_adapter, "q_input_dim", 0))
        n_text_dim = 3

        n_numeric = self._fit_metadata_2d(
            raw_metadata.get("n_numeric"), n_numeric_dim, batch_size, device, dtype
        )
        q_vector = self._fit_metadata_2d(
            raw_metadata.get("q_vector"), q_input_dim, batch_size, device, dtype
        )

        n_text_ids = raw_metadata.get("n_text_ids")
        if n_text_ids is None:
            n_text_ids = torch.zeros(batch_size, n_text_dim, device=device, dtype=torch.long)
        else:
            if isinstance(n_text_ids, str):
                parts = [it.strip() for it in n_text_ids.split(",") if it.strip() != ""]
                n_text_ids = [int(it) for it in parts]
            n_text_ids = n_text_ids if isinstance(n_text_ids, torch.Tensor) else torch.tensor(n_text_ids)
            n_text_ids = n_text_ids.to(device=device, dtype=torch.long)
            if n_text_ids.ndim == 1:
                n_text_ids = n_text_ids.unsqueeze(0)
            if n_text_ids.shape[0] == 1 and batch_size > 1:
                n_text_ids = n_text_ids.repeat(batch_size, 1)
            if n_text_ids.shape[0] != batch_size:
                n_text_ids = n_text_ids[:1].repeat(batch_size, 1)
            if n_text_ids.shape[1] < n_text_dim:
                pad = torch.zeros(
                    batch_size,
                    n_text_dim - n_text_ids.shape[1],
                    device=device,
                    dtype=torch.long,
                )
                n_text_ids = torch.cat([n_text_ids, pad], dim=1)
            elif n_text_ids.shape[1] > n_text_dim:
                n_text_ids = n_text_ids[:, :n_text_dim]
        n_text_vocab = max(int(getattr(self.physics_adapter, "n_text_vocab_size", 2048)) - 1, 0)
        n_text_ids = torch.clamp(n_text_ids, min=0, max=n_text_vocab)

        metadata = {
            "label_id": label_ids,
            "n_numeric": n_numeric,
            "n_text_ids": n_text_ids,
            "q_vector": q_vector,
        }
        if label_name != "":
            metadata["label_name"] = label_name
        return metadata
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        """
        重写父类的 from_pretrained 方法，确保返回 PhysicsInformedWanVideoPipeline 实例
        """
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline (使用 PhysicsInformedWanVideoPipeline 而不是 WanVideoPipeline)
        pipe = PhysicsInformedWanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")
        
        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)
        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe
    
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        """
        重写 __call__: 在单次 PINN 推理路径上进行 denoise + decode。
        """
        from tqdm import tqdm as _tqdm
        
        progress_bar_cmd = kwargs.get("progress_bar_cmd", _tqdm)
        tiled = kwargs.get("tiled", True)
        tile_size = kwargs.get("tile_size", (30, 52))
        tile_stride = kwargs.get("tile_stride", (15, 26))
        denoising_strength = kwargs.get("denoising_strength", 1.0)
        sigma_shift = kwargs.get("sigma_shift", 5.0)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        cfg_scale = kwargs.get("cfg_scale", 5.0)
        cfg_merge = kwargs.get("cfg_merge", False)
        switch_DiT_boundary = kwargs.get("switch_DiT_boundary", 0.875)
        vace_reference_image = kwargs.get("vace_reference_image", None)
        
        prompt = args[0] if args else kwargs.get("prompt", "")
        
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": kwargs.get("tea_cache_l1_thresh"),
            "tea_cache_model_id": kwargs.get("tea_cache_model_id", ""),
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "tea_cache_l1_thresh": kwargs.get("tea_cache_l1_thresh"),
            "tea_cache_model_id": kwargs.get("tea_cache_model_id", ""),
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": kwargs.get("input_image"),
            "end_image": kwargs.get("end_image"),
            "input_video": kwargs.get("input_video"), "denoising_strength": denoising_strength,
            "control_video": kwargs.get("control_video"), "reference_image": kwargs.get("reference_image"),
            "camera_control_direction": kwargs.get("camera_control_direction"),
            "camera_control_speed": kwargs.get("camera_control_speed", 1/54),
            "camera_control_origin": kwargs.get("camera_control_origin", (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)),
            "vace_video": kwargs.get("vace_video"), "vace_video_mask": kwargs.get("vace_video_mask"),
            "vace_reference_image": vace_reference_image, "vace_scale": kwargs.get("vace_scale", 1.0),
            "seed": kwargs.get("seed"), "rand_device": kwargs.get("rand_device", "cpu"),
            "height": kwargs.get("height", 480), "width": kwargs.get("width", 832),
            "num_frames": kwargs.get("num_frames", 81),
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": kwargs.get("motion_bucket_id"),
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": kwargs.get("sliding_window_size"),
            "sliding_window_stride": kwargs.get("sliding_window_stride"),
            "pinn_metadata": kwargs.get("pinn_metadata"),
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]

        self._final_latent = inputs_shared["latents"].detach().clone()

        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video
        
    def get_physics_weight(self):
        """获取当前的物理损失权重（带预热）"""
        if not self.enable_physics_constraint:
            return 0.0
        
        if self.current_step < self.physics_warmup_steps:
            # 线性预热
            alpha = self.current_step / self.physics_warmup_steps
            return self.lambda_physics * alpha
        else:
            return self.lambda_physics
    
    def training_loss_with_physics(self, **inputs):
        """
        带物理约束的训练损失
        
        Loss = Loss_FlowMatching + λ * Loss_Physics
        """
        # 1. 标准 Flow Matching 损失
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        # 直接生成 timestep 值，而不是索引
        timestep = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,), device=self.device).to(dtype=self.torch_dtype)
        
        # 构造 x_t
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        inputs["latents"].requires_grad_(True)  # 重要：需要计算梯度
        
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        
        # 准备模型字典（需要传入 dit, motion_controller, vace）
        models = {
            "dit": self.dit,
            "motion_controller": self.motion_controller,
            "vace": self.vace
        }
        
        # 模型预测
        noise_pred = self.model_fn(**models, **inputs, timestep=timestep)
        
        # Flow Matching 损失
        loss_fm = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss_fm = loss_fm * self.scheduler.training_weight(timestep)
        
        # 2. 物理约束损失
        physics_weight = self.get_physics_weight()
        
        if physics_weight > 0:
            # 识别材质类型
            prompt = inputs.get("prompt", [""])[0] if isinstance(inputs.get("prompt"), list) else inputs.get("prompt", "")
            material_type = self.material_classifier.classify(prompt)
            
            # 计算 PDE 残差
            z_t = inputs["latents"]
            v_pred = noise_pred  # 预测的速度场
            
            loss_physics, physics_info = self.compute_physics_loss(z_t, v_pred, material_type)
            
            # 总损失
            total_loss = loss_fm + physics_weight * loss_physics
            
            # 更新步数
            self.current_step += 1
            
            return total_loss, {
                'loss_fm': loss_fm.item(),
                'loss_physics': loss_physics.item(),
                'physics_weight': physics_weight,
                'material_type': material_type,
                **physics_info
            }
        else:
            self.current_step += 1
            return loss_fm, {'loss_fm': loss_fm.item(), 'physics_weight': 0.0}
    
    def compute_physics_loss(self, z_t, v_pred, material_type):
        """
        计算物理损失
        
        Args:
            z_t: latent at time t [B, C, T, H, W]
            v_pred: predicted velocity field [B, C, T, H, W]
            material_type: material type string
        
        Returns:
            loss_physics: scalar
            info: dict with loss components
        """
        if material_type == 'fluid':
            return self.pde_residuals.fluid_residual(z_t, v_pred)
        elif material_type == 'rigid':
            return self.pde_residuals.rigid_residual(z_t, v_pred)
        elif material_type == 'elastic':
            return self.pde_residuals.elastic_residual(z_t, v_pred)
        elif material_type == 'particle':
            return self.pde_residuals.particle_residual(z_t, v_pred)
        elif material_type == 'mixed':
            # 混合材质：计算所有类型的平均
            loss_fluid, info_fluid = self.pde_residuals.fluid_residual(z_t, v_pred)
            loss_rigid, info_rigid = self.pde_residuals.rigid_residual(z_t, v_pred)
            loss_total = (loss_fluid + loss_rigid) * 0.5
            info = {k: v * 0.5 for k, v in {**info_fluid, **info_rigid}.items()}
            return loss_total, info
        else:
            # 默认流体
            return self.pde_residuals.fluid_residual(z_t, v_pred)
    
    def load_pinn_plugin(self, checkpoint_path, device=None, enable_tracking=True):
        """
        加载 PINN 插件并将 adapter 接入推理流程。
        每个去噪步骤都会实时计算 PDE 残差（散度/涡量），记录到 self.physics_tracking。
        
        Args:
            checkpoint_path: PINN plugin checkpoint 路径
            device: 设备（默认用 pipeline 的 device）
            enable_tracking: 是否在推理时追踪物理场指标（默认开启）
        """
        if device is None:
            device = self.device
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'physics_adapter_state_dict' in checkpoint:
            latent_dim = 16
            self.initialize_physics_adapter(latent_dim=latent_dim)
            load_result = self.physics_adapter.load_state_dict(
                checkpoint['physics_adapter_state_dict'],
                strict=False,
            )
            if len(load_result.missing_keys) > 0:
                print(f"  PhysicsAdapter missing keys: {load_result.missing_keys[:8]}")
            if len(load_result.unexpected_keys) > 0:
                print(f"  PhysicsAdapter unexpected keys: {load_result.unexpected_keys[:8]}")
            if "expert_usage_ema" in set(load_result.missing_keys):
                # Backward compatibility for old checkpoints saved without buffers.
                with torch.no_grad():
                    self.physics_adapter.expert_usage_ema.fill_(
                        1.0 / max(float(self.physics_adapter.num_phenomena), 1.0)
                    )
                print("  Initialized missing buffer: expert_usage_ema")
            self.physics_adapter.to(dtype=self.torch_dtype, device=device)
            self.physics_adapter.eval()
            print(f"  PhysicsAdapter loaded (dtype={self.torch_dtype})")
        
        if 'pde_residuals_state_dict' in checkpoint:
            self.pde_residuals.load_state_dict(checkpoint['pde_residuals_state_dict'])
            print(f"  PDE Residuals loaded")
        
        if 'config' in checkpoint:
            print(f"  Plugin config: {checkpoint['config']}")
        
        original_model_fn = self.model_fn
        adapter = self.physics_adapter
        pipeline_ref = self
        
        _step = [0]
        _snapshot_steps = {1, 2, 5, 10, 13, 25, 50, 75, 100}
        
        # 每步都保存 v_original，供外部调试分析使用
        _last_v_original = [None]
        
        def model_fn_with_adapter(**kwargs):
            """每个去噪步骤: DiT → PhysicsAdapter → PDE 残差 tracking"""
            adapter_metadata_input = kwargs.pop("pinn_metadata", None)
            v_original = original_model_fn(**kwargs)
            
            if adapter is None:
                return v_original
            
            latents = kwargs.get("latents")
            if latents is None:
                return v_original
            
            adapter.to(device=v_original.device, dtype=v_original.dtype)
            adapter_metadata = pipeline_ref._prepare_adapter_metadata(
                adapter_metadata_input,
                batch_size=v_original.shape[0],
                device=v_original.device,
                dtype=v_original.dtype,
            )
            v_corrected = adapter(v_original, latents, metadata=adapter_metadata)
            
            _step[0] += 1
            step = _step[0]
            
            with torch.no_grad():
                # 始终保存最新的 v_original，用于推理结束后的像素空间对比
                _last_v_original[0] = v_original.detach().clone()
                
                if not (enable_tracking and pipeline_ref.physics_tracking is not None):
                    return v_corrected
                
                t = pipeline_ref.physics_tracking
                v_of = v_original.float()
                v_cf = v_corrected.float()
                diff = v_cf - v_of
                scale_val = adapter.scale.item()
                corr_ratio = diff.abs().mean().item() / (v_of.abs().mean().item() + 1e-10)
                
                div_orig = _compute_divergence_sq(v_of)
                div_corr = _compute_divergence_sq(v_cf)
                vor_orig = _compute_vorticity_sq(v_of)
                vor_corr = _compute_vorticity_sq(v_cf)
                smooth_orig = _compute_temporal_smoothness(v_of)
                smooth_corr = _compute_temporal_smoothness(v_cf)
                
                t["steps"].append(step)
                t["scale"].append(scale_val)
                t["correction_ratio"].append(corr_ratio)
                t["div_before"].append(div_orig)
                t["div_after"].append(div_corr)
                t["vor_before"].append(vor_orig)
                t["vor_after"].append(vor_corr)
                t["smooth_before"].append(smooth_orig)
                t["smooth_after"].append(smooth_corr)
                
                if step in _snapshot_steps:
                    snap = _build_spatial_snapshot(step, v_of, v_cf, diff, adapter)
                    t["snapshots"].append(snap)
                
                if step <= 3 or step % 10 == 0:
                    div_d = (div_corr - div_orig) / (div_orig + 1e-10) * 100
                    print(f"  [Step {step:3d}] scale={scale_val:.6f}  "
                          f"corr={corr_ratio:.4%}  "
                          f"div: {div_orig:.6f}→{div_corr:.6f} ({div_d:+.1f}%)")
            
            return v_corrected
        
        self.model_fn = model_fn_with_adapter
        self._last_v_original = _last_v_original  # 暴露给外部
        print("  model_fn wrapped with PhysicsAdapter + PDE tracking")
        print("  PINN plugin loaded successfully")
    
    def reset_tracking(self):
        """在每次推理前调用，重置物理场追踪记录。"""
        self.physics_tracking = {
            "steps": [], "scale": [], "correction_ratio": [],
            "div_before": [], "div_after": [],
            "vor_before": [], "vor_after": [],
            "smooth_before": [], "smooth_after": [],
            "snapshots": [],
        }
    
    def save_physics_report(
        self,
        output_path: str,
        video_frames=None,
        attention_overlay: bool = True,
        attention_alpha: float = 0.45,
        attention_video_fps: int = 15,
        attention_use_motion_weighted: bool = True,
        attention_motion_percentile: float = 90.0,
    ):
        """
        推理完成后调用，生成物理场验证报告。
        
        Args:
            output_path: 输出路径前缀
            video_frames: PINN 版本的视频帧 (PIL Image list)
        """
        t = self.physics_tracking
        if t is None or len(t["steps"]) == 0:
            print("  No physics tracking data to report.")
            return
        
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np
        import os
        
        base = output_path.replace(".png", "")
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        
        # ═══════════ 图1: 标量曲线（latent 空间指标） ═══════════
        self._plot_scalar_curves(t, f"{base}_curves.png")
        
        snapshots = t.get("snapshots", [])
        
        # ═══════════ 图2: 关注区域（raw_correction） ═══════════
        if snapshots:
            self._plot_attention_maps_from_snapshots(
                snapshots, f"{base}_attention_maps.png"
            )
            if video_frames:
                attention_maps = self._build_attention_maps_for_video(
                    snapshots=snapshots,
                    n_frames=len(video_frames),
                    frame_height=video_frames[0].height,
                    frame_width=video_frames[0].width,
                    use_motion_weighted=attention_use_motion_weighted,
                    motion_percentile=attention_motion_percentile,
                )
                if attention_maps is not None:
                    self._plot_attention_overlay_on_frames(
                        video_frames=video_frames,
                        attention_maps=attention_maps,
                        path=f"{base}_attention_overlay.png",
                        alpha=attention_alpha,
                    )
                    if attention_overlay:
                        self._export_attention_overlay_video(
                            video_frames=video_frames,
                            attention_maps=attention_maps,
                            path=f"{base}_attention_overlay.mp4",
                            alpha=attention_alpha,
                            fps=attention_video_fps,
                        )

        # ═══════════ 图3: 通道分析 ═══════════
        if snapshots:
            self._plot_channel_analysis(snapshots, f"{base}_channels.png")
        
        # ═══════════ 数值摘要 ═══════════
        div_b = np.mean(t["div_before"])
        div_a = np.mean(t["div_after"])
        div_r = (1 - div_a / (div_b + 1e-10)) * 100
        scale_avg = np.mean(t["scale"])
        ratio_avg = np.mean(t["correction_ratio"]) * 100
        
        print(f"\n  ┌────────────────────── Physics Summary ──────────────────────┐")
        print(f"  │  Divergence  before PINN  = {div_b:.6f}                      │")
        print(f"  │  Divergence  after  PINN  = {div_a:.6f}                      │")
        print(f"  │  Divergence  reduction    = {div_r:+.2f}%                     │")
        print(f"  │  adapter.scale  avg       = {scale_avg:.6f}                  │")
        print(f"  │  correction ratio avg     = {ratio_avg:.4f}%                 │")
        if div_r > 1:
            print(f"  │  ✓ PINN 有效降低了物理违约                                  │")
        elif div_r > -1:
            print(f"  │  ~ PINN 对物理场影响不大 (scale可能太小)                     │")
        else:
            print(f"  │  ✗ PINN 可能在恶化物理一致性 (需检查训练)                    │")
        print(f"  └─────────────────────────────────────────────────────────────┘")
    
    # ────────── 子图绘制方法 ──────────
    
    @staticmethod
    def _plot_scalar_curves(t: dict, path: str):
        """标量指标随去噪步变化曲线。"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        steps = t["steps"]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("PINN Physics Tracking — Scalar Metrics over Denoising Steps",
                      fontsize=14, fontweight="bold")
        
        ax = axes[0, 0]
        ax.plot(steps, t["div_before"], "b--", lw=1.5, label="Before PINN", alpha=0.8)
        ax.plot(steps, t["div_after"], "r-", lw=2, label="After PINN", alpha=0.9)
        ax.set_title("|∇·v|² Divergence (↓ = incompressible)")
        ax.set_xlabel("Step"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(steps, t["vor_before"], "b--", lw=1.5, label="Before PINN", alpha=0.8)
        ax.plot(steps, t["vor_after"], "r-", lw=2, label="After PINN", alpha=0.9)
        ax.set_title("|ω|² Vorticity (rotational energy)")
        ax.set_xlabel("Step"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        ax.plot(steps, t["smooth_before"], "b--", lw=1.5, label="Before PINN", alpha=0.8)
        ax.plot(steps, t["smooth_after"], "r-", lw=2, label="After PINN", alpha=0.9)
        ax.set_title("|dv/dt|² Temporal Smoothness (↓ = coherent)")
        ax.set_xlabel("Step"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        ax.plot(steps, t["scale"], "g-o", ms=3, lw=1.5)
        ax.axhline(0, color="gray", ls="--", alpha=0.4)
        ax.set_title("adapter.scale (learned correction strength)")
        ax.set_xlabel("Step"); ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.plot(steps, [r * 100 for r in t["correction_ratio"]], "m-o", ms=3, lw=1.5)
        ax.set_title("|Δv| / |v_orig| % (correction magnitude)")
        ax.set_xlabel("Step"); ax.set_ylabel("%"); ax.grid(True, alpha=0.3)
        
        ax = axes[1, 2]
        imps = [(1 - da / (db + 1e-10)) * 100
                for db, da in zip(t["div_before"], t["div_after"])]
        ax.bar(steps, imps, color=["green" if v > 0 else "red" for v in imps], alpha=0.7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("Divergence Reduction % (green=improved)")
        ax.set_xlabel("Step"); ax.set_ylabel("%"); ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Report] Scalar curves → {path}")
    
    @staticmethod
    def _build_attention_maps_for_video(
        snapshots: list,
        n_frames: int,
        frame_height: int,
        frame_width: int,
        percentile: float = 99.0,
        use_motion_weighted: bool = True,
        motion_percentile: float = 90.0,
    ):
        """将 snapshot 中的 correction attention 映射到视频帧尺寸。"""
        import numpy as np
        
        if not snapshots or n_frames <= 0 or frame_height <= 0 or frame_width <= 0:
            return None
        
        raw_maps = []
        raw_steps = []
        for snap in snapshots:
            # 优先使用 motion-weighted correction，保证与动态区域约束一致。
            m = None
            if use_motion_weighted:
                m = snap.get("motion_weighted_correction_map")
            if m is None:
                m = snap.get("raw_correction_map")
            if m is None:
                m = snap.get("correction_map")
            if m is None:
                continue
            m = np.asarray(m, dtype=np.float32)  # [T, H, W]
            if m.ndim != 3:
                continue
            if use_motion_weighted:
                motion_map = snap.get("motion_mask_map")
                if motion_map is not None:
                    motion_map = np.asarray(motion_map, dtype=np.float32)
                    if motion_map.ndim == 3 and motion_map.shape == m.shape:
                        threshold = np.percentile(motion_map, motion_percentile)
                        motion_soft = np.clip(
                            (motion_map - threshold) / (1.0 - threshold + 1e-8), 0.0, 1.0
                        )
                        m = m * motion_soft
            scale = np.percentile(m, percentile) + 1e-8
            raw_maps.append(np.clip(m / scale, 0.0, 1.0))
            raw_steps.append(float(snap.get("step", len(raw_steps) + 1)))
        
        if not raw_maps:
            print("  [Report] No correction_map found in snapshots (skip attention).")
            return None
        
        # 越靠后的 denoising step 与最终可见结果对应性越强，因此提高后期 snapshot 权重。
        stack = np.stack(raw_maps, axis=0)
        step_arr = np.asarray(raw_steps, dtype=np.float32)
        step_arr = np.maximum(step_arr, 1.0)
        step_weights = (step_arr / (step_arr.max() + 1e-8)) ** 2
        step_weights = step_weights / (step_weights.sum() + 1e-8)
        attention_latent = (stack * step_weights[:, None, None, None]).sum(axis=0)
        attention_latent = PhysicsInformedWanVideoPipeline._smooth_attention_volume(attention_latent)
        
        # 使用 trilinear 一次性做时间+空间插值到 [n_frames, frame_height, frame_width]
        attention_tensor = torch.from_numpy(attention_latent).unsqueeze(0).unsqueeze(0)  # [1,1,T,H,W]
        attention_video = F.interpolate(
            attention_tensor,
            size=(n_frames, frame_height, frame_width),
            mode="trilinear",
            align_corners=False,
        )[0, 0].cpu().numpy().astype(np.float32)
        
        # 全局稳健归一化，提升可视对比稳定性
        denom = np.percentile(attention_video, percentile) + 1e-8
        attention_video = np.clip(attention_video / denom, 0.0, 1.0)
        
        # 仅保留高响应区域，避免整屏雾化；这是幅值筛选，不改变空间对应关系。
        attention_video = PhysicsInformedWanVideoPipeline._sparsify_attention_maps(attention_video)
        return attention_video
    
    @staticmethod
    def _smooth_attention_volume(volume, spatial_kernel: int = 5, temporal_kernel: int = 3):
        """对 latent attention 做轻量时空平滑，抑制高频颗粒噪声。"""
        tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        k_t = max(1, int(temporal_kernel))
        k_s = max(1, int(spatial_kernel))
        tensor = F.avg_pool3d(
            tensor,
            kernel_size=(k_t, k_s, k_s),
            stride=1,
            padding=(k_t // 2, k_s // 2, k_s // 2),
        )
        return tensor[0, 0].cpu().numpy().astype(volume.dtype)
    
    @staticmethod
    def _sparsify_attention_maps(attention_maps, keep_percentile: float = 90.0):
        """按帧保留高响应区域，降低全屏均匀着色。"""
        import numpy as np
        
        if attention_maps is None:
            return None
        att = np.asarray(attention_maps, dtype=np.float32).copy()
        if att.ndim != 3:
            return att
        
        for idx in range(att.shape[0]):
            thresh = np.percentile(att[idx], keep_percentile)
            soft = np.clip((att[idx] - thresh) / (1.0 - thresh + 1e-8), 0.0, 1.0)
            att[idx] = np.clip(att[idx] * soft, 0.0, 1.0)
        return att
    
    @staticmethod
    def _compose_attention_overlay(frame, attention_map, alpha: float = 0.45, cmap_name: str = "inferno"):
        """按像素 alpha 叠加 attention，避免整帧统一染色。"""
        import matplotlib.cm as cm
        import numpy as np
        
        frame = np.asarray(frame).astype(np.float32)
        att = np.clip(np.asarray(attention_map, dtype=np.float32), 0.0, 1.0)
        att_rgb = (cm.get_cmap(cmap_name)(att)[:, :, :3] * 255.0).astype(np.float32)
        local_alpha = float(max(0.0, min(1.0, alpha))) * np.power(att, 1.5)
        local_alpha = local_alpha[:, :, None]
        blend = np.clip(frame * (1.0 - local_alpha) + att_rgb * local_alpha, 0.0, 255.0)
        return blend.astype(np.uint8)
    
    @staticmethod
    def _plot_attention_maps_from_snapshots(snapshots: list, path: str):
        """展示多个 snapshot 在 latent 空间的关注区域热图（中间时间帧）。"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        valid = []
        for snap in snapshots:
            m = snap.get("raw_correction_map")
            if m is None:
                m = snap.get("correction_map")
            if m is None:
                continue
            valid.append((snap.get("step", -1), np.asarray(m, dtype=np.float32)))
        
        if not valid:
            print("  [Report] No snapshot attention maps (skip attention_maps.png).")
            return
        
        n_rows = min(6, len(valid))
        indices = np.linspace(0, len(valid) - 1, n_rows, dtype=int)
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.0 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(
            "PINN Attention in Latent Space (raw_correction mid-time slice)",
            fontsize=13,
            fontweight="bold",
        )
        
        for row, idx in enumerate(indices):
            step, vol = valid[idx]  # vol: [T,H,W]
            vol = PhysicsInformedWanVideoPipeline._smooth_attention_volume(vol)
            t_mid = vol.shape[0] // 2
            m = vol[t_mid]
            norm = np.clip(m / (np.percentile(m, 99) + 1e-8), 0.0, 1.0)
            thresh = np.percentile(norm, 90)
            norm = np.clip((norm - thresh) / (1.0 - thresh + 1e-8), 0.0, 1.0)
            
            axes[row, 0].imshow(m, cmap="inferno", interpolation="bilinear", aspect="auto")
            axes[row, 0].set_title("Raw correction map", fontsize=10)
            axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
            axes[row, 0].set_ylabel(f"Step {step}", fontsize=10, fontweight="bold")
            
            axes[row, 1].imshow(norm, cmap="hot", interpolation="bilinear", aspect="auto")
            axes[row, 1].set_title("Normalized attention (p99)", fontsize=10)
            axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Report] Attention maps → {path}")
    
    @staticmethod
    def _plot_attention_overlay_on_frames(video_frames: list, attention_maps, path: str, alpha: float = 0.45):
        """将 attention 热图叠加到若干抽样视频帧。"""
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        
        n_total = min(len(video_frames), attention_maps.shape[0])
        if n_total <= 0:
            print("  [Report] Empty frames/maps (skip attention overlay PNG).")
            return
        
        n_rows = min(6, n_total)
        indices = np.linspace(0, n_total - 1, n_rows, dtype=int)
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 3.2))
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(
            "PINN Attention Overlay in Pixel Space",
            fontsize=14,
            fontweight="bold",
        )
        
        cmap_fn = cm.get_cmap("inferno")
        alpha = float(max(0.0, min(1.0, alpha)))
        
        for row, idx in enumerate(indices):
            frame = np.array(video_frames[idx]).astype(np.float32)
            att = np.clip(attention_maps[idx], 0.0, 1.0)
            blend = PhysicsInformedWanVideoPipeline._compose_attention_overlay(
                frame=frame,
                attention_map=att,
                alpha=alpha,
                cmap_name="inferno",
            )
            
            axes[row, 0].imshow(frame.astype(np.uint8))
            axes[row, 0].set_title("With PINN", fontsize=10)
            axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
            axes[row, 0].set_ylabel(f"Frame {idx}", fontsize=10, fontweight="bold")
            
            axes[row, 1].imshow(att, cmap="inferno", vmin=0.0, vmax=1.0)
            axes[row, 1].set_title("Attention map", fontsize=10)
            axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])
            
            axes[row, 2].imshow(blend)
            axes[row, 2].set_title("Overlay (bright=high attention)", fontsize=10)
            axes[row, 2].set_xticks([]); axes[row, 2].set_yticks([])
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Report] Attention overlay PNG → {path}")
    
    @staticmethod
    def _export_attention_overlay_video(video_frames: list, attention_maps, path: str, alpha: float = 0.45, fps: int = 15):
        """导出逐帧 attention overlay MP4。"""
        import matplotlib.cm as cm
        import numpy as np
        from PIL import Image
        from ..data.video import save_video
        
        n_total = min(len(video_frames), attention_maps.shape[0])
        if n_total <= 0:
            print("  [Report] Empty frames/maps (skip attention overlay MP4).")
            return
        
        alpha = float(max(0.0, min(1.0, alpha)))
        out_frames = []
        
        for idx in range(n_total):
            frame = np.array(video_frames[idx]).astype(np.float32)
            att = np.clip(attention_maps[idx], 0.0, 1.0)
            blend = PhysicsInformedWanVideoPipeline._compose_attention_overlay(
                frame=frame,
                attention_map=att,
                alpha=alpha,
                cmap_name="inferno",
            )
            out_frames.append(Image.fromarray(blend))
        
        save_video(out_frames, path, fps=int(max(1, fps)), quality=5)
        print(f"  [Report] Attention overlay MP4 → {path}")
    
    @staticmethod
    def _plot_spatial_maps(snapshots: list, path: str):
        """Fallback: 无视频帧时画纯 latent 空间热力图。"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        n_snap = len(snapshots)
        n_cols = 6
        fig, axes = plt.subplots(n_snap, n_cols, figsize=(n_cols * 3.5, n_snap * 3))
        if n_snap == 1:
            axes = axes[np.newaxis, :]
        
        fig.suptitle("Spatial Physics Fields (latent space, no video frames available)",
                      fontsize=13, fontweight="bold")
        
        col_titles = ["Correction |Δv|", "Div ∇·v (before)", "Div ∇·v (after)",
                       "Div improvement", "Vorticity (before)", "Vorticity (after)"]
        
        for row, snap in enumerate(snapshots):
            step = snap["step"]
            T = snap["correction_map"].shape[0]
            t_mid = T // 2
            
            corr = snap["correction_map"][t_mid]
            div_b = snap["div_field_before"][t_mid]
            div_a = snap["div_field_after"][t_mid]
            div_diff = div_b - div_a
            vor_b = snap["vor_field_before"][t_mid]
            vor_a = snap["vor_field_after"][t_mid]
            
            maps = [corr, div_b, div_a, div_diff, vor_b, vor_a]
            cmaps = ["hot", "RdBu_r", "RdBu_r", "RdBu", "RdBu_r", "RdBu_r"]
            sym = [False, True, True, True, True, True]
            
            for col, (m, cmap, is_sym) in enumerate(zip(maps, cmaps, sym)):
                ax = axes[row, col]
                if is_sym:
                    vmax = np.percentile(np.abs(m), 97) + 1e-8
                    ax.imshow(m, cmap=cmap, vmin=-vmax, vmax=vmax,
                              aspect="auto", interpolation="bilinear")
                else:
                    ax.imshow(m, cmap=cmap, aspect="auto", interpolation="bilinear")
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"Step {step}", fontsize=10, fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Report] Spatial maps (latent only) → {path}")
    
    @staticmethod
    def _plot_channel_analysis(snapshots: list, path: str):
        """
        通道级分析:
        - 每个 latent 通道的校正幅度 (哪些通道被 PINN 改得最多)
        - 通道校正随去噪步的演变
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Channel-level Correction Analysis — Which latent channels does PINN modify?",
                      fontsize=13, fontweight="bold")
        
        # 收集每个快照的通道校正
        all_steps = []
        all_ch_corr = []   # [n_snap, C]
        all_ch_div_b = []
        all_ch_div_a = []
        for snap in snapshots:
            all_steps.append(snap["step"])
            all_ch_corr.append(snap["channel_correction"])   # [C]
            all_ch_div_b.append(snap["channel_div_before"])  # [C]
            all_ch_div_a.append(snap["channel_div_after"])   # [C]
        
        all_ch_corr = np.array(all_ch_corr)    # [n_snap, C]
        all_ch_div_b = np.array(all_ch_div_b)
        all_ch_div_a = np.array(all_ch_div_a)
        C = all_ch_corr.shape[1]
        
        # 子图1: 条形图 — 所有快照平均的通道校正幅度
        ax = axes[0]
        ch_mean = all_ch_corr.mean(axis=0)
        colors = plt.cm.viridis(ch_mean / (ch_mean.max() + 1e-10))
        ax.bar(range(C), ch_mean, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Latent Channel")
        ax.set_ylabel("Mean |correction|")
        ax.set_title("Per-channel correction magnitude\n(which channels PINN modifies most)")
        ax.set_xticks(range(C))
        ax.grid(True, axis="y", alpha=0.3)
        
        # 子图2: 热力图 — 通道 × 步骤
        ax = axes[1]
        im = ax.imshow(all_ch_corr.T, aspect="auto", cmap="hot",
                        interpolation="nearest")
        ax.set_xlabel("Snapshot index (early → late)")
        ax.set_ylabel("Latent Channel")
        ax.set_title("Channel correction over denoising steps")
        ax.set_xticks(range(len(all_steps)))
        ax.set_xticklabels([f"s{s}" for s in all_steps], fontsize=8)
        ax.set_yticks(range(C))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 子图3: 散度通道改善
        ax = axes[2]
        ch_improve = (all_ch_div_b.mean(axis=0) - all_ch_div_a.mean(axis=0))
        colors_imp = ["green" if v > 0 else "red" for v in ch_improve]
        ax.bar(range(C), ch_improve, color=colors_imp, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Latent Channel")
        ax.set_ylabel("Div(before) - Div(after)")
        ax.set_title("Per-channel divergence improvement\n(green = PINN reduces divergence)")
        ax.set_xticks(range(C))
        ax.grid(True, axis="y", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Report] Channel analysis → {path}")
    
    @staticmethod
    def _plot_encoder_activation(snapshots: list, path: str):
        """
        可视化 adapter 内部 physics_encoder 的激活分布:
        - 激活热力图 (top-k 激活通道)
        - 激活分布直方图
        - 激活随去噪步变化
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        has_feat = any("encoder_activation" in s for s in snapshots)
        if not has_feat:
            print("  [Report] No encoder activation data (skip encoder plot).")
            return
        
        fig, axes = plt.subplots(2, len(snapshots), figsize=(4 * len(snapshots), 8),
                                  squeeze=False)
        fig.suptitle("Physics Encoder Internal Activation — Where does the adapter 'look'?",
                      fontsize=13, fontweight="bold")
        
        for col, snap in enumerate(snapshots):
            step = snap["step"]
            feat = snap.get("encoder_activation")  # [hidden_dim, T, H, W]
            if feat is None:
                continue
            
            T = feat.shape[1]
            t_mid = T // 2
            
            # 取中间时间帧，对所有 hidden_dim 通道求 L2 norm → [H, W]
            feat_frame = feat[:, t_mid, :, :]  # [hidden_dim, H, W]
            activation_map = np.sqrt((feat_frame ** 2).mean(axis=0))  # [H, W]
            
            # 子图上: 激活热力图
            ax = axes[0, col]
            im = ax.imshow(activation_map, cmap="inferno", aspect="auto",
                           interpolation="bilinear")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Step {step}", fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel("Encoder activation\n(L2 norm over hidden_dim)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            
            # 子图下: 激活分布直方图
            ax = axes[1, col]
            ax.hist(activation_map.ravel(), bins=50, color="darkorange",
                    alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.set_xlabel("Activation magnitude")
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution (step {step})")
            ax.axvline(activation_map.mean(), color="red", ls="--", lw=1.5,
                        label=f"mean={activation_map.mean():.4f}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Report] Encoder activation → {path}")
    
    def set_physics_weight(self, weight):
        """设置物理损失权重"""
        self.lambda_physics = weight
        print(f"Physics loss weight set to: {weight}")
    
    def enable_physics(self):
        """启用物理约束"""
        self.enable_physics_constraint = True
        print("Physics constraint enabled")
    
    def disable_physics(self):
        """禁用物理约束"""
        self.enable_physics_constraint = False
        print("Physics constraint disabled")
    
    def initialize_physics_adapter(self, latent_dim=16):
        """初始化物理适配器（在模型加载后调用）"""
        if self.physics_adapter is None:
            self.physics_adapter = PhysicsAdapter(
                latent_dim=latent_dim,
                hidden_dim=64,
                material_types=4
            ).to(self.device)
            print(f"✓ Physics Adapter initialized (latent_dim={latent_dim})")
        return self.physics_adapter
    
    def freeze_original_model(self):
        """冻结原始模型参数（插件模式）"""
        # 冻结 DiT
        if self.dit is not None:
            for param in self.dit.parameters():
                param.requires_grad = False
            print("✓ Original DiT frozen")
        
        # 冻结 DiT2
        if self.dit2 is not None:
            for param in self.dit2.parameters():
                param.requires_grad = False
            print("✓ Original DiT2 frozen")
        
        # 冻结 VAE
        if self.vae is not None:
            for param in self.vae.parameters():
                param.requires_grad = False
            print("✓ VAE frozen")
        
        # 冻结 Text Encoder
        if self.text_encoder is not None:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("✓ Text Encoder frozen")
        
        print("=" * 60)
        print("PLUGIN MODE: Only PINN components will be trained")
        print("=" * 60)


# 导出用于训练的模型函数
def model_fn_wan_video_pinn(
    dit,
    motion_controller=None,
    vace=None,
    latents=None,
    timestep=None,
    context=None,
    clip_feature=None,
    y=None,
    **kwargs
):
    """
    Physics-Informed 版本的模型函数
    与原版相同，只是确保支持梯度计算
    """
    # 确保 latents 可以计算梯度
    if latents is not None and not latents.requires_grad:
        latents = latents.requires_grad_(True)
    
    # 调用原始模型函数
    output = model_fn_wan_video(
        dit=dit,
        motion_controller=motion_controller,
        vace=vace,
        latents=latents,
        timestep=timestep,
        context=context,
        clip_feature=clip_feature,
        y=y,
        **kwargs
    )
    
    return output


# ═══════════════════════════════════════════════════════════════════════════
# Latent-space 物理量计算工具函数
# ═══════════════════════════════════════════════════════════════════════════

def _compute_divergence_sq(v: torch.Tensor) -> float:
    """散度²均值 (标量)。v: [B, C, T, H, W]"""
    div = _compute_divergence_field(v)
    return div.pow(2).mean().item()


def _compute_vorticity_sq(v: torch.Tensor) -> float:
    """涡量²均值 (标量)。v: [B, C, T, H, W]"""
    curl = _compute_vorticity_field(v)
    return curl.pow(2).mean().item()


def _compute_temporal_smoothness(v: torch.Tensor) -> float:
    """时间平滑性 (标量): mean(|dv/dt|²)"""
    if v.shape[2] <= 1:
        return 0.0
    dv_dt = v[:, :, 1:] - v[:, :, :-1]
    return dv_dt.pow(2).mean().item()


def _compute_divergence_field(v: torch.Tensor) -> torch.Tensor:
    """散度场 ∇·v。v: [B, C, T, H, W] → [B, 1, T, H, W]"""
    dv_dh = v[:, :, :, 1:, :] - v[:, :, :, :-1, :]
    dv_dh = F.pad(dv_dh, (0, 0, 0, 1))
    dv_dw = v[:, :, :, :, 1:] - v[:, :, :, :, :-1]
    dv_dw = F.pad(dv_dw, (0, 1))
    return dv_dh.mean(dim=1, keepdim=True) + dv_dw.mean(dim=1, keepdim=True)


def _compute_vorticity_field(v: torch.Tensor) -> torch.Tensor:
    """涡量场 ω (2D curl z 分量)。v: [B, C, T, H, W] → [B, 1, T, H-1, W-1]"""
    C = v.shape[1]
    half_C = max(C // 2, 1)
    dv_dh = v[:, :, :, 1:, :-1] - v[:, :, :, :-1, :-1]
    dv_dw = v[:, :, :, :-1, 1:] - v[:, :, :, :-1, :-1]
    curl = dv_dw[:, :half_C].mean(dim=1, keepdim=True) - dv_dh[:, half_C:].mean(dim=1, keepdim=True)
    return curl


def _compute_channel_divergence(v: torch.Tensor):
    """逐通道散度²。v: [B, C, T, H, W] → numpy [C]"""
    import numpy as np
    C = v.shape[1]
    ch_divs = []
    for c in range(C):
        vc = v[:, c:c+1]  # [B, 1, T, H, W]
        dh = vc[:, :, :, 1:, :] - vc[:, :, :, :-1, :]
        dw = vc[:, :, :, :, 1:] - vc[:, :, :, :, :-1]
        dh = F.pad(dh, (0, 0, 0, 1))
        dw = F.pad(dw, (0, 1))
        div_c = dh + dw
        ch_divs.append(div_c.pow(2).mean().item())
    return np.array(ch_divs, dtype=np.float32)


def _build_spatial_snapshot(step, v_orig, v_corr, diff, adapter):
    """
    构建一个去噪步骤的完整空间快照 (全部转到 CPU numpy float32 省显存)。
    
    Args:
        step: 当前去噪步
        v_orig: [B, C, T, H, W] float  — adapter 校正前
        v_corr: [B, C, T, H, W] float  — adapter 校正后
        diff:   v_corr - v_orig
        adapter: PhysicsAdapter 实例（读取 _cache 中间特征）
    
    Returns:
        dict 包含各种空间 tensor (numpy, 去掉 batch 维)
    """
    import numpy as np
    
    b = 0  # 只取 batch 0
    
    # 校正强度图: 对 C 通道取均值 → [T, H, W]
    correction_map = diff[b].abs().mean(dim=0).cpu().numpy().astype(np.float32)

    # 自监督动态区域：速度时序变化 + 空间梯度能量
    if v_orig.shape[2] > 1:
        temporal = (v_orig[b:b + 1, :, 1:] - v_orig[b:b + 1, :, :-1]).abs().mean(dim=1)  # [1, T-1, H, W]
        temporal = F.pad(temporal, (0, 0, 0, 0, 0, 1))[0]
    else:
        temporal = torch.zeros_like(v_orig[b, 0])
    if v_orig.shape[3] > 1:
        grad_h = (v_orig[b:b + 1, :, :, 1:, :] - v_orig[b:b + 1, :, :, :-1, :]).abs().mean(dim=1)
        grad_h = F.pad(grad_h, (0, 0, 0, 1, 0, 0))[0]
    else:
        grad_h = torch.zeros_like(v_orig[b, 0])
    if v_orig.shape[4] > 1:
        grad_w = (v_orig[b:b + 1, :, :, :, 1:] - v_orig[b:b + 1, :, :, :, :-1]).abs().mean(dim=1)
        grad_w = F.pad(grad_w, (0, 1, 0, 0, 0, 0))[0]
    else:
        grad_w = torch.zeros_like(v_orig[b, 0])
    motion_map = 0.65 * temporal + 0.35 * (0.5 * (grad_h + grad_w))
    motion_den = torch.quantile(motion_map.reshape(-1), 0.9) + 1e-6
    motion_map = torch.clamp(motion_map / motion_den, 0.0, 1.0)
    motion_mask_map = motion_map.cpu().numpy().astype(np.float32)
    
    # 逐通道校正幅度 [C]
    channel_correction = diff[b].abs().mean(dim=(1, 2, 3)).cpu().numpy().astype(np.float32)
    
    # 散度场 [T, H, W]
    div_before = _compute_divergence_field(v_orig)[b, 0].cpu().numpy().astype(np.float32)
    div_after  = _compute_divergence_field(v_corr)[b, 0].cpu().numpy().astype(np.float32)
    
    # 涡量场 [T, H-1, W-1]
    vor_before = _compute_vorticity_field(v_orig)[b, 0].cpu().numpy().astype(np.float32)
    vor_after  = _compute_vorticity_field(v_corr)[b, 0].cpu().numpy().astype(np.float32)
    
    # 逐通道散度
    ch_div_before = _compute_channel_divergence(v_orig)
    ch_div_after  = _compute_channel_divergence(v_corr)
    
    snap = {
        "step": step,
        "correction_map": correction_map,          # [T, H, W]
        "motion_mask_map": motion_mask_map,        # [T, H, W]
        "motion_weighted_correction_map": (correction_map * motion_mask_map).astype(np.float32),
        "channel_correction": channel_correction,   # [C]
        "div_field_before": div_before,             # [T, H, W]
        "div_field_after": div_after,               # [T, H, W]
        "vor_field_before": vor_before,             # [T, H-1, W-1]
        "vor_field_after": vor_after,               # [T, H-1, W-1]
        "channel_div_before": ch_div_before,        # [C]
        "channel_div_after": ch_div_after,          # [C]
    }
    
    # adapter 内部 physics_encoder 激活 (如果有缓存)
    if hasattr(adapter, "_cache") and "physics_feat" in adapter._cache:
        feat = adapter._cache["physics_feat"].float()  # [B, hidden_dim, T, H, W]
        snap["encoder_activation"] = feat[b].cpu().numpy().astype(np.float32)
    
    # adapter 内部 raw_correction (未乘 scale)
    if hasattr(adapter, "_cache") and "raw_correction" in adapter._cache:
        raw = adapter._cache["raw_correction"].float()  # [B, C, T, H, W]
        raw_map = raw[b].abs().mean(dim=0).cpu().numpy().astype(np.float32)
        snap["raw_correction_map"] = raw_map
        snap["motion_weighted_correction_map"] = (raw_map * motion_mask_map).astype(np.float32)
    
    return snap
