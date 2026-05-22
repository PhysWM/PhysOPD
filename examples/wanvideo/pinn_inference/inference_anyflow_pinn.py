#!/usr/bin/env python3
"""
Run a PILA PhysicsAdapter on top of an AnyFlow Wan2.1 bidirectional model.

This script intentionally keeps the AnyFlow sampler, transformer, and VAE intact.
The only intervention is:

    AnyFlow transformer velocity -> PILA PhysicsAdapter -> AnyFlow scheduler step

It is meant for transfer diagnostics, not for training.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from einops import rearrange
from tqdm import tqdm


DEFAULT_ANYFLOW_ROOT = "/home/dataset-assist-0/algorithm/cong.wang/try/AnyFlow"
DEFAULT_PILA_ROOT = "/home/dataset-assist-0/algorithm/cong.wang/try/PILA_MoE"
DEFAULT_PINN_CKPT = (
    "/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/"
    "models/train/wan21_stage2_fullpinn8/step-18500.pt"
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_pila_adapter_modules(pila_root):
    """Load only the PILA adapter files without importing the full diffsynth package."""
    diffsynth_pkg = types.ModuleType("diffsynth")
    models_pkg = types.ModuleType("diffsynth.models")
    diffsynth_pkg.__path__ = [str(Path(pila_root) / "diffsynth")]
    models_pkg.__path__ = [str(Path(pila_root) / "diffsynth" / "models")]
    sys.modules.setdefault("diffsynth", diffsynth_pkg)
    sys.modules.setdefault("diffsynth.models", models_pkg)

    contracts = _load_module(
        "diffsynth.models.pinn_contracts",
        str(Path(pila_root) / "diffsynth" / "models" / "pinn_contracts.py"),
    )
    adapter_mod = _load_module(
        "diffsynth.models.pinn_adapter",
        str(Path(pila_root) / "diffsynth" / "models" / "pinn_adapter.py"),
    )
    return contracts, adapter_mod


def load_auto_label_utils(pila_root):
    return _load_module(
        "pila_anyflow_auto_label_utils",
        str(Path(pila_root) / "examples" / "wanvideo" / "pinn_inference" / "auto_label_utils.py"),
    )


def safe_text(value):
    return "" if value is None else str(value).strip()


def hash_to_id(text, modulo):
    if modulo <= 1:
        return 0
    text = safe_text(text).lower()
    if text == "":
        return 0
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], byteorder="big", signed=False) % (modulo - 1)) + 1


def parse_numeric_range(text):
    import re

    matches = re.findall(r"-?\d+(?:\.\d+)?", safe_text(text).lower())
    if not matches:
        return 0.0, 0.0, 0.0, 0.0
    values = [float(x) for x in matches]
    return min(values), max(values), sum(values) / len(values), 1.0


def encode_q_field(text, dim):
    import re

    vec = [0.0 for _ in range(dim)]
    text = safe_text(text).lower()
    if text == "" or dim <= 0:
        return vec
    tokens = re.split(r"[,;|/]| and |\.", text)
    for token in [re.sub(r"\s+", " ", t).strip() for t in tokens if t.strip()]:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:8], byteorder="big", signed=False) % dim
        vec[idx] = 1.0
    return vec


def load_metadata(metadata_json, phenomenon_labels, n_text_vocab_size=2048, q_dim=64):
    if metadata_json is None:
        metadata_json = {
            "label": "Rigid Body",
            "n0": "velocity 1 to 4",
            "n1": "mass 0.5 to 2",
            "n2": "friction 0.3 to 0.7",
            "q0": "sliding, collision",
            "q1": "rigid transform",
            "q2": "stable shape and contact",
            "q3": "yes",
            "q4": "rigid body motion",
        }
    elif isinstance(metadata_json, dict):
        metadata_json = dict(metadata_json)
    elif os.path.exists(metadata_json):
        with open(metadata_json, "r", encoding="utf-8") as f:
            metadata_json = json.load(f)
    else:
        metadata_json = json.loads(metadata_json)

    if "n_numeric" in metadata_json and "q_vector" in metadata_json:
        encoded = dict(metadata_json)
    else:
        label_lookup = {name: idx for idx, name in enumerate(phenomenon_labels)}
        label_ids = metadata_json.get("label_ids")
        if label_ids is None:
            label_id = metadata_json.get("label_id")
            label_ids = [int(label_id)] if label_id is not None else None
        labels = [
            part.strip()
            for part in safe_text(
                metadata_json.get("label", metadata_json.get("label_name", "Rigid Body"))
            ).split(",")
            if part.strip()
        ]
        if label_ids is None:
            label_ids = [label_lookup[x] for x in labels if x in label_lookup]
        else:
            label_ids = [int(x) for x in label_ids]
        if not label_ids:
            label_ids = [label_lookup.get("Rigid Body", 0)]
        if not labels:
            inverse_lookup = {idx: name for name, idx in label_lookup.items()}
            labels = [inverse_lookup.get(int(label_ids[0]), "Rigid Body")]

        n_numeric = []
        n_raw = [
            metadata_json.get("n0", ""),
            metadata_json.get("n1", ""),
            metadata_json.get("n2", ""),
        ]
        for value in n_raw:
            n_numeric.extend(parse_numeric_range(value))

        q_vector = [0.0 for _ in range(q_dim)]
        for key in ("q0", "q1", "q2", "q4"):
            q_encoded = encode_q_field(metadata_json.get(key, ""), q_dim)
            q_vector = [min(1.0, a + b) for a, b in zip(q_vector, q_encoded)]
        q3 = safe_text(metadata_json.get("q3", "")).lower()
        if q3 in {"yes", "true", "1"} and q_dim > 0:
            q_vector[0] = 1.0
        elif q3 in {"no", "false", "0"} and q_dim > 1:
            q_vector[1] = 1.0

        encoded = {
            "label_name": labels[0] if labels else "Rigid Body",
            "label_id": int(label_ids[0]),
            "label_ids": [int(x) for x in label_ids],
            "n_numeric": n_numeric,
            "n_text_ids": [hash_to_id(x, n_text_vocab_size) for x in n_raw],
            "q_vector": q_vector,
        }

    return {
        "label_id": int(encoded.get("label_id", 0)),
        "label_ids": [int(x) for x in encoded.get("label_ids", [encoded.get("label_id", 0)])],
        "n_numeric": torch.tensor(encoded["n_numeric"], dtype=torch.float32),
        "n_text_ids": torch.tensor(encoded["n_text_ids"], dtype=torch.long),
        "q_vector": torch.tensor(encoded["q_vector"], dtype=torch.float32),
        "label_name": encoded.get("label_name", ""),
    }


def resolve_prompt_and_metadata(args, phenomenon_labels):
    auto_utils = load_auto_label_utils(args.pila_root)
    prompt_preview = auto_utils.prompt_preview
    phenomenon_to_id = {name: idx for idx, name in enumerate(phenomenon_labels)}

    original_prompt = args.prompt
    effective_prompt = original_prompt
    if args.disable_prompt_refinement:
        print("[prompt] refinement disabled; using original prompt")
    else:
        refiner = auto_utils.PromptVideoPromptRefiner(
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            api_key_env=args.llm_api_key_env,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
        )
        result = refiner.refine(original_prompt)
        effective_prompt = result.get("refined_prompt") or original_prompt
        if result.get("used_refinement"):
            print(
                "[prompt] refined via LLM: "
                f"original='{prompt_preview(original_prompt, limit=100)}', "
                f"refined='{prompt_preview(effective_prompt, limit=160)}'"
            )
        else:
            print("[prompt] LLM refinement unavailable or unchanged; using original prompt")
            if result.get("error"):
                print(f"[prompt] refinement warning: {result['error']}")

    metadata_mode = "default_rigid"
    if args.metadata_json is not None:
        metadata = load_metadata(args.metadata_json, phenomenon_labels)
        metadata_mode = "manual_metadata"
    elif args.auto_label_from_prompt:
        inferer = auto_utils.PromptPhysicsLabelInferer(
            phenomenon_labels,
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            api_key_env=args.llm_api_key_env,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
            default_label=args.default_label,
        )
        label_result = inferer.infer(effective_prompt)
        minimal_metadata = auto_utils.build_minimal_label_metadata(
            label_result["labels"],
            phenomenon_to_id,
            default_label=args.default_label,
        )
        metadata = load_metadata(minimal_metadata, phenomenon_labels)
        metadata_mode = "llm_auto_label"
        print(
            "[routing] auto labels via LLM: "
            f"labels={label_result['labels']}, raw={label_result['raw_labels']}, "
            f"label_ids={metadata.get('label_ids')}, fallback={label_result['fallback_used']}"
        )
        if label_result.get("error"):
            print(f"[routing] label warning: {label_result['error']}")
    else:
        metadata = load_metadata(None, phenomenon_labels)
        print("[routing] no metadata/auto-label requested; using default Rigid Body routing")

    print(
        f"[routing] metadata_mode={metadata_mode}, "
        f"label_name={metadata.get('label_name')}, label_ids={metadata.get('label_ids')}"
    )
    return effective_prompt, metadata, metadata_mode


def state_shape(state_dict, key):
    value = state_dict.get(key)
    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


def infer_adapter_arch(state_dict, default_num_phenomena, default_attr_dim):
    encoder_shape = state_shape(state_dict, "physics_encoder_shared.0.conv1.weight")
    if encoder_shape is None:
        raise RuntimeError("Missing physics_encoder_shared.0.conv1.weight in adapter checkpoint.")
    arch = {
        "latent_dim": int(encoder_shape[1]),
        "hidden_dim": int(encoder_shape[0]),
        "num_phenomena": default_num_phenomena,
        "n_numeric_dim": 12,
        "q_input_dim": 64,
        "n_text_vocab_size": 2048,
        "physics_attr_dim": default_attr_dim,
    }
    for key, field, axis in [
        ("shared_attribute_head.2.weight", "physics_attr_dim", 0),
        ("n_text_embedding.weight", "n_text_vocab_size", 0),
        ("q_projector.weight", "q_input_dim", 1),
        ("q_proj.0.weight", "q_input_dim", 1),
        ("n_projector.weight", "n_numeric_dim", 1),
        ("n_numeric_proj.0.weight", "n_numeric_dim", 1),
        ("expert_router.2.weight", "num_phenomena", 0),
    ]:
        shape = state_shape(state_dict, key)
        if shape is not None and len(shape) > axis:
            arch[field] = int(shape[axis])
    return arch


def load_physics_adapter(checkpoint_path, contracts, adapter_mod, device, dtype, moe_top_k_override=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    state = checkpoint.get("physics_adapter_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"No physics_adapter_state_dict in {checkpoint_path}")

    arch = infer_adapter_arch(
        state,
        default_num_phenomena=len(contracts.PHENOMENON_LABELS),
        default_attr_dim=contracts.PHYSICS_ATTR_DIM,
    )
    moe_top_k = int(config.get("moe_top_k", 4))
    if moe_top_k_override is not None:
        moe_top_k = int(moe_top_k_override)

    adapter = adapter_mod.PhysicsAdapter(
        latent_dim=int(config.get("latent_dim", arch["latent_dim"])),
        hidden_dim=int(config.get("adapter_hidden_dim", arch["hidden_dim"])),
        num_phenomena=int(config.get("num_phenomena", arch["num_phenomena"])),
        n_numeric_dim=int(config.get("n_numeric_dim", arch["n_numeric_dim"])),
        q_input_dim=int(config.get("q_input_dim", arch["q_input_dim"])),
        n_text_vocab_size=int(config.get("n_text_vocab_size", arch["n_text_vocab_size"])),
        physics_attr_dim=int(config.get("physics_attr_dim", arch["physics_attr_dim"])),
        moe_top_k=moe_top_k,
        physics_state_mode=str(config.get("physics_state_mode", "x0_hat")),
        use_sigma_gate=bool(config.get("use_sigma_gate", True)),
        sigma_gate_curve=str(config.get("sigma_gate_curve", "quadratic")),
        use_sigma_conditioning=bool(config.get("use_sigma_conditioning", True)),
        sigma_conditioning_dim=int(config.get("sigma_conditioning_dim", arch["hidden_dim"])),
        sigma_gate_floor=float(config.get("sigma_gate_floor", 0.05)),
        strict_physical_state_contract=True,
        core_ablation_mode=str(config.get("core_ablation_mode", "full")),
    )
    result = adapter.load_state_dict(state, strict=False)
    allowed_missing = {"expert_usage_ema", "rl_reward_ema"}
    missing = [x for x in result.missing_keys if x not in allowed_missing]
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        print(f"[warn] adapter load missing={missing[:10]} unexpected={unexpected[:10]}")

    if hasattr(adapter, "set_ablation_modes"):
        ablation_flags = config.get("ablation_flags", {})
        adapter.set_ablation_modes(
            use_moe=not bool(ablation_flags.get("ablate_disable_moe", False)),
            label_only_mode=bool(ablation_flags.get("ablate_label_only_router", False)),
        )
    adapter.moe_top_k = max(0, min(int(moe_top_k), int(adapter.num_phenomena)))
    adapter.to(device=device, dtype=dtype)
    adapter.eval()
    return adapter, config


def metadata_to_device(metadata, device, dtype):
    out = dict(metadata)
    out["n_numeric"] = out["n_numeric"].to(device=device, dtype=dtype)
    out["n_text_ids"] = out["n_text_ids"].to(device=device, dtype=torch.long)
    out["q_vector"] = out["q_vector"].to(device=device, dtype=dtype)
    return out


def attach_adapter_to_bidirectional_anyflow(pipe, adapter, metadata, correction_scale=1.0):
    def training_rollout_with_adapter(
        self,
        context_sequence=None,
        num_inference_steps=50,
        grad_timestep=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        guidance_scale=1.0,
    ):
        if grad_timestep is not None:
            raise NotImplementedError("This diagnostic script only supports full inference rollout.")
        self._guidance_scale = guidance_scale
        if negative_prompt_embeds is not None:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        context_length = context_sequence.shape[1] if context_sequence is not None else None
        device = self._execution_device
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_train_timesteps = float(self.scheduler.config.num_train_timesteps)
        adapter_stats = []

        for i, t in enumerate(tqdm(timesteps[:-1], desc="AnyFlow+PILA")):
            r = timesteps[i + 1]
            if t == r:
                continue

            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            timestep = t.expand(latent_model_input.shape[0]).unsqueeze(-1)
            timestep = timestep.repeat((1, latent_model_input.shape[1]))
            if self.use_mean_velocity:
                r_timestep = r.expand(latent_model_input.shape[0]).unsqueeze(-1)
                r_timestep = r_timestep.repeat((1, latent_model_input.shape[1]))
            else:
                r_timestep = timestep

            if context_sequence is not None:
                latent_model_input[:, :context_length, ...] = context_sequence
                timestep[:, :context_length] = 0

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                r_timestep=r_timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
                is_causal=False,
            )[0]

            if self.do_classifier_free_guidance:
                noise_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            sigma_scalar = (t / num_train_timesteps).to(device=latents.device, dtype=latents.dtype)
            sigma = sigma_scalar.reshape(1).repeat(noise_pred.shape[0])
            state_for_physics = latents - sigma_scalar.view(1, 1, 1, 1, 1) * noise_pred
            v_bcthw = rearrange(noise_pred, "b t c h w -> b c t h w")
            x_bcthw = rearrange(state_for_physics, "b t c h w -> b c t h w")
            corrected_bcthw = adapter(
                v_bcthw,
                x_bcthw,
                sigma=sigma,
                metadata=metadata_to_device(metadata, v_bcthw.device, v_bcthw.dtype),
            )
            if correction_scale != 1.0:
                corrected_bcthw = v_bcthw + correction_scale * (corrected_bcthw - v_bcthw)
            correction = corrected_bcthw - v_bcthw
            denom = v_bcthw.detach().float().reshape(v_bcthw.shape[0], -1).norm(dim=1).mean().clamp_min(1e-6)
            ratio = correction.detach().float().reshape(correction.shape[0], -1).norm(dim=1).mean() / denom
            adapter_stats.append(float(ratio.cpu()))
            noise_pred = rearrange(corrected_bcthw, "b c t h w -> b t c h w")

            latents = self.scheduler.step(noise_pred, latents, t, r)

        self._pila_adapter_stats = adapter_stats
        return latents

    pipe.training_rollout = types.MethodType(training_rollout_with_adapter, pipe)
    return pipe


def run_one(pipe, prompt, output_path, args, adapter=None, metadata=None):
    if adapter is not None:
        attach_adapter_to_bidirectional_anyflow(
            pipe,
            adapter=adapter,
            metadata=metadata,
            correction_scale=args.correction_scale,
        )
    generator = torch.Generator(args.device).manual_seed(args.seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    export_to_video(result.frames[0], output_video_path=output_path, fps=args.fps)
    stats = getattr(pipe, "_pila_adapter_stats", None)
    if stats:
        print(
            f"[adapter] mean correction ratio={sum(stats) / len(stats):.6f}, "
            f"max={max(stats):.6f}, steps={len(stats)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anyflow_root", default=DEFAULT_ANYFLOW_ROOT)
    parser.add_argument("--pila_root", default=DEFAULT_PILA_ROOT)
    parser.add_argument(
        "--model_path",
        default=f"{DEFAULT_ANYFLOW_ROOT}/experiments/pretrained_models/AnyFlow-Wan2.1-T2V-1.3B-Diffusers",
    )
    parser.add_argument("--pinn_checkpoint", default=DEFAULT_PINN_CKPT)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--metadata_json", default=None)
    parser.add_argument("--auto_label_from_prompt", action="store_true")
    parser.add_argument("--disable_prompt_refinement", action="store_true")
    parser.add_argument("--llm_model", default="gpt-5.4")
    parser.add_argument("--llm_base_url", default="http://35.220.164.252:3888/v1")
    parser.add_argument("--llm_api_key", default="sk-8viAj2SPNHZ4W0E4BcKSfdOwXr1xVzpcheUHDIPweBi4EEqB")
    parser.add_argument("--llm_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--llm_timeout", type=float, default=30.0)
    parser.add_argument("--llm_max_retries", type=int, default=2)
    parser.add_argument("--default_label", default="Fluid")
    parser.add_argument("--output_dir", default="outputs/anyflow_pinn_transfer")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--correction_scale", type=float, default=1.0)
    parser.add_argument("--moe_top_k", type=int, default=None)
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, args.anyflow_root)
    from far.pipelines.pipeline_wan_anyflow import WanAnyFlowPipeline

    if "AnyFlow-FAR" in args.model_path:
        raise NotImplementedError("This first diagnostic script targets bidirectional AnyFlow-Wan models.")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    os.makedirs(args.output_dir, exist_ok=True)

    contracts, adapter_mod = load_pila_adapter_modules(args.pila_root)
    effective_prompt, metadata, metadata_mode = resolve_prompt_and_metadata(args, contracts.PHENOMENON_LABELS)
    adapter, adapter_config = load_physics_adapter(
        args.pinn_checkpoint,
        contracts=contracts,
        adapter_mod=adapter_mod,
        device=args.device,
        dtype=dtype,
        moe_top_k_override=args.moe_top_k,
    )
    print(f"[adapter] config core={adapter_config.get('core_ablation_mode')} moe_top_k={adapter.moe_top_k}")
    print(f"[prompt] effective='{effective_prompt}'")

    if not args.skip_baseline:
        print("[1/2] Running AnyFlow baseline")
        pipe = WanAnyFlowPipeline.from_pretrained(args.model_path).to(args.device, dtype=dtype)
        run_one(pipe, effective_prompt, os.path.join(args.output_dir, "anyflow_baseline.mp4"), args)
        del pipe
        torch.cuda.empty_cache()

    print("[2/2] Running AnyFlow + PILA adapter")
    pipe = WanAnyFlowPipeline.from_pretrained(args.model_path).to(args.device, dtype=dtype)
    run_one(
        pipe,
        effective_prompt,
        os.path.join(args.output_dir, "anyflow_pila_adapter.mp4"),
        args,
        adapter=adapter,
        metadata=metadata,
    )
    print(f"[done] outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
