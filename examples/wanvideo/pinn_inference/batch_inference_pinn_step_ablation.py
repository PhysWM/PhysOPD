"""
批量视频生成脚本 - 根据CSV中的caption按顺序生成视频

支持 ID 范围，便于多卡并行：
    CUDA_VISIBLE_DEVICES=0 python batch_inference_pinn.py --start_id 1 --end_id 421 ...
    CUDA_VISIBLE_DEVICES=1 python batch_inference_pinn.py --start_id 422 --end_id 842 ...

输出命名: 0001.mp4, 0002.mp4, ... (与CSV行号一一对应)
"""
import argparse
import csv
import sys
import os
import json
import re
import hashlib
import time
from pathlib import Path
from typing import Any, Iterable

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_pinn import PhysicsInformedWanVideoPipeline
from diffsynth.pipelines.wan_video_new import ModelConfig
from auto_label_utils import (
    PromptPhysicsLabelInferer,
    PromptVideoPromptRefiner,
    build_minimal_label_metadata,
    prompt_preview,
)


PHENOMENON_LABELS = [
    # 10 physics-based categories aligned with Table 1
    "Rigid Body", "Elastic", "Fluid", "Compressible Flow", "Phase Change",
    "Collision/Contact", "Granular", "Fracture", "Thermal", "Optical",
]
PHENOMENON_TO_ID = {name: idx for idx, name in enumerate(PHENOMENON_LABELS)}
PHENOMENON_NAME_LOOKUP = {name.lower(): name for name in PHENOMENON_LABELS}
PHENOMENON_ALIAS = {}
DEFAULT_LLM_MODEL = "gpt-5.4"
DEFAULT_LLM_BASE_URL = "http://14.103.68.46/v1/chat/completions"
DEFAULT_ROUTER_SAMPLE_IDS = [
    1, 2, 3, 4, 5,
    26, 27, 28, 29, 30,
    41, 42, 43, 44, 45,
    46, 47, 48, 49, 50,
    56, 57, 58, 59, 60,
    131, 132, 133, 134, 135,
]


def parse_excluded_expert_names(text):
    if text is None:
        return []
    if isinstance(text, str):
        parts = [
            part.strip()
            for part in text.replace(";", ",").split(",")
            if part.strip()
        ]
    else:
        parts = [str(part).strip() for part in text if str(part).strip()]
    lookup = {label.casefold(): label for label in PHENOMENON_LABELS}
    unknown = [part for part in parts if part.casefold() not in lookup]
    if unknown:
        raise ValueError(f"Unknown excluded expert names: {unknown}. Known experts: {PHENOMENON_LABELS}")
    normalized = []
    seen = set()
    for part in parts:
        canonical = lookup[part.casefold()]
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def allowed_phenomenon_labels(excluded_expert_names):
    excluded = set(parse_excluded_expert_names(excluded_expert_names))
    return [label for label in PHENOMENON_LABELS if label not in excluded]


def _label_ids_from_value(value):
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, str):
        value = _parse_vector(value, cast_type=int) or []
    elif not isinstance(value, (list, tuple)):
        value = [value]
    ids = []
    for item in value:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return ids


def _label_ids_from_names(text):
    ids = []
    for part in str(text or "").split(","):
        canonical = PHENOMENON_NAME_LOOKUP.get(part.strip().lower())
        if canonical in PHENOMENON_TO_ID:
            ids.append(PHENOMENON_TO_ID[canonical])
    return ids


def sanitize_metadata_labels(metadata, excluded_expert_names, default_label="Fluid"):
    if not isinstance(metadata, dict):
        return metadata
    excluded_names = parse_excluded_expert_names(excluded_expert_names)
    if not excluded_names:
        return metadata
    excluded_ids = {PHENOMENON_TO_ID[name] for name in excluded_names}
    allowed_ids = [idx for idx in range(len(PHENOMENON_LABELS)) if idx not in excluded_ids]
    if not allowed_ids:
        raise ValueError("All physics experts were excluded; at least one expert must remain available.")
    fallback_id = PHENOMENON_TO_ID.get(default_label, allowed_ids[0])
    if fallback_id not in allowed_ids:
        fallback_id = allowed_ids[0]

    candidate_ids = _label_ids_from_value(metadata.get("label_ids"))
    if not candidate_ids:
        candidate_ids = _label_ids_from_value(metadata.get("label_id"))
    if not candidate_ids:
        candidate_ids = _label_ids_from_names(metadata.get("label_name", metadata.get("label", "")))

    filtered = []
    seen = set()
    for label_id in candidate_ids:
        if label_id < 0 or label_id >= len(PHENOMENON_LABELS):
            continue
        if label_id in excluded_ids or label_id in seen:
            continue
        filtered.append(label_id)
        seen.add(label_id)
    if not filtered:
        filtered = [fallback_id]

    sanitized = dict(metadata)
    sanitized["label_ids"] = filtered
    sanitized["label_id"] = int(filtered[0])
    sanitized["label_name"] = ", ".join(PHENOMENON_LABELS[idx] for idx in filtered)
    return sanitized


def cuda_synchronize_if_needed(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def reset_cuda_peak_memory_if_needed(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def cuda_memory_gb(device):
    if not (torch.cuda.is_available() and str(device).startswith("cuda")):
        return None, None
    return (
        float(torch.cuda.max_memory_allocated()) / 1e9,
        float(torch.cuda.max_memory_reserved()) / 1e9,
    )


def append_performance_record(path, record: dict[str, Any]):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def performance_record_exists(
    path,
    sample_id: int,
    benchmark_repeat: int | None = None,
    benchmark_phase: str | None = None,
) -> bool:
    if path is None or not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(record.get("sample_id", -1)) != int(sample_id):
                    continue
                if benchmark_repeat is not None and record.get("benchmark_repeat") != benchmark_repeat:
                    continue
                if benchmark_phase is not None and record.get("benchmark_phase") != benchmark_phase:
                    continue
                return True
    except OSError:
        return False
    return False


def _safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def _default_inference_llm_api_key():
    """Use the same direct default key as inference_pinn.py without duplicating it."""
    inference_path = Path(__file__).with_name("inference_pinn.py")
    try:
        text = inference_path.read_text(encoding="utf-8")
    except OSError:
        return None
    marker = 'parser.add_argument("--llm_api_key"'
    start = text.find(marker)
    if start < 0:
        return None
    end = text.find(")", start)
    snippet = text[start:end if end > start else start + 500]
    default_marker = 'default="'
    default_start = snippet.find(default_marker)
    if default_start < 0:
        return None
    default_start += len(default_marker)
    default_end = snippet.find('"', default_start)
    if default_end < 0:
        return None
    value = snippet[default_start:default_end].strip()
    return value or None


def _parse_vector(text, cast_type=float):
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        return [cast_type(v) for v in text]
    text = str(text).strip()
    if text == "":
        return None
    items = [it.strip() for it in text.split(",") if it.strip() != ""]
    if not items:
        return None
    return [cast_type(it) for it in items]


def _normalize_label(label):
    """Normalize label: strip whitespace and standardize spacing."""
    clean = _safe_text(label)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _hash_to_id(text, modulo):
    if modulo <= 1:
        return 0
    text = _safe_text(text).lower()
    if text == "":
        return 0
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    stable_hash = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (stable_hash % (modulo - 1)) + 1


def _parse_numeric_range(text):
    text = _safe_text(text).lower()
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(matches) == 0:
        return 0.0, 0.0, 0.0, 0.0
    values = [float(x) for x in matches]
    min_val = min(values)
    max_val = max(values)
    mean_val = sum(values) / max(len(values), 1)
    return min_val, max_val, mean_val, 1.0


def _encode_q_field(text, dim):
    vec = [0.0 for _ in range(max(dim, 0))]
    text = _safe_text(text).lower()
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


def _is_encoded_metadata(metadata):
    if not isinstance(metadata, dict):
        return False
    encoded_keys = {"label_id", "label_ids", "n_numeric", "n_text_ids", "q_vector"}
    return any(key in metadata for key in encoded_keys)


def _normalize_encoded_metadata(metadata):
    normalized = dict(metadata)
    if "label_id" in normalized and _safe_text(normalized["label_id"]) != "":
        normalized["label_id"] = int(normalized["label_id"])
    if isinstance(normalized.get("label_ids"), str):
        normalized["label_ids"] = _parse_vector(normalized.get("label_ids"), cast_type=int)
    if isinstance(normalized.get("n_numeric"), str):
        normalized["n_numeric"] = _parse_vector(normalized.get("n_numeric"), cast_type=float)
    if isinstance(normalized.get("n_text_ids"), str):
        normalized["n_text_ids"] = _parse_vector(normalized.get("n_text_ids"), cast_type=int)
    if isinstance(normalized.get("q_vector"), str):
        normalized["q_vector"] = _parse_vector(normalized.get("q_vector"), cast_type=float)
    return normalized


def encode_raw_metadata(raw_metadata, n_text_vocab_size=2048, q_dim=64):
    if not isinstance(raw_metadata, dict):
        return None
    label_name = _normalize_label(raw_metadata.get("label", raw_metadata.get("label_name", "")))
    label_id = raw_metadata.get("label_id")
    if label_id is None or _safe_text(label_id) == "":
        label_ids = []
        for part in label_name.split(","):
            part = part.strip()
            canonical_part = PHENOMENON_NAME_LOOKUP.get(part.lower())
            if canonical_part in PHENOMENON_TO_ID:
                label_ids.append(PHENOMENON_TO_ID[canonical_part])
        if not label_ids:
            if label_name != "":
                raise ValueError(f"Unknown inference label metadata: {label_name!r}")
            label_ids = [PHENOMENON_TO_ID["Fluid"]]
        label_id = int(label_ids[0])
    else:
        if isinstance(label_id, str):
            label_ids = [int(x.strip()) for x in label_id.split(",") if x.strip()]
            if not label_ids:
                label_ids = [PHENOMENON_TO_ID["Fluid"]]
        else:
            label_ids = [int(label_id)]
        label_id = int(label_ids[0])

    n_raw_0 = raw_metadata.get("n0", raw_metadata.get("n1", ""))
    n_raw_1 = raw_metadata.get("n1", raw_metadata.get("n2", ""))
    n_raw_2 = raw_metadata.get("n2", raw_metadata.get("n3", ""))

    n_numeric = []
    for value in (n_raw_0, n_raw_1, n_raw_2):
        n_min, n_max, n_mean, n_valid = _parse_numeric_range(value)
        n_numeric.extend([n_min, n_max, n_mean, n_valid])

    n_text_ids = [
        _hash_to_id(n_raw_0, n_text_vocab_size),
        _hash_to_id(n_raw_1, n_text_vocab_size),
        _hash_to_id(n_raw_2, n_text_vocab_size),
    ]

    q_vector = [0.0 for _ in range(q_dim)]
    for key in ("q0", "q1", "q2", "q4"):
        encoded = _encode_q_field(raw_metadata.get(key, ""), q_dim)
        q_vector = [min(1.0, qv + ev) for qv, ev in zip(q_vector, encoded)]
    q3 = _safe_text(raw_metadata.get("q3", "")).lower()
    if q_dim > 0:
        if q3 in {"yes", "true", "1"}:
            q_vector[0] = 1.0
        elif q3 in {"no", "false", "0"} and q_dim > 1:
            q_vector[1] = 1.0

    return {
        "label_name": label_name,
        "label_id": label_id,
        "label_ids": label_ids,
        "n_numeric": n_numeric,
        "n_text_ids": n_text_ids,
        "q_vector": q_vector,
    }


def extract_prompt_from_row(row):
    if not isinstance(row, dict):
        return ""
    prompt = _safe_text(row.get("prompt", ""))
    if prompt.startswith('"') and prompt.endswith('"'):
        prompt = prompt[1:-1]
    if prompt != "":
        return prompt
    caption = _safe_text(row.get("caption", ""))
    if caption.startswith('"') and caption.endswith('"'):
        caption = caption[1:-1]
    return caption


def source_sample_id_from_row(row, fallback_id: int):
    if not isinstance(row, dict):
        return int(fallback_id)
    for key in ("source_sample_id", "original_sample_id", "phygenbench_id", "source_id"):
        value = _safe_text(row.get(key, ""))
        if value == "":
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return int(fallback_id)


def build_wan_model_configs(model_id):
    text_encoder = ModelConfig(
        model_id=model_id,
        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
        offload_device="cpu",
    )
    vae = ModelConfig(
        model_id=model_id,
        origin_file_pattern="Wan2.1_VAE.pth",
        offload_device="cpu",
    )

    if "Wan2.1" in model_id:
        return [
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
            ),
            text_encoder,
            vae,
        ], "single_expert"

    return [
        ModelConfig(
            model_id=model_id,
            origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
            offload_device="cpu",
        ),
        ModelConfig(
            model_id=model_id,
            origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
            offload_device="cpu",
        ),
        text_encoder,
        vae,
    ], "dual_expert"


def metadata_from_row(row):
    if not isinstance(row, dict):
        return None, "none"
    if _is_encoded_metadata(row):
        return _normalize_encoded_metadata(row), "encoded_row"
    raw_keys = {"label", "label_name", "n0", "n1", "n2", "n3", "q0", "q1", "q2", "q3", "q4"}
    has_raw = any(_safe_text(row.get(key, "")) != "" for key in raw_keys)
    if not has_raw:
        return None, "none"
    return encode_raw_metadata(row), "raw_row_encoded"


def load_global_metadata(metadata_json=None, metadata_csv=None):
    if metadata_json is None and metadata_csv is None:
        return None, "none"
    if metadata_json is not None:
        if os.path.exists(metadata_json):
            with open(metadata_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            payload = json.loads(metadata_json)
    else:
        with open(metadata_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            payload = next(reader, None)
            if payload is None:
                raise ValueError(f"metadata_csv is empty: {metadata_csv}")
    if _is_encoded_metadata(payload):
        return _normalize_encoded_metadata(payload), "encoded_global"
    return encode_raw_metadata(payload), "raw_global_encoded"


def load_rows_from_csv(csv_path: str) -> list[dict]:
    """读取CSV，返回每行字典（顺序与CSV一致，不含header）。"""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def scan_existing_ids(output_dir: Path) -> set[int]:
    """扫描输出目录内已存在的视频 ID（解析文件名数字部分）"""
    existing_ids: set[int] = set()
    for path in output_dir.glob("*.mp4"):
        stem = path.stem
        if stem.isdigit():
            existing_ids.add(int(stem))
    return existing_ids


def parse_sample_ids(text):
    text = _safe_text(text)
    if not text:
        return None
    if text == "router30":
        return list(DEFAULT_ROUTER_SAMPLE_IDS)
    items = text.replace(",", " ").split()
    return [int(item) for item in items]


def _tensor_to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            if value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                return int(value.item())
            return float(value.float().item())
        if value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            return value.tolist()
        return value.float().tolist()
    if isinstance(value, (list, tuple)):
        return [_tensor_to_python(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _tensor_to_python(item) for key, item in value.items()}
    return value


def _dense_expert_weights(cache: dict[str, Any], num_experts: int) -> list[list[float]]:
    indices = cache.get("active_expert_indices")
    weights = cache.get("active_expert_weights")
    if not isinstance(indices, torch.Tensor) or not isinstance(weights, torch.Tensor):
        return []
    indices = indices.detach().long().cpu()
    weights = weights.detach().float().cpu()
    if indices.ndim == 1:
        indices = indices.unsqueeze(0)
    if weights.ndim == 1:
        weights = weights.unsqueeze(0)
    dense = torch.zeros((indices.shape[0], num_experts), dtype=torch.float32)
    valid = (indices >= 0) & (indices < num_experts)
    safe_indices = indices.clamp(0, max(num_experts - 1, 0))
    dense.scatter_add_(1, safe_indices, weights * valid.float())
    return dense.tolist()


def _dense_cache_vectors(cache: dict[str, Any], key: str, num_experts: int) -> list[list[float]]:
    value = cache.get(key)
    if not isinstance(value, torch.Tensor):
        return []
    value = value.detach().float().cpu()
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[-1] != num_experts:
        return []
    return value.tolist()


class RouterTraceHook:
    """Capture real PhysicsAdapter router cache without changing the pipeline."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.original_forward = adapter.forward
        self.records: list[dict[str, Any]] = []
        self.sample_context: dict[str, Any] = {}
        self.step = 0
        self.num_experts = int(getattr(adapter, "num_phenomena", len(PHENOMENON_LABELS)))
        self.installed = False

    def install(self):
        if self.installed:
            return

        def traced_forward(*args, **kwargs):
            result = self.original_forward(*args, **kwargs)
            self.step += 1
            cache = getattr(self.adapter, "_cache", {})
            if isinstance(cache, dict):
                self.records.append(self._record_from_cache(cache, kwargs))
            return result

        self.adapter.forward = traced_forward
        self.installed = True

    def start_sample(self, context: dict[str, Any]):
        self.sample_context = dict(context)
        self.step = 0

    def _record_from_cache(self, cache: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        dense_batch = _dense_expert_weights(cache, self.num_experts)
        first_dense = dense_batch[0] if dense_batch else [0.0 for _ in range(self.num_experts)]
        route_logits_batch = _dense_cache_vectors(cache, "route_logits", self.num_experts)
        first_route_logits = route_logits_batch[0] if route_logits_batch else []
        return {
            **self.sample_context,
            "step": int(self.step),
            "sigma": _tensor_to_python(kwargs.get("sigma")),
            "label_ids": _tensor_to_python(cache.get("label_ids")),
            "active_label_ids": _tensor_to_python(cache.get("active_label_ids")),
            "active_expert_indices": _tensor_to_python(cache.get("active_expert_indices")),
            "active_expert_weights": _tensor_to_python(cache.get("active_expert_weights")),
            "router_topk_weights": _tensor_to_python(cache.get("router_topk_weights")),
            "route_logits": _tensor_to_python(cache.get("route_logits")),
            "route_logits_10d": [float(x) for x in first_route_logits],
            "route_logits_10d_batch": route_logits_batch,
            "router_logits_10d": [float(x) for x in first_route_logits],
            "expert_weights_10d": [float(x) for x in first_dense],
            "expert_weights_10d_batch": dense_batch,
            "phenomenon_labels": PHENOMENON_LABELS,
        }


def collapse_router_forward_records(records: list[dict[str, Any]], expected_steps: int) -> list[dict[str, Any]]:
    if expected_steps <= 0 or len(records) <= expected_steps:
        return records
    if len(records) % expected_steps != 0:
        return records
    group_size = len(records) // expected_steps
    collapsed = []
    for step_idx in range(expected_steps):
        chunk = records[step_idx * group_size:(step_idx + 1) * group_size]
        base = dict(chunk[-1])
        vectors = [row.get("expert_weights_10d", []) for row in chunk]
        if all(isinstance(vec, list) and len(vec) == len(PHENOMENON_LABELS) for vec in vectors):
            mean_vec = [
                float(sum(vec[i] for vec in vectors) / len(vectors))
                for i in range(len(PHENOMENON_LABELS))
            ]
            base["expert_weights_10d"] = mean_vec
            base["expert_weights_10d_batch"] = [mean_vec]
        logits_vectors = [row.get("route_logits_10d", []) for row in chunk]
        if all(isinstance(vec, list) and len(vec) == len(PHENOMENON_LABELS) for vec in logits_vectors):
            mean_logits = [
                float(sum(vec[i] for vec in logits_vectors) / len(logits_vectors))
                for i in range(len(PHENOMENON_LABELS))
            ]
            base["route_logits_10d"] = mean_logits
            base["route_logits_10d_batch"] = [mean_logits]
            base["router_logits_10d"] = mean_logits
        base["step"] = step_idx + 1
        base["forward_calls_collapsed"] = group_size
        base["raw_forward_steps"] = [row.get("step") for row in chunk]
        collapsed.append(base)
    return collapsed


def append_router_trace(path: Path, records: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="批量生成视频 - 从CSV读取caption，支持ID范围多卡并行"
    )
    # CSV 与 范围
    parser.add_argument(
        "--csv",
        type=str,
        default="videophy_test_public.csv",
        help="CSV 文件路径（需包含 caption 列）",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        required=True,
        help="起始 ID（1-based，对应 CSV 第2行）",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        required=True,
        help="结束 ID（1-based，含该行）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_videos",
        help="输出目录",
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        default=None,
        help="逗号/空格分隔的 1-based CSV ID；设置为 router30 使用四类物理均衡子集。",
    )

    # 模型
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/train/pinn_plugin_low_noise/pinn_plugin_final.pt",
        help="PINN checkpoint 路径",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B",
        help="Model ID",
    )

    # 生成参数
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
        help="Negative prompt",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0, help="固定随机种子，与单条 inference 保持一致")
    parser.add_argument(
        "--seed_base",
        dest="seed",
        type=int,
        help="兼容旧参数名；当前语义已改为固定随机种子，等价于 --seed",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument(
        "--observable_inspection_only",
        action="store_true",
        help="Run full Wan inference but only record encoder observable diagnostics; do not apply adapter correction to v_original.",
    )
    parser.add_argument(
        "--pinn_step_range",
        type=str,
        default=None,
        help="1-indexed inclusive denoising steps where PINN correction is active, e.g. 41-50, 31-50, 1-50, none.",
    )
    parser.add_argument(
        "--moe_top_k",
        type=int,
        default=None,
        help="Override checkpoint MoE active expert count at runtime. Allows 0.",
    )
    parser.add_argument(
        "--excluded_expert_names",
        type=str,
        default="",
        help='Comma-separated expert names to exclude from routing and auto labels, e.g. "Granular,Fracture".',
    )
    parser.add_argument(
        "--performance_metrics_path",
        type=str,
        default=None,
        help="JSONL path for per-sample generation/save/time/memory records. Default: output_dir/performance_metrics.jsonl.",
    )
    parser.add_argument("--benchmark_name", type=str, default=None, help="Optional benchmark tag written to performance records.")
    parser.add_argument("--benchmark_repeat", type=int, default=None, help="Optional repeat id written to performance records.")
    parser.add_argument(
        "--benchmark_phase",
        type=str,
        default="measure",
        choices=("measure", "warmup"),
        help="Benchmark phase for selected samples. Warmup samples are usually passed via --benchmark_warmup_sample_ids.",
    )
    parser.add_argument(
        "--benchmark_warmup_sample_ids",
        type=str,
        default="1",
        help="Optional comma/space separated 1-based CSV IDs to run before measured samples in the same loaded model process.",
    )

    # 设备
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--metadata_json",
        type=str,
        default=None,
        help="Global metadata JSON string or JSON file path (raw or encoded).",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default=None,
        help="Global metadata CSV path (raw or encoded); first row will be used.",
    )
    parser.add_argument(
        "--auto_label_from_prompt",
        action="store_true",
        help="Call an OpenAI-compatible LLM API to infer routing labels when row/global metadata is absent.",
    )
    parser.add_argument("--llm_model", type=str, default=DEFAULT_LLM_MODEL, help="LLM model name. Falls back to OPENAI_MODEL or LLM_MODEL.")
    parser.add_argument("--llm_base_url", type=str, default=DEFAULT_LLM_BASE_URL, help="OpenAI-compatible base URL. Falls back to OPENAI_BASE_URL or LLM_BASE_URL.")
    parser.add_argument("--llm_api_key", type=str, default=None, help="Optional direct API key. Prefer env vars for normal batch runs.")
    parser.add_argument("--llm_api_key_env", type=str, default="OPENAI_API_KEY", help="Environment variable name that stores the API key.")
    parser.add_argument("--llm_timeout", type=float, default=30.0, help="LLM request timeout in seconds.")
    parser.add_argument("--llm_max_retries", type=int, default=2, help="Maximum retry count for LLM requests.")
    parser.add_argument(
        "--disable_prompt_refinement",
        action="store_true",
        help="Disable LLM prompt refinement and use the raw CSV caption directly.",
    )
    parser.add_argument("--skip_existing", action="store_true", help="若输出文件已存在则跳过")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="遇到单条推理失败时继续后续样本。默认关闭，失败直接中止。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：扫描输出目录，仅生成范围内缺失的 ID",
    )
    parser.add_argument("--export_router_trace", action="store_true", help="导出真实 PINN adapter router trace JSONL。")
    parser.add_argument("--router_trace_only", action="store_true", help="只保存 router trace，不保存视频文件。")
    parser.add_argument("--router_trace_path", type=str, default=None, help="router trace JSONL 输出路径；默认 output_dir/router_trace.jsonl。")
    parser.add_argument("--append_router_trace", action="store_true", help="允许向已有 router trace JSONL 追加。")

    args = parser.parse_args()
    excluded_expert_names = parse_excluded_expert_names(args.excluded_expert_names)
    allowed_labels = allowed_phenomenon_labels(excluded_expert_names)
    if args.moe_top_k is not None:
        if args.moe_top_k < 0:
            print(f"Error: --moe_top_k must be >= 0, got {args.moe_top_k}")
            sys.exit(1)
        if args.moe_top_k > len(allowed_labels):
            print(
                f"Error: --moe_top_k={args.moe_top_k} exceeds available experts "
                f"after exclusions ({len(allowed_labels)}): {allowed_labels}"
            )
            sys.exit(1)
    export_router_trace = bool(args.export_router_trace or args.router_trace_only)
    if export_router_trace and args.llm_api_key is None:
        args.llm_api_key = _default_inference_llm_api_key()

    # 解析路径
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    performance_metrics_path = (
        Path(args.performance_metrics_path)
        if args.performance_metrics_path
        else output_dir / "performance_metrics.jsonl"
    )
    if not performance_metrics_path.is_absolute():
        performance_metrics_path = project_root / performance_metrics_path
    performance_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    router_trace_path = None
    if export_router_trace:
        router_trace_path = Path(args.router_trace_path) if args.router_trace_path else output_dir / "router_trace.jsonl"
        if not router_trace_path.is_absolute():
            router_trace_path = project_root / router_trace_path
        router_trace_path.parent.mkdir(parents=True, exist_ok=True)
        if router_trace_path.exists() and not args.append_router_trace:
            print(f"Error: router trace already exists: {router_trace_path}")
            print("Use --append_router_trace or choose a new --router_trace_path/--output_dir.")
            sys.exit(1)

    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)

    # 读取 caption
    rows = load_rows_from_csv(str(csv_path))
    total = len(rows)
    if total == 0:
        print("Error: No captions found in CSV")
        sys.exit(1)

    sample_ids = parse_sample_ids(args.sample_ids)
    warmup_sample_ids = parse_sample_ids(args.benchmark_warmup_sample_ids)
    if sample_ids:
        invalid_ids = [vid_id for vid_id in sample_ids if vid_id < 1 or vid_id > total]
        if invalid_ids:
            print(f"Error: sample_ids out of range [1, {total}]: {invalid_ids}")
            sys.exit(1)
        selected_id_list = list(dict.fromkeys(sample_ids))
        start_id = min(selected_id_list)
        end_id = max(selected_id_list)
    else:
        start_id = max(1, args.start_id)
        end_id = min(total, args.end_id)
        if start_id > end_id:
            print(f"Error: start_id ({start_id}) > end_id ({end_id}) or out of range [1, {total}]")
            sys.exit(1)
        selected_id_list = list(range(start_id, end_id + 1))
    if warmup_sample_ids:
        invalid_warmup_ids = [vid_id for vid_id in warmup_sample_ids if vid_id < 1 or vid_id > total]
        if invalid_warmup_ids:
            print(f"Error: benchmark_warmup_sample_ids out of range [1, {total}]: {invalid_warmup_ids}")
            sys.exit(1)

    print("=" * 80)
    print(f"Batch PINN Step-Ablation Inference: IDs {start_id}-{end_id} / {total} total")
    if sample_ids:
        print(f"Selected sample IDs: {selected_id_list}")
    if warmup_sample_ids:
        print(f"Benchmark warmup sample IDs: {warmup_sample_ids}")
    if args.benchmark_name is not None or args.benchmark_repeat is not None:
        print(
            f"Benchmark: name={args.benchmark_name}, repeat={args.benchmark_repeat}, "
            f"phase={args.benchmark_phase}"
        )
    print(f"CSV: {csv_path}")
    print(f"Output: {output_dir}")
    print(f"PINN step range: {args.pinn_step_range or 'all'}")
    print(f"MoE top-k override: {args.moe_top_k if args.moe_top_k is not None else 'checkpoint default'}")
    print(f"Excluded experts: {excluded_expert_names if excluded_expert_names else 'none'}")
    print(f"Allowed auto-label experts: {allowed_labels}")
    print(f"Performance metrics: {performance_metrics_path}")
    if export_router_trace:
        print(f"Router trace: {router_trace_path}")
    print("=" * 80)

    # 1. 加载模型（只加载一次）
    print("\n[1/3] Loading model...")
    model_configs, loader_mode = build_wan_model_configs(args.model_id)
    print(f"  Model loader mode: {loader_mode}")
    pipe = PhysicsInformedWanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=model_configs,
    )
    pipe.enable_vram_management()

    print(f"[2/3] Loading PINN plugin from {args.checkpoint_path}...")
    pipe.load_pinn_plugin(
        args.checkpoint_path,
        device=args.device,
        enable_tracking=True,
        observable_inspection_only=args.observable_inspection_only,
        moe_top_k_override=args.moe_top_k,
        excluded_expert_names=excluded_expert_names,
        pinn_step_range=args.pinn_step_range,
    )
    router_hook = None
    if export_router_trace:
        if pipe.physics_adapter is None:
            raise RuntimeError("Cannot export router trace because physics_adapter is not loaded.")
        router_hook = RouterTraceHook(pipe.physics_adapter)
        router_hook.install()
        print("  Router trace hook installed on PhysicsAdapter.forward")
    global_metadata, global_mode = load_global_metadata(
        metadata_json=args.metadata_json,
        metadata_csv=args.metadata_csv,
    )
    global_metadata = sanitize_metadata_labels(global_metadata, excluded_expert_names)
    if global_metadata is not None:
        print(f"  Using global PINN metadata (mode={global_mode}).")
    prompt_refiner = None
    if not args.disable_prompt_refinement:
        prompt_refiner = PromptVideoPromptRefiner(
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            api_key_env=args.llm_api_key_env,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
        )
    llm_inferer = None
    if args.auto_label_from_prompt:
        llm_inferer = PromptPhysicsLabelInferer(
            allowed_labels,
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            api_key_env=args.llm_api_key_env,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
            default_label="Fluid",
        )
    effective_moe_top_k = (
        int(getattr(pipe.physics_adapter, "moe_top_k"))
        if pipe.physics_adapter is not None and hasattr(pipe.physics_adapter, "moe_top_k")
        else args.moe_top_k
    )
    effective_excluded_expert_names = (
        list(getattr(pipe.physics_adapter, "excluded_expert_names", excluded_expert_names))
        if pipe.physics_adapter is not None
        else list(excluded_expert_names)
    )

    ids_to_process: Iterable[int]
    if args.resume:
        existing_ids = scan_existing_ids(output_dir)
        ids_to_process = [i for i in selected_id_list if i not in existing_ids]
        print(
            f"[3/3] Resume enabled: existing {len(existing_ids)} files, "
            f"remaining {len(ids_to_process)} in range"
        )
        if not ids_to_process:
            print("All done in range. Nothing to generate.")
            return
    else:
        ids_to_process = selected_id_list

    run_items = [("warmup", int(vid_id)) for vid_id in warmup_sample_ids]
    run_items.extend((args.benchmark_phase, int(vid_id)) for vid_id in ids_to_process)

    print(f"[3/3] Generating videos {start_id}-{end_id}...")
    if warmup_sample_ids:
        print(f"    Warmup runs: {len(warmup_sample_ids)}; measured runs: {len(list(ids_to_process))}")
    for benchmark_phase, vid_id in run_items:
        benchmark_is_warmup = benchmark_phase == "warmup"
        idx = vid_id - 1  # 0-based index
        row = rows[idx]
        caption = extract_prompt_from_row(row)
        source_sample_id = source_sample_id_from_row(row, vid_id)
        out_name = f"warmup_{vid_id:04d}.mp4" if benchmark_is_warmup else f"{vid_id:04d}.mp4"
        out_path = output_dir / out_name

        if args.skip_existing and out_path.exists():
            phase_text = "warmup" if benchmark_is_warmup else "measure"
            print(f"  [{vid_id:4d}/{total}] ({phase_text}) Skip (exists): {out_name}")
            lookup_repeat = args.benchmark_repeat if args.benchmark_name is not None or args.benchmark_repeat is not None else None
            lookup_phase = benchmark_phase if args.benchmark_name is not None or args.benchmark_repeat is not None or benchmark_is_warmup else None
            if not performance_record_exists(performance_metrics_path, vid_id, lookup_repeat, lookup_phase):
                append_performance_record(performance_metrics_path, {
                    "sample_id": int(vid_id),
                    "source_sample_id": int(source_sample_id),
                    "output_path": str(out_path),
                    "skipped": True,
                    "success": True,
                    "error": None,
                    "benchmark_name": args.benchmark_name,
                    "benchmark_repeat": args.benchmark_repeat,
                    "benchmark_phase": benchmark_phase,
                    "warmup": bool(benchmark_is_warmup),
                    "model_id": args.model_id,
                    "checkpoint_path": str(args.checkpoint_path),
                    "pinn_step_range": args.pinn_step_range or "all",
                    "requested_moe_top_k": args.moe_top_k,
                    "moe_top_k": effective_moe_top_k,
                    "excluded_expert_names": effective_excluded_expert_names,
                    "gpu_id": os.getenv("CUDA_VISIBLE_DEVICES", ""),
                    "cuda_device_index": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
                    "generation_seconds": None,
                    "save_seconds": None,
                    "total_seconds": None,
                    "peak_memory_allocated_gb": None,
                    "peak_memory_reserved_gb": None,
                })
            continue

        seed = args.seed
        original_caption = caption
        effective_prompt = caption
        prompt_refinement = None
        if prompt_refiner is not None:
            prompt_refinement = prompt_refiner.refine(caption)
            effective_prompt = prompt_refinement.get("refined_prompt") or caption
        pinn_metadata = global_metadata
        metadata_mode = global_mode
        if pinn_metadata is None:
            pinn_metadata, metadata_mode = metadata_from_row(row)
        llm_result = None
        if pinn_metadata is None and llm_inferer is not None:
            llm_result = llm_inferer.infer(effective_prompt)
            pinn_metadata = build_minimal_label_metadata(
                llm_result["labels"],
                PHENOMENON_TO_ID,
                default_label="Fluid",
            )
            metadata_mode = "llm_auto_label"
        pinn_metadata = sanitize_metadata_labels(pinn_metadata, excluded_expert_names)
        phase_text = "warmup" if benchmark_is_warmup else "measure"
        print(f"  [{vid_id:4d}/{total}] ({phase_text}) {original_caption[:60]}{'...' if len(original_caption) > 60 else ''} -> {out_name}")
        if prompt_refinement is not None:
            if prompt_refinement.get("used_refinement"):
                print(
                    f"    refined prompt: '{prompt_preview(effective_prompt, limit=160)}'"
                )
            elif prompt_refinement.get("error"):
                print(f"    prompt refine warning: {prompt_refinement['error']}")
        if pinn_metadata is not None:
            print(f"    metadata mode: {metadata_mode}")
            print(
                f"    routing labels: {pinn_metadata.get('label_ids')} "
                f"({pinn_metadata.get('label_name', '')})"
            )
        if llm_result is not None:
            print(
                f"    llm prompt='{prompt_preview(effective_prompt)}', raw_labels={llm_result['raw_labels']}, "
                f"final_labels={llm_result['labels']}, fallback={llm_result['fallback_used']}"
            )
            if llm_result.get("error"):
                print(f"    llm warning: {llm_result['error']}")

        total_start = time.perf_counter()
        generation_seconds = None
        save_seconds = None
        total_seconds = None
        peak_allocated_gb = None
        peak_reserved_gb = None
        try:
            pipe.reset_tracking()
            cuda_synchronize_if_needed(args.device)
            reset_cuda_peak_memory_if_needed(args.device)
            trace_start = len(router_hook.records) if router_hook is not None else 0
            if router_hook is not None:
                router_hook.start_sample({
                    "sample_id": int(vid_id),
                    "source_sample_id": int(source_sample_id),
                    "benchmark_name": args.benchmark_name,
                    "benchmark_repeat": args.benchmark_repeat,
                    "benchmark_phase": benchmark_phase,
                    "warmup": bool(benchmark_is_warmup),
                    "prompt": original_caption,
                    "effective_prompt": effective_prompt,
                    "metadata_mode": metadata_mode,
                    "metadata": pinn_metadata,
                    "llm_result": llm_result if isinstance(llm_result, dict) else None,
                    "llm_labels": llm_result.get("labels") if isinstance(llm_result, dict) else None,
                    "model_id": args.model_id,
                    "checkpoint_path": str(args.checkpoint_path),
                    "pinn_step_range": args.pinn_step_range or "all",
                    "requested_moe_top_k": args.moe_top_k,
                    "moe_top_k": effective_moe_top_k,
                    "excluded_expert_names": effective_excluded_expert_names,
                    "num_inference_steps": int(args.num_inference_steps),
                    "seed": int(seed),
                })
            generation_start = time.perf_counter()
            video = pipe(
                prompt=effective_prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                seed=seed,
                cfg_scale=args.cfg_scale,
                tiled=True,
                pinn_metadata=pinn_metadata,
            )
            cuda_synchronize_if_needed(args.device)
            generation_seconds = time.perf_counter() - generation_start
            if router_hook is not None and router_trace_path is not None:
                trace_records = router_hook.records[trace_start:]
                trace_records = collapse_router_forward_records(trace_records, args.num_inference_steps)
                append_router_trace(router_trace_path, trace_records)
                if len(trace_records) != args.num_inference_steps:
                    print(
                        f"    router trace warning: wrote {len(trace_records)} rows, "
                        f"expected {args.num_inference_steps}"
                    )
            save_start = time.perf_counter()
            if not args.router_trace_only:
                save_video(video, str(out_path), fps=args.fps, quality=args.quality)
                cuda_synchronize_if_needed(args.device)
                save_seconds = time.perf_counter() - save_start
            else:
                save_seconds = 0.0
            total_seconds = time.perf_counter() - total_start
            peak_allocated_gb, peak_reserved_gb = cuda_memory_gb(args.device)
            append_performance_record(performance_metrics_path, {
                "sample_id": int(vid_id),
                "source_sample_id": int(source_sample_id),
                "output_path": str(out_path),
                "skipped": False,
                "success": True,
                "error": None,
                "benchmark_name": args.benchmark_name,
                "benchmark_repeat": args.benchmark_repeat,
                "benchmark_phase": benchmark_phase,
                "warmup": bool(benchmark_is_warmup),
                "prompt": original_caption,
                "effective_prompt": effective_prompt,
                "metadata_mode": metadata_mode,
                "metadata_label_ids": pinn_metadata.get("label_ids") if isinstance(pinn_metadata, dict) else None,
                "metadata_label_name": pinn_metadata.get("label_name") if isinstance(pinn_metadata, dict) else None,
                "model_id": args.model_id,
                "checkpoint_path": str(args.checkpoint_path),
                "pinn_step_range": args.pinn_step_range or "all",
                "requested_moe_top_k": args.moe_top_k,
                "moe_top_k": effective_moe_top_k,
                "excluded_expert_names": effective_excluded_expert_names,
                "num_inference_steps": int(args.num_inference_steps),
                "seed": int(seed),
                "height": int(args.height),
                "width": int(args.width),
                "num_frames": int(args.num_frames),
                "fps": int(args.fps),
                "gpu_id": os.getenv("CUDA_VISIBLE_DEVICES", ""),
                "cuda_device_index": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
                "generation_seconds": generation_seconds,
                "save_seconds": save_seconds,
                "total_seconds": total_seconds,
                "peak_memory_allocated_gb": peak_allocated_gb,
                "peak_memory_reserved_gb": peak_reserved_gb,
            })
        except Exception as e:
            try:
                cuda_synchronize_if_needed(args.device)
                peak_allocated_gb, peak_reserved_gb = cuda_memory_gb(args.device)
            except Exception:
                pass
            total_seconds = time.perf_counter() - total_start
            append_performance_record(performance_metrics_path, {
                "sample_id": int(vid_id),
                "source_sample_id": int(source_sample_id),
                "output_path": str(out_path),
                "skipped": False,
                "success": False,
                "error": str(e),
                "benchmark_name": args.benchmark_name,
                "benchmark_repeat": args.benchmark_repeat,
                "benchmark_phase": benchmark_phase,
                "warmup": bool(benchmark_is_warmup),
                "prompt": original_caption,
                "effective_prompt": effective_prompt,
                "metadata_mode": metadata_mode,
                "metadata_label_ids": pinn_metadata.get("label_ids") if isinstance(pinn_metadata, dict) else None,
                "metadata_label_name": pinn_metadata.get("label_name") if isinstance(pinn_metadata, dict) else None,
                "model_id": args.model_id,
                "checkpoint_path": str(args.checkpoint_path),
                "pinn_step_range": args.pinn_step_range or "all",
                "requested_moe_top_k": args.moe_top_k,
                "moe_top_k": effective_moe_top_k,
                "excluded_expert_names": effective_excluded_expert_names,
                "num_inference_steps": int(args.num_inference_steps),
                "seed": int(seed),
                "height": int(args.height),
                "width": int(args.width),
                "num_frames": int(args.num_frames),
                "fps": int(args.fps),
                "gpu_id": os.getenv("CUDA_VISIBLE_DEVICES", ""),
                "cuda_device_index": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
                "generation_seconds": generation_seconds,
                "save_seconds": save_seconds,
                "total_seconds": total_seconds,
                "peak_memory_allocated_gb": peak_allocated_gb,
                "peak_memory_reserved_gb": peak_reserved_gb,
            })
            print(f"  [{vid_id:4d}/{total}] ERROR: {e}")
            if args.continue_on_error:
                continue
            raise

    print("\n" + "=" * 80)
    print(f"Done: {start_id}-{end_id}")
    print("=" * 80)


if __name__ == "__main__":
    main()
