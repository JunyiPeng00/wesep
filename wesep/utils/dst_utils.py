from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from wesep.modules.common.speaker import PreEmphasis
from wesep.utils.score import cal_SISNRi

try:
    from wespeaker.models.speaker_model import get_speaker_model as _get_speaker_model
except ModuleNotFoundError:
    _get_speaker_model = None


def set_deterministic_seed(seed: int, *, warn_only: bool = True) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except TypeError:
        torch.use_deterministic_algorithms(True)


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        self.optimizer = optimizer
        self.warmup_steps = max(int(warmup_steps), 0)
        self.total_steps = max(int(total_steps), 1)
        self.min_lr = float(min_lr)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_step = -1

    def _lr_scale(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)
        if self.total_steps <= self.warmup_steps:
            return 1.0
        progress = float(step - self.warmup_steps) / float(
            self.total_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self, step: int) -> float:
        self.last_step = int(step)
        scale = self._lr_scale(self.last_step)
        current_lr = 0.0
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            target_lr = self.min_lr + (base_lr - self.min_lr) * scale
            group["lr"] = target_lr
            current_lr = target_lr
        return current_lr


def build_adamw_param_groups(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups: list[dict[str, Any]] = []
    if decay_params:
        param_groups.append(
            {"params": decay_params, "lr": float(lr), "weight_decay": float(weight_decay)}
        )
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "lr": float(lr), "weight_decay": 0.0})
    return param_groups


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def maybe_autocast(enabled: bool, *, dtype: torch.dtype = torch.bfloat16):
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def compute_batch_si_sdri(
    estimates: torch.Tensor,
    targets: torch.Tensor,
    mixtures: torch.Tensor,
) -> float:
    total = 0.0
    count = 0
    estimates_np = estimates.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    mixtures_np = mixtures.detach().cpu().numpy()
    for est, tgt, mix in zip(estimates_np, targets_np, mixtures_np):
        _, sisdri = cal_SISNRi(est, tgt, mix)
        total += float(sisdri)
        count += 1
    return total / max(count, 1)


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_file_logger(log_path: str | os.PathLike[str], *, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("[ %(levelname)s : %(asctime)s ] - %(message)s")
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


class FrozenSpeakerEncoder(nn.Module):
    def __init__(
        self,
        *,
        model_name: str = "ECAPA_TDNN_GLOB_c512",
        ckpt_path: str = "",
        embed_dim: int = 192,
        feat_dim: int = 80,
        pooling_func: str = "ASTP",
        sample_rate: int = 16000,
        win: int = 512,
        stride: int = 128,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if _get_speaker_model is None:
            raise ModuleNotFoundError(
                "wespeaker is required for FrozenSpeakerEncoder but is not installed"
            )

        self.embed_dim = int(embed_dim)
        self.freeze = bool(freeze)
        self.pre_emphasis = PreEmphasis()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=win,
            win_length=win,
            hop_length=stride,
            f_min=20,
            window_fn=torch.hamming_window,
            n_mels=feat_dim,
        )
        self.encoder = _get_speaker_model(model_name)(
            embed_dim=embed_dim,
            feat_dim=feat_dim,
            pooling_func=pooling_func,
        )
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state = self.encoder.state_dict()
            for key in state.keys():
                if key in checkpoint:
                    state[key] = checkpoint[key]
            self.encoder.load_state_dict(state, strict=False)

        if self.freeze:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            self.eval()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if wav.ndim != 2:
            raise ValueError(f"Expected wav [B, T], got {tuple(wav.shape)}")
        feat = self.pre_emphasis(wav)
        feat = self.mel(feat) + 1.0e-8
        feat = feat.log()
        feat = feat - feat.mean(dim=-1, keepdim=True)
        feat = feat.transpose(1, 2).contiguous()
        embedding = self.encoder(feat)
        if isinstance(embedding, tuple):
            embedding = embedding[-1]
        return embedding


def resolve_gender_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        logging.warning("speaker metadata file does not exist: %s", source)
        return {}
    mapping: dict[str, str] = {}
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        parts = [part.strip() for part in stripped.split("|")]
        if len(parts) >= 2 and parts[0].isdigit():
            mapping[parts[0]] = parts[1].upper()
    return mapping
