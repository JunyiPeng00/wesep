from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def find_wesep_root(start: Path) -> Path:
    for candidate in (start.parent, *start.parents):
        if (candidate / "wesep").is_dir() and (candidate / "examples").is_dir():
            return candidate
    raise RuntimeError(f"Failed to locate WESEP root from {start}")


WESEP_ROOT = find_wesep_root(Path(__file__).resolve())
# Optional: a local wespeaker checkout (only needed if wespeaker is not pip-installed).
_IMPORT_ROOTS = [WESEP_ROOT]
if os.environ.get("WESPEAKER_ROOT", ""):
    _IMPORT_ROOTS.append(Path(os.environ["WESPEAKER_ROOT"]).expanduser())
for import_root in _IMPORT_ROOTS:
    if import_root.exists() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from wesep.dataset.dataset import Dataset, tse_collate_fn, tse_collate_fn_2spk
from wesep.models import get_model
from wesep.utils.dst_utils import (
    AverageMeter,
    FrozenSpeakerEncoder,
    WarmupCosineScheduler,
    build_adamw_param_groups,
    build_file_logger,
    compute_batch_si_sdri,
    maybe_autocast,
    set_deterministic_seed,
    write_json,
)
from wesep.utils.file_utils import (
    load_speaker_embeddings,
    read_label_file,
    read_vec_scp_file,
)
from wesep.utils.losses import parse_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DST-TSE experiments")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint to resume from")
    return parser.parse_args()


def read_config(path: str) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return resolve_config_paths(config, config_path)


def _resolve_existing_path(raw_path: str, config_path: Path) -> str:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    experiment_root = config_path.parents[2] if len(config_path.parents) > 2 else WESEP_ROOT
    search_roots = [
        Path.cwd(),
        WESEP_ROOT,
        experiment_root,
        config_path.parent,
        config_path.parent.parent,
    ]
    for root in search_roots:
        resolved = (root / raw_path).expanduser().resolve()
        if resolved.exists():
            return str(resolved)
    return raw_path


def resolve_config_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    if "paths" in config:
        for key, value in list(config["paths"].items()):
            if isinstance(value, str) and value:
                config["paths"][key] = _resolve_existing_path(value, config_path)

    model_cfg = config.get("model_args", {}).get("tse_model", {})
    for key in ("wavlm_ckpt", "spk_model_init"):
        value = model_cfg.get(key, "")
        if isinstance(value, str) and value:
            model_cfg[key] = _resolve_existing_path(value, config_path)

    speaker_cfg = config.get("speaker_encoder", {})
    ckpt_path = speaker_cfg.get("ckpt_path", "")
    if isinstance(ckpt_path, str) and ckpt_path:
        speaker_cfg["ckpt_path"] = _resolve_existing_path(ckpt_path, config_path)

    return config


def _setup_per_rank_miopen_cache(local_rank: int) -> None:
    """Per-rank MIOpen cache directory under /tmp (ROCm only; no effect on CUDA).

    The path contains a node and a local-rank component, so workers never share
    a cache directory. MIOpen reads these variables at the first convolution.
    """
    job_id = os.environ.get("SLURM_JOB_ID", "0")
    user = os.environ.get("USER", "user")
    node_key = (
        os.environ.get("SLURM_NODEID")
        or os.environ.get("SLURMD_NODENAME")
        or os.uname().nodename
    )
    node_key = str(node_key).replace("/", "_")
    cache_root = os.environ.get("SLURM_TMPDIR") or "/tmp"
    cache_dir = os.path.join(cache_root, f"{user}-miopen-{job_id}-node{node_key}-local{local_rank}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["MIOPEN_USER_DB_PATH"] = cache_dir
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = cache_dir


def init_distributed(config: dict[str, Any]) -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    _setup_per_rank_miopen_cache(local_rank)

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed DST-TSE training requires CUDA/HIP devices")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend=str(config.get("dist_backend", "nccl")),
            device_id=device,
        )
        return distributed, rank, local_rank, world_size, device

    requested_gpu = str(config.get("gpus", "0")).split(",")[0]
    device = torch.device(f"cuda:{requested_gpu}" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def log_info(logger: Any, message: str, *args: Any) -> None:
    if logger is not None:
        logger.info(message, *args)


def trace_rank(config: dict[str, Any], rank: int, message: str, *args: Any) -> None:
    if bool(config.get("train", {}).get("ddp_trace", False)):
        formatted = message % args if args else message
        print(f"[ddp-trace rank={rank}] {formatted}", file=sys.stderr, flush=True)


def distributed_barrier(device: torch.device) -> None:
    kwargs: dict[str, Any] = {}
    if device.type == "cuda" and device.index is not None:
        kwargs["device_ids"] = [device.index]
    dist.barrier(**kwargs)


def reduce_validation_metrics(
    metrics: dict[str, float],
    *,
    device: torch.device,
    distributed: bool,
) -> dict[str, float]:
    if not distributed:
        return metrics

    if device.type == "cuda":
        torch.cuda.set_device(device)
    count = float(metrics.get("val_count", 0.0))
    payload = torch.tensor(
        [
            float(metrics["val_loss"]) * count,
            float(metrics["val_si_sdri"]) * count,
            count,
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    total_count = max(float(payload[2].item()), 1.0)
    return {
        "val_loss": float(payload[0].item() / total_count),
        "val_si_sdri": float(payload[1].item() / total_count),
        "val_count": total_count,
    }


def unwrap_primary_output(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def forward_model(
    model: torch.nn.Module,
    model_name: str,
    mix: torch.Tensor,
    enroll: torch.Tensor,
    external_spk_emb: torch.Tensor | None,
) -> torch.Tensor:
    if model_name.startswith("WavLMDSTBSRNN"):
        return unwrap_primary_output(model(mix, enroll, external_spk_emb=external_spk_emb))
    if model_name == "BSRNN" and external_spk_emb is not None:
        return unwrap_primary_output(model(mix, external_spk_emb))
    return unwrap_primary_output(model(mix, enroll))


def build_speaker_encoder(config: dict[str, Any]) -> FrozenSpeakerEncoder | None:
    speaker_cfg = config.get("speaker_encoder", {})
    train_cfg = config.get("train", {})
    loss_cfg = config["model_args"]["tse_model"]
    must_enable = bool(speaker_cfg.get("enabled", False)) or float(
        loss_cfg.get("lambda_p", train_cfg.get("lambda_p", 0.0))
    ) > 0.0
    if not must_enable:
        return None
    return FrozenSpeakerEncoder(
        model_name=speaker_cfg.get("model_name", "ECAPA_TDNN_GLOB_c512"),
        ckpt_path=speaker_cfg.get("ckpt_path", ""),
        embed_dim=int(speaker_cfg.get("embed_dim", 192)),
        feat_dim=int(speaker_cfg.get("feat_dim", 80)),
        pooling_func=speaker_cfg.get("pooling_func", "ASTP"),
        sample_rate=int(config["dataset_args"].get("resample_rate", 16000)),
        win=int(speaker_cfg.get("win", 512)),
        stride=int(speaker_cfg.get("stride", 128)),
        freeze=bool(speaker_cfg.get("freeze", True)),
    )


def build_train_sources(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int] | None]:
    paths = config["paths"]
    joint_training = bool(config["model_args"]["tse_model"].get("joint_training", True))
    multi_task = bool(config["model_args"]["tse_model"].get("multi_task", False))
    if not joint_training and paths.get("train_spk_embeds"):
        return load_speaker_embeddings(paths["train_spk_embeds"], paths["train_utt2spk"]), None

    with open(paths["train_spk2utt"], "r", encoding="utf-8") as handle:
        spk2enroll = json.load(handle)
    if not multi_task:
        return spk2enroll, None
    dict_spk = {speaker: index for index, speaker in enumerate(spk2enroll.keys())}
    return spk2enroll, dict_spk


def build_eval_sources(config: dict[str, Any], split: str) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    paths = config["paths"]
    joint_training = bool(config["model_args"]["tse_model"].get("joint_training", True))
    split_prefix = "val" if split == "val" else "test"
    if not joint_training and paths.get(f"{split_prefix}_spk_embeds"):
        spk_source = read_vec_scp_file(paths[f"{split_prefix}_spk_embeds"])
    else:
        spk_source = read_label_file(paths[f"{split_prefix}_spk2utt"])
    spk1_enroll = read_label_file(paths[f"{split_prefix}_spk1_enroll"])
    spk2_enroll = read_label_file(paths[f"{split_prefix}_spk2_enroll"])
    return spk_source, spk1_enroll, spk2_enroll


def build_dataset_and_loader(
    config: dict[str, Any],
    *,
    split: str,
    speaker_source: dict[str, Any],
    spk1_enroll: dict[str, str] | None = None,
    spk2_enroll: dict[str, str] | None = None,
    dict_spk: dict[str, int] | None = None,
) -> tuple[Any, DataLoader]:
    paths = config["paths"]
    joint_training = bool(config["model_args"]["tse_model"].get("joint_training", True))
    data_key = "train_data" if split == "train" else f"{split}_data"
    if split == "train":
        whole_utt = bool(config.get("whole_utt", False))
        repeat_dataset = bool(config.get("repeat_dataset", True))
    else:
        whole_utt = bool(config.get(f"{split}_whole_utt", config.get("whole_utt", False)))
        repeat_dataset = bool(config.get(f"{split}_repeat_dataset", False))
    dataset = Dataset(
        config.get("data_type", "shard"),
        paths[data_key],
        config["dataset_args"],
        speaker_source,
        spk1_enroll,
        spk2_enroll,
        state=split,
        joint_training=joint_training,
        dict_spk=dict_spk,
        whole_utt=whole_utt,
        repeat_dataset=repeat_dataset,
    )
    loader_args = dict(config["dataloader_args"])
    split_batch_size = config.get(f"{split}_batch_size")
    if split_batch_size is not None:
        loader_args["batch_size"] = int(split_batch_size)
    elif split != "train" and whole_utt:
        # Whole-utterance eval examples are naturally variable length, while
        # the existing TSE collate path concatenates tensors without padding.
        loader_args["batch_size"] = 1

    loader = DataLoader(
        dataset,
        **loader_args,
        collate_fn=tse_collate_fn if split == "train" else tse_collate_fn_2spk,
    )
    return dataset, loader


def infer_steps_per_epoch(config: dict[str, Any], batch_size: int, split: str) -> int:
    dataset_args = config.get("dataset_args", {})
    if split == "train" and int(dataset_args.get("sample_num_per_epoch", 0)) > 0:
        sample_num = int(dataset_args["sample_num_per_epoch"])
    elif split == "train":
        with open(config["paths"]["train_utt2spk"], "r", encoding="utf-8") as handle:
            sample_num = len(handle.readlines()) // 2
    else:
        with open(config["paths"][f"{split}_spk2utt"], "r", encoding="utf-8") as handle:
            sample_num = len(handle.readlines()) // 2
    return max(sample_num // max(batch_size, 1), 1)


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    global_step: int,
    epoch: int,
    best_val_si_sdri: float,
    config: dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Atomic save: write to a temp file, then os.replace, so a crash during the
    # write cannot corrupt latest.pt.
    tmp_path = f"{os.fspath(path)}.tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": {
                "base_lrs": scheduler.base_lrs,
                "warmup_steps": scheduler.warmup_steps,
                "total_steps": scheduler.total_steps,
                "min_lr": scheduler.min_lr,
                "last_step": scheduler.last_step,
            },
            "global_step": global_step,
            "epoch": epoch,
            "best_val_si_sdri": best_val_si_sdri,
            "config": config,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def load_checkpoint_if_needed(
    checkpoint_path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
) -> tuple[int, int, float]:
    if not checkpoint_path:
        return 0, 0, -1.0e9
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.last_step = int(state["scheduler"].get("last_step", -1))
    scheduler.base_lrs = list(state["scheduler"].get("base_lrs", scheduler.base_lrs))
    return (
        int(state.get("global_step", 0)),
        int(state.get("epoch", 0)),
        float(state.get("best_val_si_sdri", -1.0e9)),
    )


def compute_dst_regularizers(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if not hasattr(model, "get_last_dynamic_stats"):
        device = next(model.parameters()).device
        zero = torch.tensor(0.0, device=device)
        return {"compute": zero, "stability": zero, "mean_gate": zero, "mean_lstar": zero, "attn_flops": zero}

    stats = model.get_last_dynamic_stats(detach=False, to_cpu=False)
    gate = stats.get("gate_probs")
    if gate is None:
        device = next(model.parameters()).device
        zero = torch.tensor(0.0, device=device)
        return {"compute": zero, "stability": zero, "mean_gate": zero, "mean_lstar": zero, "attn_flops": zero}

    cfg = config["model_args"]["tse_model"]
    gate = gate.clamp(1.0e-6, 1.0 - 1.0e-6)
    expected_depth = stats["expected_joint_depth"]
    compute_loss = expected_depth.mean() / float(getattr(model, "num_layers", gate.size(1)))
    entropy = -(gate * torch.log(gate) + (1.0 - gate) * torch.log(1.0 - gate)).mean()
    smoothness = (
        torch.square(gate[:, 1:] - gate[:, :-1]).mean()
        if gate.size(1) > 1
        else torch.tensor(0.0, device=gate.device)
    )
    stability = entropy + float(cfg.get("beta", 0.1)) * smoothness
    mean_lstar = (
        stats["hard_lstar"].float().mean()
        if "hard_lstar" in stats
        else torch.tensor(0.0, device=gate.device)
    )
    attn_flops = (
        stats["attn_flops"].float().mean()
        if "attn_flops" in stats
        else torch.tensor(0.0, device=gate.device)
    )
    return {
        "compute": compute_loss,
        "stability": stability,
        "mean_gate": gate.mean(),
        "mean_lstar": mean_lstar,
        "attn_flops": attn_flops,
    }


def compute_speaker_consistency_loss(
    speaker_encoder: FrozenSpeakerEncoder | None,
    estimates: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if speaker_encoder is None:
        return estimates.new_tensor(0.0)
    estimate_emb = F.normalize(speaker_encoder(estimates), dim=-1)
    target_emb = F.normalize(speaker_encoder(targets).detach(), dim=-1)
    return 1.0 - F.cosine_similarity(estimate_emb, target_emb, dim=-1).mean()


@torch.no_grad()
def run_validation(
    *,
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    sisdr_loss: torch.nn.Module,
    speaker_encoder: FrozenSpeakerEncoder | None,
    config: dict[str, Any],
    autocast_enabled: bool,
    rank: int = 0,
) -> dict[str, float]:
    model.eval()
    if speaker_encoder is not None:
        speaker_encoder.eval()

    loss_meter = AverageMeter()
    sisdri_meter = AverageMeter()
    max_batches = int(config.get("train", {}).get("val_batches", 0) or 0)
    trace_interval = int(config.get("train", {}).get("ddp_trace_val_interval", 0) or 0)
    for batch_idx, batch in enumerate(loader):
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        mix = batch["wav_mix"].float()
        target = batch["wav_targets"].float()
        enroll = batch["spk_embeds"].float()

        external_spk_emb = None
        if speaker_encoder is not None and bool(config.get("speaker_encoder", {}).get("use_for_separator", False)):
            external_spk_emb = speaker_encoder(enroll)

        with maybe_autocast(autocast_enabled):
            outputs = forward_model(model, model_name, mix, enroll, external_spk_emb)
            loss = sisdr_loss(outputs, target).mean()
            if float(config["model_args"]["tse_model"].get("lambda_p", 0.0)) > 0.0:
                loss = loss + float(config["model_args"]["tse_model"].get("lambda_p", 0.0)) * compute_speaker_consistency_loss(
                    speaker_encoder,
                    outputs,
                    target,
                )
        loss_meter.update(loss.item(), target.size(0))
        sisdri_meter.update(compute_batch_si_sdri(outputs, target, mix), target.size(0))
        if trace_interval > 0 and (batch_idx + 1) % trace_interval == 0:
            trace_rank(
                config,
                rank,
                "validation_batch=%d val_count=%d",
                batch_idx + 1,
                loss_meter.count,
            )
        if max_batches > 0 and batch_idx + 1 >= max_batches:
            break

    model.train()
    if speaker_encoder is not None:
        speaker_encoder.train()
    return {
        "val_loss": loss_meter.average,
        "val_si_sdri": sisdri_meter.average,
        "val_count": float(loss_meter.count),
    }


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    distributed, rank, local_rank, world_size, device = init_distributed(config)
    exp_dir = Path(config["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = build_file_logger(exp_dir / "train.log", name=f"dst-train-{exp_dir.name}") if rank == 0 else None
    writer = SummaryWriter(log_dir=exp_dir / "tensorboard") if rank == 0 else None
    if rank == 0:
        write_json(exp_dir / "config.resolved.json", config)
        with open(exp_dir / "config.resolved.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
    if distributed:
        distributed_barrier(device)

    seed = int(config.get("seed", 42))
    set_deterministic_seed(seed + rank)
    log_info(
        logger,
        "using device %s rank=%d local_rank=%d world_size=%d",
        device,
        rank,
        local_rank,
        world_size,
    )

    trace_rank(config, rank, "setup_sources_start")
    train_source, dict_spk = build_train_sources(config)
    val_source, val_spk1, val_spk2 = build_eval_sources(config, "val")
    trace_rank(config, rank, "setup_sources_done")

    trace_rank(config, rank, "train_loader_start")
    train_dataset, train_loader = build_dataset_and_loader(
        config,
        split="train",
        speaker_source=train_source,
        dict_spk=dict_spk,
    )
    trace_rank(config, rank, "train_loader_done")
    trace_rank(config, rank, "val_loader_start")
    val_dataset, val_loader = build_dataset_and_loader(
        config,
        split="val",
        speaker_source=val_source,
        spk1_enroll=val_spk1,
        spk2_enroll=val_spk2,
    )
    trace_rank(config, rank, "val_loader_done")

    model_name = config["model"]["tse_model"]
    trace_rank(config, rank, "model_build_start")
    raw_model = get_model(model_name)(**config["model_args"]["tse_model"]).to(device)
    trace_rank(config, rank, "model_build_done")
    trace_rank(config, rank, "speaker_encoder_start")
    speaker_encoder = build_speaker_encoder(config)
    if speaker_encoder is not None:
        speaker_encoder = speaker_encoder.to(device)
    trace_rank(config, rank, "speaker_encoder_done")

    train_cfg = config["train"]
    # The cosine schedule must span the actual run length; when num_epochs is
    # set, the configured total_steps (e.g. 200000) is only a fallback.
    batch_size = int(config["dataloader_args"]["batch_size"])
    configured_steps_per_epoch = int(train_cfg.get("steps_per_epoch", 0) or 0)
    steps_per_epoch = configured_steps_per_epoch or infer_steps_per_epoch(config, batch_size, "train")
    if distributed and configured_steps_per_epoch <= 0:
        steps_per_epoch = max(math.ceil(steps_per_epoch / max(world_size, 1)), 1)
    configured_num_epochs = train_cfg.get("num_epochs")
    if configured_num_epochs is not None:
        total_epochs = int(configured_num_epochs)
        total_steps = max(steps_per_epoch * total_epochs, 1)
    else:
        total_steps = int(train_cfg["total_steps"])
        total_epochs = int(math.ceil(total_steps / max(steps_per_epoch, 1)))
    optimizer = torch.optim.AdamW(
        build_adamw_param_groups(
            raw_model,
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=int(train_cfg["warmup_steps"]),
        total_steps=total_steps,
        min_lr=float(train_cfg.get("min_lr", 0.0)),
    )
    sisdr_loss = parse_loss("SISDR")[0].to(device)

    global_step, start_epoch, best_val_si_sdri = load_checkpoint_if_needed(
        args.checkpoint,
        model=raw_model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    model: torch.nn.Module = raw_model
    if distributed:
        trace_rank(config, rank, "ddp_wrap_start")
        model = DistributedDataParallel(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(config.get("find_unused_parameters", True)),
        )
        trace_rank(config, rank, "ddp_wrap_done")

    log_interval = int(train_cfg.get("log_interval", 50))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 20))
    precision = str(train_cfg.get("precision", "bf16")).lower()
    autocast_enabled = precision == "bf16" and device.type == "cuda"
    no_improve_epochs = 0

    log_info(logger, "model=%s", model_name)
    log_info(
        logger,
        "trainable params=%.2f M",
        sum(p.numel() for p in raw_model.parameters() if p.requires_grad) / 1.0e6,
    )
    log_info(logger, "steps_per_epoch=%d total_steps=%d total_epochs=%d", steps_per_epoch, total_steps, total_epochs)

    for epoch in range(start_epoch + 1, total_epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        if speaker_encoder is not None:
            speaker_encoder.train()
        epoch_loss = AverageMeter()
        epoch_sisdr_loss = AverageMeter()
        epoch_dst_compute_loss = AverageMeter()
        epoch_dst_stability_loss = AverageMeter()
        epoch_spk_weighted_loss = AverageMeter()
        epoch_spk_raw_loss = AverageMeter()
        last_batch_idx = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            last_batch_idx = batch_idx
            if global_step >= total_steps:
                break

            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            mix = batch["wav_mix"].float()
            target = batch["wav_targets"].float()
            enroll = batch["spk_embeds"].float()

            if hasattr(raw_model, "set_current_step"):
                raw_model.set_current_step(global_step)

            scheduler.step(global_step)
            optimizer.zero_grad(set_to_none=True)

            external_spk_emb = None
            if speaker_encoder is not None and bool(config.get("speaker_encoder", {}).get("use_for_separator", False)):
                external_spk_emb = speaker_encoder(enroll)

            with maybe_autocast(autocast_enabled):
                outputs = forward_model(model, model_name, mix, enroll, external_spk_emb)
                sisdr_component = sisdr_loss(outputs, target).mean()
                loss = sisdr_component
                dst_regs = compute_dst_regularizers(raw_model, config)
                dst_compute_loss = outputs.new_tensor(0.0)
                dst_stability_loss = outputs.new_tensor(0.0)
                if (
                    model_name.startswith("WavLMDSTBSRNN")
                    and config["model_args"]["tse_model"].get("stage_mode") == "dst"
                    and global_step >= int(config["model_args"]["tse_model"].get("gate_warmup_steps", 10000))
                ):
                    dst_compute_loss = float(config["model_args"]["tse_model"].get("lambda_c", 0.05)) * dst_regs["compute"]
                    dst_stability_loss = float(config["model_args"]["tse_model"].get("lambda_s", 0.01)) * dst_regs["stability"]
                    loss = loss + dst_compute_loss + dst_stability_loss

                spk_loss = outputs.new_tensor(0.0)
                spk_weighted_loss = outputs.new_tensor(0.0)
                if float(config["model_args"]["tse_model"].get("lambda_p", 0.0)) > 0.0:
                    spk_loss = compute_speaker_consistency_loss(speaker_encoder, outputs, target)
                    spk_weighted_loss = float(config["model_args"]["tse_model"].get("lambda_p", 0.0)) * spk_loss
                    loss = loss + spk_weighted_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(train_cfg.get("grad_clip", 5.0)))
            optimizer.step()
            global_step += 1

            epoch_loss.update(loss.item(), mix.size(0))
            epoch_sisdr_loss.update(float(sisdr_component.item()), mix.size(0))
            epoch_dst_compute_loss.update(float(dst_compute_loss.item()), mix.size(0))
            epoch_dst_stability_loss.update(float(dst_stability_loss.item()), mix.size(0))
            epoch_spk_weighted_loss.update(float(spk_weighted_loss.item()), mix.size(0))
            epoch_spk_raw_loss.update(float(spk_loss.item()), mix.size(0))
            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_sisdr", float(sisdr_component.item()), global_step)
                writer.add_scalar("train/loss_dst_compute", float(dst_compute_loss.item()), global_step)
                writer.add_scalar("train/loss_dst_stability", float(dst_stability_loss.item()), global_step)
                writer.add_scalar("train/loss_spk_weighted", float(spk_weighted_loss.item()), global_step)
                writer.add_scalar("train/loss_spk_raw", float(spk_loss.item()), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/si_sdri_batch", compute_batch_si_sdri(outputs, target, mix), global_step)
                writer.add_scalar("train/mean_gate_prob", float(dst_regs["mean_gate"].item()), global_step)
                writer.add_scalar("train/mean_l_star", float(dst_regs["mean_lstar"].item()), global_step)
                writer.add_scalar("train/joint_flops_est", float(dst_regs["attn_flops"].item()), global_step)

            if global_step % log_interval == 0:
                log_info(
                    logger,
                    (
                        "epoch=%d step=%d batch=%d loss_total=%.4f "
                        "loss_sisdr=%.4f loss_dst_compute=%.4f "
                        "loss_dst_stability=%.4f loss_spk=%.4f "
                        "loss_spk_raw=%.4f lr=%.6e gate=%.4f l*=%.3f"
                    ),
                    epoch,
                    global_step,
                    batch_idx,
                    loss.item(),
                    float(sisdr_component.item()),
                    float(dst_compute_loss.item()),
                    float(dst_stability_loss.item()),
                    float(spk_weighted_loss.item()),
                    float(spk_loss.item()),
                    optimizer.param_groups[0]["lr"],
                    float(dst_regs["mean_gate"].item()),
                    float(dst_regs["mean_lstar"].item()),
                )
            if batch_idx >= steps_per_epoch:
                break

        trace_rank(
            config,
            rank,
            "epoch=%d train_end batch=%d global_step=%d",
            epoch,
            last_batch_idx,
            global_step,
        )
        trace_rank(config, rank, "epoch=%d validation_start", epoch)
        # Validation has uneven per-rank lengths for iterable shard datasets.
        # Use the local module directly so DDP cannot issue per-forward
        # collectives before the explicit metric all_reduce below.
        metrics = run_validation(
            model=raw_model,
            model_name=model_name,
            loader=val_loader,
            device=device,
            sisdr_loss=sisdr_loss,
            speaker_encoder=speaker_encoder,
            config=config,
            autocast_enabled=autocast_enabled,
            rank=rank,
        )
        trace_rank(
            config,
            rank,
            "epoch=%d validation_done val_count=%.0f",
            epoch,
            metrics["val_count"],
        )
        trace_rank(config, rank, "epoch=%d validation_reduce_start", epoch)
        metrics = reduce_validation_metrics(metrics, device=device, distributed=distributed)
        trace_rank(config, rank, "epoch=%d validation_reduce_done", epoch)
        if writer is not None:
            writer.add_scalar("val/loss", metrics["val_loss"], global_step)
            writer.add_scalar("val/si_sdri", metrics["val_si_sdri"], global_step)
        log_info(
            logger,
            "epoch=%d finished train_loss=%.4f val_loss=%.4f val_si_sdri=%.4f",
            epoch,
            epoch_loss.average,
            metrics["val_loss"],
            metrics["val_si_sdri"],
        )
        if epoch % 10 == 0:
            log_info(
                logger,
                (
                    "epoch=%d loss_summary total=%.4f sisdr=%.4f "
                    "dst_compute=%.4f dst_stability=%.4f spk=%.4f "
                    "spk_raw=%.4f val_loss=%.4f val_si_sdri=%.4f"
                ),
                epoch,
                epoch_loss.average,
                epoch_sisdr_loss.average,
                epoch_dst_compute_loss.average,
                epoch_dst_stability_loss.average,
                epoch_spk_weighted_loss.average,
                epoch_spk_raw_loss.average,
                metrics["val_loss"],
                metrics["val_si_sdri"],
            )

        if rank == 0:
            save_checkpoint(
                exp_dir / "models" / "latest.pt",
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                epoch=epoch,
                best_val_si_sdri=best_val_si_sdri,
                config=config,
            )
        if metrics["val_si_sdri"] > best_val_si_sdri:
            best_val_si_sdri = metrics["val_si_sdri"]
            no_improve_epochs = 0
            if rank == 0:
                save_checkpoint(
                    exp_dir / "models" / "best.pt",
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    epoch=epoch,
                    best_val_si_sdri=best_val_si_sdri,
                    config=config,
                )
        else:
            no_improve_epochs += 1

        if global_step >= total_steps:
            log_info(logger, "reached total_steps=%d, stopping", total_steps)
            break
        if no_improve_epochs >= early_stop_patience:
            log_info(logger, "early stopping after %d unimproved epochs", no_improve_epochs)
            break

    if writer is not None:
        writer.close()
    log_info(logger, "training finished at step=%d best_val_si_sdri=%.4f", global_step, best_val_si_sdri)
    if distributed:
        distributed_barrier(device)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
