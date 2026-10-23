from __future__ import annotations

import argparse
import io
import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any
import sys

import numpy as np
import soundfile as sf
import torch
import yaml
from mir_eval.separation import bss_eval_sources
from torch.utils.data import DataLoader

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

from wesep.dataset.dataset import Dataset, tse_collate_fn_2spk
from wesep.models import get_model
from wesep.utils.dst_utils import (
    FrozenSpeakerEncoder,
    append_jsonl,
    build_file_logger,
    resolve_gender_map,
    set_deterministic_seed,
    write_json,
)
from wesep.utils.file_utils import read_label_file, read_vec_scp_file
from wesep.utils.score import cal_SISNRi

try:
    from pesq import PesqError, pesq
except ImportError:
    PesqError = None
    pesq = None

try:
    from pystoi import stoi
except ImportError:
    stoi = None


LMDB_MAP_SIZE = int(os.environ.get("EVAL_LMDB_MAP_SIZE", int(math.pow(1024, 4))))  # 1TB default (tools/make_lmdb.py); override for hosts with a virtual-memory ulimit
LMDB_COMMIT_INTERVAL = 100


class EvalAudioLmdbWriter:
    def __init__(self, path: Path, commit_interval: int = LMDB_COMMIT_INTERVAL) -> None:
        import lmdb

        self.path = path
        self.commit_interval = commit_interval
        self.env = lmdb.open(str(path), map_size=LMDB_MAP_SIZE)
        self.txn = self.env.begin(write=True)
        self.keys: list[str] = []
        self.closed = False

    def write(self, key: str, waveform: np.ndarray, sample_rate: int) -> None:
        if self.closed:
            raise RuntimeError(f"cannot write to closed LMDB audio sink: {self.path}")

        buffer = io.BytesIO()
        sf.write(
            buffer,
            np.asarray(waveform, dtype=np.float32),
            sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        self.txn.put(key.encode(), buffer.getvalue())
        self.keys.append(key)
        if len(self.keys) % self.commit_interval == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self) -> None:
        if self.closed:
            return
        self.txn.put(b"__keys__", pickle.dumps(self.keys))
        self.txn.commit()
        self.env.sync()
        self.env.close()
        self.closed = True


AUDIO_ROLES = ("est", "mix", "ref", "enroll")


def encode_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(waveform, dtype=np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def parse_save_roles(value: Any) -> list[str]:
    """eval.save_roles: list or comma string; subset of AUDIO_ROLES; default ['est']."""
    if value is None or value == "":
        return ["est"]
    roles = [r.strip().lower() for r in (value.split(",") if isinstance(value, str) else list(value))]
    roles = [r for r in roles if r]
    bad = [r for r in roles if r not in AUDIO_ROLES]
    if bad:
        raise ValueError(f"unknown eval.save_roles entries {bad}; allowed: {list(AUDIO_ROLES)}")
    if not roles:
        raise ValueError("eval.save_roles must name at least one of " + ", ".join(AUDIO_ROLES))
    return roles


class EvalAudioTarWriter:
    """All test-stage audio in ONE uncompressed tar: members '<key>/<role>.wav' + trailing 'index.json'.

    Uncompressed so members are byte-addressable via the index (offset/size) without re-reading the
    whole archive; WAV payloads are PCM_16 like the LMDB sink. A sidecar '<tar>.index.json' is also
    written next to the archive.
    """

    def __init__(self, path: Path) -> None:
        import tarfile

        self.path = path
        self.tar = tarfile.open(str(path), mode="w")
        self.entries: list[dict[str, Any]] = []
        self.closed = False

    def write(self, key: str, role: str, waveform: np.ndarray, sample_rate: int, **meta: Any) -> None:
        import tarfile

        if self.closed:
            raise RuntimeError(f"cannot write to closed tar audio sink: {self.path}")
        payload = encode_wav_bytes(waveform, sample_rate)
        member = f"{key}/{role}.wav"
        info = tarfile.TarInfo(name=member)
        info.size = len(payload)
        info.mtime = int(time.time())
        offset_data = self.tar.fileobj.tell() + tarfile.BLOCKSIZE  # data starts after the 512-byte header
        self.tar.addfile(info, io.BytesIO(payload))
        self.entries.append({
            "key": key, "role": role, "member": member, "sample_rate": int(sample_rate),
            "num_samples": int(np.asarray(waveform).shape[-1]), "offset": int(offset_data), "size": len(payload),
            **meta,
        })

    def close(self) -> None:
        import tarfile

        if self.closed:
            return
        index = json.dumps({"format": "dst-tse-eval-audio-tar/v1", "roles": AUDIO_ROLES, "entries": self.entries},
                           ensure_ascii=False, indent=1).encode()
        info = tarfile.TarInfo(name="index.json")
        info.size = len(index)
        info.mtime = int(time.time())
        self.tar.addfile(info, io.BytesIO(index))
        self.tar.close()
        Path(str(self.path) + ".index.json").write_bytes(index)
        self.closed = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DST-TSE experiments")
    parser.add_argument("--config", required=True, help="Resolved YAML config")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output_dir", default="", help="Override output directory")
    parser.add_argument("--max_samples", type=int, default=0, help="Stop after this many separated target samples")
    parser.add_argument("--wav_format", default="", help="Override eval.wav_format (tar|lmdb|wav)")
    parser.add_argument("--save_roles", default="", help="Override eval.save_roles, comma list of est,mix,ref,enroll")
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
    if not speaker_cfg.get("enabled", False) and float(
        config["model_args"]["tse_model"].get("lambda_p", 0.0)
    ) <= 0.0:
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


def build_eval_dataset(config: dict[str, Any], split: str) -> DataLoader:
    paths = config["paths"]
    joint_training = bool(config["model_args"]["tse_model"].get("joint_training", True))
    if not joint_training and paths.get(f"{split}_spk_embeds"):
        speaker_source = read_vec_scp_file(paths[f"{split}_spk_embeds"])
    else:
        speaker_source = read_label_file(paths[f"{split}_spk2utt"])
    spk1_enroll = read_label_file(paths[f"{split}_spk1_enroll"])
    spk2_enroll = read_label_file(paths[f"{split}_spk2_enroll"])
    dataset = Dataset(
        config.get("data_type", "shard"),
        paths[f"{split}_data"],
        config["dataset_args"],
        speaker_source,
        spk1_enroll,
        spk2_enroll,
        state=split,
        joint_training=joint_training,
        whole_utt=True,
        repeat_dataset=False,
    )
    return DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=tse_collate_fn_2spk)


def compute_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    sdr, _, _, _ = bss_eval_sources(reference[np.newaxis, :], estimate[np.newaxis, :], compute_permutation=False)
    return float(sdr[0])


def compute_optional_metrics(
    target: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
) -> tuple[float | None, float | None]:
    pesq_score = None
    stoi_score = None
    if pesq is not None and sample_rate in {8000, 16000}:
        try:
            pesq_score = float(
                pesq(
                    sample_rate,
                    target,
                    estimate,
                    mode="wb" if sample_rate == 16000 else "nb",
                    on_error=PesqError.RETURN_VALUES if PesqError is not None else None,
                )
            )
        except Exception:
            pesq_score = None
    if stoi is not None:
        try:
            stoi_score = float(stoi(target, estimate, fs_sig=sample_rate))
        except Exception:
            stoi_score = None
    return pesq_score, stoi_score


def summarise_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "si_snr",
        "si_sdri",
        "sdr",
        "sdri",
        "pesq",
        "stoi",
        "spk_cos",
        "l_star",
        "avg_gate_prob",
        "joint_depth_expected",
        "attn_flops",
        "peak_attn_mem_bytes",
    ]
    summary: dict[str, Any] = {"num_samples": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None
    return summary


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    set_deterministic_seed(int(config.get("seed", 42)))

    model_name = config["model"]["tse_model"]
    output_dir = Path(args.output_dir or (Path(config["exp_dir"]) / f"eval_{args.split}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_file_logger(output_dir / "eval.log", name=f"dst-eval-{output_dir.name}")

    requested_gpu = str(config.get("gpus", "0")).split(",")[0]
    device = torch.device(f"cuda:{requested_gpu}" if torch.cuda.is_available() else "cpu")
    logger.info("using device %s", device)

    model = get_model(model_name)(**config["model_args"]["tse_model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    speaker_encoder = build_speaker_encoder(config)
    if speaker_encoder is not None:
        speaker_encoder = speaker_encoder.to(device)
        speaker_encoder.eval()

    gender_map = resolve_gender_map(config.get("eval", {}).get("speaker_meta_path", ""))
    loader = build_eval_dataset(config, args.split)
    audio_dir = output_dir / "audio"
    eval_cfg = config.get("eval", {})
    save_audio = bool(eval_cfg.get("save_wav", False))
    wav_format = str(args.wav_format or eval_cfg.get("wav_format", "tar")).strip().lower()
    save_roles = parse_save_roles(args.save_roles or eval_cfg.get("save_roles"))
    audio_lmdb_writer: EvalAudioLmdbWriter | None = None
    audio_tar_writer: EvalAudioTarWriter | None = None
    if save_audio and wav_format == "tar":
        audio_tar_writer = EvalAudioTarWriter(output_dir / "audio.tar")
    elif save_audio and wav_format == "wav":
        audio_dir.mkdir(parents=True, exist_ok=True)
    elif save_audio and wav_format == "lmdb":
        audio_lmdb_writer = EvalAudioLmdbWriter(output_dir / "audio.lmdb")
    elif save_audio:
        raise ValueError(f"unsupported eval.wav_format={wav_format!r}; expected 'tar', 'lmdb' or 'wav'")
    if save_audio:
        logger.info("saving test audio: format=%s roles=%s", wav_format, save_roles)

    rows: list[dict[str, Any]] = []
    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists():
        predictions_path.unlink()

    sample_rate = int(config["dataset_args"].get("resample_rate", 16000))
    processed_samples = 0
    try:
        for batch_idx, batch in enumerate(loader):
            mix = batch["wav_mix"].float().to(device)
            target = batch["wav_targets"].float().to(device)
            enroll = batch["spk_embeds"].float().to(device)

            target_cpu = target.detach().cpu().numpy()
            mix_cpu = mix.detach().cpu().numpy()

            if target_cpu.shape[0] >= 2:
                interferers = np.stack([target_cpu[1], target_cpu[0]], axis=0)
            else:
                interferers = mix_cpu - target_cpu

            for sample_idx in range(mix.size(0)):
                mix_one = mix[sample_idx : sample_idx + 1]
                target_one = target[sample_idx : sample_idx + 1]
                enroll_one = enroll[sample_idx : sample_idx + 1]

                external_spk_emb = None
                if speaker_encoder is not None and bool(config.get("speaker_encoder", {}).get("use_for_separator", False)):
                    external_spk_emb = speaker_encoder(enroll_one)

                start_time = time.perf_counter()
                with torch.no_grad():
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    estimate = forward_model(model, model_name, mix_one, enroll_one, external_spk_emb)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start_time
                estimate_np = estimate[0].detach().cpu().numpy()
                target_np = target_one[0].detach().cpu().numpy()
                mix_np = mix_one[0].detach().cpu().numpy()
                interferer_np = interferers[sample_idx]

                si_snr, si_sdri = cal_SISNRi(estimate_np, target_np, mix_np)
                sdr = compute_sdr(target_np, estimate_np)
                mix_sdr = compute_sdr(target_np, mix_np)
                pesq_score, stoi_score = compute_optional_metrics(target_np, estimate_np, sample_rate)

                spk_cos = None
                target_interferer_spk_cos = None
                if speaker_encoder is not None:
                    estimate_emb = torch.nn.functional.normalize(speaker_encoder(estimate), dim=-1)
                    target_emb = torch.nn.functional.normalize(speaker_encoder(target_one), dim=-1)
                    interferer_emb = torch.nn.functional.normalize(
                        speaker_encoder(torch.from_numpy(interferer_np).unsqueeze(0).to(device)),
                        dim=-1,
                    )
                    spk_cos = float(torch.nn.functional.cosine_similarity(estimate_emb, target_emb, dim=-1).mean().item())
                    target_interferer_spk_cos = float(
                        torch.nn.functional.cosine_similarity(target_emb, interferer_emb, dim=-1).mean().item()
                    )

                stats = {}
                if hasattr(model, "get_last_dynamic_stats"):
                    stats = model.get_last_dynamic_stats(detach=True, to_cpu=True)

                utterance_key = batch["key"][sample_idx]
                speaker_id = batch["spk"][sample_idx]
                gender = gender_map.get(str(speaker_id))
                companion_speaker = batch["spk"][1 - sample_idx] if len(batch["spk"]) > 1 else None
                audio_key = f"{batch_idx:04d}_{utterance_key}_{speaker_id}"
                if save_audio:
                    role_audio = {
                        "est": estimate_np,
                        "mix": mix_np,
                        "ref": target_np,
                        "enroll": enroll_one[0].detach().cpu().numpy(),
                    }
                    for role in save_roles:
                        wav = role_audio[role]
                        if audio_tar_writer is not None:
                            audio_tar_writer.write(audio_key, role, wav, sample_rate, utt_id=utterance_key, spk=str(speaker_id))
                        elif wav_format == "wav":
                            suffix = ".wav" if role == "est" else f".{role}.wav"
                            sf.write(audio_dir / f"{audio_key}{suffix}", wav, sample_rate)
                        elif audio_lmdb_writer is not None:
                            audio_lmdb_writer.write(audio_key if role == "est" else f"{audio_key}/{role}", wav, sample_rate)

                row = {
                    "utt_id": utterance_key,
                    "split": args.split,
                    "model_name": model_name,
                    "si_snr": float(si_snr),
                    "si_sdri": float(si_sdri),
                    "sdr": float(sdr),
                    "sdri": float(sdr - mix_sdr),
                    "pesq": pesq_score,
                    "stoi": stoi_score,
                    "dnsmos": None,
                    "spk_cos": spk_cos,
                    "l_star": float(stats.get("hard_lstar", torch.tensor([0.0]))[0].item())
                    if "hard_lstar" in stats
                    else None,
                    "avg_gate_prob": float(stats.get("mean_gate_prob", torch.tensor([0.0]))[0].item())
                    if "mean_gate_prob" in stats
                    else None,
                    "joint_depth_expected": float(stats.get("expected_joint_depth", torch.tensor([0.0]))[0].item())
                    if "expected_joint_depth" in stats
                    else None,
                    "attn_flops": float(stats.get("attn_flops", torch.tensor([0.0]))[0].item())
                    if "attn_flops" in stats
                    else None,
                    "peak_attn_mem_bytes": float(
                        stats.get("peak_attn_mem_bytes", torch.tensor([0.0]))[0].item()
                    )
                    if "peak_attn_mem_bytes" in stats
                    else None,
                    "rtf_cpu": None,
                    "rtf_gpu": float(elapsed / max(len(mix_np) / sample_rate, 1.0e-8)),
                    "snr_mix": float(
                        10.0
                        * np.log10(
                            (np.mean(target_np**2) + 1.0e-8) / (np.mean(interferer_np**2) + 1.0e-8)
                        )
                    ),
                    "same_gender": None if gender is None or companion_speaker is None else bool(gender == gender_map.get(str(companion_speaker))),
                    "enroll_dur_s": float(enroll_one.shape[-1]) / float(sample_rate),
                    "target_interferer_spk_cos": target_interferer_spk_cos,
                }
                rows.append(row)
                append_jsonl(predictions_path, [row])
                processed_samples += 1
                logger.info(
                    "utt=%s target=%s si_sdri=%.3f l*=%.2f flops=%.1f",
                    utterance_key,
                    speaker_id,
                    row["si_sdri"],
                    row["l_star"] if row["l_star"] is not None else -1.0,
                    row["attn_flops"] if row["attn_flops"] is not None else -1.0,
                )
                if args.max_samples > 0 and processed_samples >= args.max_samples:
                    break
            if args.max_samples > 0 and processed_samples >= args.max_samples:
                break
    finally:
        if audio_lmdb_writer is not None:
            audio_lmdb_writer.close()
        if audio_tar_writer is not None:
            audio_tar_writer.close()

    summary = summarise_rows(rows)
    write_json(output_dir / "metrics.json", summary)
    logger.info("evaluation finished: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
