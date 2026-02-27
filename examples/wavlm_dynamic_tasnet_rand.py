#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WavLMDynamicTasNet end-to-end sanity script with random signals.

Usage (set converted WavLM checkpoint path first):
    export WAVLM_CKPT=/path/to/wavlm_base.hf.pth
    cd wesep && PYTHONPATH=. python examples/wavlm_dynamic_tasnet_rand.py

Optional env vars:
    WAVLM_CKPT   Required; path to converted WavLM .pth
    BATCH        Batch size (default 2)
    LEN_SAMPLES  Samples per segment at 16 kHz (default 32000, 2 s)
    BACKWARD     If 1, run one backward step (default 0)
"""

from __future__ import annotations

import os
import sys

# Allow running from wesep repo root
if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
    sys.path.insert(0, os.getcwd())

import torch

from wesep.models.wavlm_dynamic_tasnet import (
    WavLMDynamicTasNetConfig,
    WavLMDynamicTasNet,
)


def main() -> None:
    ckpt = os.environ.get("WAVLM_CKPT", "").strip()
    if not ckpt or not os.path.isfile(ckpt):
        print("Set env WAVLM_CKPT to the path of your converted WavLM checkpoint (.pth)")
        print("Example: export WAVLM_CKPT=/path/to/wavlm_base.hf.pth")
        sys.exit(1)

    batch = int(os.environ.get("BATCH", "2"))
    num_samples = int(os.environ.get("LEN_SAMPLES", "32000"))
    do_backward = os.environ.get("BACKWARD", "0") == "1"

    cfg = WavLMDynamicTasNetConfig(
        wavlm_name="wavlm_base",
        wavlm_ckpt=ckpt,
        wavlm_frozen=True,
        encoder_dim=512,
        kernel_size=320,
        stride=160,
        sep_tcn_channels=256,
        num_tcn_layers=3,
    )
    model = WavLMDynamicTasNet(cfg)
    model.eval()

    mix_wav = torch.randn(batch, num_samples)
    enroll_wav = torch.randn(batch, num_samples)

    print("WavLMDynamicTasNet random-signal sanity check")
    print("  mix_wav:   ", mix_wav.shape)
    print("  enroll_wav:", enroll_wav.shape)

    with torch.no_grad():
        est = model(mix_wav, enroll_wav)

    print("  est:       ", est.shape)
    assert est.shape == mix_wav.shape, (est.shape, mix_wav.shape)
    assert torch.isfinite(est).all(), "Output contains NaN/Inf"

    mse = ((est - mix_wav) ** 2).mean().item()
    print("  MSE(est, mix):", round(mse, 6))

    if do_backward:
        model.train()
        mix_wav = torch.randn(1, 16000, requires_grad=True)
        enroll_wav = torch.randn(1, 16000)
        est = model(mix_wav, enroll_wav)
        loss = (est ** 2).mean()
        loss.backward()
        print("  backward: OK")

    print("done.")


if __name__ == "__main__":
    main()
