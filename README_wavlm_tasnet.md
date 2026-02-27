## WavLMTasNet in wesep

This document describes how to run and verify the WavLM+TasNet speaker-conditioned
separation model implemented in `wesep`.

### Requirements

- Python environment: `conda activate wedefense`
- Dependencies: `torch>=1.12.0`, `torchaudio>=0.12.0`

### Prepare WavLM checkpoint

The model expects a locally converted WavLM checkpoint (a `.pth` file storing
`{\"config\": ..., \"state_dict\": ...}`) compatible with the helpers in
`wesep.modules.wavlm_frontend`. You can reuse the conversion scripts already
present in your WavLM / wespeaker experiments, or adapt
`wespeaker_hubert/wespeaker/frontend/wav2vec2/utils/import_huggingface_wavlm.py`.

Once you have such a file, point the environment variable `WAVLM_CKPT` to it:

```bash
export WAVLM_CKPT=/path/to/wavlm_base_converted.pth
```

### Simple forward demo

You can also run a quick forward-pass sanity check (using random noise) by
executing:

```bash
conda activate wedefense
python -m wesep.models.wavlm_tasnet
```

Make sure `WavLMTasNetConfig.wavlm_ckpt` is set to your converted checkpoint
path inside `wavlm_tasnet.py`. The script will:

- Build the model.
- Run a forward pass on random mixtures and references.
- Print the output tensor shape.

---

## WavLMDynamicTasNet

同仓库中另一模型：内部说话人嵌入 + hybrid mask + QKb + TCN，接口为 `forward_concat(wav, enroll_len)` 或 `forward(mix_wav, enroll_wav)`。

### 随机信号示例脚本

```bash
conda activate wedefense
cd wesep
export WAVLM_CKPT=/path/to/wavlm_base.hf.pth
PYTHONPATH=. python examples/wavlm_dynamic_tasnet_rand.py
```

可选环境变量：`BATCH=2`、`LEN_SAMPLES=32000`、`BACKWARD=1`（跑一步反向）。

