# In-context enrollment for target speech extraction (Libri2Mix)

This recipe trains and evaluates the in-context enrollment system of
"In-Context Enrollment for Target Speech Extraction with Pre-Trained Self-Supervised Learning Models"
(system S4 in the paper): `confs/in_context_bi_d4.yaml`, model `WavLMDSTBSRNN`
(`wesep/models/wavlm_dst_bsrnn.py`).

## Method (as implemented)

The enrollment, a 0.1 s zero segment and the mixture are concatenated and passed through
a frozen WavLM Base+. Layers 1-4 (joint depth d = 4) encode them jointly with
bidirectional attention (`joint_mask_mode: full`, no attention mask). At each joint layer,
an 8-head MHFA pools only that layer's enrollment frames (`nb_layer=1`); the front-end
output is not pooled. The layer-4 embedding (256-d) is used twice: it conditions the
mixture-only layers 5-12 through two learned projections that add an input shift and a
cosine key bias (`use_kv_bias: true`), and it conditions every block of the BSRNN extractor
(6 blocks, dimension 128) through FiLM. The mixture representation passed to the extractor
is a learned softmax-weighted sum of the 12 transformer-layer outputs for the mixture
frames (the front-end output is not included). WavLM stays frozen; MHFA, the layer
weights, the projections and the extractor are trained with negative SI-SDR only.

`WavLMDSTBSRNN` also contains experimental joint-depth selection modes (`stage_mode` other
than `fixed_stage`) and their gate modules. They are not configured or supported by this
recipe; the gate modules are still built (unused) so that parameter initialisation matches
the reported run.

## Result

Libri2Mix test set (16 kHz, min, clean; 3,000 mixtures, both speakers as targets = 6,000
cases), trained on train-100. FR = percentage of cases with SI-SDRi < 0 dB.

| System | SI-SDRi (dB) | SDRi (dB) | FR (%) | STOI | PESQ |
|---|---|---|---|---|---|
| S3: separately encoded enrollment, MHFA over all layers (paper baseline, not in this recipe) | 15.66 | 16.24 | 3.00 | 0.934 | 2.82 |
| S4: in-context, d = r = 4, bidirectional (`confs/in_context_bi_d4.yaml`) | 17.04 | 17.53 | 1.53 | 0.951 | 3.02 |

No trained checkpoint is released.

## Requirements

Tested with Python 3.12 and PyTorch/torchaudio 2.6.0 (ROCm 6.2.4 build). Also needed:
`pyyaml`, `tensorboard`, `soundfile`, `mir_eval`, `pesq`, `pystoi`, `transformers` and
`huggingface_hub` (stage 3 only), and `wespeaker`
(`pip install git+https://github.com/wenet-e2e/wespeaker.git`) for the frozen ECAPA-TDNN
speaker encoder. Instead of installing wespeaker, `WESPEAKER_ROOT` can point to a local
checkout. `torch.distributed.init_process_group(device_id=...)` needs PyTorch >= 2.3.

## Reproduce

All commands run from this directory.

1. Generate Libri2Mix (16 kHz, `min`, train-100/dev/test) with
   [LibriMix](https://github.com/JorisCos/LibriMix). Only the clean condition is used.
2. Data lists and shards (stages 1-2):

   ```bash
   bash run.sh --stage 1 --stop_stage 2 --mix_data_path /path/to/Libri2Mix/wav16k/min
   ```

   Stage 1 runs `local/prepare_data.sh` from `../v2`. Training enrollments are drawn on the
   fly from all train-100 utterances of the target speaker (`spk2enroll.json`). Development
   and test enrollments come from fixed lists (`spk1.enroll`, `spk2.enroll`), which equal
   the BUT SpeakerBeam `map_mixture2enrollment` lists downloaded by that script
   (needs `wget`). Stage 2 writes 100 mixtures per shard (139 shards for train-100); with
   fewer shards than GPUs x data-loader workers, some ranks would receive no data.
3. Pre-trained models (stage 3): converts `microsoft/wavlm-base-plus` to
   `convert/wavlm_base_plus.hf.pth` and downloads the WeSpeaker VoxCeleb ECAPA-TDNN (512)
   model (`Wespeaker/wespeaker-voxceleb-ecapa-tdnn512`, revision `89426e1`) to
   `wespeaker_models/voxceleb_ECAPA512/`.

   ```bash
   bash run.sh --stage 3 --stop_stage 3
   ```

   The ECAPA-TDNN model is frozen and does not enter the loss or the extractor input
   (`use_for_separator: false`, `lambda_p: 0`). It is built in training, which keeps the
   random-number stream of the reported run, and in evaluation it gives the `spk_cos` and
   `target_interferer_spk_cos` columns of `predictions.jsonl`.
4. Training (stage 4), evaluation (stage 5) and summary (stage 6):

   ```bash
   bash run.sh --stage 4 --stop_stage 6 --gpus 0,1,2,3,4,5,6,7
   ```

   `batch_size` in the config is per GPU. The reported run used a global batch of 64
   (2 nodes x 8 GPUs x 4); keep `nodes x GPUs x batch_size = 64`, e.g. 8 GPUs with
   `batch_size: 8`, because the number of steps per epoch and the learning-rate schedule
   follow from it. For several nodes, run the same command on every node with
   `--nnodes N --node_rank i --master_addr <host> --master_port <port>` (the experiment
   directory must be on a shared file system). Resume with `--checkpoint exp/.../models/latest.pt`.
   Evaluation uses `models/best.pt` and writes `eval_test/predictions.jsonl` and
   `eval_test/metrics.json`; `summarize.py` adds FR.

## Training settings

- 150 epochs of 218 steps at a global batch of 64 mixtures (about one pass over the 13,900
  train-100 mixtures per epoch; 32,700 steps in total). Both speakers of a mixture are used
  as targets. 3 s random chunks.
- AdamW, peak learning rate 1e-3, weight decay 1e-2 (no decay for biases, normalisation
  and other 1-D parameters), linear warm-up over 5,000 steps from 2.5e-5, then cosine decay
  to 2.5e-5.
- Gradient-norm clipping at 5.0; fp32; seed 42 (offset by the rank).
- Within a training batch, all enrollments are truncated to the shortest one.
- After each epoch, the model is scored on random 3 s chunks of the 3,000 development
  mixtures; `models/best.pt` is the checkpoint with the highest mean development SI-SDRi.
- Reference cost: about 32 h on 16 AMD MI250X GCDs.

## Evaluation

Whole utterances, batch size 1. For each test mixture, the enrollments of the two speakers
are truncated from the end to the shorter one. Reported metrics: SI-SDRi, SDRi
(`mir_eval`), wide-band PESQ, STOI, and FR (percentage of cases with SI-SDRi < 0 dB).
With `eval.save_wav: true` the estimates are written to `eval_test/audio.tar` (about 1 GB).

## Notes for AMD ROCm

The code runs on CUDA and ROCm builds of PyTorch. On ROCm, `train.py` gives every rank its
own MIOpen cache directory under `/tmp`. On GPU, the LSTMs in the extractor run with
cuDNN/MIOpen disabled, as in the reported run.

## Citation

```bibtex
@inproceedings{peng2027incontext,
  title     = {In-Context Enrollment for Target Speech Extraction with Pre-Trained Self-Supervised Learning Models},
  author    = {Peng, Junyi and Fan, Lichun and Su, Hang and Luan, Jian},
  booktitle = {Proc. ICASSP 2027},
  year      = {2027},
  note      = {to appear}
}
```
