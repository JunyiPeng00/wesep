#!/bin/bash

# In-context enrollment TSE on Libri2Mix (16 kHz, min, clean).
# Run from this directory, e.g.
#   bash run.sh --stage 1 --stop_stage 3 --mix_data_path /path/to/Libri2Mix/wav16k/min
#   bash run.sh --stage 4 --stop_stage 6 --gpus 0,1,2,3,4,5,6,7

. ./path.sh || exit 1

stage=1
stop_stage=6

# Data preparation
mix_data_path=/path/to/Libri2Mix/wav16k/min
data=data
noise_type=clean
num_utts_per_shard=100

# Training / evaluation
config=confs/in_context_bi_d4.yaml
exp_dir=exp/in_context_bi_d4
gpus="0,1,2,3,4,5,6,7"
checkpoint=          # resume training from this checkpoint (e.g. exp/.../models/latest.pt)
eval_checkpoint=     # default: ${exp_dir}/models/best.pt
eval_split=test
max_samples=0

# Multi-node training (run the same command on every node with its own node_rank)
nnodes=1
node_rank=0
master_addr=127.0.0.1
master_port=29500

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: prepare Libri2Mix lists and fixed dev/test enrollments ..."
  # Same steps as ../v2; the dev/test enrollment map is downloaded from BUTSpeechFIT/speakerbeam.
  ./local/prepare_data.sh --mix_data_path ${mix_data_path} \
    --data ${data} \
    --noise_type ${noise_type} \
    --stage 1 \
    --stop-stage 2
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: write shards (${num_utts_per_shard} mixtures per shard) ..."
  for dset in train-100 dev test; do
    python tools/make_shard_list_premix.py --num_utts_per_shard ${num_utts_per_shard} \
      --num_threads 16 \
      --prefix shards \
      --shuffle \
      ${data}/${noise_type}/${dset}/wav.scp ${data}/${noise_type}/${dset}/utt2spk \
      ${data}/${noise_type}/${dset}/shards ${data}/${noise_type}/${dset}/shard.list
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: download and convert WavLM Base+, download the WeSpeaker ECAPA-TDNN512 model ..."
  python -c "from wesep.modules.wavlm_frontend.convert_hf_ssl_models import convert_hf_ssl_model; \
convert_hf_ssl_model(model_name='wavlm_base_plus', exp_dir='convert', hf_cache_dir='convert', local_files_only=False)"
  python -c "from huggingface_hub import hf_hub_download; \
[hf_hub_download(repo_id='Wespeaker/wespeaker-voxceleb-ecapa-tdnn512', filename=f, \
revision='89426e111abc9ffcc40279baf808e087693c6e9b', local_dir='wespeaker_models/voxceleb_ECAPA512') \
for f in ('avg_model.pt', 'config.yaml')]"
  echo "c5659521b311ce284751b7a7e8897e9287b7b9131db00ad58f35807b49515f6a  wespeaker_models/voxceleb_ECAPA512/avg_model.pt" |
    sha256sum -c - || exit 1
fi

num_gpus=$(echo "${gpus}" | awk -F ',' '{print NF}')
runtime_config="${exp_dir}/config.run.yaml"

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: training ..."
  mkdir -p "${exp_dir}"
  if [ "${nnodes}" -le 1 ]; then
    export CUDA_VISIBLE_DEVICES="${gpus}"
  fi
  local_gpus=$(seq -s ',' 0 $((num_gpus - 1)))
  if [ "${node_rank}" -eq 0 ]; then
    cp "${config}" "${runtime_config}"
    sed -i "s#^exp_dir: .*#exp_dir: ${exp_dir}#" "${runtime_config}"
    sed -i "s#^gpus: .*#gpus: \"${local_gpus}\"#" "${runtime_config}"
  else
    # Other nodes wait for node 0 to write the runtime config (shared file system).
    for _ in $(seq 1 120); do
      [ -s "${runtime_config}" ] && break
      sleep 1
    done
  fi
  train_args=(--config "${runtime_config}")
  if [ -n "${checkpoint}" ]; then
    train_args+=(--checkpoint "${checkpoint}")
  fi
  # The global batch is nnodes x num_gpus x batch_size (reported run: 64 = 16 x 4).
  if [ "${nnodes}" -gt 1 ]; then
    torchrun --nnodes=${nnodes} --nproc_per_node=${num_gpus} --node_rank=${node_rank} \
      --master_addr=${master_addr} --master_port=${master_port} \
      train.py "${train_args[@]}"
  elif [ "${num_gpus}" -gt 1 ]; then
    torchrun --standalone --nnodes=1 --nproc_per_node=${num_gpus} \
      train.py "${train_args[@]}"
  else
    python train.py "${train_args[@]}"
  fi
fi

if [ "${nnodes}" -gt 1 ] && [ "${node_rank}" -ne 0 ]; then
  exit 0
fi

output_dir="${exp_dir}/eval_${eval_split}"

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: evaluation on ${eval_split} ..."
  export CUDA_VISIBLE_DEVICES=$(echo "${gpus}" | cut -d ',' -f 1)
  eval_checkpoint="${eval_checkpoint:-${exp_dir}/models/best.pt}"
  eval_args=(
    --config "${exp_dir}/config.resolved.yaml"
    --checkpoint "${eval_checkpoint}"
    --split "${eval_split}"
    --output_dir "${output_dir}"
  )
  if [ "${max_samples}" -gt 0 ]; then
    eval_args+=(--max_samples "${max_samples}")
  fi
  python eval.py "${eval_args[@]}"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: summary ..."
  python summarize.py "${output_dir}/predictions.jsonl" | tee "${output_dir}/summary.json"
fi
