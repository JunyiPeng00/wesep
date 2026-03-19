#!/bin/bash

# Copyright 2023 Shuai Wang (wangshuai@cuhk.edu.cn)

. ./path.sh || exit 1

# General configuration
stage=-1
stop_stage=-1

# Data preparation related
data=data
fs=16k
min_max=min
noise_type="clean"
data_type="shard" # shard/raw
Libri2Mix_dir=/share5/users/dream.peng/Data/Librispeech/Librimix_Data/Libri2Mix
mix_data_path="${Libri2Mix_dir}/wav${fs}/${min_max}"

# Training related
gpus=[1,2]
use_gan_loss=false
config=confs/bsrnn.yaml
config=confs/wavlm_tasnet.yaml
exp_dir=exp/wavlm_tasnet_v2/baseline
if [ -z "${config}" ] && [ -f "${exp_dir}/config.yaml" ]; then
  config="${exp_dir}/config.yaml"
fi

# TSE model initialization related
checkpoint=

# Inferencing and scoring related
save_results=true
use_pesq=true
use_dnsmos=true
dnsmos_use_gpu=true

# Model average related
num_avg=5

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --mix_data_path ${mix_data_path} \
    --data ${data} \
    --noise_type ${noise_type} \
    --stage 1 \
    --stop-stage 2
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in train-100 dev test; do
    #  for dset in train-360; do
    python tools/make_shard_list_premix.py --num_utts_per_shard 1000 \
      --num_threads 16 \
      --prefix shards \
      --shuffle \
      ${data}/$noise_type/$dset/wav.scp ${data}/$noise_type/$dset/utt2spk \
      ${data}/$noise_type/$dset/shards ${data}/$noise_type/$dset/shard.list
  done
fi



if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/latest_checkpoint.pt" ]; then
    checkpoint="${exp_dir}/models/latest_checkpoint.pt"
  fi
  if ${use_gan_loss}; then
    train_script=wesep/bin/train_gan.py
  else
    train_script=wesep/bin/train.py
  fi
  export OMP_NUM_THREADS=8
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    ${train_script} --config $config \
    --exp_dir ${exp_dir} \
    --gpus $gpus \
    --num_avg ${num_avg} \
    --data_type "${data_type}" \
    --train_data ${data}/$noise_type/train-100/${data_type}.list \
    --train_utt2spk ${data}/$noise_type/train-100/single.utt2spk \
    --train_spk2utt ${data}/$noise_type/train-100/spk2enroll.json \
    --val_data ${data}/$noise_type/dev/${data_type}.list \
    --val_spk1_enroll ${data}/$noise_type/dev/spk1.enroll \
    --val_spk2_enroll ${data}/$noise_type/dev/spk2.enroll \
    --val_spk2utt ${data}/$noise_type/dev/single.wav.scp \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_best_model.pt
  python wesep/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg} \
    --mode best \
    --epochs "141,143,145,147,149"
fi
if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/avg_best_model.pt" ]; then
  checkpoint="${exp_dir}/models/avg_best_model.pt"
fi


# shellcheck disable=SC2215
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Start inferencing ..."
  python wesep/bin/infer.py --config $config \
    --fs ${fs} \
    --gpus 0 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/$noise_type/test/${data_type}.list \
    --test_spk1_enroll ${data}/$noise_type/test/spk1.enroll \
    --test_spk2_enroll ${data}/$noise_type/test/spk2.enroll \
    --test_spk2utt ${data}/$noise_type/test/single.wav.scp \
    --save_wav ${save_results} \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Start scoring ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  ./tools/score.sh --dset "${data}/$noise_type/test" \
    --exp_dir "${exp_dir}" \
    --fs ${fs} \
    --use_pesq "${use_pesq}" \
    --use_dnsmos "${use_dnsmos}" \
    --dnsmos_use_gpu "${dnsmos_use_gpu}" \
    --n_gpu "${num_gpus}"
fi
