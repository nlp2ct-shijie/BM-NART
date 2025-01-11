#!/bin/bash

savedir=path/to/checkpoints
dataset=path/to/binarized_data
userdir=BM_NART
task=BM_NART_TASK
criterion=BM_NART_LOSS
arch=BM_NART_CTC
max_token=max_token
batch_size=batch_size
max_epoch=max_epoch
update_freq=update_freq
output_bleu="bleu.log"  # the file to save the valid bleu
output_file="train.log"  # the file to save the training log
lr=0.0005
warmup_updates=10000
warmup_init_lr=1e-07
check_freq=5

python train-auto.py --savedir ${savedir} --dataset ${dataset} --userdir ${userdir} --task ${task} --criterion ${criterion} --arch ${arch} \
		   --max-token ${max_token} --max-epochs {max_epoch} --update-freq ${update_freq} \
		   --output-bleu ${output_bleu} --output-files ${output_file} \
		   --lr ${lr} --warmup-updates ${warmup_updates} --warmup-init-lr ${warmup_init_lr} \
		   --at-weights 0.8 0.7 0.6 0.5 0.4 --check-freq ${check_freq} \
		   --with-curr --saving-type 'epoch'