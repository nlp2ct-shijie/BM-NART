# BM-NART
## Overview
### Model Introduciton
We propose a Bidirectional Multitask Non-Autoregressive  Transformer (BM-NART) model, which  enhances the NART decoder model with  a weak twin-decoder block, providing AR prediction supervision signal in both directions.
![overall_model_arch](https://ooo.0x0.ooo/2024/10/31/ODwf0S.png)
### The Main Result
Our proposed fully BM-NART model overperforms the ART models on four lanuage pairs, WMT14 En-De, WMT16 En-Ro, Ro-En, and IWSLT14 De-En. Meanwhile, it also shows comparable translation quality on WMT14 De-En. In short, we gain comparable translation quality with the ART models with the same scale and significant inference speedup (13.7× - 19.2×) compared with the previous NART models.
![Speedup ratio and translation quality](https://ooo.0x0.ooo/2024/10/31/OHMo7v.jpg)
## Requirement & Installation
- Python >= 3.7
- Pytorch >= 1.10.1
```shell
cd path/to/BM-NART-main
pip install -e .
```
In our implement, we need to use the ctcdecode to support the CTC beam search and the torch_imputer to gain the best alignment. You can follow the below instruction to instsall them.
```shell
cd path/to/BM-NART-main/ctcdecode
pip install .
cd path/to/BM-NART-main/imputer-pytorch
pip install .
```
## Data Processing
We provide the BiKD dataset of IWSLT14 DE-EN, WMT14 En-De/De-En, WMT16-En-Ro/Ro/En we use in our experiment. Please refer the the following [link](https://drive.google.com/drive/folders/1TX9Pi-m-h_JjC5p7kfy6IcD33B3nIHZ-?usp=drive_link) for BiKD dataset downloading.

Then, to generate the binarized data used for the later training, you can run the following script. (We also provide the shell file in BM-NART-main/run/Preprocess.sh)
```shell
input_dir=path/to/raw_data         # directory of raw text data
data_dir=path/to/binarized_data    # directory of the generated binarized data
src=src                            # source language id
tgt=tgt                            # target language id
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${input_dir}/train.{src}-{tgt} --validpref ${input_dir}/valid.{src}-{tgt} --testpref ${input_dir}/test.{src}-{tgt} \
    --nwordssrc 40000 --nwordstgt 40000 \
    --srcdict ${input_dir}/dict.${src}.txt --tgtdict {input_dir}/dict.${tgt}.txt \
    --destdir ${data_dir} --workers 4
```
## Training and Evaluating
We have integrated the complete process of train, average checkpoint, and inference into `path/to/BM-NART/main/run/train-auto.py`. You can directly run the program, then it will automatically start training with the default bidirectional configuration, average the best 5 checkpoints and finally inference the best average checkpoint. You can run the following script to start training. (We also provide the shell file in BM-NART-main/run/train-auto-py.sh)
```shell
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
```
We set commonly used parameters here. If you want to set more detailed parameters, such as dropout rate, bidirectional sharing settings, etc., you can go to `get_train_order` function in `train-auto.py` to set them.
After finishing the training, you will see a file named `gen.best5.out` which records the inference result.
## Model Directory Structure
```
BM-NART
├── criterions
│   └── BM_NART_LOSS.py                      # BM_NART loss composed of Bi-AR loss and NAR loss
├── tasks
│   ├── BM_NART_TASK.py                      # The BM_NART task used to process the Bi-AR samples and NAR samples 
│   ├── nat_ctc_task.py                      # The original nat_ctc task
├── models
    ├── BM_NART_CTC.py                       # The CTC model that introduces the Bi-AR 
    ├── BM_NART_GLAT.py                      # The GLAT+CTC model that introduces the Bi-AR 
    ├── BM_NART_N4.py                        # The BM_NART GLAT+CTC model with only 4 encoders and decoders
    ├── nat_ctc.py                           # The original nat_ctc model
    ├── backward_multihead_attention.py      # The implementation of backward multihead attention
    ├── TransformerBackwardDecoder.py        # The implementation of backward transformer decoder
    ├── TransformerBackwardDecoderLayer.py   # The implementation of backward transformer decoder layer
```
