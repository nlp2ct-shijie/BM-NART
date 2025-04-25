# BM-NART
Implementation of the paper "**Bidirectional Multitask Learning for Non-Autoregressive Machine Translation**".
## Background
#### Abstract
Non-Autoregressive Transformer (NART) models generate tokens independently, resulting in lower translation quality than the Autoregressive Transformer (ART) model. To enhance the generation quality, prior Multitask Learning (MTL) frameworks have incorporated a directional Autoregressive (AR) prediction task in conjunction with the Non-Autoregressive (NAR) task. This work proposes further enhancing the NART model with Bidirectional Autoregressive (Bi-AR) prediction tasks. We propose the Bidirectional Multitask Non-Autoregressive Transformer (BM-NART) framework, which enhances the NART decoder model with a weak twin-decoder block, providing AR prediction supervision signal in both directions. To accommodate the bidirectional decoder, we further enhance the Autoregressive Knowledge Distillation (ARKD) with the introduction of Bidirectional Knowledge Distillation (BiKD), which employs dual directional teacher models to provide Bi-AR knowledge distillation data. The experiment confirms that with BiKD, the BM-NART framework achieves generation quality comparable to ART models in BLEU and BERTScore while retaining the advantage of high parallel generation, achieving a 13.7-20 times acceleration with various parameter scalings. Our LLM-based analysis further reveals that the BM-NART framework surpasses the ART model in handling ambiguous translations, knowledge-dependent translations, and reducing hallucinations, illustrating the substantial potential of future NART models.
#### Model Introduciton
We propose a Bidirectional Multitask Non-Autoregressive  Transformer (BM-NART) model, which  enhances the NART decoder model with  a weak twin-decoder block, providing AR prediction supervision signal in both directions.
![overall_model_arch](https://github.com/nlp2ct-shijie/BM-NART/blob/main/Assets/BM_NART_Architecture.png)
#### Practical Advantages
- BM-NART further enhances the NART models (GLAT/GLAT+CTC) without introducing additional inference costs.
- BM-NART achieves translation performance comparable to or even surpassing that of ART models across all language pairs.
- With equivalent model scales, BM-NART achieves significant inference speedup (13.7× – 19.2×) compared to ART models.

<center>
![Speedup ratio and translation quality](https://github.com/nlp2ct-shijie/BM-NART/blob/main/Assets/Result_of_BM_NART.png)
</center>


## Requirement & Installation
- Python >= 3.7
- Pytorch >= 1.10.1
```shell
cd path/to/BM-NART-main
pip install -e .
```
In our implementation, we use the ctcdecode to support the CTC beam search and the torch_imputer to gain the best alignment. Follow the below instruction to instsall both packages:
```shell
cd path/to/BM-NART-main/ctcdecode
pip install .
cd path/to/BM-NART-main/imputer-pytorch
pip install .
```
## Data Processing
- We release the BiKD dataset of IWSLT14 DE-EN, WMT14 En-De/De-En, WMT16-En-Ro/Ro/En. 
- Refer to the following link for BiKD data downloading:  [https://drive.google.com/drive/folders/1TX9Pi-m-h_JjC5p7kfy6IcD33B3nIHZ-?usp=drive_link](https://drive.google.com/drive/folders/1TX9Pi-m-h_JjC5p7kfy6IcD33B3nIHZ-?usp=drive_link)
- To generate the binarized data, please run the shell file `path/to/BM-NART/main/run/Preprocess.py`
```shell
input_dir=path/to/raw_data      # directory of raw text data
data_dir=path/to/binarized_data   # directory of the generated binarized data
src=src                            # source language id
tgt=tgt                            # target language id
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${input_dir}/train.{src}-{tgt} --validpref ${input_dir}/valid.{src}-{tgt} --testpref ${input_dir}/test.{src}-{tgt} \
    --nwordssrc 40000 --nwordstgt 40000 \
    --srcdict ${input_dir}/dict.${src}.txt --tgtdict {input_dir}/dict.${tgt}.txt \
    --destdir ${data_dir} --workers 4
```

## Training and Evaluating
-  We have integrated the complete process of train, average checkpoint, and inference into `path/to/BM-NART/main/run/train-auto.py`. Run the script `BM-NART-main/run/train-auto-py.sh` to start training.
-  The commonly used parameters can be set here. If you want to change more detailed parameters, such as dropout rate, bidirectional sharing settings, etc., refer to `get_train_order` function in `train-auto.py`.
- After finishing the training, a file named `gen.best5.out` which records the inference result will be generated.
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
check_freq=5 # check frequency of validation bleu

python train-auto.py --savedir ${savedir} --dataset ${dataset} --userdir ${userdir} --task ${task} --criterion ${criterion} --arch ${arch} \
		   --max-token ${max_token} --max-epochs {max_epoch} --update-freq ${update_freq} \
		   --output-bleu ${output_bleu} --output-files ${output_file} \
		   --lr ${lr} --warmup-updates ${warmup_updates} --warmup-init-lr ${warmup_init_lr} \
		   --at-weights 0.8 0.7 0.6 0.5 0.4 --check-freq ${check_freq} \
		   --with-curr --saving-type 'epoch'
```

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
## Citing
Please kindly cite us if you find our papers or codes useful.
