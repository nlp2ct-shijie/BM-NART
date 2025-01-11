#!/bin/bash

input_dir=path/to/raw_data      # directory of raw text data
data_dir=path/to/binarized_data   # directory of the generated binarized data
src=src                            # source language id
tgt=tgt                            # target language id
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${input_dir}/train.{src}-{tgt} --validpref ${input_dir}/valid.{src}-{tgt} --testpref ${input_dir}/test.{src}-{tgt} \
    --nwordssrc 40000 --nwordstgt 40000 \
    --srcdict ${input_dir}/dict.${src}.txt --tgtdict {input_dir}/dict.${tgt}.txt \
    --destdir ${data_dir} --workers 4