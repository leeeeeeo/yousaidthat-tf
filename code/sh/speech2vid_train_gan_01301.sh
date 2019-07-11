#!/bin/sh
set -e
date_now="`date +%F-%H-%M`"
name="speech2vid_train_gan_0130"
python -u ../speech2vid_train_gan_multigpu.py \
--tfrecords \
/workspace/liuhan/work/avasyn/data/lrs2_matlab0_0128/lrs2_matlab0_0128_passnan.tfrecords \
/workspace/liuhan/work/avasyn/data/lrs2_matlab0_0129/lrs2_matlab0_0129_passnan.tfrecords \
--gpu 1,2,3 \
--is_training 0 \
--early_stopping_folder "/workspace/liuhan/work/avasyn/data/lrs2_matlab0_0127" \
--bn 1 \
--audio 1 \
--audiofc 1 \
--face 1 \
--facefc 1 \
--decoder 1 \
--lip 1 \
--name $name \
--baselr 0.0001 \
--audiolr 0 \
--identitylr 0.05 \
--bnlr 0 \
--idnum 1 \
--xavier 0 \
--mode gan \
--l1weight 100 \
--ganweight 1 \
--batchsize 32 \
--max_to_keep 10 \
--save_freq 5000 \
--early_stopping_freq 5000 \
--early_stopping_size 0 \
2>&1 | tee ../../logs/$name"_"$date_now.log $@
