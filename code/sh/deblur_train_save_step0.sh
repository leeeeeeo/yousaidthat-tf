#!/bin/sh
set -e
date_now="`date +%F-%H-%M`"
name="deblur_train_step0"
python -u ../deblur_train_save_step0.py \
--tfrecords \
/media/server009/data/dataset/lrs2_matlab_1221/lrs2_matlab_1221_passnan.tfrecords \
--gpu 1 \
--use_pretrained_conv 1 \
--name $name \
--baselr 0 \
--loss_func mse \
--batchsize 1 \
2>&1 | tee ../../logs/$name"_"$date_now.log $@