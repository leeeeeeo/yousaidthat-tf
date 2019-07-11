#!/bin/sh
set -e
date_now="`date +%F-%H-%M`"
name="deblur_train_0123"
python -u ../deblur_train.py \
--tfrecords \
/media/server009/data/dataset/lrs2_matlab_1221/lrs2_matlab_1221_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1220/lrs2_matlab_1220_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1219/lrs2_matlab_1219_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1218/lrs2_matlab_1218_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1217/lrs2_matlab_1217_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1216/lrs2_matlab_1216_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1215/lrs2_matlab_1215_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1214/lrs2_matlab_1214_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1213/lrs2_matlab_1213_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1212/lrs2_matlab_1212_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1211/lrs2_matlab_1211_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1210/lrs2_matlab_1210_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1209/lrs2_matlab_1209_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1208/lrs2_matlab_1208_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1207/lrs2_matlab_1207_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1206/lrs2_matlab_1206_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1205/lrs2_matlab_1205_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1204/lrs2_matlab_1204_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1203/lrs2_matlab_1203_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1202/lrs2_matlab_1202_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1201/lrs2_matlab_1201_passnan.tfrecords \
--gpu 1 \
--early_stopping_folder "/media/server009/data/dataset/lrs2_matlab_1222" \
--use_pretrained_conv 1 \
--name $name \
--baselr 0.0001 \
--loss_func mse \
--max_to_keep 50 \
--ckpt "../../models/code15-server009/deblur_train_0123-135213" \
2>&1 | tee ../../logs/$name"_"$date_now.log $@