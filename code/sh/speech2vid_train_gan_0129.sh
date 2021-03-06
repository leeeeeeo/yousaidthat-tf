#!/bin/sh
set -e
date_now="`date +%F-%H-%M`"
name="speech2vid_train_gan_0129"
python -u ../speech2vid_train_gan_halfface.py \
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
--gpu 0 \
--is_training 0 \
--early_stopping_folder "/media/server009/data/dataset/lrs2_matlab_1222" \
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
--mode gan_halfface \
--l1weight 100 \
--ganweight 1 \
--batchsize 32 \
--max_to_keep 10 \
--save_freq 5000 \
--early_stopping_freq 5000 \
--early_stopping_size 0 \
2>&1 | tee ../../logs/$name"_"$date_now.log $@
