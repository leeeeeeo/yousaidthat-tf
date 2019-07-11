python ../deblur_train.py \
--tfrecords \
/media/server009/data/dataset/lrs2_matlab_1222/lrs2_matlab_1222_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1221/lrs2_matlab_1221_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1220/lrs2_matlab_1220_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1219/lrs2_matlab_1219_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1218/lrs2_matlab_1218_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1217/lrs2_matlab_1217_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1216/lrs2_matlab_1216_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1215/lrs2_matlab_1215_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1214/lrs2_matlab_1214_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1213/lrs2_matlab_1213_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1212/lrs2_matlab_1212_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1211/lrs2_matlab_1211_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1210/lrs2_matlab_1210_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1209/lrs2_matlab_1209_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1208/lrs2_matlab_1208_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1207/lrs2_matlab_1207_passnan.tfrecords \
/media/server009/data/dataset/lrs2_matlab_1206/lrs2_matlab_1206_passnan.tfrecords /media/server009/data/dataset/lrs2_matlab_1205/lrs2_matlab_1205_passnan.tfrecords \
--gpu 1 \
--use_pretrained_conv 1 \
--name train_deblur_0121 \
--baselr 0.0000001 \
--ckpt "../models/train_deblur_0121-133127"