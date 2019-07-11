cd ..
python evaluation_deblur.py \
--ckpt "../models/code15-server009/deblur_train_0123-135213" \
--gpu 1 \
--input "../images/faceimg1/faceimg1_1_blur.png" \
--use_pretrained_conv 1
