cd ..
python evaluation_gan.py \
--device ssh7 \
--gpu 1 \
--bn 1 \
--audio 1 \
--audiofc 1 \
--face 1 \
--facefc 1 \
--decoder 1 \
--lip 1 \
--matlab 1 \
--xavier 0 \
--mp4 1 \
--face_detection 0 \
--idnum 1  \
--images ../images/trump0907 \
--output train_12191 \
--ckpt "../models/code15-server009/speech2vid_train_gan_0123-201522_idnum1"
