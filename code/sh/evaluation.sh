cd ..
python evaluation.py \
--device server009 \
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
--mode deconv \
--output train_1216 \
--ckpt "models/train_1216-212411-idnum1"
