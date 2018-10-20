import tensorflow as tf
import deblur_inference
from deblur_utils import (
    input_setup,
    checkpoint_dir,
    read_data,
    merge,
    checkimage,
    imsave
)

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.1  # 0.99
TRAINING_STEPS = 30000
MODEL_SAVE_PATH = './'
MODEL_NAME = 'deblur.ckpt'
DATA_DIR = '/Users/lls/Downloads/train_data/291/'
EPOCH = 1500
IMAGE_SIZE = 41
LABEL_SIZE = 41
C_DIM = 3


def main(argv=None):
    images = tf.placeholder(
        tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, C_DIM], name='images')
    labels = tf.placeholder(
        tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, C_DIM], name='labels')

    input_, label_ = read_data(DATA_DIR)

    pred = deblur_inference.inference(input_)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step*BATCH_SIZE, len(input_)*100, LEARNING_RATE_DECAY, staircase=True)
    loss = tf.reduce_mean(tf.square(label_-input_-pred))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    counter = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for ep in range(EPOCH):
            batch_idxs = len(input_)
            for idx in range(0, batch_idxs):
                batch_images = input_[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                batch_labels = label_[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                counter += 1
                _, err = sess.run([train_op, loss], feed_dict={
                                  images: batch_images, labels: batch_labels})

                if counter % 10 == 0:
                    print 'epoch [{}], step [{}], loss [{}]'.format(
                        ep+1, counter, err)
                if counter % 500 == 0:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                                                  MODEL_NAME), global_step=global_step)


if __name__ == "__main__":
    tf.app.run()
