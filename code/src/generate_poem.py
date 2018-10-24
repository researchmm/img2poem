from model import *
import config
import os

ckpt_file = '../model/ckpt/epoch03_lr0.000005.ckpt'
sess_config = tf.ConfigProto()
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpus)

sess = tf.InteractiveSession(config=sess_config)

batch_size = 1
model = SeqGAN(sess, batch_size)

model.load_params(ckpt_file)

def generate_poem(img_feature):
    return model.test_one_image(img_feature)
