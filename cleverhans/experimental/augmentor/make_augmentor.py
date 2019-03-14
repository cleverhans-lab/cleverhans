import sys
import tensorflow as tf
sess = tf.Session()
with sess.as_default():
  _, raw, out = sys.argv
  from cleverhans.serial import save, load
  raw = load(raw)
  from augmentor import Augmentor
  model = Augmentor(raw)
  save(out, model)
