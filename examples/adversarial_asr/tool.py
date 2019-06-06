from tensorflow.python import pywrap_tensorflow
import numpy as np
import tensorflow as tf
from lingvo.core import asr_frontend
from lingvo.core import py_utils

def _MakeLogMel(audio, sample_rate): 
  audio = tf.expand_dims(audio, axis=0)
  static_sample_rate = 16000
  mel_frontend = _CreateAsrFrontend()
  with tf.control_dependencies(
      [tf.assert_equal(sample_rate, static_sample_rate)]):
    log_mel, _ = mel_frontend.FPropDefaultTheta(audio)
  return log_mel

def _CreateAsrFrontend():
  p = asr_frontend.MelFrontend.Params()
  p.sample_rate = 16000.
  p.frame_size_ms = 25.
  p.frame_step_ms = 10.
  p.num_bins = 80
  p.lower_edge_hertz = 125.
  p.upper_edge_hertz = 7600.
  p.preemph = 0.97
  p.noise_scale = 0.
  p.pad_end = False
  # Stack 3 frames and sub-sample by a factor of 3.
  p.left_context = 2
  p.output_stride = 3
  return p.cls(p)

def create_features(input_tf, sample_rate_tf, mask_freq):
    """
    Return:
        A tensor of features with size (batch_size, max_time_steps, 80)
    """
    
    features_list = []      
    # unstact the features with placeholder    
    input_unpack = tf.unstack(input_tf, axis=0)
    for i in range(len(input_unpack)):
        features = _MakeLogMel(input_unpack[i], sample_rate_tf)
        features = tf.reshape(features, shape=[-1, 80]) 
        features = tf.expand_dims(features, dim=0)  
        features_list.append(features)
    features_tf = tf.concat(features_list, axis=0) 
    features_tf = features_tf * mask_freq
    return features_tf    
          
def create_inputs(model, features, tgt, batch_size, mask_freq):    
    tgt_ids, tgt_labels, tgt_paddings = model.GetTask().input_generator.StringsToIds(tgt)
    
    # we expect src_inputs to be of shape [batch_size, num_frames, feature_dim, channels]
    src_paddings = tf.zeros([tf.shape(features)[0], tf.shape(features)[1]], dtype=tf.float32)
    src_paddings = 1. - mask_freq[:,:,0]
    src_frames = tf.expand_dims(features, dim=-1)

    inputs = py_utils.NestedMap()
    inputs.tgt = py_utils.NestedMap(
        ids=tgt_ids,
        labels=tgt_labels,
        paddings=tgt_paddings,
        weights=1.0 - tgt_paddings)
    inputs.src = py_utils.NestedMap(src_inputs=src_frames, paddings=src_paddings)
    inputs.sample_ids = tf.zeros([batch_size])
    return inputs

class Transform(object):
    '''
    Return: PSD
    '''    
    def __init__(self):
        self.scale = 8. / 3.
        self.frame_length = int(FLAGS.window_size)
        self.frame_step = int(FLAGS.window_size//4)
    
    def __call__(self, x, psd_max_ori):
        win = tf.contrib.signal.stft(x, self.frame_length, self.frame_step)
        z = self.scale *tf.abs(win / FLAGS.window_size)
        psd = tf.square(z)
        PSD = tf.pow(10., 9.6) / tf.reshape(psd_max_ori, [-1, 1, 1]) * psd
        return PSD