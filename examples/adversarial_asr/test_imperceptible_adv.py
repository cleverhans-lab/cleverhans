import tensorflow as tf
from lingvo import model_imports
from lingvo import model_registry
from lingvo.core import py_utils
import six
import os
import re
import tarfile
import numpy as np
from lingvo.core import asr_frontend
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import subprocess
import scipy.io.wavfile as wav
from absl import flags
from absl import app
from lingvo.core import cluster_factory

flags.DEFINE_string('input', 'read_data.txt',
                    'the text file saved the dir of audios and the corresponding original and targeted transcriptions')
flags.DEFINE_integer('batch_size', '5',
                    'batch_size to do the testing')
flags.DEFINE_string('checkpoint', "./model/ckpt-00908156",
                    'location of checkpoint')
flags.DEFINE_string('stage', "stage2", 'which stage to test')
flags.DEFINE_integer('window_size', '2048', 'window size in spectrum analysis')
FLAGS = flags.FLAGS

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

def Read_input(data, batch_size):
    """
    Returns: 
        audios_np: a numpy array of size (batch_size, max_length) in float
        sample_rate: a numpy array  
        trans: an array includes the targeted transcriptions (batch_size,)
        masks_freq: a numpy array to mask out the padding features in frequency domain 
    """
    audios = []
    lengths = []
    
    for i in range(batch_size):
        name, _  = data[0,i].split(".")
        sample_rate_np, audio_temp = wav.read("./" + str(name) + "_" + FLAGS.stage + ".wav")

        # read the wav form range from [-32767, 32768] or [-1, 1]
        if max(audio_temp) < 1:
            audio_np = audio_temp * 32768
            
        else:
            audio_np = audio_temp
        length = len(audio_np)
        
        audios.append(audio_np)
        lengths.append(length)
    
    max_length = max(lengths)   
    lengths_freq = (np.array(lengths) // 2 + 1) // 240 * 3
    max_length_freq = max(lengths_freq)
    masks_freq = np.zeros([batch_size, max_length_freq, 80])
    
    # combine the audios into one array
    audios_np = np.zeros([batch_size, max_length])  
    
    for i in range(batch_size):
        audios_np[i, :lengths[i]] = audios[i]
        masks_freq[i, :lengths_freq[i], :] = 1
        
    audios_np = audios_np.astype(float)
    trans = data[2, :]
    
    return audios_np, sample_rate_np, trans, masks_freq

def create_features(input_tf, sample_rate_tf, mask_freq):
    """
    Return:
        A tensor of features with size (batch_size, max_time_steps, 80)
    """
    
    features_list = []      
    # unstact the features with placeholder    
    input_unpack = tf.unstack(input_tf, axis=0)
    #length_unpack = tf.unstack(lengths_tf, axis=0)
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
    
    # src_inputs has the shape [batch_size, num_frames, feature_dim, channels]
    src_paddings = tf.zeros([tf.shape(features)[0], tf.shape(features)[1]], dtype=tf.float32)
    
    src_paddings = ( 1 - mask_freq[:,:, 0])   
    src_frames = tf.expand_dims(features, dim=-1)

    inputs = py_utils.NestedMap()
    inputs.tgt = py_utils.NestedMap(
        ids=tgt_ids,
        labels=tgt_labels,
        paddings=tgt_paddings,
        weights=1.0 - tgt_paddings)
    inputs.src = py_utils.NestedMap(src_inputs=src_frames, paddings=src_paddings)
    inputs.sample_ids = tf.zeros([batch_size])
    return inputs, src_paddings
      

def main(argv):
    data = np.loadtxt(FLAGS.input, dtype=str, delimiter=",")
    # calculate the number of loops to run the test
    num = len(data[0])
    batch_size = FLAGS.batch_size
    num_loops = num / batch_size 
    assert num % batch_size == 0
      
        
    with tf.device("/gpu:0"):
        tf.set_random_seed(1234)
        tfconf = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=tfconf) as sess:           
            params = model_registry.GetParams('asr.librispeech.Librispeech960Wpm', 'Test')
            params.cluster.worker.gpus_per_replica = 1
            cluster = cluster_factory.Cluster(params.cluster)
            with cluster, tf.device(cluster.GetPlacer()):
                params.vn.global_vn = False
                params.random_seed = 1234
                params.is_eval = True
                model = params.cls(params)
                task = model.GetTask()
                saver = tf.train.Saver()
                saver.restore(sess, FLAGS.checkpoint)
                
                # define the placeholders
                input_tf = tf.placeholder(tf.float32, shape=[batch_size, None])
                tgt_tf = tf.placeholder(tf.string)
                sample_rate_tf = tf.placeholder(tf.int32) 
                mask_tf = tf.placeholder(tf.float32, shape=[batch_size, None, 80])
                               
                # generate the features and inputs
                features = create_features(input_tf, sample_rate_tf, mask_tf)
                shape = tf.shape(features)
                inputs, src_paddings = create_inputs(model, features, tgt_tf, batch_size, mask_tf)
                
                # loss
                metrics = task.FPropDefaultTheta(inputs)              
                loss = tf.get_collection("per_loss")[0]  
                
                # prediction
                decoded_outputs = task.Decode(inputs)
                dec_metrics_dict = task.CreateDecoderMetrics()                
                       
                correct = 0                              
                for l in range(num_loops):                    
                    data_sub = data[:, l * batch_size:(l + 1) * batch_size]                                       
                    audios_np, sample_rate, tgt_np, mask_freq  = Read_input(data_sub, batch_size)                     
                    feed_dict={input_tf: audios_np, 
                               sample_rate_tf: sample_rate, 
                               tgt_tf: tgt_np, 
                               mask_tf: mask_freq}
                    
                    losses = sess.run(loss, feed_dict)  
                    predictions = sess.run(decoded_outputs, feed_dict)
                    
                    task.PostProcessDecodeOut(predictions, dec_metrics_dict)
                                        
                    for i in range(batch_size):                                           
                        print("pred:{}".format(predictions['topk_decoded'][i, 0]))
                        print("targ:{}".format(tgt_np[i].lower()))
                        print("true: {}".format(data_sub[1, i].lower()))

                        if predictions['topk_decoded'][i,0] == tgt_np[i].lower():
                            correct += 1
                            print("------------------------------")
                            print("example {} succeeds".format(i))
                        
                print("num of examples succeed: {}".format(correct))
                print("success rate: {}".format(correct / float(num) * 100))
             

if __name__ == '__main__':
    app.run(main)
