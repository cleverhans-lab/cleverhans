import tensorflow as tf
from lingvo import model_imports
from lingvo import model_registry
import numpy as np
import scipy.io.wavfile as wav
import generate_masking_threshold as generate_mask
from tool import create_features, create_inputs
import time
from lingvo.core import cluster_factory
from absl import flags
from absl import app

flags.DEFINE_string('input', 'read_data.txt',
                    'the text file saved the dir of audios and the corresponding original and targeted transcriptions')
flags.DEFINE_integer('batch_size', '5',
                    'batch_size to do the testing')
flags.DEFINE_string('checkpoint', "./model/ckpt-00908156",
                    'location of checkpoint')
flags.DEFINE_string('stage', "stage2", 'which stage to test')
flags.DEFINE_boolean('adv', 'True', 'to test adversarial examples or clean examples')

FLAGS = flags.FLAGS

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
        if FLAGS.adv:
            sample_rate_np, audio_temp = wav.read("./" + str(name) + "_" + FLAGS.stage + ".wav")
        else:
            sample_rate_np, audio_temp = wav.read("./" + str(name) + ".wav")

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
    if FLAGS.adv:
        trans = data[2, :]
    else:
        trans = data[1, :]
    
    return audios_np, sample_rate_np, trans, masks_freq

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
                inputs = create_inputs(model, features, tgt_tf, batch_size, mask_tf)
                
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
                    wer_value = dec_metrics_dict['wer'].value * 100.
                    
                    for i in range(batch_size):                                           
                        print("pred:{}".format(predictions['topk_decoded'][i, 0]))
                        print("targ:{}".format(tgt_np[i].lower()))
                        print("true: {}".format(data_sub[1, i].lower()))

                        if predictions['topk_decoded'][i,0] == tgt_np[i].lower():
                            correct += 1
                            print("------------------------------")
                            print("example {} succeeds".format(i))

                    print("Now, the WER is: {0:.2f}%".format(wer_value))    
                print("num of examples succeed: {}".format(correct))
                print("success rate: {}%".format(correct / float(num) * 100))
             

if __name__ == '__main__':
    app.run(main)
