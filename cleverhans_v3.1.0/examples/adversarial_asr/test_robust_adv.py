import tensorflow as tf
from lingvo import model_imports
from lingvo import model_registry
import numpy as np
import scipy.io.wavfile as wav
import generate_masking_threshold as generate_mask
from tool import create_features, create_inputs, create_speech_rir
import time
from lingvo.core import cluster_factory
from absl import flags
from absl import app

flags.DEFINE_string("root_dir", "./", "location of Librispeech")
flags.DEFINE_string('input', 'read_data.txt',
                    'the text file saved the dir of audios and the corresponding original and targeted transcriptions')
flags.DEFINE_string('rir_dir', 'LibriSpeech/test-clean/3575/170457/3575-170457-0013',
                    'directory of generated room reverberations')
flags.DEFINE_string('checkpoint', './model/ckpt-00908156',
                    'location of checkpoint')

flags.DEFINE_integer('batch_size', '5',
                    'batch_size to do the testing')
flags.DEFINE_string('stage', 'stage2', 'which step to test')
flags.DEFINE_boolean('adv', 'True', 'to test adversarial examples or clean examples')
flags.DEFINE_integer('num_test_rooms', '100',
                    'batch_size to do the testing')
flags.DEFINE_integer('num_train_rooms', '1000',
                    'batch_size to do the testing')

FLAGS = flags.FLAGS

def Read_input(data, batch_size):
    """
    Returns: 
        audios_np: a numpy array of size (batch_size, max_length) in float
        sample_rate: a numpy array  
        trans: an array includes the targeted transcriptions (batch_size,)
    """
    audios = []
    lengths = []

    for i in range(batch_size):
        name, _  = data[0,i].split(".")

        if FLAGS.adv:  
            sample_rate_np, delta = wav.read("./" + str(name) + "_robust_perturb_" + FLAGS.stage + ".wav")
            _, audio_orig = wav.read("./" + str(name) + ".wav")  
            if max(delta) < 1:
                delta = delta * 32768
            audio_np = audio_orig + delta
        else:
            sample_rate_np, audio_np = wav.read("./" + str(name) + ".wav")

        length = len(audio_np)
        
        audios.append(audio_np)
        lengths.append(length)       
    
    max_length = max(lengths)  
    masks = np.zeros([batch_size, max_length]) 
    lengths_freq = (np.array(lengths) // 2 + 1) // 240 * 3
    max_length_freq = max(lengths_freq)
    masks_freq = np.zeros([batch_size, max_length_freq, 80])
    
    # combine the audios into one array
    audios_np = np.zeros([batch_size, max_length])  
    
    for i in range(batch_size):
        audios_np[i, :lengths[i]] = audios[i]
        masks[i, :lengths[i]] = 1
        masks_freq[i, :lengths_freq[i], :] = 1
        
    audios_np = audios_np.astype(float)

    if FLAGS.adv:
        trans = data[2, :]
    else:
        trans = data[1, :]

    lengths = np.array(lengths).astype(np.int32)
    
    return audios_np, sample_rate_np, trans, masks_freq, lengths, max_length, masks


def Readrir(num_room):
    '''
    Return:
        rir: a numpy array of the room reverberation 
        (make sure the test rooms are different from training rooms)
        
    '''        
    index = num_room + FLAGS.num_train_rooms + 1      
    _, rir = wav.read(FLAGS.root_dir + FLAGS.rir_dir + "_rir_" + str(index) + ".wav")   
    return rir

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
                rir_tf = tf.placeholder(tf.float32)
                lengths = tf.placeholder(np.int32, shape=[batch_size,])
                maxlen = tf.placeholder(np.int32)
                mask = tf.placeholder(dtype=np.float32, shape=[batch_size, None]) 
                
                # generate the features and inputs
                new_input = create_speech_rir(input_tf, rir_tf, lengths, maxlen, batch_size) * mask
                features = create_features(new_input, sample_rate_tf, mask_tf)
                shape = tf.shape(features)
                inputs = create_inputs(model, features, tgt_tf, batch_size, mask_tf)
                
                # loss
                metrics = task.FPropDefaultTheta(inputs)              
                loss = tf.get_collection("per_loss")[0]  
                
                # prediction
                decoded_outputs = task.Decode(inputs)
                dec_metrics_dict = task.CreateDecoderMetrics()

                success_rates = []
                for num_room in range(FLAGS.num_test_rooms):
                    correct = 0
                    rir = Readrir(num_room) 

                    for l in range(num_loops):                    
                        data_sub = data[:, l * batch_size:(l + 1) * batch_size]                                           
                        audios_np, sample_rate, tgt_np, mask_freq, lengths_np, max_len, masks  = Read_input(data_sub, batch_size)  

                        feed_dict={input_tf: audios_np, 
                                   sample_rate_tf: sample_rate, 
                                   tgt_tf: tgt_np, 
                                   mask_tf: mask_freq,
                                   rir_tf: rir, 
                                   lengths: lengths_np,
                                   maxlen: max_len,
                                   mask: masks}
                        
                        losses = sess.run(loss, feed_dict)  
                        predictions = sess.run(decoded_outputs, feed_dict)
                        
                        task.PostProcessDecodeOut(predictions, dec_metrics_dict)
                        wer_value = dec_metrics_dict['wer'].value * 100.

                        for i in range(batch_size):   
                            print("example: {}, loss_ce: {}".format(l*batch_size + i, losses[i]))
                            print("pred:{}".format(predictions['topk_decoded'][i, 0]))
                            print("targ:{}".format(tgt_np[i].lower()))
                            print("true: {}".format(data_sub[1, i].lower()))

                            if predictions['topk_decoded'][i,0] == tgt_np[i].lower():
                                correct += 1

                        print("--------------------------------") 
                        print("Now, the WER is: {0:.2f}%".format(wer_value))                                             
                                                
                    print("num of examples succeed for room {}: {}".format(num_room, correct))
                    success_rate = correct / float(num) * 100
                    print("success rate for room {}: {}%".format(num_room, success_rate))
                    
                    success_rates.append(success_rate)
                success_ave = float(sum(success_rates))/len(success_rates)
                print("success rate overall: {}%".format(success_ave))                  
                    
if __name__ == '__main__':
    app.run(main)
