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
import random

# data directory
flags.DEFINE_string("root_dir", "./", "location of Librispeech")
flags.DEFINE_string('input', 'read_data.txt', 
                    'the text file saved the dir of audios and the corresponding original and targeted transcriptions')
flags.DEFINE_string('rir_dir', 'LibriSpeech/test-clean/3575/170457/3575-170457-0013',
                    'directory of generated room reverberations')

# data processing
flags.DEFINE_integer('max_length_dataset', '223200', 'the length of the longest audio in the whole dataset')
flags.DEFINE_float('initial_bound', '2000.', 'initial l infinity norm for adversarial perturbation')
flags.DEFINE_integer('num_rir', '1000', 'number of room reverberations used in training')
flags.DEFINE_integer('num_counter', '2', 'the initial number of required successful rooms')
flags.DEFINE_integer('num_rooms', '10', 'the initial number of rooms to test')
flags.DEFINE_integer('max_delta', '300', 'the max delta added to the max l infinity norm')

# training parameters
flags.DEFINE_string('checkpoint', "./model/ckpt-00908156", 'location of checkpoint')
flags.DEFINE_integer('batch_size', '5', 'batch size')                   
flags.DEFINE_float('lr_stage1', '50', 'learning_rate for stage 1')
flags.DEFINE_float('lr_stage2', '5', 'learning_rate for stage 2')
flags.DEFINE_integer('num_iter_stage1', '2000', 'number of iterations in stage 1')
flags.DEFINE_integer('num_iter_stage2', '4000', 'number of iterations in stage 2')
flags.DEFINE_integer('num_gpu', '0', 'which gpu to run')


FLAGS = flags.FLAGS

def ReadFromWav(data, batch_size):
    """
    Returns: 
        audios_np: a numpy array of size (batch_size, max_length) in float
        trans: a numpy array includes the targeted transcriptions (batch_size, )
        max_length: the max length of the batch of audios
        sample_rate_np: a numpy array
        masks: a numpy array of size (batch_size, max_length)
        masks_freq: a numpy array of size (batch_size, max_length_freq, 80)
        lengths: a list of the length of original audios
    """
    audios = []
    lengths = []
       
    # read the .wav file
    for i in range(batch_size):
        sample_rate_np, audio_temp = wav.read(FLAGS.root_dir + str(data[0, i]))
        # read the wav form range from [-32767, 32768] or [-1, 1]
        if max(audio_temp) < 1:
            audio_np = audio_temp * 32768
        else:
            audio_np = audio_temp
            
        length = len(audio_np)
        
        audios.append(audio_np)
        lengths.append(length)
    
    max_length = max(lengths)
   
    # pad the input audio
    audios_np = np.zeros([batch_size, max_length])  
    masks = np.zeros([batch_size, max_length])
    lengths_freq = (np.array(lengths) // 2 + 1) // 240 * 3
    max_length_freq = max(lengths_freq)
    masks_freq = np.zeros([batch_size, max_length_freq, 80])
    for i in range(batch_size):
        audio_float = audios[i].astype(float)
        audios_np[i, :lengths[i]] = audio_float   
        masks[i, :lengths[i]] = 1
        masks_freq[i, :lengths_freq[i], :] = 1
    
    # read the transcription
    trans = data[2, :]
    lengths = np.array(lengths).astype(np.int32)
    
    return audios_np, trans, max_length, sample_rate_np, masks, masks_freq, lengths

def Readrir():
    '''
    Return:
        rir: a numpy array of the room reverberation
        
    '''        
    index = random.randint(1, FLAGS.num_rir)       
    _, rir = wav.read(FLAGS.root_dir + FLAGS.rir_dir + "_rir_" + str(index) + ".wav")   
    return rir


class Attack:
    def __init__(self, sess, batch_size=1,
                 lr_stage1=100, lr_stage2=0.1, num_iter_stage1=1000, num_iter_stage2=4000, th=None,
                        psd_max_ori=None):
       
        self.sess = sess
        self.num_iter_stage1 = num_iter_stage1
        self.num_iter_stage2 = num_iter_stage2
        self.batch_size = batch_size     
        self.lr_stage1 = lr_stage1  
        self.lr_stage2 = lr_stage2       
        
        tf.set_random_seed(1234)
        params = model_registry.GetParams('asr.librispeech.Librispeech960Wpm', 'Test')
        params.random_seed = 1234
        params.is_eval = True
        params.cluster.worker.gpus_per_replica = 1
        cluster = cluster_factory.Cluster(params.cluster)
        with cluster, tf.device(cluster.GetPlacer()):
            model = params.cls(params)
            self.delta_large = tf.Variable(np.zeros((batch_size, FLAGS.max_length_dataset), dtype=np.float32), name='qq_delta')
            
            # placeholders
            self.input_tf = tf.placeholder(tf.float32, shape=[batch_size, None], name='qq_input')
            self.tgt_tf = tf.placeholder(tf.string)
            self.rir = tf.placeholder(tf.float32)

            self.sample_rate_tf = tf.placeholder(tf.int32, name='qq_sample_rate')                
            self.mask = tf.placeholder(dtype=np.float32, shape=[batch_size, None], name='qq_mask')   
            self.mask_freq = tf.placeholder(dtype=np.float32, shape=[batch_size, None, 80])
            self.noise = tf.placeholder(np.float32, shape=[batch_size, None], name="qq_noise")
            self.maxlen = tf.placeholder(np.int32)
            self.lr = tf.placeholder(np.float32)
            self.lengths = tf.placeholder(np.int32, shape=[batch_size,])
            
            # variable
            self.rescale = tf.Variable(np.ones((batch_size,1), dtype=np.float32) * FLAGS.initial_bound, name='qq_rescale')    
            
            # extract the delta
            self.delta = tf.slice(tf.identity(self.delta_large), [0, 0], [batch_size, self.maxlen])                      
            self.apply_delta = tf.clip_by_value(self.delta, -self.rescale, self.rescale) 
            self.before_rir = tf.clip_by_value(self.apply_delta * self.mask + self.input_tf, -2**15, 2**15-1)
            self.new_input = create_speech_rir(self.before_rir, self.rir, self.lengths, self.maxlen, self.batch_size) * self.mask
            self.pass_in = tf.clip_by_value(self.new_input + self.noise, -2**15, 2**15-1)            
       
            # generate the inputs that are needed for the lingvo model
            self.features = create_features(self.pass_in, self.sample_rate_tf, self.mask_freq)
            self.inputs = create_inputs(model, self.features, self.tgt_tf, self.batch_size, self.mask_freq)  
        
            task = model.GetTask()
            metrics = task.FPropDefaultTheta(self.inputs)

            # self.celoss with the shape (batch_size)
            self.celoss = tf.get_collection("per_loss")[0]         
            self.decoded = task.Decode(self.inputs)       
        
        self.optimizer1 = tf.train.AdamOptimizer(self.lr)             
        grad1, var1 = self.optimizer1.compute_gradients(self.celoss, [self.delta_large])[0]             
        self.train1 = self.optimizer1.apply_gradients([(tf.sign(grad1), var1)])
             
        
    def attack_stage1(self, audios, trans, maxlen, sample_rate, masks, masks_freq, num_loop, data, lengths):
        """
        The first stage saves the adversarial examples that can successfully attack one room
        """
        
        sess = self.sess       
        # initialize and load the pretrained model
        sess.run(tf.initializers.global_variables())
        saver = tf.train.Saver([x for x in tf.global_variables() if x.name.startswith("librispeech")])
        saver.restore(sess, FLAGS.checkpoint)
                     
        # reassign the variables 
        sess.run(tf.assign(self.rescale, np.ones((self.batch_size, 1), dtype=np.float32) * FLAGS.initial_bound))        
        sess.run(tf.assign(self.delta_large, np.zeros((self.batch_size, FLAGS.max_length_dataset), dtype=np.float32)))
             
        noise = np.zeros(audios.shape)
        rir = Readrir()
        feed_dict = {self.input_tf: audios, 
                     self.tgt_tf: trans, 
                     self.sample_rate_tf: sample_rate,                   
                     self.mask: masks, 
                     self.mask_freq: masks_freq,
                     self.noise: noise, 
                     self.maxlen: maxlen,
                     self.lengths: lengths,
                     self.rir: rir,
                     self.lr: self.lr_stage1}
        losses, predictions = sess.run((self.celoss, self.decoded), feed_dict)
       
        # show the initial predictions 
        for i in range(self.batch_size):
            print("example: {}, loss: {}".format(num_loop * self.batch_size + i, losses[i]))
            print("pred:{}".format(predictions['topk_decoded'][i, 0]))
            print("targ:{}".format(trans[i].lower()))
            print("true: {}".format(data[1, i].lower()))
                   
       
        # We'll make a bunch of iterations of gradient descent here
        now = time.time()
        MAX = self.num_iter_stage1
        loss_th = [np.inf] * self.batch_size
        final_adv = [None] * self.batch_size
        final_perturb = [None] * self.batch_size
        clock = 0
        
        for i in range(1, MAX + 1):           
            now = time.time()

            rir = Readrir()
            feed_dict = {self.input_tf: audios, 
                     self.tgt_tf: trans, 
                     self.sample_rate_tf: sample_rate,                   
                     self.mask: masks, 
                     self.mask_freq: masks_freq,
                     self.noise: noise, 
                     self.maxlen: maxlen,
                     self.lengths: lengths,
                     self.rir: rir,
                     self.lr: self.lr_stage1}
            losses, predictions = sess.run((self.celoss, self.decoded), feed_dict)

            # Actually do the optimization                           
            sess.run(self.train1, feed_dict)   
            if i % 10 == 0:
                d, apply_delta, cl, predictions, new_input = sess.run((self.delta, self.apply_delta, self.celoss, self.decoded, self.new_input), feed_dict)
             
            for ii in range(self.batch_size): 
                if i % 100 == 0:
                    print("example: {}".format(num_loop * self.batch_size + ii))
                    print("iteration: {}. loss {}".format(i, cl[ii]))
                    print("pred:{}".format(predictions['topk_decoded'][ii, 0]))
                    print("targ:{}".format(trans[ii].lower()))
               
                if i % 10 == 0:                                      
                    if predictions['topk_decoded'][ii, 0] == trans[ii].lower():
                        print("-------------------------------True--------------------------")
                        rescale = sess.run(self.rescale)
                        # update rescale
                        if i % 10 == 0:
                            if rescale[ii]  > np.max(np.abs(d[ii])):                            
                                rescale[ii] = np.max(np.abs(d[ii]))                   
                            rescale[ii] *= .8
                            print("Iteration i=%d, worked ii=%d celoss=%f bound=%f"%(i, ii, cl[ii], rescale[ii]))   
                            sess.run(tf.assign(self.rescale, rescale))    

                        # save the best adversarial example
                        final_adv[ii] = new_input[ii]
                        final_perturb[ii] = apply_delta[ii]
                        print("Stage 1: save the example at iteration i=%d example ii=%d celoss=%f bound=%f" % (i, ii, cl[ii], rescale[ii]))
                                                                                                                                
                # in case no final_delta return        
                if (i == MAX-1 and final_adv[ii] is None):
                    final_adv[ii] = new_input[ii] 
                    final_perturb[ii] = apply_delta[ii]            
         
            if i % 10 == 0:
                print("ten iterations take around {} ".format(clock))
                clock = 0
             
            clock += time.time() - now
            
        return final_adv, final_perturb 

    def attack_stage2(self, audios, trans, adv, rescales, maxlen, sample_rate, masks, masks_freq, num_loop, data, lengths):       
        sess = self.sess       
        # initialize and load the pretrained model
        sess.run(tf.initializers.global_variables())
        saver = tf.train.Saver([x for x in tf.global_variables() if x.name.startswith("librispeech")])
        saver.restore(sess, FLAGS.checkpoint)
             
        # reassign the variables
        sess.run(tf.assign(self.delta_large, adv))  
        sess.run(tf.assign(self.rescale, rescales)) 
         
        noise = np.zeros(audios.shape)
        rir = Readrir()
        feed_dict = {self.input_tf: audios, 
                     self.tgt_tf: trans, 
                     self.sample_rate_tf: sample_rate,                   
                     self.mask: masks, 
                     self.mask_freq: masks_freq,
                     self.noise: noise, 
                     self.maxlen: maxlen,
                     self.lengths: lengths,
                     self.rir: rir,
                     self.lr: self.lr_stage2}
        losses, predictions = sess.run((self.celoss, self.decoded), feed_dict)
       
        # show the initial predictions 
        for i in range(self.batch_size):
            print("example: {}, loss: {}".format(num_loop * self.batch_size + i, losses[i]))
            print("pred:{}".format(predictions['topk_decoded'][i, 0]))
            print("targ:{}".format(trans[i].lower()))
            print("true: {}".format(data[1, i].lower()))
                   
       
        # We'll make a bunch of iterations of gradient descent here
        now = time.time()
        MAX = self.num_iter_stage2
        loss_th = [np.inf] * self.batch_size
        final_adv = [None] * self.batch_size
        final_perturb = [None] * self.batch_size
        num_counters = [FLAGS.num_counter] * self.batch_size
        num_rooms = [FLAGS.num_rooms] * self.batch_size
        clock = 0
        
        for i in range(1, MAX + 1):           
            now = time.time()

            rir = Readrir()
            feed_dict = {self.input_tf: audios, 
                     self.tgt_tf: trans, 
                     self.sample_rate_tf: sample_rate,                   
                     self.mask: masks, 
                     self.mask_freq: masks_freq,
                     self.noise: noise, 
                     self.maxlen: maxlen,
                     self.lengths: lengths,
                     self.rir: rir,
                     self.lr: self.lr_stage2}
            losses, predictions = sess.run((self.celoss, self.decoded), feed_dict)

            # Actually do the optimization                           
            sess.run(self.train1, feed_dict)   
            if i % 10 == 0:
                d, apply_delta, cl, predictions, new_input = sess.run((self.delta, self.apply_delta, self.celoss, self.decoded, self.new_input), feed_dict)
             
            for ii in range(self.batch_size): 
                if i % 100 == 0:
                    print("example: {}".format(num_loop * self.batch_size + ii))
                    print("iteration: {}. loss {}".format(i, cl[ii]))
                    print("pred:{}".format(predictions['topk_decoded'][ii, 0]))
                    print("targ:{}".format(trans[ii].lower()))
                               
                sum_counter = 0
                if i % 10 == 0:   
                    for counter in range(num_rooms[ii]): 
                        if predictions['topk_decoded'][ii, 0] == trans[ii].lower():   
                            sum_counter += 1    
                            print("succeed %d times for example %d" % (sum_counter, ii))                
                            rir = Readrir()
                            feed_dict = {self.input_tf: audios, 
                                         self.tgt_tf: trans, 
                                         self.sample_rate_tf: sample_rate,                   
                                         self.mask: masks, 
                                         self.mask_freq: masks_freq,
                                         self.noise: noise, 
                                         self.maxlen: maxlen,
                                         self.lengths: lengths,
                                         self.rir: rir,
                                         self.lr: self.lr_stage2}
                            predictions = sess.run(self.decoded, feed_dict)

                        if sum_counter == num_counters[ii]:                      
                            print("-------------------------------True--------------------------")                            
                            print(" The num_counter is %d for example %d" % (num_counters[ii], ii))
                            num_counters[ii] += 1
                            if num_counters[ii] > num_rooms[ii]:
                                num_rooms[ii] += 1
                            # save the best adversarial example
                            final_adv[ii] = new_input[ii]
                            final_perturb[ii] = apply_delta[ii]                           
                            print("Stage 2: save the example at iteration i=%d example ii=%d celoss=%f" % (i, ii, cl[ii]))
                                                                                                                                
                # in case no final_delta return        
                if (i == MAX-1 and final_adv[ii] is None):
                    final_adv[ii] = new_input[ii] 
                    final_perturb[ii] = apply_delta[ii]            
         
            if i % 10 == 0:
                print("ten iterations take around {} ".format(clock))
                clock = 0             
            clock += time.time() - now
            
        return final_adv, final_perturb

def main(argv):
    data = np.loadtxt(FLAGS.input, dtype=str, delimiter=",")
    data = data[:, FLAGS.num_gpu * 10 : (FLAGS.num_gpu + 1) * 10]
    num = len(data[0])
    batch_size = FLAGS.batch_size
    num_loops = num / batch_size 
    assert num % batch_size == 0
    
    with tf.device("/gpu:0"):
        tfconf = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=tfconf) as sess: 
            # set up the attack class
            attack = Attack(sess, 
                            batch_size=batch_size,
                            lr_stage1=FLAGS.lr_stage1,
                            lr_stage2=FLAGS.lr_stage2,
                            num_iter_stage1=FLAGS.num_iter_stage1,
                            num_iter_stage2=FLAGS.num_iter_stage2)

            for l in range(num_loops):
                data_sub = data[:, l * batch_size:(l + 1) * batch_size] 
                               
                # stage 1
                # all the output are numpy arrays
                audios, trans, maxlen, sample_rate, masks, masks_freq, lengths = ReadFromWav(data_sub, batch_size)                                                                      
                adv_example, perturb = attack.attack_stage1(audios, trans, maxlen, sample_rate, masks, masks_freq, l, data_sub, lengths)
                
                # save the adversarial examples in stage 1 that can only successfully attack one simulated room
                for i in range(batch_size):
                    print("Final distortion for stage 1", np.max(np.abs(adv_example[i][:lengths[i]] - audios[i, :lengths[i]])))                                      
                    name, _ = data_sub[0, i].split(".")                    
                    saved_name = FLAGS.root_dir + str(name) + "_robust_speechrir_stage1.wav"                     
                    adv_example_float =  adv_example[i] / 32768.
                    wav.write(saved_name, 16000, np.array(np.clip(adv_example_float[:lengths[i]], -2**15, 2**15-1)))

                    saved_name = FLAGS.root_dir + str(name) + "_robust_perturb_stage1.wav"                     
                    perturb_float =  perturb[i] / 32768.
                    wav.write(saved_name, 16000, np.array(np.clip(perturb_float[:lengths[i]], -2**15, 2**15-1)))
                    print(saved_name) 

                # stage 2                
                # read the adversarial examples saved in stage 1
                adv = np.zeros([batch_size, FLAGS.max_length_dataset])
                adv[:, :maxlen] = adv_example - audios
                rescales = np.max(np.abs(adv), axis=1) + FLAGS.max_delta
                rescales = np.expand_dims(rescales, axis=1)

                audios, trans, maxlen, sample_rate, masks, masks_freq, lengths = ReadFromWav(data_sub, batch_size)                                                                      
                adv_example, perturb = attack.attack_stage2(audios, trans, adv, rescales, maxlen, sample_rate, masks, masks_freq, l, data_sub, lengths)
                
                # save the adversarial examples in stage 2 that can successfully attack a set of simulated rooms
                for i in range(batch_size):
                    print("Final distortion for stage 2", np.max(np.abs(adv_example[i][:lengths[i]] - audios[i, :lengths[i]])))                                      
                    name, _ = data_sub[0, i].split(".")                    
                    saved_name = FLAGS.root_dir + str(name) + "_robust_speechrir_stage2.wav"                     
                    adv_example_float =  adv_example[i] / 32768.
                    wav.write(saved_name, 16000, np.array(np.clip(adv_example_float[:lengths[i]], -2**15, 2**15-1)))

                    saved_name = FLAGS.root_dir + str(name) + "_robust_perturb_stage2.wav"                     
                    perturb_float =  perturb[i] / 32768.
                    wav.write(saved_name, 16000, np.array(np.clip(perturb_float[:lengths[i]], -2**15, 2**15-1)))
                    print(saved_name) 


if __name__ == '__main__':
    app.run(main)
    
    
