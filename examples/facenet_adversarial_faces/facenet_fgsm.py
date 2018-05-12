import facenet

import tensorflow as tf
import numpy as np
from cleverhans.model import Model
from cleverhans.attacks import FastGradientMethod

import set_loader


class InceptionResnetV1Model(Model):
    model_path = "models/facenet/20170512-110547/20170512-110547.pb"

    def __init__(self):
        super(InceptionResnetV1Model, self).__init__()

        # Load Facenet CNN
        facenet.load_model(self.model_path)
        # Save input and output tensors references
        graph = tf.get_default_graph()
        self.face_input = graph.get_tensor_by_name("input:0")
        self.embedding_output = graph.get_tensor_by_name("embeddings:0")

    def convert_to_classifier(self):
        # Create victim_embedding placeholder
        self.victim_embedding_input = tf.placeholder(
            tf.float32,
            shape=(None, 128))

        # Squared Euclidean Distance between embeddings
        distance = tf.reduce_sum(
            tf.square(self.embedding_output - self.victim_embedding_input),
            axis=1)

        # Convert distance to a softmax vector
        # 0.99 out of 4 is the distance threshold for the Facenet CNN
        threshold = 0.99
        score = tf.where(
            distance > threshold,
            0.5 + ((distance - threshold) * 0.5) / (4.0 - threshold),
            0.5 * distance / threshold)
        reverse_score = 1.0 - score
        self.softmax_output = tf.transpose(tf.stack([reverse_score, score]))

        # Save softmax layer
        self.layer_names = []
        self.layers = []
        self.layers.append(self.softmax_output)
        self.layer_names.append('probs')

    def fprop(self, x, set_ref=False):
        return dict(zip(self.layer_names, self.layers))


with tf.Graph().as_default():
    with tf.Session() as sess:
        # Load model
        model = InceptionResnetV1Model()
        # Convert to classifier
        model.convert_to_classifier()

        # Load pairs of faces and their labels in one-hot encoding
        faces1, faces2, labels = set_loader.load_testset(1000)

        # Create victims' embeddings using Facenet itself
        graph = tf.get_default_graph()
        phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
        feed_dict = {model.face_input: faces2,
                     phase_train_placeholder: False}
        victims_embeddings = sess.run(
            model.embedding_output, feed_dict=feed_dict)

        # Define FGSM for the model
        steps = 1
        eps = 0.01
        alpha = eps / steps
        fgsm = FastGradientMethod(model)
        fgsm_params = {'eps': alpha,
                       'clip_min': 0.,
                       'clip_max': 1.}
        adv_x = fgsm.generate(model.face_input, **fgsm_params)

        # Run FGSM
        adv = faces1
        for i in range(steps):
            print("FGSM step " + str(i + 1))
            feed_dict = {model.face_input: adv,
                         model.victim_embedding_input: victims_embeddings,
                         phase_train_placeholder: False}
            adv = sess.run(adv_x, feed_dict=feed_dict)

        # Test accuracy of the model
        batch_size = graph.get_tensor_by_name("batch_size:0")

        feed_dict = {model.face_input: faces1,
                     model.victim_embedding_input: victims_embeddings,
                     phase_train_placeholder: False,
                     batch_size: 64}
        real_labels = sess.run(model.softmax_output, feed_dict=feed_dict)

        accuracy = np.mean(
            (np.argmax(labels, axis=-1)) == (np.argmax(real_labels, axis=-1))
        )
        print('Accuracy: ' + str(accuracy * 100) + '%')

        # Test accuracy against adversarial examples
        feed_dict = {model.face_input: adv,
                     model.victim_embedding_input: victims_embeddings,
                     phase_train_placeholder: False,
                     batch_size: 64}
        adversarial_labels = sess.run(
            model.softmax_output, feed_dict=feed_dict)

        same_faces_index = np.where((np.argmax(labels, axis=-1) == 0))
        different_faces_index = np.where((np.argmax(labels, axis=-1) == 1))

        accuracy = np.mean(
            (np.argmax(labels[same_faces_index], axis=-1)) ==
            (np.argmax(adversarial_labels[same_faces_index], axis=-1))
        )
        print('Accuracy against adversarial examples for '
              + 'same person faces (dodging): '
              + str(accuracy * 100)
              + '%')

        accuracy = np.mean(
            (np.argmax(labels[different_faces_index], axis=-1)) == (
                np.argmax(adversarial_labels[different_faces_index], axis=-1))
        )
        print('Accuracy against adversarial examples for '
              + 'different people faces (impersonation): '
              + str(accuracy * 100)
              + '%')
