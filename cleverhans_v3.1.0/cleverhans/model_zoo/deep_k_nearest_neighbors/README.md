# Deep k-Nearest Neighbors

Code in `dknn.py` shows how to reproduce one of the results from the
following paper, using the MNIST dataset. It can be
adapted to use other datasets and any attack supported by CleverHans.
The LSH backend used in the paper is FALCONN.

**Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning**
*Nicolas Papernot, Patrick McDaniel*

Deep neural networks(DNNs) enable innovative applications of machine learning
like image recognition, machine translation, or malware detection. However, deep
learning is often criticized for its lack of robustness in adversarial
settings(e.g., vulnerability to adversarial inputs) and general inability to
rationalize its predictions. In this work, we exploit the structure of deep
learning to enable new learning-based inference and decision strategies that
achieve desirable properties such as robustness and interpretability. We take a
first step in this direction and introduce the Deep k-Nearest Neighbors(DkNN).
This hybrid classifier combines the k-nearest neighbors algorithm with
representations of the data learned by each layer of the DNN: a test input is
compared to its neighboring training points according to the distance that
separates them in the representations. We show the labels of these neighboring
points afford confidence estimates for inputs outside the model's training
manifold, including on malicious inputs like adversarial examples--and therein
provides protections against inputs that are outside the models understanding.
This is because the nearest neighbors can be used to estimate the nonconformity
of, i.e., the lack of support for, a prediction in the training data. The
neighbors also constitute human-interpretable explanations of predictions. We
evaluate the DkNN algorithm on several datasets, and show the confidence
estimates accurately identify inputs outside the model, and that the
explanations provided by nearest neighbors are intuitive and useful in
understanding model failures.
