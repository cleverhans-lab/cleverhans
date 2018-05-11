# FGSM against Facenet
This directory contains an implementation of the FGSM attack against one of the face recognition CNNs of the Facenet library. You can find a detailed article [here](https://brunolopezgarcia.github.io/2018/05/09/Crafting-adversarial-faces.html).

## How to run

In order to run this example you need to do the following:

1. Install the Facenet library, download and align the LFW faces and download a pretrained Facenet model. You can do that following the next Facenet tutorial: https://github.com/davidsandberg/facenet/wiki/Validate-on-LFW
(Running the validate_on_lfw script is not necessary). The datasets and the models folders must be in the folder of this example.

2. Change the following line in the facenet_fgsm.py script with the name of the .pb file of the model you downloaded:
    ```
    model_path = "models/facenet/20170512-110547/20170512-110547.pb"
    ```

3. Run the script:
    ```
    python facenet_fgsm.py
    ```
