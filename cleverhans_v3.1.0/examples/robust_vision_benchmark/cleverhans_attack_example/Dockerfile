FROM python:3.6

# set workdir to the home directory
WORKDIR /root

# install required packages
RUN pip3 install --no-cache-dir foolbox
RUN pip3 install --no-cache-dir robust_vision_benchmark
RUN pip3 install --no-cache-dir -e git+http://github.com/tensorflow/cleverhans.git#egg=cleverhans
RUN pip3 install tensorflow

# add your model script
COPY main.py main.py
COPY utils.py utils.py

CMD ["python3", "./main.py"]
