FROM runpod/tensorflow

SHELL ["/bin/bash", "-c"]
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

# RUN apt upgrade -y
RUN ln -s /usr/local/bin/pip /usr/bin/pip


# Stable diffusion
RUN pip install --force-reinstall -v "tensorflow==2.11.0"
RUN pip install keras_cv
RUN pip install pillow numpy
RUN pip install -v "protobuf==3.20.0"
RUN apt update
RUN apt intall -y ffmpeg

# Upscaling
RUN git clone https://github.com/xinntao/Real-ESRGAN.git /realesrgan --depth 1
RUN apt-get -y install libsm6 libxext6  
RUN pip install basicsr
RUN pip install facexlib
RUN pip install gfpgan
WORKDIR /realesrgan
RUN pip install -r requirements.txt
RUN python setup.py develop
WORKDIR /
ADD setup.py setup.py
RUN python setup.py
ADD python python