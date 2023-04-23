# Setup
docker tensroflow-2.11.0-gpu
- `ln -s /usr/local/bin/pip /usr/bin/pip`
- `pip3 install --force-reinstall -v "tensorflow==2.11.0"`
- `pip3 install keras_cv`
- `pip3 install pillow matplotlib numpy`
- `pip install -v "protobuf==3.20.0"`
- Wait `while ps -p $PID > /dev/null; do sleep 1; done`
- Zip `tar -czvf data.tar.gz ./data`

- Fix update
```
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
```

- apt-get install ffmpeg libsm6 libxext6  -y
- wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip

# Local Setup
pip install pillow image numpy python-ffmpeg

&$env:ffmpeg -f concat -safe 0 -i ./tmp_list.txt -r 7 -c:v libx264 -crf 20 -pix_fmt yuv420p try2_up_part.mp4

# Frame interpolation
- Flowframes