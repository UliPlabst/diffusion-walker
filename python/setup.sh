rm /etc/apt/sources.list.d/cuda.list
apt-key del 7fa2af80
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

ln -s /usr/local/bin/pip /usr/bin/pip

git config --global user.email "up@mogular.com"
git config --global user.name "UliPlabst"

pip install --force-reinstall -v "tensorflow==2.11.0"
pip install keras_cv
pip install pillow
pip install -v "protobuf==3.20.0"

apt update
apt install -y ffmpeg 

apt-get -y install libsm6 libxext6  
pip install basicsr
pip install facexlib
pip install gfpgan

git clone https://github.com/xinntao/Real-ESRGAN.git ./realesrgan --depth 1
cd /realesrgan
pip install -r requirements.txt
python setup.py develop
cd ..

python setup.py -v1 -gan
