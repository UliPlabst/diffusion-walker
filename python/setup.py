from tensorflow import keras
import sys
import os

def initV1():
  text_encoder_weights_fpath = keras.utils.get_file(
    origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_encoder.h5",  # noqa: E501
    file_hash="4789e63e07c0e54d6a34a29b45ce81ece27060c499a709d556c7755b42bb0dc4",  # noqa: E501
  )
  diffusion_model_weights_fpath = keras.utils.get_file(
    origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5",  # noqa: E501
    file_hash="8799ff9763de13d7f30a683d653018e114ed24a6a819667da4f5ee10f9e805fe",  # noqa: E501
  )
  
def initV2():
  text_encoder_weights_fpath = keras.utils.get_file(
    origin="https://huggingface.co/ianstenbit/keras-sd2.1/resolve/main/text_encoder_v2_1.h5",  # noqa: E501
    file_hash="985002e68704e1c5c3549de332218e99c5b9b745db7171d5f31fcd9a6089f25b",  # noqa: E501
  )
  diffusion_model_weights_fpath = keras.utils.get_file(
    origin="https://huggingface.co/ianstenbit/keras-sd2.1/resolve/main/diffusion_model_v2_1.h5",  # noqa: E501
    file_hash="c31730e91111f98fe0e2dbde4475d381b5287ebb9672b1821796146a25c5132d",  # noqa: E501
  )
  
  
def initRealesR():
  os.system('wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ./realesrgan/weights')
  
if("-gan" in sys.argv):
  initRealesR()
if("-v1" in sys.argv):
  initV1()
if("-v2" in sys.argv):
  initV2()