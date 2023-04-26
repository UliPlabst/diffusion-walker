python ./python/main.py
echo "@@@ RUNNING UPSAMPLING"
python ./realesrgan/inference_realesrgan.py -n RealESRGAN_x4plus -i ./data -o ./data_up