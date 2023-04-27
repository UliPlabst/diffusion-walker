import keras_cv
from tensorflow import keras
import tensorflow as tf
from PIL import Image
from model import model
import numpy as np

class ZoomParams:
  mask_width = 120
  
zoom_params = ZoomParams()

def shrink_and_paste_on_blank(current_image, mask_width):
  height = current_image.height
  width = current_image.width
  
  blank_image = np.zeros((width, height, 3), dtype=np.uint8)
  shrinked = current_image.resize((height - 2 * mask_width, width - 2 * mask_width))
  shrinked = np.array(shrinked)
  blank_image[mask_width : height-mask_width, mask_width : width-mask_width, :] = shrinked
  
  return blank_image
  
def create_mask(width, height, mask_width):
  correction = 8
  
  res = np.ones((1, height, width), dtype=np.uint8)
  res[:, 0 : mask_width + correction, :] = 0
  res[:, :, 0 : mask_width + correction] = 0
  res[:, height - mask_width:, :] = 0
  res[:, :, width - mask_width:] = 0
  res[:, height - mask_width:, width - mask_width:] = 0
  return res

def create_zoom_out_image(current_image, prompt):
  global zoom_params
  current_image = shrink_and_paste_on_blank(current_image, zoom_params.mask_width)
  mask = create_mask(width, height, zoom_params.mask_width)

  return model.inpaint(
    prompt,
    current_image,
    mask,
    model.get_noise()
  )
  
  
width = 512
height = 512
  
prompt = "Circuitous: A pattern that imitates the look of a winding or circuitous path, with a repeating pattern of curves and twists.. | colorful | abstract"
model.batch_size = 1
current_image = model.generate_image_batch(
  model.encode(prompt),
  model.get_noise()
)[0]

all_images = [ current_image ]

for i in range(10):
  img_curr = Image.fromarray(current_image)
  zoom_out = create_zoom_out_image(img_curr, prompt)
  
    
