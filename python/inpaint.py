import keras_cv
from tensorflow import keras
import tensorflow as tf
from PIL import Image, ImageFilter
from model import model
import numpy as np
from utils import save_images, set_data_dir, save_video
import math

class ZoomParams:
  mask_width = 120
  correction = 8
  
zoom_params = ZoomParams()

def shrink_and_paste_on_blank(current_image, mask_width):
  current_img = Image.fromarray(current_image)
  height = current_img.height
  width = current_img.width
  
  blank_image = np.zeros((width, height, 3), dtype=np.uint8)
  shrinked = current_img.resize((height - 2 * mask_width, width - 2 * mask_width))
  shrinked = np.array(shrinked)
  blank_image[mask_width : height-mask_width, mask_width : width-mask_width, :] = shrinked
  
  return blank_image
  
def create_mask(width, height, mask_width):
  global zoom_params
  correction = zoom_params.correction
  
  res = np.ones((1, height, width), dtype=np.uint8)
  res[:, 0 : mask_width + correction, :] = 0
  res[:, :, 0 : mask_width + correction] = 0
  res[:, height - mask_width:, :] = 0
  res[:, :, width - mask_width:] = 0
  res[:, height - mask_width:, width - mask_width:] = 0
  return res
  
def create_blend_mask(width, height, mask_width, blend):
  margin = blend
  mask = np.zeros((height, width), dtype=np.uint8)  
  mask[mask_width + margin : width - mask_width - margin, mask_width + margin : height - mask_width - margin] =255
  
  mask_img = Image.fromarray(mask)
  return mask_img.filter(ImageFilter.GaussianBlur(blend)).convert("L")
  
def create_zoom_out_interpolation(current_image, zoomout_image, num_frames = 120):
  global zoom_params
  
  mask_width = zoom_params.mask_width
  inner_img = Image.fromarray(current_image)
  outer_img = Image.fromarray(zoomout_image)
  
    
  width = inner_img.width
  height = inner_img.height
  
  factor = width / (width - 2 * mask_width)
  
  big_height = round(outer_img.height * factor)
  big_width = round(outer_img.width * factor)
  
  margin_left = math.floor((big_width - width) / 2)
  margin_top = math.floor((big_height - height) / 2)
  
  add_margin = 0
  
  big_img = outer_img.resize((big_height, big_width))
  big_image = np.array(big_img)
  
  blend_mask = create_blend_mask(big_width, big_height, margin_left, 10)
  
  blank_image = np.zeros((big_img.width, big_img.height, 3), dtype=np.uint8)
  blank_image[margin_left : margin_left + inner_img.width, margin_top : margin_top + inner_img.height, :] = inner_img
  blank_img = Image.fromarray(blank_image)
  blank_img.save("blank.jpg")
  blend_mask.save("blend.png")
  # big_image[
  #   margin_left + add_margin : margin_left + width - add_margin , 
  #   margin_top + add_margin : margin_top + height - add_margin, 
  #   :
  # ] = current_image[add_margin : width-add_margin, add_margin : width-add_margin]
  # big_img = Image.fromarray(big_image)
  big_img.paste(blank_img, blend_mask)
  big_img.save("./big.jpg")
  
  frames = [ inner_img ]
  width_step = (big_img.width - width) / num_frames
  height_step = (big_img.height - height) / num_frames
  for i in range(num_frames - 1, 1, -1):
    left = width_step * i / 2
    top = height_step * i / 2
    right = big_img.width - (width_step * i / 2)
    bottom = big_img.height - (width_step * i / 2)
    frame = big_img.crop([left, top, right, bottom])
    frame = frame.resize((height, width), resample = Image.LANCZOS)
    frames.append(frame)
  frames.append(outer_img)
  return frames
    
def create_zoom_out_image(current_image, prompt, num_resamples = 3) :
  global zoom_params
  current_image = shrink_and_paste_on_blank(current_image, zoom_params.mask_width)
  mask = create_mask(width, height, zoom_params.mask_width)

  return model.inpaint(
    prompt,
    current_image,
    mask,
    model.get_noise(),
    num_resamples=num_resamples
  )
  
  
width = 512
height = 512
  

set_data_dir("./data")

prompt = "A planet with a forest of towering mushrooms | scify | cyber punk | render | colorful | highly detailed | neon | sharp"
model.batch_size = 1
current_image = None
zoomed_out = None

current_image = None
for i in range(10):
  if(True):
    if(current_image is None):
      model.init()
      current_image = model.generate_image_batch(
        model.encode(prompt),
        model.get_noise()
      )[0]
    zoomed_out = create_zoom_out_image(current_image, prompt, num_resamples = 5)
  else:
    current_image = np.array(Image.open("./start.jpg"))
    zoomed_out = np.array(Image.open("./zoomed.jpg"))
    

  Image.fromarray(current_image).save("./start.jpg")
  Image.fromarray(zoomed_out).save("./zoomed.jpg")

  frames = create_zoom_out_interpolation(current_image, zoomed_out)
  save_images(frames)
  current_image = np.array(frames[-1])
  
save_video(framerate=30)