from typing import List, Optional, Union
import numpy as np
import warnings
import numpy as np
import tensorflow as tf
import os
from PIL import Image

image_cnt = 0
data_dir = "./data"

class UtilParams:
   save_latest = True
   
util_params = UtilParams()

def set_data_dir(dir):
  global data_dir
  if(os.path.isdir(dir) == False):
    os.makedirs(dir)
  data_dir = dir

def set_image_cnt(num):
  global image_cnt
  image_cnt = num
  
def get_image_cnt():
  global image_cnt
  return image_cnt

def export_as_gif(filename, images, frames_per_second=10, rubber_band=False):
    if rubber_band:
        images += images[2:-1][::-1]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )
    
def interpolate_frames(image1, image2, frames):
  i1 = np.array(image1)
  i2 = np.array(image2)
  result = add_frames_linear_interp([i1, i2], nmb_frames_target=frames)
  images = [ Image.fromarray(img) for img in result ]
  save_images(images)
  return images

def save_images(images):
  global image_cnt
  for i in images:
    print(f"Saving image {image_cnt}")
    i.save(f"{data_dir}/{image_cnt}.jpg")
    image_cnt += 1
  if(util_params.save_latest == True and len(images) > 0):
    last = images[-1]
    last.save("./latest.jpg")
  return images
  
def ensure_batches(num):
  if(int(num) != num):
    raise Exception(f"{num} is not integer")
  return int(num)

def save_video(pattern = "%d.jpg", crf = 26, framerate = 7):
  os.system(f"ffmpeg -framerate {framerate} -i '{data_dir}/{pattern}' -c:v libx264 -crf {crf} -pix_fmt yuv420p result.mp4")

def add_frames_linear_interp(
        list_imgs: List[np.ndarray],
        fps_target: Union[float, int] = None,
        duration_target: Union[float, int] = None,
        nmb_frames_target: int = None):
    r"""
    Helper function to cheaply increase the number of frames given a list of images,
    by virtue of standard linear interpolation.
    The number of inserted frames will be automatically adjusted so that the total of number
    of frames can be fixed precisely, using a random shuffling technique.
    The function allows 1:1 comparisons between transitions as videos.
    Args:
        list_imgs: List[np.ndarray)
            List of images, between each image new frames will be inserted via linear interpolation.
        fps_target:
            OptionA: specify here the desired frames per second.
        duration_target:
            OptionA: specify here the desired duration of the transition in seconds.
        nmb_frames_target:
            OptionB: directly fix the total number of frames of the output.
    """

    # Sanity
    if nmb_frames_target is not None and fps_target is not None:
        raise ValueError("You cannot specify both fps_target and nmb_frames_target")
    if fps_target is None:
        assert nmb_frames_target is not None, "Either specify nmb_frames_target or nmb_frames_target"
    if nmb_frames_target is None:
        assert fps_target is not None, "Either specify duration_target and fps_target OR nmb_frames_target"
        assert duration_target is not None, "Either specify duration_target and fps_target OR nmb_frames_target"
        nmb_frames_target = fps_target * duration_target

    # Get number of frames that are missing
    nmb_frames_diff = len(list_imgs) - 1
    nmb_frames_missing = nmb_frames_target - nmb_frames_diff - 1

    if nmb_frames_missing < 1:
        return list_imgs

    list_imgs_float = [img.astype(np.float32) for img in list_imgs]
    # Distribute missing frames, append nmb_frames_to_insert(i) frames for each frame
    mean_nmb_frames_insert = nmb_frames_missing / nmb_frames_diff
    constfact = np.floor(mean_nmb_frames_insert)
    remainder_x = 1 - (mean_nmb_frames_insert - constfact)
    nmb_iter = 0
    while True:
        nmb_frames_to_insert = np.random.rand(nmb_frames_diff)
        nmb_frames_to_insert[nmb_frames_to_insert <= remainder_x] = 0
        nmb_frames_to_insert[nmb_frames_to_insert > remainder_x] = 1
        nmb_frames_to_insert += constfact
        if np.sum(nmb_frames_to_insert) == nmb_frames_missing:
            break
        nmb_iter += 1
        if nmb_iter > 100000:
            print("add_frames_linear_interp: issue with inserting the right number of frames")
            break

    nmb_frames_to_insert = nmb_frames_to_insert.astype(np.int32)
    list_imgs_interp = []
    for i in range(len(list_imgs_float) - 1):
        img0 = list_imgs_float[i]
        img1 = list_imgs_float[i + 1]
        list_imgs_interp.append(img0.astype(np.uint8))
        list_fracts_linblend = np.linspace(0, 1, nmb_frames_to_insert[i] + 2)[1:-1]
        for fract_linblend in list_fracts_linblend:
            img_blend = interpolate_linear(img0, img1, fract_linblend).astype(np.uint8)
            list_imgs_interp.append(img_blend.astype(np.uint8))
        if i == len(list_imgs_float) - 2:
            list_imgs_interp.append(img1.astype(np.uint8))

    return list_imgs_interp
    
    
def save_tensor(name, tensor):
   tf.io.write_file(name, tf.io.serialize_tensor(tensor))
   
def load_tensor(name, dtype = tf.dtypes.float16):
  serialized_tensor = tf.io.read_file(name)
  tensor = tf.io.parse_tensor(serialized_tensor, out_type=dtype)
  return tensor
  
def interpolate_linear(p0, p1, fract_mixing):
    r"""
    Helper function to mix two variables using standard linear interpolation.
    Args:
        p0:
            First tensor / np.ndarray for interpolation
        p1:
            Second tensor / np.ndarray  for interpolation
        fract_mixing: float
            Mixing coefficient of interval [0, 1].
            0 will return in p0
            1 will return in p1
            0.x will return a linear mix between both.
    """
    reconvert_uint8 = False
    if type(p0) is np.ndarray and p0.dtype == 'uint8':
        reconvert_uint8 = True
        p0 = p0.astype(np.float64)

    if type(p1) is np.ndarray and p1.dtype == 'uint8':
        reconvert_uint8 = True
        p1 = p1.astype(np.float64)

    interp = (1 - fract_mixing) * p0 + fract_mixing * p1

    if reconvert_uint8:
        interp = np.clip(interp, 0, 255).astype(np.uint8)

    return interp
    