import keras_cv
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import math
import random
from PIL import Image
from utils import add_frames_linear_interp, export_as_gif, save_images, ensure_batches, save_video, interpolate_frames, get_image_cnt, set_image_cnt, set_data_dir, save_tensor, load_tensor
from prompts import get_next_prompt, set_prompt_file, set_prompt_index, set_prompt_transformer

keras.mixed_precision.set_global_policy("mixed_float16")
batch_size = 3
model = keras_cv.models.StableDiffusion(jit_compile=True)
seed = 12345
noise_shape = (512 // 8, 512 // 8, 4)

start_encoding = None
start_noise = None
encoding = None
noise = None

def generate_noise_rotation_batch(noise1, noise2, steps, circle_fraction = 1):
  walk_scale_x = tf.cos(tf.linspace(0.0, circle_fraction * 2, steps) * math.pi)
  walk_scale_x = tf.cast(walk_scale_x, dtype=tf.float64)
  walk_scale_y = tf.sin(tf.linspace(0.0, circle_fraction * 2, steps) * math.pi)
  walk_scale_y = tf.cast(walk_scale_y, dtype=tf.float64)
  noise_x = tf.tensordot(walk_scale_x, noise1, axes=0)
  noise_y = tf.tensordot(walk_scale_y, noise2, axes=0)
  noise = tf.add(noise_x, noise_y)
  
  batches = ensure_batches(steps / batch_size)
  batched_noise = tf.split(noise, batches)
  return [batched_noise, noise[-1]]


def generate_interpolated_encodings(encoding_1, encoding_2, steps):
  global batch_size
  interpolated_encodings = tf.linspace(encoding_1, encoding_2, steps)
  batches = ensure_batches(steps / batch_size)
  return tf.split(interpolated_encodings, batches)
 
  
def generate_encoding_walk(encoding, steps, step_size):
  global model
  global batch_size
  step = tf.ones_like(encoding) * step_size
  walked_encodings = []
  for step_index in range(steps):
      walked_encodings.append(encoding)
      encoding += step
  walked_encodings = tf.stack(walked_encodings)
  batches = ensure_batches(steps / batch_size)
  batched_encodings = tf.split(walked_encodings, batches)
  return [batched_encodings, encoding]


def create_image_batch(encoding, noise):
  global model
  global batch_size
  images = [
    Image.fromarray(img)
    for img in model.generate_image(
        encoding,
        batch_size=batch_size,
        num_steps=25,
        diffusion_noise=noise
    )
  ]
  save_images(images)
  return images


def encode(prompt):
  global model
  return tf.squeeze(model.encode_text(prompt))
 
  
def walk_steps(encoding, noise, steps, step_size):
  batches = ensure_batches(steps / batch_size)
  [batched_encodings, res_encoding] = generate_encoding_walk(encoding, steps, step_size)
  
  allimages = []
  for batch in range(batches):
    images = create_image_batch(batched_encodings[batch], noise)
    allimages += images
  return [allimages, res_encoding]


def get_noise():
  global noise_shape
  global seed
  return tf.random.normal(noise_shape, seed=seed, dtype=tf.float64)


def walk_steps_return(encoding, noise, steps, step_size):
  [images, _] = walk_steps(encoding, noise, steps, step_size)
  reverse_images = images[:]
  reverse_images.reverse()
  
  save_images(reverse_images)
  all_images = []
  all_images += images
  all_images += reverse_images
  return [all_images, encoding]


def rotate_noise(encoding, noise1, noise2, steps, circle_fraction=1):
  global batch_size
  batches = ensure_batches(steps / batch_size)
  [batched_noise, result_noise] = generate_noise_rotation_batch(noise1, noise2, steps, circle_fraction = circle_fraction)
  images = []
  for batch in range(batches):
    images += create_image_batch(encoding, batched_noise[batch])
  return [images, result_noise]


def rotate_noise_iter(encoding, noise, steps, iterations = 1, circle_fraction = .25):
  global batch_size
  global noise_shape
  global seed
  all_images = []
  for i in range(iterations):
    noise2 = get_noise()
    [images, result_noise] = rotate_noise(encoding, noise, noise2, steps, circle_fraction = circle_fraction)
    all_images += images
    noise = result_noise
  return [all_images, noise]
  
  
def change_noise_with_walk(encoding, noise1, noise2, step_size, steps, interpolation_steps):
  [walk_images, result_encoding] = walk_steps(encoding, noise1, steps, step_size)
  image_cnt = get_image_cnt()
  set_image_cnt(image_cnt + interpolation_steps)
  [return_walk_images, _] = walk_steps(result_encoding, noise2, steps, -1 * step_size)
  set_image_cnt(image_cnt)
  interpolated_images = interpolate_frames(walk_images[-1], return_walk_images[0], interpolation_steps)
  set_image_cnt(image_cnt + interpolation_steps + steps)

  all_images = []
  all_images += walk_images
  all_images += interpolated_images
  all_images += return_walk_images
  return [all_images, noise2]


def interpolate_encodings_and_rotate_noise(encoding1, encoding2, noise1, noise2, steps, circle_fraction = 1):
  global batch_size
  batches = ensure_batches(steps / batch_size)
  batched_encodings = generate_interpolated_encodings(encoding1, encoding2, steps)
  [batched_noise, result_noise] = generate_noise_rotation_batch(noise1, noise2, steps, circle_fraction = circle_fraction)
  images = []
  for batch in range(batches):
    images += create_image_batch(batched_encodings[batch], batched_noise[batch])
  return [images, encoding2, result_noise]


def interpolate_encodings(encoding1, encoding2, noise, steps):
  global batch_size
  batches = ensure_batches(steps / batch_size)
  batched_encodings = generate_interpolated_encodings(encoding1, encoding2, steps)
  images = []
  for batch in range(batches):
    images += create_image_batch(batched_encodings[batch], noise)
  return [images, encoding2]


def get_steps(min, max):
  global batch_size
  steps = random.randint(min, max)
  steps = steps // batch_size * batch_size
  print(f"@@ steps = {steps}")
  return steps


def get_float_param(min, max, name):
  res = random.uniform(min, max)
  print(f"@@ param {name} = {res}")
  return res


def get_int_param(min, max, name):
  res = random.randint(min, max)
  print(f"@@ param {name} = {res}")
  return res
  
  
def get_step_fn():
  rn = random.randint(0, 100)
  if(rn < 10):
    return 1 # 10% change noise with walk
  elif(rn < 40):
    return 2 # 30% interpolate encodings
  elif(rn < 60):
    return 3 # 20% interpolate and rotate
  elif(rn < 75):
    return 4 # 15% rotate
  elif(rn < 85):
    return 5 # 10% rotate iter
  elif(rn < 95):
    return 6 # 5%
  else:
    return 7 # 5%
  

def next_step(current_encoding, current_noise):
  step = get_step_fn()
  
  if(step == 1): #change_noise_with_walk
    print("@@ [step fn] change_noise_with_walk")
    steps = get_steps(42, 60)
    noise_2 = get_noise()
    distance = get_float_param(0.25, 0.25, "distance")
    step_size = distance / steps
    [_, res_noise] = change_noise_with_walk(current_encoding, current_noise, noise_2, step_size, steps, 12)
    current_noise = res_noise
    
  elif(step == 2): #interpolate_encodings
    print("@@ [step fn] interpolate_encodings")
    steps = get_steps(180, 360)
    prompt = get_next_prompt()
    if(prompt is None):
      return None
    encoding_2 = encode(prompt)
    [_, res_encoding] = interpolate_encodings(current_encoding, encoding_2, current_noise, steps)
    current_encoding = res_encoding
    
  elif(step == 3): #interpolate_encodings_and_rotate_noise
    print("@@ [step fn] interpolate_encodings_and_rotate_noise")
    steps = get_steps(180, 360)
    prompt = get_next_prompt()
    if(prompt is None):
      return None
    encoding_2 = encode(prompt)
    noise_2 = get_noise()
    [_, res_encoding, res_noise] = interpolate_encodings_and_rotate_noise(current_encoding, encoding_2, current_noise, noise_2, steps, .25)
    current_encoding = res_encoding
    current_noise = res_noise
    
  elif(step == 4): #rotate_noise
    print("@@ [step fn] rotate_noise")
    steps = get_steps(150, 210)
    noise_2 = get_noise()
    [_, result_noise] = rotate_noise(current_encoding, current_noise, noise_2, steps, 1)
    current_noise = result_noise
    
  elif(step == 5): #rotate_noise_iter
    steps = get_steps(90, 120)
    iter = get_int_param(1, 3, "iter")
    steps = steps * iter
    print("@@ [step fn] rotate_noise_iter")
    [_, result_noise] = rotate_noise_iter(current_encoding, current_noise, steps, iter, .25)
    current_noise = result_noise
    
  elif(step == 6): #walk_steps_return with positive step
    print("@@ [step fn] walk_steps_return with positive step")
    distance = get_float_param(0.1, 0.15, "distance")
    steps = get_steps(42, 60)
    step_size = distance / steps
    [_, result_encoding] = walk_steps_return(current_encoding, current_noise, steps,  step_size)
    current_encoding = result_encoding
    
  elif(step == 7): #walk_steps_return with negative step
    print("@@ [step fn] walk_steps_return with negative step")
    distance = get_float_param(0.1, 0.15, "distance")
    steps = get_steps(42, 60)
    step_size = distance / steps
    [_, result_encoding] = walk_steps_return(current_encoding, current_noise, steps,  -1 * step_size)
    current_encoding = result_encoding
    
  return [current_encoding, current_noise]


def restore(prompt_index, image_count):
  print(f"Restoring with prompt_index {prompt_index} and image_count {image_count}")
  global start_encoding
  global start_noise
  global encoding
  global noise
  
  set_prompt_index(prompt_index)
  set_image_cnt(image_count)
  
  start_encoding = load_tensor("./start_encoding")
  start_noise = load_tensor("./start_noise", tf.dtypes.float64)
  
  encoding = load_tensor("./current_encoding")
  noise = load_tensor("./current_noise", tf.dtypes.float64)


def setup():
  global start_encoding
  global start_noise
  global encoding
  global noise
  start_encoding = encode(get_next_prompt())
  start_noise = get_noise()
  save_tensor("./start_encoding", start_encoding)
  save_tensor("./start_noise", start_noise)
  encoding = start_encoding
  noise = start_noise


def run_steps(end_steps = 120):
  global encoding
  global start_encoding
  global noise
  global start_noise
  step = 0
  while(True):
    res = next_step(encoding, noise)
    if(res is None):
      break
    [encoding, noise] = res
    print(f"Finished step {step}, image_cnt={get_image_cnt()}")
    save_tensor("./current_encoding", encoding)
    save_tensor("./current_noise", noise)
    step += 1

  print("@@ [step fn] Moving to origin")
  interpolate_encodings_and_rotate_noise(encoding, start_encoding, noise, start_noise, end_steps, .25)



set_data_dir("./data1")
set_prompt_file("./star_trek.md")

def prompt_transformer(x):
  return x + "| scify | cyber punk | render | colorful | highly detailed | neon | sharp"

set_prompt_transformer(prompt_transformer)
run_steps()