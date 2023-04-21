import keras_cv
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import math
import random
from PIL import Image
from utils import set_data_dir
from prompts import set_prompt_file, set_prompt_transformer
from steps import run_steps, setup, restore

set_data_dir("./data")
set_prompt_file("./star_trek.md")

def prompt_transformer(x):
  return x + " | scify | cyber punk | render | colorful | highly detailed | neon | sharp"

set_prompt_transformer(prompt_transformer)
setup()
run_steps()