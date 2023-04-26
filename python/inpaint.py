import keras_cv
from tensorflow import keras
import tensorflow as tf

keras.mixed_precision.set_global_policy("mixed_float16")
model = keras_cv.models.StableDiffusion(img_height=128, img_width=128, jit_compile=True)

prompt = "Circuitous: A pattern that imitates the look of a winding or circuitous path, with a repeating pattern of curves and twists.. | colorful | abstract"

model.text_to_image(
  prompt,
  None,
  1,
  25
)

# model.inpaint(
#     "Circuitous: A pattern that imitates the look of a winding or circuitous path, with a repeating pattern of curves and twists.. | colorful | abstract",
    
# )