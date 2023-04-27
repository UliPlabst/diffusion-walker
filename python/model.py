
import keras_cv
import tensorflow as tf
  
class Model:
  model: keras_cv.models.StableDiffusion
  batch_size = 3
  num_steps = 25
  noise_shape = (512 // 8, 512 // 8, 4)
  seed = 123
  
  def init_model(self):
    self.model = keras_cv.models.StableDiffusion(jit_compile=True)
    
  def encode(self, prompt):
    return tf.squeeze(self.model.encode_text(prompt))
  
  def generate_image_batch(self, encoding, noise):
    return self.model.generate_image(
      encoding,
      batch_size=self.batch_size,
      num_steps=self.num_steps,
      diffusion_noise=noise
    )
  
  def get_noise(self):
    return tf.random.normal(self.noise_shape, seed=self.seed, dtype=tf.float64)
  
  def inpaint(self, prompt, image, mask, noise):
    return self.model.inpaint(
      prompt,
      image,
      mask,
      diffusion_noise=noise,
      batch_size=1
    )[0]
    
    
model = Model()