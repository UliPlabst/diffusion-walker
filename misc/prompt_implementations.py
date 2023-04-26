
#star trek
def prompt_transformer(x):
  return x + " | scify | cyber punk | render | colorful | highly detailed | neon | sharp"

#rohrschach
params.change_noise_with_walk.probability = 0
params.interpolate_encodings.probability = 0
params.interpolate_encodings_and_rotate_noise.probability = 0
params.rotate_noise.probability = 50
params.rotate_noise.step_min = 180
params.rotate_noise.step_max = 360
params.rotate_noise_iter.probability = 40
params.rotate_noise_iter.step_min = 100
params.rotate_noise_iter.step_max = 140
params.encoding_walk_neg_params.probability = 5
params.encoding_walk_pos_params.probability = 5

validate_params()
setup()
run_steps(60)

#patterns
def prompt_transformer(x):
  return x + " | colorful | abstract"

set_prompt_transformer(prompt_transformer)

params.change_noise_with_walk.probability = 10
params.interpolate_encodings.probability = 10
params.interpolate_encodings_and_rotate_noise.probability = 10
params.rotate_noise.probability = 40
params.rotate_noise.step_min = 180
params.rotate_noise.step_max = 360
params.rotate_noise_iter.probability = 20
params.rotate_noise_iter.step_min = 140
params.rotate_noise_iter.step_max = 160
params.encoding_walk_neg_params.probability = 5
params.encoding_walk_pos_params.probability = 5