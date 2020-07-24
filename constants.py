import os

ROOT_DIRECTORY='/ntfsl/data/ph'

SCORE_MATRIX = {
    'neg': 0,
    'type1': 1,
    'type2': 0.75,
    'type3': 1,
    'type4': 0.4,
}

SUBFOLDERS = list(SCORE_MATRIX.keys())

PREPROCESSED_SUBDIRECTORY=os.path.join(
    ROOT_DIRECTORY,
    'preprocessed'
)

INPUT_DIM=(256,256)
INPUT_DIM_RGB = (INPUT_DIM[0], INPUT_DIM[1], 3)


# scale inputs by 255 to achieve range of 0 to 1 instead of 0 to 255
DEFAULT_DIVIDE=True


SCHEMA='ph_img'

SOURCE_IMAGE_TABLENAME='source_img'
DISTORTION_IMAGE_TABLENAME='distortion_img_v3'
DISTORTION_FOLDER=os.path.join(
    ROOT_DIRECTORY,
    'distorted_v3'
)

MODEL_DIRECTORY=os.path.join(
    ROOT_DIRECTORY,
    'models'
)

MODEL_TABLENAME='models'

PG_DICT = {
    'SCHEMA': SCHEMA,
    'SOURCE_IMAGE_TABLENAME': SOURCE_IMAGE_TABLENAME,
    'DISTORTION_IMAGE_TABLENAME': DISTORTION_IMAGE_TABLENAME,
    'MODEL_TABLENAME': MODEL_TABLENAME,
}

# distortion sampling params
N_DISTORTION_ITERATIONS=2000
DISTORTION_SAMPLE_RATE=0.25

## training params

N_TRAINING_EPOCHS = 210
N_BATCHES_PER_EPOCH = 70
DEFAULT_TRAIN_BATCH_SIZE=80

TEST_SAMPLE_SIZE=3000
### distortion constants

#all of these are lower/upper bounds for a uniform distribution
from numpy import pi
import random
from copy import copy

warp_ax = (-14,24)
warp_ay = (-14,24)
warp_perx = (100,700)
warp_pery = (100,700)
warp_phax = (-100,100)
warp_phay = (-100,100)

wave_ax = (-14,20)
wave_ay = (-14,20)
wave_perx = (100,700)
wave_pery = (100,700)
wave_phax = (-190,190)
wave_phay = (-190,190)

rot_theta = (-50*pi/180.,50*pi/180.)
rot_offset_x = [-1,1]
rot_offset_y = [-1,1]

scale_x = [0.7,1.2]
scale_y = [0.7,1.2]
scale_x_offset = [-1,1]
scale_y_offset = [-1,1]

x_offset = [-15,15]
y_offset = [-15,15]

flip_chance = 0.5

#if tinting is used
rgb_shift = [-12,12]

#determines priority distribution for mappings

wave = (0,1)
warp = (0,1)
affine = (0,1)

#params for strokes
stroke_priority = 0.65#probability for stroke before distortion
max_strokes = 4
stroke_alpha_prob = 0.8
stroke_shape  = {'length_mean':15, 'length_sd':5, 'radius_mean':3, 'radius_sd':3, 'max_radius':7}
prob_palette=0.8#chance that borrows from palette of source image
stroke_kwargs = {'imp_shape':stroke_shape, 'prob_alpha':stroke_alpha_prob,
                 'prob_palette':prob_palette,'max_num_imps':max_strokes}

