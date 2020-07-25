# Image Regression Pipeline

## Overview

These scripts allow one to build a neural network model for purposes of predicting a numeric outcome of an image.

Each instance of this script is meant to be tuned to one specific model for training purposes.

Note that absolute paths are preferred when a path/file is required as input. The only exception is in the constants when a directory is indicated as a subdirectory of an already-defined directory.

### Future Note

I will later extend this to classification, which is mostly trivial (it requires possibly adding another option to the training script/constants, as well as changing the final layer of the network and its loss function). It can work pretty well for binary classification as it is, though.

## Steps to run

### Set up GPU libraries (optional)

These scripts run much faster when you have CUDA/CudNN set up with an NVIDIA GPU. Keras and Tensorflow are used for the computational heavy-lifting. You can generally set these up with two installs: One for the NVIDIA driver, and one for CUDA.

Make sure that you have the right driver version for the right CUDA version. Tensorflow is picky about which version of CUDA to use. As of this script, I am using CUDA 10.1 (July 2020).

You should be able to verify that you have a GPU working by following instructions from here: https://stackoverflow.com/a/48394195/1362215

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())


### Install `requirements.txt`


Note: If using Anaconda, the environment may also install CUDA for you. pip will not.

    sudo pip3 install -r requirements.txt

You may need to install additional packages to get OpenCV 2 to run.

    sudo apt install python3-opencv

Should install it for you, and you should be able to import it with

    python3 -c "import cv2; print(cv2.__version__)"


### `dbinfo.py`

You will need to set up a PostgreSQL database for this script to work, since the image and model metadata are stored for relatively straightforward use.

1. See https://www.postgresqltutorial.com/install-postgresql/ for installation help.

2. In `dbinfo.py`, replace the values with your dbname, username, host, and password (and port if it's not 5432). A more robust method of storing your password is by saving it as an environmental variable on your computer/server, which you can easily extend the script to do by using `os.environ['NAME_OF_ENVIRONMENT_VARIABLE']`.


### `constants.py`

1. Gather various images, and place them all in different folders in the same directory. e.g., `/mydirectory/group1`, `/mydirectory/group2`, etc.

2. Change `ROOT_DIRECTORY` to the root directory of your data. This is where you will store all your (folders of) images for training.

3. Change `SCORE_MATRIX` to have the names of each directory as a key, and the score to assign to it as the value.

4. Make sure that `INPUT_DIM` matches the dimensions of your image, (HEIGHT, WIDTH).

5. If you are using an alpha channel, change the third element of `INPUT_DIM_RGB` to `4` instead of `3`.

6. Change `SCHEMA` to the name of the schema you want to use in Postgres. This will help prevent table clobbering if you want to have multiple sets of this script that don't interfere with each other.

7. Adjust the batch size if needed. If your batch size is too large, it is quite possible for your GPU (or RAM if running on CPU) to be unable to handle the large amount of data. This also goes for the test sample size and RAM. Note that there is a RAM spike just before training in each round.

8. Change any other constants that you need to. The constants under `### distortion constants` are set to work with 256 x 256 images. With the exception of `rot_theta`, `rot_offset_x`, `rot_offset_y`, `scale_x`, and `scale_y`, anything above `flip_chance` can probably be scaled linearly with respect to the size of the image. I will explain what these parameters do in a later section.


## Preprocess the data with `preprocess.py`

The first piece of code needed to run is `preprocess.py`. Simply run this script, and it will create a folder with all your input images resized.

    python3 preprocess.py

You may want to modify this script in case you want to do other operations, such as cropping or creating multiple cropped subsections per image.

## Create distorted sample

In order to get the most out of your data, especially if it is a small sample, this step will create myriads of mutations of the input data. These are the different operations:

 * Scaling (expanding and shrinking)

 * Rotating (about center and around center)

 * Shifting left/right

 * Adding a transverse sine wave distort

 * Adding a longitudinal sine wave distort

 * Flipping the image

 * Adding random strokes

 * Adjusting tint (currently disabled)

 * Randomize the order of the first 3 (affine) transformations


Simply run

    python3 prepare_distorted_images.py

You can also run

    python3 prepare_distorted_images.py reset

to remove previously distorted images and their corresponding metadata in SQL.

You may cancel this script any time after the first set of images has been created. These will be used as the training sample in your data, as well as a moving test sample. (I am still working on getting an ideal test sampling procedure).

## Adjust the network

In the future I will make it a bit easier to work with the scripts without modifying them, but you may modify `network.py` and/or `train_model.py` in order to change how the network is trained.

`network.py` has the actual layers defined. You may find it more useful to completely redefine the wrapper for the network and build the network yourself, especially if you want to use skip-layers.

`train_model.py` I mostly change for adjusting the loss function if needed.

## Train model with `train_model.py`

Run

    python3 train_model.py

You can optionally include the path of a `.h5` keras network object to instantiate the network from a previously defined one. This will disregard any network-specific changes you made to `network.py` and `train_model.py`.

This script can be stopped with a **SINGLE** `ctrl+C` command. **Wait for the model to save**, and then the script will stop/can be stopped safely. The model will be recorded in the models table, and you can provide the path of that model to continue training the model.


## Use the model to evaluate on other images.

Simply run

    python3 run_model.py [input pattern] [output_filename] --model-id=[model_id]

The `--model-id` will be in the `[SCHEMA].models table`. The first model you make will have an ID of 1, the second 2, etc.

You can also provide a `--model-filename` argument instead to directly reference the `.h5` file containing the network. You can also change the `--model-tablename` argument to match a different table (including schema) that points to the correct model. This makes such a script more interoperable when you have different models trained for different purposes.


The output will have 2 columns

* full_path: path of the input file

* score: score given to the image