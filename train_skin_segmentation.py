# Try to make the training reproducible
seed = 0
import numpy as np
np.random.seed(seed)
import random as rn
rn.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)

# Setup allocating only as much GPU memory as needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(graph=tf.get_default_graph(), config=config)
from keras import backend
backend.tensorflow_backend.set_session(session)

import os
import datetime
from keras import optimizers, losses, metrics, callbacks
from keras.preprocessing.image import ImageDataGenerator

from model import unet
from losses import image_binary_crossentropy, image_categorical_crossentropy
from callbacks import TestPredictor
from data import ImageMaskGenerator, preprocess_image, preprocess_mask

# Adjustable hyperparameters
normalize_lighting = True
min_value, max_value = 0., 1.
augmentation_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)
background_as_class = True
use_custom_losses = False
optimizer = optimizers.Adam(lr=1e-4)
epochs = 100
train_batchsize = 2

# Other parameters
data_dir = "data/skin/revised"
train_dir = os.path.join(data_dir, "train")
validation_dir = os.path.join(data_dir, "validation")
test_dir = os.path.join(data_dir, "test")
height, width, channels = 480, 640, 3
classes = 1
validation_batchsize = 1
test_batchsize = 1
image_preprocessing = preprocess_image(normalize_lighting=normalize_lighting, min_value=min_value, max_value=max_value)

run_name = "run_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
weights_dir = os.path.join("weights", run_name)
if not os.path.isdir(weights_dir):
    print("Create directory " + weights_dir + " for saving weights")
    os.makedirs(weights_dir)
results_dir = os.path.join("results", run_name)
if not os.path.isdir(results_dir):
    print("Create directory " + results_dir + " for saving results")
    os.makedirs(results_dir)

print("\nTrain dataset statistics:")
train_generator = ImageMaskGenerator(
    train_dir,
    augmentation_args=augmentation_args,
    image_preprocessing=image_preprocessing,
    mask_preprocessing=preprocess_mask,
    background_as_class=background_as_class,
    target_size=(height, width),
    image_color_mode="rgb",
    mask_color_mode="grayscale",
    image_subdirectory="image",
    mask_subdirectory="label",
    batch_size=train_batchsize,
    seed=seed
)
print("\nValidation dataset statistics:")
validation_generator = ImageMaskGenerator(
    validation_dir,
    augmentation_args=augmentation_args,
    image_preprocessing=image_preprocessing,
    mask_preprocessing=preprocess_mask,
    background_as_class=background_as_class,
    target_size=(height, width),
    image_color_mode="rgb",
    mask_color_mode="grayscale",
    image_subdirectory="image",
    mask_subdirectory="label",
    batch_size=validation_batchsize,
    seed=seed
)
print("\nTest dataset statistics:")
test_generator = ImageDataGenerator(preprocessing_function=image_preprocessing).flow_from_directory(
    test_dir,
    target_size=(height, width),
    color_mode='rgb',
    classes=["."],
    class_mode=None,
    batch_size=test_batchsize,
    shuffle=False
)

model = unet(
    input_size=(height, width, channels),
    classes=classes,
    background_as_class=background_as_class
)
if background_as_class is True:
    if use_custom_losses is True:
        loss = image_categorical_crossentropy
    else:
        loss = losses.categorical_crossentropy
    metric = metrics.categorical_accuracy
    metric_name = "categorical_accuracy"
else:
    if use_custom_losses is True:
        loss = image_binary_crossentropy
    else:
        loss = losses.binary_crossentropy
    metric = metrics.binary_accuracy
    metric_name = "binary_accuracy"
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
# model.summary()

callbacks = [
    callbacks.ModelCheckpoint(os.path.join(weights_dir, 'best_loss.hdf5'),
                              monitor="loss", verbose=1, save_best_only=True),
    callbacks.ModelCheckpoint(os.path.join(weights_dir, 'best_acc.hdf5'),
                              monitor=metric_name, verbose=1, save_best_only=True),
    callbacks.ModelCheckpoint(os.path.join(weights_dir, 'best_val_loss.hdf5'),
                              monitor="val_loss", verbose=1, save_best_only=True),
    callbacks.ModelCheckpoint(os.path.join(weights_dir, 'best_val_acc.hdf5'),
                              monitor="val_" + metric_name, verbose=1, save_best_only=True),
    TestPredictor(test_generator, results_dir, save_prefix="out-", background_as_class=background_as_class)
]

print("\nTraining")
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size
)
print("Train loss:", history.history["loss"])
print("Train accuracy:", history.history[metric_name])
print("Train validation loss:", history.history["val_loss"])
print("Train validation accuracy:", history.history["val_" + metric_name])

print("\nEvaluation")
evaluation = model.evaluate_generator(
    validation_generator,
    steps=10 * validation_generator.samples / validation_generator.batch_size,
    verbose=1
)
print("Evaluation loss:", evaluation[0])
print("Evaluation accuracy:", evaluation[1])
