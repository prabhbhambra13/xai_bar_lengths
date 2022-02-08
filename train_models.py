#!/usr/bin/env python
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import Xception, ResNet50V2, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print('TensorFlow Version:', tf.__version__)
print('No. GPUs Available:', len(tf.config.list_logical_devices('GPU')))

tf.config.optimizer.set_jit(True)

with open('params.yaml', 'r') as fd:
    params = yaml.safe_load(fd)

data_params = params['data']
train_params = params['train']
xai_params = params['xai']
augmentation_params = data_params['augmentation']

model_type = params['model'].lower()
path = data_params['path']
seed = data_params['random_seed']
img_height = data_params['img_height']
img_width = data_params['img_width']
img_depth = data_params['img_depth']
batch_size = train_params['batch_size']
epochs = train_params['epochs']
lr = train_params['optimizer']['lr']
decay = train_params['optimizer']['decay']
factor = train_params['callbacks']['reduce_lr']['factor']
patience_lr = train_params['callbacks']['reduce_lr']['patience']
patience_es = train_params['callbacks']['early_stopping']['patience']

df = pd.read_csv(f'{path}/training_solutions_rev1.csv', header=0)
df['GalaxyID'] = df['GalaxyID'].astype('str') + '.jpg'
n_classes = len(df.columns) - 1

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=augmentation_params['rotation_range'],
    fill_mode=augmentation_params['fill_mode'],
    horizontal_flip=augmentation_params['horizontal_flip'],
    vertical_flip=augmentation_params['vertical_flip'],
    width_shift_range=augmentation_params['width_shift_range'],
    height_shift_range=augmentation_params['height_shift_range'],
    brightness_range=augmentation_params['brightness_range'],
    validation_split=augmentation_params['validation_split']
)

train_ds = datagen.flow_from_dataframe(
    dataframe=df,
    directory=f'{path}/images_training_rev1',
    x_col='GalaxyID',
    y_col=df.columns[1:],
    target_size=(img_height, img_width),
    class_mode='raw',
    seed=seed,
    subset='training',
    shuffle=True,
    batch_size=batch_size
)

val_ds = datagen.flow_from_dataframe(
    dataframe=df,
    directory=f'{path}/images_training_rev1',
    x_col='GalaxyID',
    y_col=df.columns[1:],
    target_size=(img_height, img_width),
    class_mode='raw',
    seed=seed,
    subset='validation',
    shuffle=True,
    batch_size=batch_size
)

step_size_train = train_ds.n // train_ds.batch_size
step_size_val = val_ds.n // val_ds.batch_size

if model_type == 'vgg16':
    base_model = VGG16(include_top=False, input_shape=(img_height, img_width, img_depth))
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(16*n_classes, activation='swish')(x)
    x = layers.Dense(8*n_classes, activation='swish')(x)
    x = layers.Dense(n_classes, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)

elif model_type == 'resnet50v2':
    base_model = ResNet50V2(include_top=False, input_shape=(img_height, img_width, img_depth), pooling='avg')
    x = layers.Dense(n_classes, activation='sigmoid')(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)

elif model_type == 'xception':
    base_model = Xception(include_top=False, input_shape=(img_height, img_width, img_depth), pooling='avg')
    x = layers.Dense(n_classes, activation='sigmoid')(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    
else:
    NotImplementedError('Please select one of "vgg16", "resnet50v2", or "xception" as the model.')

for l in model.layers:
    if hasattr(l, 'activation'):
        if l.activation == activations.relu:
            l.activation = activations.swish

Path(f'./out/{model_type}/').mkdir(parents=True, exist_ok=True)

adam = optimizers.Adam(lr=lr, decay=decay)
mse = losses.MeanSquaredError()
rmse = metrics.RootMeanSquaredError()

model.compile(
    optimizer=adam,
    loss=mse,
    metrics=[rmse]
)

model.summary()

reduce_lr = callbacks.ReduceLROnPlateau(factor=factor, patience=patience_lr)
early_stopping = callbacks.EarlyStopping(patience=patience_es)
csv_logger = callbacks.CSVLogger(f'./out/{model_type}/training_log.csv')

model.fit(
    train_ds,
    steps_per_epoch=step_size_train,
    validation_data=val_ds,
    validation_steps=step_size_val,
    epochs=epochs,
    callbacks=[reduce_lr, early_stopping, csv_logger],
    verbose=2
)

model.save(f'./out/{model_type}/model.h5')
print('Model Saved')