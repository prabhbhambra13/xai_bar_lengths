#!/usr/bin/env python
import yaml
import csv
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

mapping = pd.read_csv(f'{path}/gz2_filename_mapping.csv', header=0)
barlengths = pd.read_csv(f'{path}/hoyle_barlengths.csv', header=0)
metadata = pd.read_csv(f'{path}/gz2sample.csv', header=0)

barlengths = barlengths.set_index('objid').join(mapping.set_index('objid')).reset_index()
barlengths = barlengths.rename(columns={'asset_id': 'GalaxyID'})
barlengths['GalaxyID'] = barlengths['GalaxyID'].astype('str') + '.jpg'
barlengths = barlengths.set_index('objid').join(metadata[['OBJID', 'PETROR90_R']].set_index('OBJID')).reset_index()

barlengths = barlengths.sample(frac=1, random_state=seed).reset_index(drop=True)

test_images = int(barlengths.shape[0] * data_params['test_split'])
train_images = int(barlengths.shape[0] - test_images)

train = barlengths.head(train_images)
test = barlengths.tail(test_images)

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
    dataframe=train,
    directory=f'{path}/images_gz2/images',
    x_col='GalaxyID',
    y_col='length_scaled',
    target_size=(img_height, img_width),
    class_mode='raw',
    seed=seed,
    subset='training',
    shuffle=True,
    batch_size=batch_size
)

val_ds = datagen.flow_from_dataframe(
    dataframe=train,
    directory=f'{path}/images_gz2/images',
    x_col='GalaxyID',
    y_col='length_scaled',
    target_size=(img_height, img_width),
    class_mode='raw',
    seed=seed,
    subset='validation',
    shuffle=True,
    batch_size=batch_size
)

step_size_train = train_ds.n // train_ds.batch_size
step_size_val = val_ds.n // val_ds.batch_size

base_model = Xception(include_top=False, input_shape=(img_height, img_width, img_depth), pooling='avg')
x = layers.Dense(1, activation='sigmoid')(base_model.output)
model = Model(inputs=base_model.input, outputs=x)

for l in model.layers:
    if hasattr(l, 'activation'):
        if l.activation == activations.relu:
            l.activation = activations.swish

Path(f'./out/control/').mkdir(parents=True, exist_ok=True)

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
csv_logger = callbacks.CSVLogger(f'./out/control/training_log.csv')

model.fit(
    train_ds,
    steps_per_epoch=step_size_train,
    validation_data=val_ds,
    validation_steps=step_size_val,
    epochs=epochs,
    callbacks=[reduce_lr, early_stopping, csv_logger],
    verbose=2
)

model.save(f'./out/control/model.h5')
print('Model Saved')

datagen = ImageDataGenerator(
    rescale=1./255
)

test_ds = datagen.flow_from_dataframe(
    dataframe=test,
    directory=f'{path}/images_gz2/images',
    x_col='GalaxyID',
    y_col='length_scaled',
    target_size=(img_height, img_width),
    class_mode='raw',
    seed=seed,
    shuffle=True,
    batch_size=batch_size
)

predictions = model.predict(test_ds)

pred = pd.DataFrame(predictions, columns=['cnn_length'])
pred['GalaxyID'] = test_ds.filenames
pred = pred.set_index('GalaxyID').join(barlengths[['objid', 'length_scaled', 'PETROR90_R', 'GalaxyID']].set_index('GalaxyID')).reset_index()

col = pred['GalaxyID']
pred.drop(labels=['GalaxyID'], axis=1, inplace=True)
pred.insert(0, 'GalaxyID', col)
col = pred['objid']
pred.drop(labels=['objid'], axis=1, inplace=True)
pred.insert(0, 'objid', col)

pred['cnn_length'] = pred['cnn_length'] * pred['PETROR90_R']
pred['length_scaled'] = pred['length_scaled'] * pred['PETROR90_R']
pred.rename(columns={'length_scaled': 'hoyle_length'}, inplace=True)
pred.drop(labels=['PETROR90_R'], axis=1, inplace=True)

pred.to_csv(f'./out/control/predictions.csv', index=False)
print('Predictions Saved')

corr = (pred['hoyle_length']).corr(pred['cnn_length'])

if not Path(f'./out/barlength_correlations_test.csv').exists():
    with open(f'./out/barlength_correlations_test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model', 'smooth_noise', 'corr_threshold', 'len_threshold', 'xinput', 'correlation'])

with open(f'./out/barlength_correlations_test.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['control', 'n/a', 'n/a', 'n/a', 'n/a', corr])