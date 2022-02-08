#!/usr/bin/env python
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print('TensorFlow Version:', tf.__version__)
print('No. GPUs Available:', len(tf.config.list_logical_devices('GPU')))

tf.config.optimizer.set_jit(True)

with open('params.yaml', 'r') as fd:
    params = yaml.safe_load(fd)

data_params = params['data']
train_params = params['train']
xai_params = params['xai']

model_type = params['model'].lower()
path = data_params['path']
seed = data_params['random_seed']
seed = data_params['random_seed']
img_height = data_params['img_height']
img_width = data_params['img_width']
img_depth = data_params['img_depth']
batch_size = train_params['batch_size']

df = pd.read_csv(f'{path}/training_solutions_rev1.csv', header=0)
df['GalaxyID'] = df['GalaxyID'].astype('str') + '.jpg'

datagen = ImageDataGenerator(
    rescale=1./255
)

train_ds = datagen.flow_from_dataframe(
    dataframe=df,
    directory=f'{path}/images_training_rev1',
    x_col='GalaxyID',
    y_col=df.columns[1:],
    target_size=(img_height, img_width),
    class_mode=None,
    seed=seed,
    batch_size=batch_size,
    shuffle=False
)

test_ds = datagen.flow_from_directory(
    directory=f'{path}/images_test_rev1',
    target_size=(img_height, img_width),
    class_mode=None,
    seed=seed,
    batch_size=batch_size,
    shuffle=False
)

if model_type == 'vgg16' or model_type == 'resnet50v2' or model_type == 'xception':
    Path(f'./out/{model_type}/').mkdir(parents=True, exist_ok=True)

    model = load_model(f'./out/{model_type}/model.h5')
    predictions_train = model.predict(train_ds)
    predictions_test = model.predict(test_ds)
    
    pred = pd.DataFrame(predictions_train, columns=df.columns[1:])
    pred['GalaxyID'] = train_ds.filenames
    col = pred['GalaxyID']
    pred.drop(labels=['GalaxyID'], axis=1, inplace=True)
    pred.insert(0, 'GalaxyID', col)
    pred.to_csv(f'./out/{model_type}/predictions_train.csv', index=False)
    print('Train Predictions Saved')

    pred = pd.DataFrame(predictions_test, columns=df.columns[1:])
    pred['filenames'] = test_ds.filenames
    pred['GalaxyID'] = pred['filenames'].str.slice(5, -4).astype('int32')
    col = pred['GalaxyID']
    pred.drop(labels=['filenames'], axis=1, inplace=True)
    pred.drop(labels=['GalaxyID'], axis=1, inplace=True)
    pred.insert(0, 'GalaxyID', col)
    pred.to_csv(f'./out/{model_type}/predictions_test.csv', index=False)
    print('Test Predictions Saved')

elif model_type == 'ensemble':
    Path(f'./out/{model_type}/').mkdir(parents=True, exist_ok=True)

    predictions_train_v = pd.read_csv('./out/vgg16/predictions_train.csv')
    predictions_train_r = pd.read_csv('./out/resnet50v2/predictions_train.csv')
    predictions_train_x = pd.read_csv('./out/xception/predictions_train.csv')

    predictions_train = (predictions_train_v + predictions_train_r + predictions_train_x)
    predictions_train['GalaxyID'] = predictions_train_x['GalaxyID']
    for col in predictions_train.select_dtypes(include=['float64']).columns:
        predictions_train[col] = predictions_train[col] / 3
    print('Train Predictions Saved')

    predictions_test_v = pd.read_csv('./out/vgg16/predictions_test.csv')
    predictions_test_r = pd.read_csv('./out/resnet50v2/predictions_test.csv')
    predictions_test_x = pd.read_csv('./out/xception/predictions_test.csv')

    predictions_test = (predictions_test_x + predictions_test_r + predictions_test_v)
    predictions_test['GalaxyID'] = predictions_test_x['GalaxyID']
    for col in predictions_test.select_dtypes(include=['float64']).columns:
        predictions_test[col] = predictions_test[col] / 3
    print('Test Predictions Saved')
    
else:
    NotImplementedError('Please select one of "vgg16", "resnet50v2", "xception", or "ensemble" as the model.')
