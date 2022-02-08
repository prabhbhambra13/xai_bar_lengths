#!/usr/bin/env python
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tf_keras_vis.saliency import Saliency

def lookup_class(class_name):
    return df.columns.get_loc(class_name) - 1

def class_score(output, class_index):
    return output[:,class_index]

def score(class_index):
    return lambda x: class_score(x, class_index)

def model_modifier(m):
    m.layers[-1].activation = activations.linear
    return m

def barlength_cams(galaxy_id, model):

    img = load_img(f'{path}/images_gz2/images/{galaxy_id}')
    img = img_to_array(img) * 1./255
    img = tf.expand_dims(img, axis=0)
    img = tf.image.resize(img, [img_height, img_width])

    class_index = lookup_class('Class3.1')
    
    saliency = Saliency(
        model,
        model_modifier=model_modifier,
        clone=False
    )
    cam = saliency(
        score(class_index),
        img,
        smooth_samples=smooth_samples,
        smooth_noise=smooth_noise
    )

    cam = (cam - cam.min()) / (cam.max() - cam.min())
    tf.keras.backend.clear_session()

    return cam

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
img_height = data_params['img_height']
img_width = data_params['img_width']
img_depth = data_params['img_depth']
smooth_samples = xai_params['smoothgrad']['smooth_samples']
smooth_noise = xai_params['smoothgrad']['smooth_noise']

df = pd.read_csv(f'{path}/training_solutions_rev1.csv', header=0)
df['GalaxyID'] = df['GalaxyID'].astype('str') + '.jpg'

mapping = pd.read_csv(f'{path}/gz2_filename_mapping.csv', header=0)
barlengths = pd.read_csv(f'{path}/hoyle_barlengths.csv', header=0)
metadata = pd.read_csv(f'{path}/gz2sample.csv', header=0)

barlengths = barlengths.set_index('objid').join(mapping.set_index('objid')).reset_index()
barlengths = barlengths.rename(columns={'asset_id': 'GalaxyID'})
barlengths['GalaxyID'] = barlengths['GalaxyID'].astype('str') + '.jpg'
barlengths = barlengths.set_index('objid').join(metadata[['OBJID', 'PETROR90_R']].set_index('OBJID')).reset_index()

pbar = tqdm(total=barlengths.shape[0])

if model_type == 'vgg16' or model_type == 'resnet50v2' or model_type == 'xception':
    Path(f'./out/{model_type}/heatmaps/{smooth_noise}/').mkdir(parents=True, exist_ok=True)
    model = load_model(f'./out/{model_type}/model.h5')

    for row in barlengths.itertuples():
        galaxy_id = row.GalaxyID
        cam = barlength_cams(galaxy_id, model)
        np.save(f'./out/{model_type}/heatmaps/{smooth_noise}/{galaxy_id}', cam)
        pbar.update(1)

else:
    NotImplementedError('Please select one of "vgg16", "resnet50v2", or "xception" as the model.')

pbar.close()