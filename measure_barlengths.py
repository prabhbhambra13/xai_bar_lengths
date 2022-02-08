#!/usr/bin/env python
import yaml
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def ximage(cam, galaxy_id):

    img = load_img(f'{path}/images_gz2/images/{galaxy_id}')
    img = img_to_array(img)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.image.rgb_to_grayscale(img).numpy()
    img = img[:,:,0] * 1./255
    cam = cam * img
    
    return cam

def barlength_mask(cam):
    
    for heat_threshold in np.arange(0.5, 1.0, 0.0001):
        x, y = np.asarray(cam > heat_threshold).nonzero()
        corr = abs(np.corrcoef(x, y)[1, 0])
        if corr > c_t or len(x) < l_t:
            break
    cam_mask = np.where(cam > heat_threshold, 1, 0)
    
    return cam_mask

def barlength_calc(cam_mask):
    
    x, y = np.where(cam_mask > 0)
    length = 1
    for i in range(len(x)):
        for j in range(len(x)):
            l = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            length = l if l > length else length
    length = length * 424 / img_height

    return length

def measure_barlength(cam):

    cam_mask = barlength_mask(cam)
    xai_length = barlength_calc(cam_mask)

    return xai_length

def save_summary_and_calculate_corr(data, model_type, prefix, corr_threshold, len_threshold, ximage):

    if prefix != '':
        prefix = f'{prefix}_'
    summary = pd.DataFrame(data, columns=['objid', 'GalaxyID', 'hoyle_length', 'xai_length'])
    summary.to_csv(f'./out/{model_type}/barlengths/{prefix}{val_test}_{smooth_noise}_{corr_threshold}_{len_threshold}_{ximage}.csv', index=False)
    corr = (summary['hoyle_length']).corr(summary['xai_length'])

    return corr

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
img_height = data_params['img_height']
img_width = data_params['img_width']
img_depth = data_params['img_depth']
smooth_samples = xai_params['smoothgrad']['smooth_samples']
smooth_noise = xai_params['smoothgrad']['smooth_noise']
xinput = xai_params['smoothgrad']['xinput']
corr_threshold = xai_params['barlength_mask']['corr_threshold']
len_threshold = xai_params['barlength_mask']['len_threshold']
val_test = xai_params['val_test'].lower()
validation_split = xai_params['validation_split']

mapping = pd.read_csv(f'{path}/gz2_filename_mapping.csv', header=0)
barlengths = pd.read_csv(f'{path}/hoyle_barlengths.csv', header=0)
metadata = pd.read_csv(f'{path}/gz2sample.csv', header=0)

barlengths = barlengths.set_index('objid').join(mapping.set_index('objid')).reset_index()
barlengths = barlengths.rename(columns={'asset_id': 'GalaxyID'})
barlengths['GalaxyID'] = barlengths['GalaxyID'].astype('str') + '.jpg'
barlengths = barlengths.set_index('objid').join(metadata[['OBJID', 'PETROR90_R']].set_index('OBJID')).reset_index()
barlengths = barlengths.sample(frac=1, random_state=seed)

val_images = int(barlengths.shape[0] * validation_split)
test_images = int(barlengths.shape[0] - val_images)

if val_test == 'val':
    barlengths = barlengths.head(val_images)
elif val_test == 'test':
    barlengths = barlengths.tail(test_images)
else:
    KeyError('Please select one of "val", or "test" for the val_test parameter.')

if not Path(f'./out/barlength_correlations_{val_test}.csv').exists():
    with open(f'./out/barlength_correlations_{val_test}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model', 'smooth_noise', 'corr_threshold', 'len_threshold', 'xinput', 'correlation'])

for model_type in ['vgg16', 'resnet50v2', 'xception', 'ensemble']:
    Path(f'./out/{model_type}/barlengths/').mkdir(parents=True, exist_ok=True)

dic = {}
if val_test == 'val':
    for model_type in ['vgg16', 'resnet50v2', 'xception', 'averageheatmaps', 'averagemeasurements']:
        for c_t in corr_threshold:
            for l_t in len_threshold:
                for xinput in ['True', 'False']:
                    dic[f'{model_type}_{c_t}_{l_t}_{xinput}'] = []

    pbar = tqdm(total=barlengths.shape[0])

    for row in barlengths.itertuples():    
        objid = row.objid
        galaxy_id = row.GalaxyID
        petror90_r = row.PETROR90_R
        length_scaled = row.length_scaled

        cam_v = np.load(f'./out/vgg16/heatmaps/{smooth_noise}/{galaxy_id}.npy')[0]
        cam_r = np.load(f'./out/resnet50v2/heatmaps/{smooth_noise}/{galaxy_id}.npy')[0]
        cam_x = np.load(f'./out/xception/heatmaps/{smooth_noise}/{galaxy_id}.npy')[0]

        for c_t in corr_threshold:
            for l_t in len_threshold:

                    cam = cam_v
                    xai_length = measure_barlength(cam)
                    avg_length += xai_length / 3
                    dic[f'vgg16_{c_t}_{l_t}_False'].append([objid, galaxy_id, length_scaled * petror90_r, xai_length * petror90_r])

                    cam = ximage(cam, galaxy_id)
                    xai_length = measure_barlength(cam)
                    avg_length_ximage += xai_length / 3
                    dic[f'vgg16_{c_t}_{l_t}_True'].append([objid, galaxy_id, length_scaled * petror90_r, xai_length * petror90_r])

                    cam = cam_x
                    xai_length = measure_barlength(cam)
                    avg_length = xai_length / 3
                    dic[f'xception_{c_t}_{l_t}_False'].append([objid, galaxy_id, length_scaled * petror90_r, xai_length * petror90_r])

                    cam = cam_r
                    xai_length = measure_barlength(cam)
                    avg_length += xai_length / 3
                    dic[f'resnet50v2_{c_t}_{l_t}_False'].append([objid, galaxy_id, length_scaled * petror90_r, xai_length * petror90_r])

                    cam = ximage(cam, galaxy_id)
                    xai_length = measure_barlength(cam)
                    avg_length_ximage += xai_length / 3
                    dic[f'resnet50v2_{c_t}_{l_t}_True'].append([objid, galaxy_id, length_scaled * petror90_r, xai_length * petror90_r])

                    cam = ximage(cam, galaxy_id)
                    xai_length = measure_barlength(cam)
                    avg_length_ximage = xai_length / 3
                    dic[f'xception_{c_t}_{l_t}_True'].append([objid, galaxy_id, length_scaled * petror90_r, xai_length * petror90_r])

                    cam = (cam_x + cam_r + cam_v) / 3
                    xai_length = measure_barlength(cam)
                    dic[f'averageheatmaps_{c_t}_{l_t}_False'].append([objid, galaxy_id, length_scaled * petror90_r, xai_length * petror90_r])

                    cam = ximage(cam, galaxy_id)
                    xai_length = measure_barlength(cam)
                    dic[f'averageheatmaps_{c_t}_{l_t}_True'].append([objid, galaxy_id, length_scaled * petror90_r, xai_length * petror90_r])

                    dic[f'averagemeasurements_{c_t}_{l_t}_False'].append([objid, galaxy_id, length_scaled * petror90_r, avg_length * petror90_r])

                    dic[f'averagemeasurements_{c_t}_{l_t}_True'].append([objid, galaxy_id, length_scaled * petror90_r, avg_length_ximage * petror90_r])
        
        pbar.update(1)

    pbar.close()

else:
    model_type = params['model'].lower()
    xinput = xai_params['smoothgrad']['xinput']
    c_t = corr_threshold
    l_t = len_threshold
    dic[f'{model_type}_{c_t}_{l_t}_{xinput}'] = []

    pbar = tqdm(total=barlengths.shape[0])

    for row in barlengths.itertuples():    
        objid = row.objid
        galaxy_id = row.GalaxyID
        petror90_r = row.PETROR90_R
        length_scaled = row.length_scaled

        cam_v = np.load(f'./out/vgg16/heatmaps/{smooth_noise}/{galaxy_id}.npy')[0]
        cam_r = np.load(f'./out/resnet50v2/heatmaps/{smooth_noise}/{galaxy_id}.npy')[0]
        cam_x = np.load(f'./out/xception/heatmaps/{smooth_noise}/{galaxy_id}.npy')[0]

        if model_type == 'vgg16':
            cam = cam_v
            if xinput:
                cam = ximage(cam, galaxy_id)
            xai_length = measure_barlength(cam)

        elif model_type == 'resnet50v2':
            cam = cam_r
            if xinput:
                cam = ximage(cam, galaxy_id)
            xai_length = measure_barlength(cam)

        elif model_type == 'exception':
            cam = cam_x
            if xinput:
                cam = ximage(cam, galaxy_id)
            xai_length = measure_barlength(cam)

        elif model_type == 'averageheatmaps':
            cam = (cam_x + cam_r + cam_v) / 3
            if xinput:
                cam = ximage(cam, galaxy_id)
            xai_length = measure_barlength(cam)

        elif model_type == 'averagemeasurements':
            if xinput:
                cam_v = ximage(cam_v, galaxy_id)
                cam_r = ximage(cam_r, galaxy_id)
                cam_x = ximage(cam_x, galaxy_id)
            xai_length = 0
            for cam in [cam_v, cam_r, cam_x]:
                xai_length += measure_barlength(cam) / 3

        else:
            KeyError('Please select one of "vgg16", "resnet50v2", "xception", "averageheatmaps", "averagemeasurements" as the model.')
        
        dic[f'{model_type}_{c_t}_{l_t}_{xinput}'].append([objid, galaxy_id, length_scaled * petror90_r, xai_length * petror90_r])
        
        pbar.update(1)
    
    pbar.close()

for key, data in dic.items():
    parts = key.split('_')
    prefix = parts[0]
    corr_threshold = parts[1]
    len_threshold = parts[2]
    ximage = parts[3]

    if prefix == 'averageheatmaps' or prefix == 'averagemeasurements':
        model_type = 'ensemble'
    else:
        model_type = prefix

    corr = save_summary_and_calculate_corr(data, model_type, prefix, corr_threshold, len_threshold, ximage)

    with open(f'./out/barlength_correlations_{val_test}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([prefix, smooth_noise, corr_threshold, len_threshold, ximage, corr])
