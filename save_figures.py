#!/usr/bin/env python
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tf_keras_vis.saliency import Saliency

def load_image(galaxy_id, source):

    img = load_img(f'{path}/{source}/{galaxy_id}')
    img = img_to_array(img)
    img = tf.image.resize(img, [224, 224])
    img = img.numpy() * 1./255

    return img

def lookup_class(class_name):
    return df.columns.get_loc(class_name) - 1

def class_score(output, class_index):
    return output[:,class_index]

def score(class_index):
    return lambda x: class_score(x, class_index)

def model_modifier(m):
    m.layers[-1].activation = activations.linear
    return m

def load_cam(galaxy_id):

    cam_v = np.load(f'./out/vgg16/heatmaps/0.1/{galaxy_id}.npy')[0]
    cam_r = np.load(f'./out/resnet50v2/heatmaps/0.1/{galaxy_id}.npy')[0]
    cam_x = np.load(f'./out/xception/heatmaps/0.1/{galaxy_id}.npy')[0]
    cam = (cam_v + cam_r + cam_x) / 3

    return cam

def barlength_mask(cam, corr_threshold, len_threshold):
    
    for heat_threshold in np.arange(0.5, 1.0, 0.0001):
        x, y = np.asarray(cam > heat_threshold).nonzero()
        corr = abs(np.corrcoef(x, y)[1, 0])
        if corr > corr_threshold or len(x) < len_threshold:
            break
    cam_mask = np.where(cam > heat_threshold, 1, 0)
    
    return cam_mask

def plot_scatter(df, title, fname):

    sns.set_style("white")
    rc = {'figure.figsize':(10,7),'font.size': 20}
    plt.rcParams.update(rc)
    lims = [0, 15]

    reg = LinearRegression(fit_intercept=False).fit(df['hoyle_length'].values.reshape(-1,1), df['measured_length'].values.reshape(-1,1), sample_weight=1. / (df['std_hoyle_length'].values + df['std_measured_length'].values))
    g = sns.FacetGrid(df, height=10, despine=False, xlim=lims, ylim=lims)
    g.map_dataframe(sns.scatterplot, x="hoyle_length", y="measured_length", color='xkcd:dull blue', edgecolors='face', marker='.')
    g.map_dataframe(sns.kdeplot, x="hoyle_length", y="measured_length", color='xkcd:navy', levels=5)
    plt.plot(np.linspace(0,40), np.linspace(0,40), color='xkcd:black', linestyle='-')
    plt.plot(np.linspace(0,40), (reg.coef_ * np.linspace(0,40) + reg.intercept_).reshape(-1), color='xkcd:red', linestyle='-')

    plt.suptitle(title)
    plt.xlabel(r'Hoyle Bar Length /$h^{-1}~\rm{kpc}$')
    plt.ylabel(r'Measured Length /$h^{-1}~\rm{kpc}$')

    plt.tight_layout()
    g.fig.subplots_adjust(top=0.94)
    g.savefig(f'./out/figures/{fname}.png')

with open('params.yaml', 'r') as fd:
    params = yaml.safe_load(fd)

data_params = params['data']
train_params = params['train']
xai_params = params['xai']

path = data_params['path']
seed = data_params['random_seed']

Path(f'./out/figures/').mkdir(parents=True, exist_ok=True)

model_v = load_model('./out/vgg16/model.h5')
model_r = load_model('./out/resnet50v2/model.h5')
model_x = load_model('./out/xception/model.h5')
model_list = [model_v, model_r, model_x]

df = pd.read_csv(f'{path}/training_solutions_rev1.csv', header=0)
df['GalaxyID'] = df['GalaxyID'].astype('str') + '.jpg'

v1_measurements = pd.read_csv('./out/ensemble/barlengths/averagemeasurements_test_0.1_0.85_40_False_v1.csv', header=0)
v2_measurements = pd.read_csv('./out/ensemble/barlengths/averagemeasurements_test_0.1_0.85_40_False_v2.csv', header=0)
cnn_measurements = pd.read_csv('./out/control/predictions.csv', header=0)

pred = pd.read_csv(f'./out/ensemble/predictions_train.csv', header=0)

# Fig 1

galaxy_id = '1110.jpg'
img = load_image(galaxy_id, 'images_gz2/images')

cam_bar = np.zeros((224, 224))
cam_bulge = np.zeros((224, 224))
cam_spiral = np.zeros((224, 224))
class_dict = {'Class3.1': cam_bar, 'Class5.3': cam_bulge, 'Class11.3': cam_spiral}

for m in model_list:
    saliency = Saliency(
        m,
        model_modifier=model_modifier,
        clone=False
    )
    
    for class_name, class_cam in class_dict.items():
        class_index = lookup_class(class_name)
        cam = saliency(
            score(class_index),
            img,
            smooth_samples=256,
            smooth_noise=0.1
        )
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        class_cam += cam[0]

    tf.keras.backend.clear_session()

sns.set_style("white")
rc = {'figure.figsize':(20, 6), 'font.size': 25}
plt.rcParams.update(rc)

fig, ax = plt.subplots(1, 4)

img_show = img[63:163, 63:163]
ax[0].imshow(img_show)

cam_bar = class_dict['Class3.1'][63:163, 63:163] / 3
ax[1].imshow(cam_bar, cmap='jet')
ax[1].imshow(img_show, alpha=0.4)

cam_bulge = class_dict['Class5.3'][63:163, 63:163] / 3
ax[2].imshow(cam_bulge, cmap='jet')
ax[2].imshow(img_show, alpha=0.4)

cam_spiral = class_dict['Class11.3'][63:163, 63:163] / 3
ax[3].imshow(cam_spiral, cmap='jet')
ax[3].imshow(img_show, alpha=0.4)

for i, a in enumerate(fig.axes):
    a.set_xticks([])
    a.set_yticks([])

ax[0].set_title('Image')
ax[1].set_title('Bar')
ax[2].set_title('Bulge')
ax[3].set_title('Spiral Arms')

plt.tight_layout()
plt.savefig('./out/figures/class_heatmaps.png')
plt.close()

# Fig 2

log = {}
for m in ['resnet50v2', 'xception', 'vgg16']:
    log[m] = pd.read_csv(f'./out/{m}/training_log.csv', header=0)

sns.set_style("white")
rc = {'figure.figsize':(10, 7), 'font.size': 20}
plt.rcParams.update(rc)

fig, ax = plt.subplots()
plt.plot(log['vgg16']['epoch'], log['vgg16']['root_mean_squared_error'], color='xkcd:red', linestyle='-', label='VGG16 Training Set')
plt.plot(log['vgg16']['epoch'], log['vgg16']['val_root_mean_squared_error'], color='xkcd:red', linestyle='--', label='VGG16 Validation Set')
plt.plot(log['resnet50v2']['epoch'], log['resnet50v2']['root_mean_squared_error'], color='xkcd:green', linestyle='-', label='ResNet50v2 Training Set')
plt.plot(log['resnet50v2']['epoch'], log['resnet50v2']['val_root_mean_squared_error'], color='xkcd:green', linestyle='--', label='ResNet50v2 Validation Set')
plt.plot(log['xception']['epoch'], log['xception']['root_mean_squared_error'], color='xkcd:blue', linestyle='-', label='Xception Training Set')
plt.plot(log['xception']['epoch'], log['xception']['val_root_mean_squared_error'], color='xkcd:blue', linestyle='--', label='Xception Validation Set')
plt.xlabel('Epochs')
plt.ylabel('Root Mean Squared Error')
plt.legend()
plt.tight_layout()
plt.savefig('./out/figures/learning_curves.png')
plt.close()

# Fig 3

sns.set_style("white")
rc = {'figure.figsize':(15, 15), 'font.size': 25}
plt.rcParams.update(rc)

fig, ax = plt.subplots(3, 3)
for i, galaxy_id in zip(range(3), ['245259.jpg', '226234.jpg', '206000.jpg']):

    img = load_image(galaxy_id, 'images_gz2/images')
    cam = load_cam(galaxy_id)
    mask = barlength_mask(cam, 0.85, 30)
    
    img_show = img[63:163, 63:163]
    ax[i,0].imshow(img_show)

    cam_show = cam[63:163, 63:163]
    ax[i,1].imshow(cam_show, cmap='jet')
    ax[i,1].imshow(img_show, alpha=0.4)

    mask_show = mask[63:163, 63:163]
    ax[i,2].imshow(mask_show)
    ax[i,2].imshow(img_show, alpha=0.4)

for i, a in enumerate(fig.axes):
    a.set_xticks([])
    a.set_yticks([])

ax[0,0].set_title('Image')
ax[0,1].set_title('SmoothGrad Heatmap')
ax[0,2].set_title('Isolated Bar')

plt.tight_layout()
plt.savefig('./out/figures/barlength_heatmaps.png')
plt.close()

# Fig 4

v1_measurements['xai_length'] = v1_measurements['xai_length'] * 0.02 * 0.396
v2_measurements['xai_length'] = v2_measurements['xai_length'] * 0.02 * 0.396

xai_measurements = v1_measurements.rename(columns={'xai_length': 'v1_length'})
xai_measurements['v2_length'] = v2_measurements['xai_length']
xai_measurements['measured_length'] = (xai_measurements['v1_length'] + xai_measurements['v2_length']) / 2
xai_measurements['hoyle_length'] = xai_measurements['hoyle_length'] * 0.7
xai_measurements['std_hoyle_length'] = xai_measurements['hoyle_length'] * 0.17

xai_measurements['spread'] = abs(xai_measurements['v1_length'] - xai_measurements['v2_length'])
xai_measurements['bin'] = pd.qcut(xai_measurements['measured_length'], 10)
bins = pd.qcut(xai_measurements['measured_length'], 10).unique()
map = xai_measurements.groupby('bin').mean().reset_index()[['bin', 'spread']].rename(columns={'spread': 'std_measured_length'})
xai_measurements = xai_measurements.merge(map, how='left', on='bin')

cnn_measurements['hoyle_length'] = cnn_measurements['hoyle_length'] * 0.7
cnn_measurements['std_hoyle_length'] = cnn_measurements['hoyle_length'] * 0.17
cnn_measurements['std_measured_length'] = 1
cnn_measurements.rename(columns={'cnn_length': 'measured_length'}, inplace=True)
cnn_measurements['measured_length'] = cnn_measurements['measured_length'] * 0.7

plot_scatter(xai_measurements, 'Explainable Deep Learning Method', 'xai_scatter')
plot_scatter(cnn_measurements, 'Direct Deep Learning Method', 'cnn_scatter')

# Fig 5

galaxy_ids = ['102271.jpg', '120250.jpg']

cam_smooth = np.zeros((224, 224))
cam_features = np.zeros((224, 224))
class_dict = {'Class1.1': cam_smooth, 'Class1.2': cam_features}

for galaxy_id, (class_name, class_cam) in zip(galaxy_ids, class_dict.items()):
    img = load_image(galaxy_id, 'images_training_rev1')

    for m in model_list:
        saliency = Saliency(
            m,
            model_modifier=model_modifier,
            clone=False
        )
        class_index = lookup_class(class_name)
        cam = saliency(
            score(class_index),
            img,
            smooth_samples=256,
            smooth_noise=0.1
        )
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        class_cam += cam[0]

        tf.keras.backend.clear_session()

sns.set_style("white")
rc = {'figure.figsize':(10, 12), 'font.size': 25}
plt.rcParams.update(rc)

fig, ax = plt.subplots(2, 2)

kaggle = df[df['GalaxyID'] == galaxy_ids[0]]['Class1.1'].values[0]
model = pred[pred['GalaxyID'] == galaxy_ids[0]]['Class1.1'].values[0]

img_show = load_image(galaxy_ids[0], 'images_training_rev1')
ax[0,0].imshow(img_show)
ax[0,0].set_xlabel('Galaxy Zoo Consensus:\nModel Prediction:', horizontalalignment='right', x=1.0, fontsize='small')

cam_smooth = class_dict['Class1.1'] / 3
ax[0,1].imshow(cam_smooth, cmap='jet')
ax[0,1].imshow(img_show, alpha=0.4)
ax[0,1].set_xlabel(f'{kaggle:.0%} Smooth\n{model:.0%} Smooth', horizontalalignment='left', x=0.0, fontsize='small')

kaggle = df[df['GalaxyID'] == galaxy_ids[1]]['Class1.2'].values[0]
model = pred[pred['GalaxyID'] == galaxy_ids[1]]['Class1.2'].values[0]

img_show = load_image(galaxy_ids[1], 'images_training_rev1')
ax[1,0].imshow(img_show)
ax[1,0].set_xlabel('Galaxy Zoo Consensus:\nModel Prediction:', horizontalalignment='right', x=1.0, fontsize='small')

cam_smooth = class_dict['Class1.2'] / 3
ax[1,1].imshow(cam_smooth, cmap='jet')
ax[1,1].imshow(img_show, alpha=0.4)
ax[1,1].set_xlabel(f'{kaggle:.0%} Features\n{model:.0%} Features', horizontalalignment='left', x=0.0, fontsize='small')

for i, a in enumerate(fig.axes):
    a.set_xticks([])
    a.set_yticks([])

ax[0,0].set_title('Image')
ax[0,1].set_title('Smooth/Features')

plt.tight_layout()
plt.savefig('./out/figures/misclassification_heatmaps.png')
plt.close()

misclass = len(df[((df['Class1.1'] > 0.5) & (pred['Class1.1'] < 0.1)) | ((df['Class1.1'] < 0.5) & (pred['Class1.1'] > 0.5))])
total = len(df)

print(f'Percent Missclassified: {misclass/total:.0%}')

# Appendix A

sns.set_style("white")
rc = {'figure.figsize':(20, 25), 'font.size': 25}
plt.rcParams.update(rc)

fig, ax = plt.subplots(5, 4)

samples = df[(df['Class3.1'] > 0.5) & (df['Class5.3'] > 0.5) & (df['Class11.3'] > 0.5)]
samples = samples.sample(n=5, random_state=seed)

for i, (_, galaxy) in zip(range(5), samples.iterrows()):

    img = load_image(galaxy.GalaxyID, 'images_training_rev1')

    cam_bar = np.zeros((224, 224))
    cam_bulge = np.zeros((224, 224))
    cam_spiral = np.zeros((224, 224))
    class_dict = {'Class3.1': cam_bar, 'Class5.3': cam_bulge, 'Class11.3': cam_spiral}

    for m in model_list:
        saliency = Saliency(
            m,
            model_modifier=model_modifier,
            clone=False
        )
        
        for class_name, class_cam in class_dict.items():
            class_index = lookup_class(class_name)
            cam = saliency(
                score(class_index),
                img,
                smooth_samples=256,
                smooth_noise=0.1
            )
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            class_cam += cam[0]

        tf.keras.backend.clear_session()

    img_show = img[63:163, 63:163]
    ax[i,0].imshow(img_show)

    cam_bar = class_dict['Class3.1'][63:163, 63:163] / 3
    ax[i,1].imshow(cam_bar, cmap='jet')
    ax[i,1].imshow(img_show, alpha=0.4)

    cam_bulge = class_dict['Class5.3'][63:163, 63:163] / 3
    ax[i,2].imshow(cam_bulge, cmap='jet')
    ax[i,2].imshow(img_show, alpha=0.4)

    cam_spiral = class_dict['Class11.3'][63:163, 63:163] / 3
    ax[i,3].imshow(cam_spiral, cmap='jet')
    ax[i,3].imshow(img_show, alpha=0.4)

for i, a in enumerate(fig.axes):
    a.set_xticks([])
    a.set_yticks([])

ax[0,0].set_title('Image')
ax[0,1].set_title('Bar')
ax[0,2].set_title('Bulge')
ax[0,3].set_title('Spiral Arms')

plt.tight_layout()
plt.savefig('./out/figures/appendix_a.png')
plt.close()

# Appendix B

sns.set_style("white")
rc = {'figure.figsize':(15, 25), 'font.size': 25}
plt.rcParams.update(rc)

fig, ax = plt.subplots(5, 3)

samples = v1_measurements.sample(n=5, random_state=seed)

for i, (_, galaxy) in zip(range(5), samples.iterrows()):

    img = load_image(galaxy.GalaxyID, 'images_gz2/images')
    cam = load_cam(galaxy.GalaxyID)
    mask = barlength_mask(cam, 0.85, 30)
    
    img_show = img[63:163, 63:163]
    ax[i,0].imshow(img_show)

    cam_show = cam[63:163, 63:163]
    ax[i,1].imshow(cam_show, cmap='jet')
    ax[i,1].imshow(img_show, alpha=0.4)

    mask_show = mask[63:163, 63:163]
    ax[i,2].imshow(mask_show)
    ax[i,2].imshow(img_show, alpha=0.4)

for i, a in enumerate(fig.axes):
    a.set_xticks([])
    a.set_yticks([])

ax[0,0].set_title('Image')
ax[0,1].set_title('SmoothGrad Heatmap')
ax[0,2].set_title('Isolated Bar')

plt.tight_layout()
plt.savefig('./out/figures/appendix_b.png')
plt.close()