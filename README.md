# Explaining deep learning of galaxy morphology with saliency mapping.

This repository contains the code used in the paper [Explaining deep learning of galaxy morphology with saliency mapping](https://arxiv.org/abs/2110.08288).

## Dependencies

- `python 3.7.9`
- `pyyaml 5.4.1`
- `numpy 1.19.5`
- `pandas 1.2.1`
- `tensorflow 2.4.0`
- `tqdm 4.56.1`
- `tf-keras-vis 0.6.0`
- `seaborn 0.11.1`
- `scikit-learn 0.24.1`

## Data

The following data sets were used for this project:

- [Galaxy Zoo - The Galaxy Challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge)
- [Galaxy Zoo 2: Images from Original Sample](https://zenodo.org/record/3565489)
- [SDSS metadata for GZ2](https://data.galaxyzoo.org/)
- [Bar lengths: Table 1](https://data.galaxyzoo.org/)

In order to run the code, please organise your data directory as follows:

```
parent_directory
├──gz2_filename_mapping.csv
├──gz2sample.csv    
├──hoyle_barlengths.csv
├──training_solutions_rev1.csv
├──images_gz2
│   └──images
│       ├──100.jpg
│       ├──1000.jpg
│       ├──...
├──images_test_rev1
│   └──images
│       ├──100018.jpg
│       ├──140111.jpg
│       ├──...
├──images_training_rev1
│   ├──100008.jpg
│   ├──139923.jpg
│   ├──...
```

Note that `all_ones_benchmark`, `all_zeroes_benchmark`, and `central_pixel_benchmark` files from the Kaggle dataset are not needed. Also note that after unzipping `images_test_rev1` the user needs to create a futher directory called `images` inside the unzip location and move all the images into this subdirectory. This is required for the use of the [`flow_from_directory`](https://www.tensorflow.org/versions/r2.4/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory) method in `Keras`.

## Usage

All user inputs are handled by editing the relevant entry in the `params.yaml` file. The two universal entries are `data:path` (which you should replace with the path to the data directory described above), and `data:random_seed` (which is used to set the random seed throughtout the repository).

### `train_models.py`

This script trains and saves the models using the Kaggle "Galaxy Zoo - The Galaxy Challenge" dataset. It outputs `.h5` models as well as `.csv` training logs in the `./out/~model~/` directory.

The relevant entries in `params.yaml` are:

- `model` - The model to be trained. One of `vgg16`, `resnet50v2`, or `xception`.
- `data`
  - `augmentation` - How to augment the training data. Please see the `Keras` documentation for [`ImageDataGenerator`](https://www.tensorflow.org/versions/r2.4/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) for more information on the individual entries.
  - `img_height`, `img_width`, and `img_depth` - The dimensions to which the training images will be downscaled to. These should be set to 224, 224, and 4 respectively, and should not be changed.
- `train`
  - `epochs` - The number of epoch to train for.
  - `batch_size` - The batch size to use for training.
  - `optimizer`
    - `lr` - The learning rate of the Adam optimiser.
    - `decay` - The decay rate of the Adam optimiser.
  - `callbacks`
    - `reduce_lr` - Parameters for the [`ReduceLROnPlateau`](https://www.tensorflow.org/versions/r2.4/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau) callback.
    - `early_stopping` - Parameters for the [`EarlyStopping`](https://www.tensorflow.org/versions/r2.4/api_docs/python/tf/keras/callbacks/EarlyStopping) callback.

### `test_models.py`

This script evaluates the models on the test partition of the Kaggle data set. It outputs `.csv` predictions on both the training and test sets in the `./out/~model~/` directory.

The relevant entries in `params.yaml` are:

- `model` - The model to be evaluated. One of `vgg16`, `resnet50v2`, `xception`, or `ensemble`. The script has to be run with all other values before it is run with the value `ensemble`.

### `generate_heatmaps.py`

This script saves SmoothGrad attention maps highlighting the bar, for images in the Hoyle bar length catalogue. It outputs attention maps in the `./out/~model~/heatmaps/~smooth_noise~/` directory in `.npy` format.

The relevant entries in `params.yaml` are:

- `model` - The model to use in creating the attention maps. One of `vgg16`, `resnet50v2`, or `xception`.
- `xai`
  - `smoothgrad`
    - `smooth_samples` - The number of samples to use in the SmoothGrad implementation.
    - `smooth_noise` - The noise spread to use in the SmoothGrad implementation.

### `measure_barlengths.py`

This script measures the bar lengths of galaxies from SmoothGrad attention maps and saves down the measured values. It outputs bar length measurements in the `./out/~model~/barlengths/` directory, as well as correlations between the measured bar lengths and the catalogue values in either `./out/barlength_correlations_val.csv` or `./out/barlength_correlations_test.csv` depending on the value of `val_test`.

The relevant entries in `params.yaml` are:

- `model` - The model to use when measuring bar lengths. One of `vgg16`, `resnet50v2`, `xception`, `averageheatmaps`, or `averagemeasurements`.
- `xai`
  - `val_test` - Whether to work on the validation or test set of the bar length catalogue. One of `val` or `test`.
  - `validation_split` - The proportion of galaxies in the bar length catalogue to attribute to the validation set.
  - `smoothgrad`
    - `smooth_noise` - The noise spread of the attention maps to use for measuring the bar lengths. These attention maps should have been created in the previous step.
    - `xinput` - Whether or not to multiply the attention map by the input image. Accepts `true` or `false`.
  - `barlength_mask`
    - `corr_threshold` - The correlation threshold over which the bars will be considered linearly distributed. When `val_test` is set to `val` this entry expects a list over which it will perform a hyperparameter search over (along with the values in `len_threshold`). If `val_test` is set to `test`, only a single value is expected.
    - `len_threshold` - The threshold for the number of pixels required to define a bar length. When `val_test` is set to `val` this entry expects a list over which it will perform a hyperparameter search over (along with the values in `corr_threshold`). If `val_test` is set to `test`, only a single value is expected.

### `control_model.py`

This script trains and evaluates an Xception based CNN to predict bar lengths. It outputs a model to `./out/control/model.h5`, a `.csv` containing bar length predictions on the test set to `./out/control/predictions.csv`, as well as the correlation between the predicted bar lengths and the catalogue values in `./out/barlength_correlations_test.csv`.

The relevant entries in `params.yaml` are:

- `data`
  - `augmentation` - How to augment the training data. Please see the `Keras` documentation for [`ImageDataGenerator`](https://www.tensorflow.org/versions/r2.4/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) for more information on the individual entries.
  - `img_height`, `img_width`, and `img_depth` - The dimensions to which the training images will be downscaled to. Once set, these values should not change.
  - `test_split` - The proportion of galaxies in the bar length catalogue to attribute to the test set.
- `train`
  - `epochs` - The number of epoch to train for.
  - `batch_size` - The batch size to use for training.
  - `optimizer`
    - `lr` - The learning rate of the Adam optimiser.
    - `decay` - The decay rate of the Adam optimiser.
  - `callbacks`
    - `reduce_lr` - Parameters for the [`ReduceLROnPlateau`](https://www.tensorflow.org/versions/r2.4/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau) callback.
    - `early_stopping` - Parameters for the [`EarlyStopping`](https://www.tensorflow.org/versions/r2.4/api_docs/python/tf/keras/callbacks/EarlyStopping) callback.

### `save_figures.py`

This script generates the figures used in the paper. It outputs `.png` figures to `./out/figures/`.

The only relevant entries in `params.yaml` is `data:path`.