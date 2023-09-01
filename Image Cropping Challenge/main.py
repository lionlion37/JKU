"""
Preprocess images and train as well as test models with them.
"""

import os
import numpy as np
import torch
import glob

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import ImageYielderFromDirectory, ImageYielderFromList
from utils import validate_images, data_augmentation, crop_images
from utils import TrainDataset, collate_fn
from utils import normalize
from utils import train, evaluate

from architectures import ModelA, ModelB, CustomCNN, MyScndCNN, AlexNet


# ----------------------------------------------------------------------------------------------------------------------

np.random.seed(42)
torch.random.manual_seed(42)

# ----------------------------------------------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
print(f'Device: {device}')

# ----------------------------------------------------------------------------------------------------------------------

log_path = os.path.join('.', 'log')

input_dir = os.path.join('.', 'data')
output_dir = os.path.join('.', 'validated_images', 'all')
train_name = 'all_images_train'
test_name = 'all_images_test'
validation_name = 'all_images_validation'
general_name = 'all_images'

n_augmentations = 3
crops_per_image_train = 2
crops_per_image_test = 3
crops_per_image_val = 3

# set either of those to True if files produced by preprocessing functions should be replaced by new ones
overwrite_augmentation = False
overwrite_cropping = False
overwrite_normalization = False

# set to True if targets should be full images, not just cropped out parts
full_images = True

# set to True if targets should be full images, but with all pixels, except those of the crop part, set to 0
exp = False

# set to True if training should take place
train_model = True

# set to True if all models in folder 'models' should be evaluated to find the best one
test_all_models = False

# ----------------------------------------------------------------------------------------------------------------------

# hyperparameters
learning_rate = 0.0001
n_updates = 5000
num_workers = 5
train_batch_size = 60
test_batch_size = 60
validation_batch_size = 60
validate_at = 50

########################################################################################################################
########################################################################################################################


# validate images and copy to folder 'validated_images' ----------------------------------------------------------------

n_images = validate_images(name=general_name, input_dir=input_dir, output_dir=output_dir, log_path=log_path)
image_yielder_dir = ImageYielderFromDirectory(input_dir=output_dir)

########################################################################################################################


print('\n############################################## augmentation ###############################################\n')

# split the images in train set, test set and validation set with ratio 70:15:15 ---------------------------------------

images = [image for image in image_yielder_dir.get_images()]

train_set, testing_sets = train_test_split(images, shuffle=True, test_size=0.3)
test_set, validation_set = train_test_split(testing_sets, shuffle=True, test_size=0.5)

# augment and resize train images, resize test and validation images ---------------------------------------------------

# ################## WARNING: If images are newly augmented, they also have to be cropped out again !! #################

# train images have to split in two parts due to memory size
train_images_augmented_1 = data_augmentation(input_images=train_set[:8000], name=train_name + '_1',
                                             n_augmentations=n_augmentations, overwrite=overwrite_augmentation)
train_images_augmented_2 = data_augmentation(input_images=train_set[8000:], name=train_name + '_2',
                                             n_augmentations=n_augmentations, overwrite=overwrite_augmentation)
train_images_augmented = train_images_augmented_1 + train_images_augmented_2
image_yielder_train = ImageYielderFromList(image_list=train_images_augmented)

# test images
test_set_resized = data_augmentation(input_images=test_set, name=test_name, n_augmentations=1, is_test_set=True,
                                     overwrite=overwrite_augmentation)
image_yielder_test = ImageYielderFromList(image_list=test_set_resized)

# validation images
validation_set_resized = data_augmentation(input_images=validation_set, name=validation_name, n_augmentations=1,
                                           is_test_set=True, overwrite=overwrite_augmentation)
image_yielder_validation = ImageYielderFromList(image_list=validation_set_resized)

########################################################################################################################


print('\n################################################ cropping #################################################\n')

# crop train, test and validation images and save in list as numpy array of transposed shape (X, Y) --------------------

train_image_list, train_crop_list, train_missing = crop_images(name=train_name, image_yielder=image_yielder_train,
                                                               crops_per_image=crops_per_image_train,
                                                               overwrite=overwrite_cropping)

test_image_list, test_crop_list, test_missing = crop_images(name=test_name, image_yielder=image_yielder_test,
                                                            crops_per_image=crops_per_image_test,
                                                            overwrite=overwrite_cropping)

validation_image_list, validation_crop_list, validation_missing = crop_images(name=validation_name,
                                                                              image_yielder=image_yielder_validation,
                                                                              crops_per_image=crops_per_image_val,
                                                                              overwrite=overwrite_cropping)

########################################################################################################################


print('\n############################################### normalizing ###############################################\n')

# normalize images -----------------------------------------------------------------------------------------------------

train_image_list, train_means, train_stds = normalize(input=train_image_list, file_name=train_name,
                                                      overwrite=overwrite_normalization)
test_image_list, test_means, test_stds = normalize(input=test_image_list, file_name=test_name,
                                                   overwrite=overwrite_normalization)
validation_image_list, val_means, val_stds = normalize(input=validation_image_list, file_name=validation_name,
                                                       overwrite=overwrite_normalization)

# normalize full train images
if full_images:
    train_images_normal, train_tg_means, train_tg_stds = normalize(input=train_images_augmented, file_name='train_aug',
                                                                   overwrite=overwrite_normalization)
    test_images_normal, test_tg_means, test_tg_stds = normalize(input=test_set_resized, file_name='test_aug',
                                                                overwrite=overwrite_normalization)
    validation_images_normal, val_tg_means, val_tg_stds = normalize(input=validation_set_resized,  file_name='val_aug',
                                                                    overwrite=overwrite_normalization)

# normalize just cropped out parts of images (*_missing)
else:
    train_images_normal, train_tg_means, train_tg_stds = normalize(input=train_missing, file_name='train_aug_exp',
                                                                   overwrite=overwrite_normalization)
    test_images_normal, test_tg_means, test_tg_stds = normalize(input=test_missing, file_name='test_aug_exp',
                                                                overwrite=overwrite_normalization)
    validation_images_normal, val_tg_means, val_tg_stds = normalize(input=validation_missing,  file_name='val_aug_exp',
                                                                    overwrite=overwrite_normalization)

########################################################################################################################


# create target lists --------------------------------------------------------------------------------------------------

# transpose targets (full images) from (Y, X) to (X, Y) and append to list *_targets
if full_images:
    train_targets = []
    for image in train_images_normal:
        for _ in range(crops_per_image_train):
            train_targets.append(np.array(image).transpose(1, 0))

    test_targets = []
    for image in test_images_normal:
        for _ in range(crops_per_image_test):
            test_targets.append(np.array(image).transpose(1, 0))

    validation_targets = []
    for image in validation_images_normal:
        for _ in range(crops_per_image_val):
            validation_targets.append(np.array(image).transpose(1, 0))

# transpose targets (only cropped out parts) from (Y, X) to (X, Y) and append to list *_targets
else:
    train_targets = []
    for image in train_images_normal:
        train_targets.append(np.array(image).transpose(1, 0))

    test_targets = []
    for image in test_images_normal:
        test_targets.append(np.array(image).transpose(1, 0))

    validation_targets = []
    for image in validation_images_normal:
        validation_targets.append(np.array(image).transpose(1, 0))

# create Datasets ------------------------------------------------------------------------------------------------------

# train dataset, len(train_dataset) = 105 991
train_dataset = TrainDataset(image_list=train_image_list, crop_list=train_crop_list, target_list=train_targets, exp=exp)

# test dataset, len(test_dataset) = 11 356
test_dataset = TrainDataset(image_list=test_image_list, crop_list=test_crop_list, target_list=test_targets, exp=False)

# validation dataset, len(validation_dataset) = 11 356
validation_dataset = TrainDataset(image_list=validation_image_list, crop_list=validation_crop_list,
                                  target_list=validation_targets, exp=False)

# create DataLoaders ---------------------------------------------------------------------------------------------------

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, collate_fn=collate_fn)

test_loader = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=test_batch_size,
                         collate_fn=collate_fn)


validation_loader = DataLoader(validation_dataset, shuffle=False, num_workers=num_workers,
                               batch_size=validation_batch_size, collate_fn=collate_fn)

########################################################################################################################


print('\n################################################ training #################################################\n')

# Training -------------------------------------------------------------------------------------------------------------

if train_model:

    # either load a model or create a new instance

    # cnn = SimpleCNN()
    cnn = torch.load(os.path.join('.', 'models', '2CNN-v3.1.pt'))

    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    trained_cnn = train(model=cnn, dataloader=train_loader, valloader=validation_loader, optimizer=optimizer,
                        loss_function=loss_function, n_updates=n_updates, validate_at=validate_at,
                        missing_targets_val=validation_missing, val_means=val_means, val_stds=val_stds,
                        name='2CNN-final', device=device)

    # save trained model
    torch.save(trained_cnn, os.path.join('.', 'models', '2CNN-final.pt'))

########################################################################################################################

    print('\n################################################# testing ################################################'
          '#\n')

    # Testing ----------------------------------------------------------------------------------------------------------

    loss_crops, loss_whole = evaluate(model=trained_cnn, dataloader=test_loader, targets=test_missing,
                                      to_denormalize=True, means=test_means, stds=test_stds, with_tqdm=True,
                                      device=device)

    print(f'loss on just cropped out parts: {loss_crops}, loss on whole imgaes: {loss_whole}')

########################################################################################################################


# Test all models ------------------------------------------------------------------------------------------------------

if test_all_models:

    print('\n################################################# testing ################################################'
          '#\n')

    loss_crops_dict = {}
    loss_whole_dict = {}

    # load paths of every model in folder 'models'
    model_paths = glob.glob(os.path.join('models', '**'), recursive=True)

    for path in model_paths:

        if not path.endswith('.pt'):
            continue

        # evaluate model
        model = torch.load(path)
        loss_crops, loss_whole = evaluate(model=model, dataloader=test_loader, targets=test_missing,
                                          to_denormalize=True, means=test_means, stds=test_stds, with_tqdm=True,
                                          device=device)

        loss_crops_dict[path] = loss_crops
        loss_whole_dict[path] = loss_whole

    print(f'Best model loss_crops: {min(loss_crops_dict)}\nBest model loss_whole: {min(loss_whole_dict)}')

print('Done!')
