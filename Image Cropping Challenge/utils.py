"""
Functions and classes for preprocessing, training and evaluation.
"""

import os
import numpy as np
import random
import glob
import dill as pickle
from PIL import Image
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from packages.clean_up_img import ex2 as clean_up
from packages.crop_img import ex4 as create_data

# ----------------------------------------------------------------------------------------------------------------------
# Validate and copy images ---------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def validate_images(name: str, input_dir: str, output_dir: str, log_path: str = './log'):
    """
    Validates image files in given input folder and copies them to a given output folder which is created. A logfile
    will be created in a pre-defined directory. Moreover, the number of copied images will be stored in a config folder
    as a .pkl file. If the given output folder already exists, the corresponding number of images will be loaded and no
    validation or copying takes place.

    Will create a new folder -name- in config folder and log folder if they don't already exist.

    :param name: name for image tranche; will be used for config and log files
    :param input_dir: directory with image files to validate
    :param output_dir: directory to copy validated image files to
    :param log_path: path of log file
    :return: number of validated images
    """

    # create -name- folder in config directory if not existing
    if not os.path.isdir(os.path.join('.', 'config', name)):
        os.mkdir(os.path.join('.', 'config', name))

    logfile = os.path.join(log_path, name + '_validation.log')

    if not os.path.isdir(output_dir):

        n_images = clean_up(input_dir, output_dir, logfile)
        n_images = int(n_images)

        # safe number of images
        with open(os.path.join('config', name, 'n_images_' + str(name) + '.pkl'), 'wb') as f:
            pickle.dump(n_images, f)

        print(f'Validated and copied {n_images} images to {output_dir}!')

    else:
        # load number of images
        with open(os.path.join('config', name, 'n_images_' + str(name) + '.pkl'), 'rb') as f:
            n_images = pickle.load(f)

        print(f'Image directory {output_dir} already exists with {n_images} images!')

    return n_images


# ----------------------------------------------------------------------------------------------------------------------
# augment images -------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def data_augmentation(input_images: list, n_augmentations: int, name: str, overwrite: bool = False,
                      is_test_set: bool = False):
    """
    Will perform a random horizontal flip, a random vertical flip, a random rotation and random cropping on the images
    given by an image_yielder. Moreover, it will resize the given images to a height and width between 70 and 100 px.

    :param input_images: list of images to augment
    :param n_augmentations: number of augmentations per image
    :param name: name for corresponding image tranche; will be used for config and log files
    :param overwrite: set to True if files previously created by this function should be removed
    :param is_test_set: define if given images should only be resized (test set) or also augmented (train set)
    :return: list of augmented images
    """
    image_list = []

    if overwrite:
        if os.path.isfile(os.path.join('.', 'config', name + '_images_augmented.pkl')):
            os.system(f'rm {os.path.join(".", "config", name + "_images_augmented.pkl")}')

    # data augmentation transform chain for train set
    transform_chain = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=180)
    ])

    # augment images if pickle file does not already exist
    if not os.path.isfile(os.path.join('.', 'config', name + '_images_augmented.pkl')):
        for t, image in tqdm(enumerate(input_images), desc=f'{name}: Augmenting images', total=len(input_images)):
            laczos = False
            for n in range(n_augmentations):
                height = random.randrange(start=70, stop=101)
                width = random.randrange(start=70, stop=101)

                if not is_test_set and laczos:
                    image_pp = transform_chain(image)
                    random_cropper = transforms.RandomResizedCrop(size=(height, width))
                    image_pp = random_cropper(image_pp)

                else:
                    image_pp = Image.Image.resize(image, size=(height, width), resample=Image.LANCZOS)
                    laczos = True

                if np.std(image_pp) == 0 or np.mean(image_pp) == 0:
                    image_pp = Image.Image.resize(image, size=(height, width), resample=Image.LANCZOS)
                    laczos = True

                image_list.append(image_pp)

        # save augmented images
        with open(os.path.join('.', 'config', name + '_images_augmented.pkl'), 'wb') as f:
            pickle.dump(image_list, f)

        print(f'{name}: Successfully augmented and saved the given images!')

    # load existing images
    else:
        with open(os.path.join('.', 'config', name + '_images_augmented.pkl'), 'rb') as f:
            image_list = pickle.load(f)

        print(f'{name}: Found existing pickle augmentation file with same name! Loading these images.')

    return image_list


# ----------------------------------------------------------------------------------------------------------------------
# crop images ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def crop_images(name: str, image_yielder, crops_per_image: int = 1, overwrite: bool = False, device: str = 'cuda:0'):
    """
    Will generate random, odd crop_size tuples in range [5, 21] if there is not already a pickle file with the same name
    existing in the config folder. When there is an existing file, this will be loaded.

    Moreover, the function will generate random crop_center tuples so that the cropped-out part with the corresponding
    crop size is more than 20 pixels away from the end of the image. This will only happen if there is not already a
    pickle file with the same name existing in the config folder. When there is an existing file, this will be loaded.

    crop_size and crop_center tuples will be generated 'crop_per_image'-times per image.

    Afterwards, the given images will be cropped accordingly and appended to an image-array. Moreover, a crop_array and
    target_array are returned. All those three arrays will be saved as a pickle file with the name -name- in the folder
    'preprocessed_images'. If such a file already exists, the arrays will just be loaded and not newly created.

    Will create a new folder -name- in config folder if it doesn't already exist.

    :param name: name for corresponding image tranche; will be used for config and log files
    :param image_yielder: yields original images
    :param crops_per_image: number of times a image should be cropped (makes it possible to reuse images multiple times)
    :param overwrite: set to True if files previously created by this function should be removed
    :param device: gpu or cpu
    :return: image_arrays - cropped images, crop_arrays - position of cropping, target_arrays - right
    values for cropped field
    """

    # create -name- folder in config directory if not already existing
    if not os.path.isdir(os.path.join('.', 'config', name)):
        os.mkdir(os.path.join('.', 'config', name))

    # delete any previous files if overwrite = True
    if overwrite:
        if os.path.isfile(os.path.join('config', name, 'crop_sizes.pkl')):
            os.system(f'rm {os.path.join(".", "config", name, "crop_sizes.pkl")}')
        if os.path.isfile(os.path.join('config', name, 'crop_centers.pkl')):
            os.system(f'rm {os.path.join("config", name, "crop_centers.pkl")}')
        if os.path.isfile(os.path.join('.', 'preprocessed_images', name + '.pkl')):
            os.system(f'rm {os.path.join(".", "preprocessed_images", name + ".pkl")}')

    # ------------------------------------------------------------------------------------------------------------------

    # create random odd crop_size tuples in range [5, 21] if not already existing in config folder
    if not os.path.isfile(os.path.join('config', name, 'crop_sizes.pkl')):

        crop_sizes = np.empty(shape=(len(image_yielder), crops_per_image, 2))

        for n, _ in enumerate(crop_sizes):
            for crop in range(crops_per_image):
                size_tpl = [0, 0]  # (Y, X)
                size_tpl[0] = int(random.randrange(start=5, stop=23, step=2))  # Y
                size_tpl[1] = int(random.randrange(start=5, stop=23, step=2))  # X
                crop_sizes[n, crop] = size_tpl

        # safe crop_size tuples
        crop_sizes_config = {'crop_sizes': crop_sizes, 'crops_per_image': crops_per_image}
        with open(os.path.join('config', name, 'crop_sizes.pkl'), 'wb') as f:
            pickle.dump(crop_sizes_config, f)

        print(f'{name}: Generated {crops_per_image} crop size(s) per image!')

    else:
        # load crop_size tuples
        with open(os.path.join('config', name, 'crop_sizes.pkl'), 'rb') as f:
            crop_sizes_config = pickle.load(f)
            crop_sizes = crop_sizes_config['crop_sizes']
            crops_per_image_sizes = crop_sizes_config['crops_per_image']

        print(f'{name}: Crop sizes config file found! Loaded {crops_per_image} crop size(s) per image.')

    # ------------------------------------------------------------------------------------------------------------------

    # create np.array for random crop_center tuples/load random crop_center tuples
    if not os.path.isfile(os.path.join('config', name, 'crop_centers.pkl')):

        crop_centers = np.empty(shape=(len(image_yielder), crops_per_image, 2))
        generate_crop_centers = True

    else:
        # load crop_center tuples
        with open(os.path.join('config', name, 'crop_centers.pkl'), 'rb') as f:
            crop_centers_config = pickle.load(f)
            crop_centers = crop_centers_config['crop_centers']
            crops_per_image_centers = crop_centers_config['crops_per_image']

            # validate crops_per_image
            if crops_per_image_sizes == crops_per_image_centers:
                crops_per_image = crops_per_image_sizes
            else:
                raise AssertionError(f'{name}: crops_per_image of crop_sizes and crop_centers are not the same!')

        generate_crop_centers = False

    # ------------------------------------------------------------------------------------------------------------------

    # create/load cropped images
    if not os.path.isfile(os.path.join('.', 'preprocessed_images', name + '.pkl')):

        images_cropped = []
        crop_arrays = []
        target_arrays = []

        for n, image in tqdm(enumerate(image_yielder.get_images()), desc=f'{name}: Cropping images...',
                             total=len(image_yielder)):
            image = np.array(image)
            for crop in range(crops_per_image):

                # predefined
                crop_size = crop_sizes[n, crop]

                # generate random crop_center tuple
                if generate_crop_centers:
                    image_y = image.shape[0]  # Y
                    image_x = image.shape[1]  # X
                    crop_center = [0, 0]  # (Y, X)
                    crop_center[0] = random.randrange(start=int(20 + (int(crop_size[0] / 2)) + 1),
                                                      stop=int(image_y - (20 + (int(crop_size[0] / 2)) + 1)))  # Y
                    crop_center[1] = random.randrange(start=int(20 + (int(crop_size[1] / 2)) + 1),
                                                      stop=int(image_x - (20 + (int(crop_size[1] / 2)) + 1)))  # X
                    crop_centers[n, crop] = crop_center

                # load crop_center tuple
                else:
                    crop_center = crop_centers[n, crop]

                # crop image
                image_cropped, crop_array, target_array = create_data(
                    image_input=image, crop_size=crop_size, crop_center=crop_center)

                # transpose shape from (Y, X) to (X, Y)
                image_cropped = image_cropped.transpose(1, 0)  # --> (X, Y)
                crop_array = crop_array.transpose(1, 0)  # --> (X, Y)
                target_array = target_array.transpose(1, 0)  # --> (X, Y)

                images_cropped.append(image_cropped)
                crop_arrays.append(crop_array)
                target_arrays.append(target_array)

        if generate_crop_centers:
            crop_centers_config = {'crop_centers': crop_centers, 'crops_per_image': crops_per_image}
            with open(os.path.join('.', 'config', name, 'crop_centers.pkl'), 'wb') as f:
                pickle.dump(crop_centers_config, f)

        # safe cropped images
        image_data = {'images_cropped': images_cropped, 'crop_arrays': crop_arrays, 'target_arrays': target_arrays}
        with open(os.path.join('.', 'preprocessed_images', name + '.pkl'), 'wb') as f:
            pickle.dump(image_data, f)

        print(f'{name}: Successfully cropped the given images and saved the result!')

    else:
        # load cropped images
        with open(os.path.join('.', 'preprocessed_images', name + '.pkl'), 'rb') as f:
            image_data = pickle.load(f)
            images_cropped, crop_arrays, target_arrays = image_data['images_cropped'], image_data['crop_arrays'], \
                                                         image_data['target_arrays']

        print(f'{name}: Found existing preprocessed images with the same name! Loading these images.')

    return images_cropped, crop_arrays, target_arrays


# ----------------------------------------------------------------------------------------------------------------------
# ImageYielderFromDirectory --------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class ImageYielderFromDirectory:
    """
    Searches for image files in given input directory and returns them either with or without normalization.
    """

    def __init__(self, input_dir: str):
        self.file_paths = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
        self.file_names = [os.path.basename(path) for path in self.file_paths]

    def __len__(self):
        return len(self.file_names)

    def get_stats(self):
        """
        Generates the mean and standard deviation for every image in the input directory.
        :return: List of means and standard deviations
        """
        means = np.zeros(shape=(len(self.file_paths),), dtype=np.float64)
        stds = np.zeros(shape=(len(self.file_paths),), dtype=np.float64)

        for n, file in enumerate(self.file_paths):
            image = np.array(Image.open(file))
            means[n] = np.mean(image)
            stds[n] = np.std(image)

        return means, stds

    def get_images(self):
        """
        Creates a yielder for the images found in the input directory.
        :return: yielder of images
        """
        for n, file in enumerate(self.file_paths):
            image = Image.open(file)

            yield image

    def get_images_normal(self):
        """
        Uses get_stats() to return normalized images found in the input directory.
        :return: yielder for normalized images
        """
        means, stds = self.get_stats()
        for n, file in enumerate(self.file_paths):
            image = np.array(Image.open(file), dtype=np.float32)
            image_normal = (image - means[n]) / stds[n]

            yield image_normal


# ----------------------------------------------------------------------------------------------------------------------
# ImageYielderFromList -------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class ImageYielderFromList:

    def __init__(self, image_list: list):
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def get_images(self):
        """
        Creates a yielder for the images in the given list.

        :return: Yielder for images
        """
        for image in self.image_list:
            yield image

# ----------------------------------------------------------------------------------------------------------------------
# Normalization --------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def normalize(input: list, file_name: str, overwrite: bool = False):
    """
    Normalizes a given input list and saves the means and standard deviations as a pickle file in the folder
    -folder_name- in the config directory with the name -name- if one doesn't already exists. If this is the case, the
    existing config file will be loaded. This makes it possible to normalize a given input with previously saved values.

    :param input: list of numpy arrays
    :param file_name: name for config file
    :param overwrite: set to True if files previously created by this function should be removed
    :return: normalized input as list
    """
    means = []
    stds = []
    output = []

    if overwrite:
        if os.path.isfile(os.path.join('.', 'config', file_name + '_means_stds.pkl')):
            os.system(f'rm {os.path.join(".", "config", file_name + "_means_stds.pkl")}')

    if os.path.isfile(os.path.join('.', 'config', file_name + '_means_stds.pkl')):

        # load means and stds
        with open(os.path.join('.', 'config', file_name + '_means_stds.pkl'), 'rb') as f:
            means_stds_config = pickle.load(f)
            means = means_stds_config['means']
            stds = means_stds_config['stds']

        print(f'{file_name}: Found existing means and standard deviations! Loading these.')

    else:
        # generate means and stds
        for element in input:
            mean = np.mean(element)
            std = np.std(element)
            if std == 0:
                std = 1

            means.append(mean)
            stds.append(std)

        # saving results in same order as input
        means_stds_config = {'means': means, 'stds': stds}
        with open(os.path.join('.', 'config', file_name + '_means_stds.pkl'), 'wb') as f:
            pickle.dump(means_stds_config, f)

        print(f'{file_name}: Calculated means and standard deviations and saved the results! '
              f'Successfully normalized input.')

    # normalizing input
    for n, element in enumerate(input):
        output.append((element - means[n]) / stds[n])

    return output, means, stds


# ----------------------------------------------------------------------------------------------------------------------
# Dataset --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class TrainDataset(Dataset):

    def __init__(self, image_list: list, crop_list: list, target_list: list, exp: bool = False):
        """
        Creates a Dataset instance for training (either for train, test or validation set). If -exp- is set to True,
        the targets will not be the whole, not cropped images, but a tensor with only the cropped out part visible and
        every pixel around this set to 0.

        :param image_list: list of images as numpy arrays
        :param crop_list: list of crop numpy arrays
        :param target_list: list of targets as numpy arrays
        :param exp: set to True if you want to experiment
        """
        self.image_list = image_list
        self.crop_list = crop_list
        self.target_list = target_list
        self.exp = exp

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        """
        Returns exactly one sample, its target and the sample_id. Moreover, for each pixel in a second dimension there
        is the information if the pixel was cropped or not. This is reached by concatenating the image tensor and the
        corresponding crop tensor.

        :param index: index of desired sample
        :return: tuple with sample tensor, target tensor and sample_id tensor
        """
        image_norm = self.image_list[index]
        crop_np = self.crop_list[index]
        target_np = self.target_list[index]

        # creating tensor from numpy array
        image = torch.empty(size=(1, image_norm.shape[0], image_norm.shape[1]), dtype=torch.float32)
        crop = torch.empty(size=(1, crop_np.shape[0], crop_np.shape[1]), dtype=torch.float32)
        target = torch.empty(size=(1, target_np.shape[0], target_np.shape[1]), dtype=torch.float32)

        image[0] = torch.tensor(image_norm, dtype=torch.float32)
        crop[0] = torch.tensor(crop_np, dtype=torch.float32)
        target[0] = torch.tensor(target_np, dtype=torch.float32)

        if self.exp:
            zero_ids = (crop == 0).nonzero()

            for i in zero_ids:
                target[i[0], i[1], i[2]] = 0

        # concatenating crop and image tensor
        sample = torch.cat((image, crop), 0)

        sample_id = torch.tensor(index, dtype=torch.int)

        return sample, target, sample_id

# ----------------------------------------------------------------------------------------------------------------------


class TestDataset(Dataset):

    def __init__(self, image_list: list, crop_sizes_list: list, crop_centers_list: list):
        """
        Creates a Dataset for creating the predictions for submission and for use in 'challenge.py'.

        :param image_list: list of images as numpy arrays
        :param crop_sizes_list: list of crop sizes as tuples
        :param crop_centers_list: list of crop centers as tuples
        """
        self.image_list = image_list
        self.crop_sizes_list = crop_sizes_list
        self.crop_centers_list = crop_centers_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        """
        Returns exactly one sample and the sample_id. Moreover, for each pixel in a second dimension there
        is the information if the pixel was cropped or not. This is reached by concatenating the image tensor and the
        corresponding crop tensor.

        :param index: index of desired sample
        :return: tuple with sample tensor and sample_id tensor
        """
        image_norm = self.image_list[index]
        _, crop_array, _ = create_data(image_input=image_norm, crop_size=self.crop_sizes_list[index],
                                       crop_center=self.crop_centers_list[index])

        # creating tensor from numpy array
        image = torch.empty(size=(1, image_norm.shape[0], image_norm.shape[1]), dtype=torch.float32)
        crop = torch.empty(size=(1, image_norm.shape[0], image_norm.shape[1]), dtype=torch.float32)

        image[0] = torch.tensor(image_norm, dtype=torch.float32)
        crop[0] = torch.tensor(crop_array, dtype=torch.float32)

        # concatenating crop and image tensor
        sample = torch.cat((image, crop), 0)

        sample_id = torch.tensor(index, dtype=torch.int)

        return sample, sample_id


# ----------------------------------------------------------------------------------------------------------------------
# DataLoader -----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def collate_fn(batch_as_list: list):
    """
    Pads all images of one batch to the same size and stacks them. The same happens with the given targets.

    :param batch_as_list: list with samples containing the input image, the corresponding target and sample_id.
    :return: stacked_samples (tensor of images), stacked_targets (tensor of targets), targets (list of unchanged targets
    as tensors), shapes (original shape of input images), sample_ids
    """
    batch_size = len(batch_as_list)

    images = [entry[0] for entry in batch_as_list]
    targets = [entry[1] for entry in batch_as_list]
    sample_ids = [entry[2] for entry in batch_as_list]

    # handle images ----------------------------------------------------------------------------------------------------

    # get max height and width of images in one batch
    x_lengths_img = [img.shape[1] for img in images]
    y_lengths_img = [img.shape[2] for img in images]

    # create tensor with shapes of all images
    x_tensor = torch.empty(size=(len(x_lengths_img), 1))
    y_tensor = torch.empty(size=(len(y_lengths_img), 1))
    x_tensor[:, 0] = torch.tensor(x_lengths_img)
    y_tensor[:, 0] = torch.tensor(y_lengths_img)
    shapes = torch.cat((x_tensor, y_tensor), 1)

    x_max_img = max(x_lengths_img)
    y_max_img = max(y_lengths_img)

    # create zero-padding batch with max width and max height
    stacked_samples = torch.zeros(size=(batch_size, 2, x_max_img, y_max_img), dtype=torch.float32)

    # stack images
    for i, image in enumerate(images):
        stacked_samples[i, :, :image.shape[1], :image.shape[2]] = image

    # handle targets ---------------------------------------------------------------------------------------------------

    # get max height and width of targets in one batch
    x_lengths_tg = [tg.shape[1] for tg in targets]
    y_lengths_tg = [tg.shape[2] for tg in targets]

    x_max_tg = max(x_lengths_tg)
    y_max_tg = max(y_lengths_tg)

    # create zero-padding batch with max width and max height
    stacked_targets = torch.zeros(size=(batch_size, 1, x_max_tg, y_max_tg), dtype=torch.float32)
    stacked_targets_exp = torch.zeros(size=(batch_size, 1, x_max_tg, y_max_tg), dtype=torch.float32)

    # stack targets
    for i, target in enumerate(targets):
        stacked_targets[i, :, :target.shape[1], :target.shape[2]] = target

    return stacked_samples, stacked_targets, targets, shapes, sample_ids


def collate_fn_test(batch_as_list: list):
    """
    collate_fn function for DataLoader to create predictions for submission. Returns no targets. For use in
    'challenge.py'.

    :param batch_as_list: list with samples containing the input image and sample_id.
    :return: stacked_samples (tensor of cropped out images), shapes (original shapes of input images), sample_ids
    """

    batch_size = len(batch_as_list)

    images = [entry[0] for entry in batch_as_list]
    sample_ids = [entry[1] for entry in batch_as_list]

    # handle images ----------------------------------------------------------------------------------------------------

    # get max height and width of images in one batch
    x_lengths_img = [img.shape[1] for img in images]
    y_lengths_img = [img.shape[2] for img in images]

    # create tensor with shapes of all images
    x_tensor = torch.empty(size=(len(x_lengths_img), 1))
    y_tensor = torch.empty(size=(len(y_lengths_img), 1))
    x_tensor[:, 0] = torch.tensor(x_lengths_img)
    y_tensor[:, 0] = torch.tensor(y_lengths_img)
    shapes = torch.cat((x_tensor, y_tensor), 1)

    x_max_img = max(x_lengths_img)
    y_max_img = max(y_lengths_img)

    # create zero-padding batch with max width and max height
    stacked_samples = torch.zeros(size=(batch_size, 2, x_max_img, y_max_img), dtype=torch.float32)

    # stack images
    for i, image in enumerate(images):
        stacked_samples[i, :, :image.shape[1], :image.shape[2]] = image

    return stacked_samples, shapes, sample_ids


# ----------------------------------------------------------------------------------------------------------------------
# Training -------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def train(model: torch.nn.Module, dataloader: DataLoader, valloader: DataLoader, validate_at: int,
          missing_targets_val: list, val_means: list, val_stds: list, optimizer, loss_function, n_updates: int,
          name: str, device: torch.device = 'cuda:0'):
    """
    Trains the given model on the given data in -dataloader- under the use of a specific -optimizer- and
    -loss_function-. The training goes -n_updates- long.

    Every -validate_at- iterations, the model will be evaluated on the validation set given as
    -valloader- and under the use of -missing_targets_val-, -val_means- and -val_stds-. If the current state of the
    model performs better on the validation set than the last state, it will be saved as 'best_model.pt' in the folder
    'models'.

    The loss, gradients and parameters will be saved under the name -name- with tensorboard.

    :param model: model to train
    :param dataloader: data for model training
    :param valloader: data for evaluation
    :param validate_at: number of updates after which the model should be evaluated on the validation set
    :param missing_targets_val: targets for evaluation
    :param val_means: means for denormalization during evaluation
    :param val_stds: means for denormalization during evaluation
    :param optimizer: optimizer for training
    :param loss_function: loss function
    :param n_updates: number of updates
    :param name: name for tensorboard
    :param device: CPU or GPU
    :return: trained model
    """
    model.to(device)

    update = 0

    # define progress bar, tensorboard writer, 'best' validation loss
    update_progress_bar = tqdm(total=n_updates, desc='Training')
    writer = SummaryWriter(log_dir=os.path.join('.', 'log', 'training'))
    best_validation_loss = 10000
    best_model = model

    # go over dataloader as long n_updates is not reached
    while update < n_updates:
        for batch in dataloader:

            # training -------------------------------------------------------------------------------------------------

            # get batch
            samples, stacked_targets, targets_list, shapes, ids = batch
            crop_arrays = samples[:, 1]
            crop_arrays = crop_arrays.type(torch.bool)

            # send to device
            samples = samples.to(device)
            stacked_targets = stacked_targets.to(device)
            crop_arrays = crop_arrays.to(device)

            # create prediction on given samples, returns images with same spatial dimensions as input
            output = model(samples)

            loss = loss_function(output, stacked_targets)

            loss.backward()

            optimizer.step()

            # analysis -------------------------------------------------------------------------------------------------

            # write to tensorboard every 10 updates
            if update % 10 == 0:

                # Add losses as scalars
                writer.add_scalar(tag=f"{name}/main_loss", scalar_value=loss.cpu(), global_step=update)

                # Add weights as arrays
                for i, param in enumerate(model.parameters()):
                    writer.add_histogram(tag=f'{name}/param_{i}', values=param.cpu(), global_step=update)

                # Add gradients as arrays
                for i, param in enumerate(model.parameters()):
                    writer.add_histogram(tag=f'{name}/gradients_{i}', values=param.grad.cpu(), global_step=update)

            # evaluate model every validate_at steps
            if update % validate_at == 0 and update > 0:

                update_progress_bar.set_description(f"Testing on validation set, this may take a moment", refresh=True)

                # evaluate
                val_loss, val_loss_whole_img = evaluate(model, dataloader=valloader, targets=missing_targets_val,
                                                        to_denormalize=True, means=val_means, stds=val_stds,
                                                        with_tqdm=False, device=device)

                # write to tensorboard

                # add validation loss on just cropped out parts
                writer.add_scalar(tag=f"validation_{name}/val_loss", scalar_value=val_loss, global_step=update)

                # add validation loss on whole images
                writer.add_scalar(tag=f"validation_{name}/val_loss_whole", scalar_value=val_loss_whole_img.cpu(),
                                      global_step=update)

                # Add validation weights as arrays
                for i, param in enumerate(model.parameters()):
                    writer.add_histogram(tag=f'validation_{name}/param_{i}', values=param.cpu(), global_step=update)

                # Add validation gradients as arrays
                for i, param in enumerate(model.parameters()):
                    writer.add_histogram(tag=f'validation_{name}/gradients_{i}', values=param.grad.cpu(),
                                         global_step=update)

                # Save best model for early stopping
                if best_validation_loss > val_loss:
                    best_validation_loss = val_loss
                    best_model = model
                    torch.save(model, os.path.join('.', 'models', 'best_model.pt'))

            # ----------------------------------------------------------------------------------------------------------

            optimizer.zero_grad()

            # update progress bar
            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()
            update += 1

            if update >= n_updates:
                break

    update_progress_bar.close()

    return model

# ----------------------------------------------------------------------------------------------------------------------
# Evaluation -----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def evaluate(model: torch.nn.Module, dataloader: DataLoader, targets: np.array, to_denormalize: bool,
             means: list = None, stds: list = None, with_tqdm: bool = True, device: torch.device = 'cuda:0'):
    """
    Evaluates a given model on given data. The function also denormalizes the predictions of the model if necessary.

    :param model: neural network model to evaluate
    :param dataloader: data to evaluate model on
    :param targets: list of targets
    :param to_denormalize: set to True if predictions have to be denormalized
    :param means: means for denormalizing predictions if to_denormalize = True
    :param stds: standard deviations for denormalizing predictions if to_denormalize = True
    :param with_tqdm: set to True if a tqdm progress bar is desired
    :param device: CPU or GPU
    :return: overall_loss_targets: MSE of cropped out part, loss_whole_image: MSE of whole image
    """

    losses = []
    loss_function = torch.nn.MSELoss()
    loss_whole_img = torch.tensor(0., device=device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing', total=len(dataloader), disable=(not with_tqdm)):

            # create predictions ---------------------------------------------------------------------------------------

            samples, stacked_targets, targets_list, shapes, ids = batch
            stacked_targets = stacked_targets.to(device)
            samples = samples.to(device)

            crop_arrays = samples[:, 1]

            outputs = model(samples)

            # create loss with whole images (not just cropped out parts) as targets ------------------------------------

            loss_whole_img += (torch.stack([loss_function(output, target) for output, target in
                                                zip(outputs, stacked_targets)]).sum() / len(dataloader.dataset))

            # create loss with just cropped out parts as targets -------------------------------------------------------

            for n, prediction in enumerate(outputs):

                # get unscaled target
                index = int(ids[n])
                target = targets[index].flatten()

                # select restored part from output and convert to numpy array
                crop_array = crop_arrays[n].cpu().detach().numpy().astype(bool)
                prediction_np = prediction[0].cpu().detach().numpy()
                predicted_target = prediction_np[crop_array]

                # denormalize prediction with mean and standard deviation of input if to denormalize
                if to_denormalize:
                    mean = means[index]
                    std = stds[index]

                    predicted_target = predicted_target * std + mean

                # calculate MSE loss
                loss = mean_squared_error(target, predicted_target)
                losses.append(loss)

    # calculate mean over all losses
    overall_loss_targets = np.mean(losses)

    return overall_loss_targets, loss_whole_img
