"""
Get predictions for challenge submission.
"""

import os
import dill as pickle
import torch

from torch.utils.data import DataLoader

from utils import TestDataset, collate_fn_test
from utils import normalize


def challenge(modelname: str, file_name: str):

    predictions = []
    model = torch.load(os.path.join('.', 'models', modelname + '.pt'))

    # load and prepare data --------------------------------------------------------------------------------------------

    with open(os.path.join('.', 'challenge_testset.pkl'), 'rb') as f:
        dataset_dict = pickle.load(f)

    images = dataset_dict['images']
    crop_sizes = dataset_dict['crop_sizes']
    crop_centers = dataset_dict['crop_centers']

    # normalize data
    test_image_list, test_means, test_stds = normalize(input=images, file_name='challenge', overwrite=False)

    # create dataset
    test_dataset = TestDataset(image_list=test_image_list, crop_sizes_list=crop_sizes, crop_centers_list=crop_centers)

    # create dataloader
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=3, batch_size=10, collate_fn=collate_fn_test)

    # predict ----------------------------------------------------------------------------------------------------------

    for batch in test_loader:
        samples, shapes, ids = batch
        samples = samples.to('cuda:0')

        crop_arrays = samples[:, 1]

        outputs = model(samples)

        # crop out target part of image and append it to list
        for n, pred in enumerate(outputs):
            idi = int(ids[n])

            # convert to numpy array and normalize
            prediction_whole = pred[0].cpu().detach().numpy() * test_stds[idi] + test_means[idi]

            # get mask
            crop_array = crop_arrays[n].cpu().detach().numpy().astype(bool)

            predicted_target = prediction_whole[crop_array]
            prediction = predicted_target.reshape(crop_sizes[idi])
            prediction = prediction.astype('uint8')

            predictions.append(prediction)

    # save predictions
    with open(os.path.join('.', 'challenge_submissions', file_name + '.pkl'), 'wb') as f:
        pickle.dump(predictions, f)

    print('\nSaved submission prediction in folder "challenge_submissions"!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='name of model', type=str)
    parser.add_argument('file_name', help='name of pickle file for prediction', type=str)
    args = parser.parse_args()
    modelname = args.model_name
    file_name = args.file_name
    challenge(modelname, file_name)