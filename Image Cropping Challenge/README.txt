The goal of this challenge was to create a Machine Learning application that is able to fill out missing parts of images. 


Preprocessing

########################################################################################################################

## validate images ##

--> validated images in folder 'validated_images'
--> number of images "n_images"

variables:
- n_images: int, number of validated images

########################################################################################################################

## read images ##

--> list "images" with all images in output folder of validation as -PIL Image-

## train test split 70:15:15 ##

--> list "train_set" ("n_images" * 0.7 elements), "test_set" "n_images" * 0.15 elements), "validation_set" with
    -PIL Images- ("n_images" * 0.15 elements)

 variables:
 - images: list, all images in image folder of datatype -PIL Image-
 - train_set: list, 70% of all images with datatype -PIL Image-
 - test_set: list, 15% of all images with datatype -PIL Image-
 - validation_set: list, 15% of all images with datatype -PIL Image-

########################################################################################################################

## augmentation and resizing ##

--> list "train_images_augmented" with augmented -PIL Images- and right size # "n_images" * 0.7 * "n_augmentations"
    elements
--> list "test_set_resized", "validation_set_resized" with resized -PIL Images- # same number of elements

variables:
- train_images_augmented: list, "n_augmentations" augmented images per training image and in correct size of datatype
    -PIL-Image-
- test_set_resized: list, test_set, but all images with correct size and of datatype -PIL Image-
- validation_set_resized: list, validation_set, but all images with correct size and of datatype -PIL Image-

########################################################################################################################

## cropping ##

--> lists "train_image_list", "test_image_list", "validation_image_list" with correctly and randomly cropped out parts
    where one image was cropped "crops_per_image" times
--> lists "train_crop_list", "test_crop_list", "validation_crop_list" which are containing the information where the
    cropping has happened
--> lists "train_missing", "test_missing", "validation_missing"

########################################################################################################################

overall loss of Model A (means): ca. 3775.324
overall loss of Model B (simple CNN): ca. 1682.149 (with whole images) / ca. 2047.704 (with just cropped out parts)
MyScndCNN: 926.345 / 0.0994
v4: 870

BEST: 833

Copy the following code into main.py if Model A should be used:

# Model A --------------------------------------------------------------------------------------------------------------

losses = []
loss_function = torch.nn.MSELoss(reduction='mean')

for _, batch in tqdm(enumerate(train_loader), desc='Batches', total=len(train_loader)):

    samples, targets, shapes, ids = batch
    model_a = ModelA(targets=targets, means=train_means, ids=ids)

    predictions = model_a(samples)

    for n, prediction in enumerate(predictions):
        target = targets[n]
        loss = loss_function(prediction, target)
        losses.append(loss)


losses_np = np.array(losses)
overall_loss_model_a = np.mean(losses_np)







"""
input_dir = os.path.join('.', '0000')
output_dir = os.path.join('.', 'validated_images', '0000_test_images')
train_name = '0000_train_images'
test_name = '0000_test_images'
validation_name = '0000_validation_images'
general_name = '0000_images'
"""
