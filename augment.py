# import os
# import imgaug as ia
# import imageio.v2 as imageio
# from imgaug import augmenters as iaa

# # Define individual augmentations
# augmenters = [
#     ("fliplr", iaa.Fliplr(0.5)),  # horizontal flips
#     ("crop", iaa.Crop(percent=(0, 0.1))),  # random crops
#     ("contrast", iaa.LinearContrast((0.75, 1.5))),
#     ("multiply", iaa.Multiply((0.8, 1.2), per_channel=0.2)),
#     ("affine", iaa.Affine(
#         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#         rotate=(-25, 25),
#         shear=(-8, 8)
#     )),
#     ("elastic", iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25))
# ]

# def load_batch(batch_size):
#     # Load a batch of images (e.g., all images in a directory)
#     files = [os.path.join('facedataset', f) for f in os.listdir('facedataset') if f.endswith(('.jpg', '.jpeg', '.png'))]
#     images = [imageio.imread(f) for f in files[:batch_size]]
#     return images, files[:batch_size]

# def augment_and_save(images, filenames, augmenter, augmenter_name):
#     # Apply the augmentations and save the images in separate folders
#     images_aug = augmenter(images=images)
#     augment_folder = os.path.join('augment', augmenter_name)
#     if not os.path.exists(augment_folder):
#         os.makedirs(augment_folder)
#     for img_aug, filename in zip(images_aug, filenames):
#         base_filename = os.path.basename(filename)
#         new_filename = f"{augmenter_name}_{base_filename}"
#         imageio.imwrite(os.path.join(augment_folder, new_filename), img_aug)

# # Example usage:
# batch_size = 144  # Define your batch size
# images, filenames = load_batch(batch_size)
# for augmenter_name, augmenter in augmenters:
#     augment_and_save(images, filenames, augmenter, augmenter_name)

# import os
# import imageio.v2 as imageio
# from imgaug import augmenters as iaa

# # Define individual augmentations
# augmenters = [
#     iaa.Fliplr(0.5),  # horizontal flips
#     iaa.Crop(percent=(0, 0.1)),  # random crops
#     iaa.LinearContrast((0.75, 1.5)),
#     iaa.Multiply((0.8, 1.2), per_channel=0.2),
#     iaa.Affine(
#         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#         rotate=(-25, 25),
#         shear=(-8, 8)
#     ),
#     iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)
# ]

# def load_images():
#     # Load all images in the directory
#     files = [os.path.join('facedataset', f) for f in os.listdir('facedataset') if f.endswith(('.jpg', '.jpeg', '.png'))]
#     images = [(imageio.imread(f), f) for f in files]
#     return images

# def apply_augmentations(images):
#     # Apply each augmentation to each image and save the results
#     for augmenter in augmenters:
#         augmenter_name = augmenter.__class__.__name__
#         augment_folder = os.path.join('augment', augmenter_name)
#         if not os.path.exists(augment_folder):
#             os.makedirs(augment_folder)
#         for image, filename in images:
#             base_filename, file_ext = os.path.splitext(os.path.basename(filename))
#             image_aug = augmenter(image=image)
#             new_filename = f"{base_filename}_{augmenter_name}{file_ext}"
#             augment_folder = os.path.join('augment', base_filename)
#             if not os.path.exists(augment_folder):
#                 os.makedirs(augment_folder)
#             imageio.imwrite(os.path.join(augment_folder, new_filename), image_aug)
# # Example usage:
# images = load_images()
# apply_augmentations(images)

import os
import imageio.v2 as imageio
from imgaug import augmenters as iaa

# Define individual augmentations
augmenters = [
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Crop(percent=(0, 0.1)),  # random crops
    iaa.LinearContrast((0.75, 1.5)),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    ),
    iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)
]

def load_images():
    # Load all images in the directory
    files = [os.path.join('facedataset', f) for f in os.listdir('facedataset') if f.endswith(('.jpg', '.jpeg', '.png'))]
    images = [(imageio.imread(f), f) for f in files]
    return images

def apply_augmentations(images):
    # Apply each augmentation to each image and save the results
    augment_folder = 'augment'
    if not os.path.exists(augment_folder):
        os.makedirs(augment_folder)
    for augmenter in augmenters:
        augmenter_name = augmenter.__class__.__name__
        for image, filename in images:
            base_filename, file_ext = os.path.splitext(os.path.basename(filename))
            image_aug = augmenter(image=image)
            new_filename = f"{base_filename}_{augmenter_name}{file_ext}"
            imageio.imwrite(os.path.join(augment_folder, new_filename), image_aug)

# Example usage:
images = load_images()
apply_augmentations(images)
