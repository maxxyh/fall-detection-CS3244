import argparse
import os
import numpy as np
import shutil
import random
from glob import glob
from PIL import Image, ImageEnhance

'''
Assumes images are stored in the following structure:
-Fall
    -Fall_001
        -Fall_001_frame_001.jpg
-Not Fall
    ...

e.g. program call
python data_augmentation.py --main-folder-path=data_augmentation_test/Falls 
    --output-folder-path=data_augmentation_test/augmented_fall 
    --output-subfolder-prefix=fall
    --no-aug-data=50 
'''


def augment_imgs(imgs, output_folder, input_folder):
    '''
    give a folder containing images generate random augmentations with reasonable paramenters
    Augmentations:
        brightness
        contrast
        saturation
        color
        sharpness
        rotation
        imgfliphorizontal
    '''
    rotation_angle = random.randint(-10, 10)
    img_hor_flip = True if random.random() > 0.5 else False
    sharpness = random.uniform(0.5, 1.5)
    brightness = random.uniform(0.5, 1.5)
    contrast = random.uniform(0.5, 1.5)
    color = random.uniform(0.5, 1.5)

    print("Input folder is: ", input_folder)
    all_frames = glob(input_folder + "/*.jpg")
    for img_path in imgs:
        img = Image.open(input_folder + "/" + img_path)
        if img_hor_flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotation_angle)
        sharpness_enhancer = ImageEnhance.Sharpness(img)
        img = sharpness_enhancer.enhance(sharpness)
        brightness_enhancer = ImageEnhance.Brightness(img)
        img = brightness_enhancer.enhance(brightness)
        contrast_enhancer = ImageEnhance.Contrast(img)
        img = contrast_enhancer.enhance(contrast)
        color_enhancer = ImageEnhance.Color(img)
        img = color_enhancer.enhance(color)
        img.save(output_folder + "/" + img_path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-folder-path', type=str, default='',
                        help='Path of folder that contains classes of images to be augmented', required=True)
    parser.add_argument('--output-folder-path', type=str, default='augmented', required=True,
                        help='Path of folder that will contain augmented clips and a temporary folder for holding augmented images')
    parser.add_argument('--output-subfolder-prefix', type=str, default='fall_fall-', required=True,
                        help='Prefix for the name of subfolders that will be generated for each clip')
    parser.add_argument('--no-aug-data', type=int, default='',
                        required=True, help='Number of augmented data files required')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_arguments()

    print("Args \n", opt)
    main_folder_path = opt.main_folder_path
    output_folder_path = opt.output_folder_path
    no_augmented_data = opt.no_aug_data
    subfolder_prefix = opt.output_subfolder_prefix

    print("Output folder path", output_folder_path)
    print("Main folder path", main_folder_path)
    print("No. augmented data", no_augmented_data)

    # Checks for existing output directory and overwrites it if it exists
    if os.path.exists(output_folder_path) and os.path.isdir(output_folder_path):
        shutil.rmtree(output_folder_path)
    os.makedirs(output_folder_path, exist_ok=True)

    # obtains subfolders each containing the set of frames from a video
    video_clip_names = os.listdir(main_folder_path)
    print(f"Image folders are {video_clip_names}")
    no_of_clips_available = len(video_clip_names)

    # loops for the number of sets of augmented data that needs to be generated
    for i in range(no_augmented_data):
        output_clip_folder_path = output_folder_path + \
            "//" + subfolder_prefix + str(i + 1)
        if os.path.exists(output_clip_folder_path) and os.path.isdir(output_clip_folder_path):
            shutil.rmtree(output_clip_folder_path)
        os.makedirs(output_clip_folder_path, exist_ok=True)
        vid_clip_no_to_augment = i % no_of_clips_available
        img_names = os.listdir(os.path.join(
            main_folder_path, video_clip_names[vid_clip_no_to_augment]))
        augment_imgs(imgs=img_names, output_folder=output_clip_folder_path,
                     input_folder=main_folder_path + "//" + video_clip_names[vid_clip_no_to_augment])
