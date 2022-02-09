# fall-detection-CS3244

# Installation

Set up a anaconda environment in python 3.8. 

## Install all dependencies

Run `pip install -r requirements.txt`


Download the following from [Adrian Nunez' repo](https://github.com/AdrianNunez/Fall-Detection-with-CNNs-and-Optical-Flow)

## Download Optical Flows

[Optical Flow Download](https://drive.google.com/file/d/1YhBljXOFHdqukZW0Zp6TPbQ-brsoz7ep/view?usp=sharing) to `URFD_opticalflow` folder.

## Download saved_features

[Saved features](https://drive.google.com/file/d/1JQg6mCrV_0lQR0MSUaRlD5VusF5MLRhB/view?usp=sharing) and [labels](https://drive.google.com/file/d/1EKTpI7BzlX4qQoAyph5d5cJ1jnWU3f6n/view?usp=sharing) to `saved_features` folder.

## Download weights

[Weights file](https://drive.google.com/file/d/0B4i3D0pfGJjYNWxYTVUtNGtRcUE/view?usp=sharing) to home folder.

# Run the Model 

## Original model

`python tf2_temporalnet_urfd3.py`

## Modified model

`python VGG16.py`

# Data Augmentation

Augments video data based on:

- brightness
- color
- contrast
- horizontal flip
- saturation
- sharpness
- rotation

Changes each folder of photos with the same random augmentations mentioned above. Creates **50 sets** of augmented data using clips of data from the file path `data_augmentation_test/Falls` and outputs them into the path `data_augmentation_test/augmented_fall`. Each set of data will be created in a subfolder in the following format: `fall[video_number]`.

`python data_augmentation.py --main-folder-path=data_augmentation_test/Falls --output-folder-path=data_augmentation_test/augmented_fall --output-subfolder-prefix=fall --no-aug-data=50`

# Telegram bot

The telegram bot is currently only capable of sending a static fall image together with an alert of the location of the fall. In the future, the bot can be integrated with the model to run checks and stream video data to perform real-time predictions.

## Run the telegram bot

Include your own bot token in the `mytoken` variable.

### Run Locally

`python fall_bot.py`

*To run on a server, uncomment webhook code and include your own parameters.
