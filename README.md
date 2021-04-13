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
