# CropDoor

## Overview

This project is to detect the door in the floor plan image and save it according to it's orientation(bottom, top, left, 
right, double). To detect the door, the pre-trained model, faster_rcnn_resnet_50, has been already trained with the 
door dataset. The tensorflow version is 1.12.0. 

## Structure

- input_image

    The floor plan images to crop door

- object_detection

    * The source code to detect the door in the floor plan image
    * The faster_rcnn_model(.pb and .pbtxt)

- output_image

    The cropped door images. This directory is automatically made while running the project.
    
- utils
    
    The source code to crop the detected door and save it into the new jpg image

- main
    
    The main execution file

- requirements

    All the dependencies for this project
    
- settings
    
    Several options in it.

## Installation

- Environment

    Ubuntu 18.04, Python 3.6, tensorflow==1.12.0

- Dependency Installation

    ```
        pip3 install -r requirements.txt
    ```  

## Execution

- Please copy the floor plan image to detect the door

- Please run the following command in the this project directory

    ```
        python3 main.py
    ```
