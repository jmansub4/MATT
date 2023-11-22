import os
import re
import shutil
import supervision as sv
import numpy as np
import cv2 as cv2
import imageio
import scipy.ndimage as nd
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from autodistill_yolov8 import YOLOv8
from ultralytics import YOLO
from pathlib import Path

# Set these values to appropriate folders on your local system

source_videos = "folder path of videos in multispectral formats (ex. rgb, fuse, tir) to extract images in corresponding folders"
source_images_from_videos = "folder path of multispectral images extracted from videos"
target = "folder path of images after applying flipped transformation"
labelled_target="folder path of final labeled images"
FRAME_STRIDE = 100

# Get a list of all the multispectral video types.
# The code assumes that multispectral videos in different formats are stored in corresponding folders.
# For example, all rgb videos are stored in "rgb" folder under this video source top level folder.

video_types = os.listdir(source_videos)

# Extract images and store them in appropriate folders depending on the type (rgb, tir, fuse) from videos
# The images are named with the folder name appended with an incrementing number. For example, rgb-0000.jpg, rgb-0001.jpg etc. in "rgb" folder.
# Another example, fuse-0000.jpg, fuse-0001.jpg etc. in "fuse" folder.
# This format is pre-requisite for transposing rgb labels on tir and fuse images.

for video_type in video_types:
    print(video_type)

    video_paths = sv.list_files_with_extensions(directory=source_videos + '/' + video_type, extensions=["mov", "mp4"])
    print(video_paths)
    
    for video_path in video_paths:
        print(video_path)
        video_name = video_path.stem
        print(video_name)
        image_name_pattern = video_type + "-{:05d}.jpg"
        if not os.path.exists(source_images_from_videos + '/' + video_type):
                    os.makedirs(source_images_from_videos + '/' + video_type)
        with sv.ImageSink(target_dir_path=source_images_from_videos + '/' + video_type, image_name_pattern=image_name_pattern) as sink:
            for image in sv.get_video_frames_generator(source_path=str(video_path), stride=FRAME_STRIDE):
                sink.save_image(image=image)

# Apply flipped image transformation
for folder in (os.listdir(source_images_from_videos)):
        for file in (os.listdir(source_images_from_videos + '/' + folder)):
            try:
                image = imageio.imread(source_images_from_videos + '/' + folder + "/" + str(os.path.splitext(file)[0])+'.jpg')
                flipped_images = cv2.flip(image, 1);
                if not os.path.exists(target + '/' + folder):
                    os.makedirs(target + '/' + folder)
                cv2.imwrite(target + '/' + folder + '/'+str(os.path.splitext(file)[0])+'.jpg',cv2.cvtColor(image, cv2.COLOR_RGB2BGR));
                cv2.imwrite(target + '/' + folder + '/'+str(os.path.splitext(file)[0])+'_flipped_images.jpg',cv2.cvtColor(flipped_images, cv2.COLOR_RGB2BGR));
        
            except Exception as e:
                print(e)

# Run auto labelling using SAM only on RGB images
ontology=CaptionOntology({
    "truck": "truck",
    "car": "car"    
})

base_model = GroundedSAM(ontology=ontology)


for folder in (os.listdir(target)):
    if folder == 'rgb':
        dataset = base_model.label(
            input_folder=target + "/" + folder,
            extension=".jpg",
            output_folder=labelled_target )


# Transpose generated label files to fuse and tir normal and flipped images
# The files that are generated in labeled_target folder are in a YOLO format. But initially it only contains labels and images from 
# regular and flipped rgb images.

# Get a list of rgb images numbers currently in labeled target folder in "train" and "valid" folder after removing the prefix "rgb-"
train = [ f[f.find("-")+1 : f.find(".")] for f in os.listdir(labelled_target + "/train/images/")]
valid = [ f[f.find("-")+1 : f.find(".")] for f in os.listdir(labelled_target + "/valid/images/")]

# Get a list of fuse and tir images
fuse_images = os.listdir(target + "/fuse")
tir_images = os.listdir(target + "/tir")

# Get a list of SAM generated rgb labels from "train" and "valid" folders
rgb_train_labels = os.listdir(labelled_target + "/train/labels/")
rgb_valid_labels = os.listdir(labelled_target + "/valid/labels/")

# Filter out the fuse images list to create train and valid sets based on the number of the rgb image.
train_fuse_images = list(filter(lambda x: all(y not in x for y in valid), fuse_images))
valid_fuse_images = list(filter(lambda x: all(y not in x for y in train), fuse_images))

# Filter out the tir images list to create train and valid sets based on the number of the rgb image.
train_tir_images = list(filter(lambda x: all(y not in x for y in valid), tir_images))
valid_tir_images = list(filter(lambda x: all(y not in x for y in train), tir_images))

# Copy train fuse images in labeled target train folder
for image in (train_fuse_images):
    shutil.copy(target + "/fuse/" + image, labelled_target + "/train/images/")

# Copy train tir images in labeled target train folder
for image in (train_tir_images):
    shutil.copy(target + "/tir/" + image, labelled_target + "/train/images/")

# Copy valid fuse images in labeled target valid folder
for image in (valid_fuse_images):
    shutil.copy(target + "/fuse/" + image, labelled_target + "/valid/images/")

# Copy valid tir images in labeled target valid folder
for image in (valid_tir_images):
    shutil.copy(target + "/tir/" + image, labelled_target + "/valid/images/")

# Copy the rgb labels after filtering out the appropriate ones from the entire rgb label set as fuse and tir labels to the labeled target labels folder
for l in rgb_train_labels:
    shutil.copy(labelled_target + "/train/labels/" + l, labelled_target + "/train/labels/fuse-" + l[l.find("-")+1 : l.find(".")] + ".txt")
    shutil.copy(labelled_target + "/train/labels/" + l, labelled_target + "/train/labels/tir-" + l[l.find("-")+1 : l.find(".")] + ".txt")

# Copy the rgb labels after filtering out the appropriate ones from the entire rgb label set as fuse and tir labels to the labeled target labels folder
for l in rgb_valid_labels:
    shutil.copy(labelled_target + "/valid/labels/" + l, labelled_target + "/valid/labels/fuse-" + l[l.find("-")+1 : l.find(".")] + ".txt")
    shutil.copy(labelled_target + "/valid/labels/" + l, labelled_target + "/valid/labels/tir-" + l[l.find("-")+1 : l.find(".")] + ".txt")

# Train YOLOV8 model

target_model = YOLO("yolov8n.pt")
target_model.train(data=labelled_target + "/data.yaml", epochs=200, device="mps")
target_model.train(labelled_target + "/data.yaml", epochs=200, device="mps")

