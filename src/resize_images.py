#!/usr/bin/python
from PIL import Image
import os, sys
import torchvision
from argparse import ArgumentParser

"""
Script to resize all the images in given folder (either training, validation or test data).

Usage: resize_images.py -p [path to directory] -s [size to which image will be resized]
"""

# Globals
path = ""
size = ""
gray = False

def parse_arguments():
    global path
    global size
    global gray
    
    parser = ArgumentParser(description="Resize all files in folder")
    parser.add_argument(
        "-p", "--path", help="path to the train, valid or test directory", required=True
    )
    parser.add_argument(
        "-s", "--size", help="size to which image will be resized", required=True
    )
    parser.add_argument(
        "-g", "--gray", help="flag if grayscale is desired -> True", required=False, type=bool
    )

    args = parser.parse_args()
    path = args.path
    size = args.size
    if args.gray is not None:
        gray = args.gray
    

def resize(size):
    dirs = os.listdir(path)
    total = len(dirs)
    for i, patient in enumerate(dirs):
        if not (patient.startswith(".")):
            for study in os.listdir(os.path.join(path, patient)):
                if not (study.startswith(".")):
                    for image in os.listdir(os.path.join(path, patient, study)):
                        im = Image.open(os.path.join(path, patient, study, image))
                        imResize = torchvision.transforms.functional.resize(
                            im, size, interpolation=Image.LANCZOS
                        )
                        if gray:
                            imResize = torchvision.transforms.functional.to_grayscale(imResize)
                            
                        delete_path = os.path.join(path, patient, study, image)
                        
                        os.remove(delete_path)
                        save_path = os.path.join(path, patient, study, image.split('.')[0] + '.png')
                        print(f'patient {i+1}/{total} - path: {save_path}')
                        imResize.save(save_path)


if __name__ == "__main__":
    parse_arguments()
    resize(int(size))
