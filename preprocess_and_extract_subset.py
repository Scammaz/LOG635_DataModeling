from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import numpy as np
import imutils
import random
import pickle

DATASET_A_DIR = Path.cwd() / "DataSet_B"
LABELS = ["Cercle2", "Cercle3", "Diamant2", "Diamant5", "Hexagone3", "Hexagone4", "Triangle3", "Triangle5"]

def pretreat_subset(pretreated_subset_path: Path):
    
    form_dir = ""
    form_subdir = ""
    file_name = ""
    
    for first_level in DATASET_A_DIR.glob('*'):
        if first_level.is_dir():
            form_dir = first_level.name
            for second_level in first_level.glob('*'):
                if second_level.is_dir():
                    form_subdir = second_level.name
                    class_num = LABELS.index(form_subdir) if form_subdir in LABELS else -1
                    if(class_num >= 0):
                        print(f"{label} {class_num}")
                    
                    for file in second_level.glob('*'):
                        if file.is_file():
                            file_name = file.name
                            # preprocess image
                            image = cv2.imread(file)
                            processed_image = preprocess_image(image)
                            
                            # extract maker from that image
                            new_image = extract_marker(image, processed_image)
                            
                            # save extracted to processed dataset
                            subset_dir = pretreated_subset_path / form_dir / form_subdir / file_name
                            save_image_to(subset_dir, new_image)
                            

def preprocess_image(image):
    pass

def extract_marker(originalImage, processedImage):
    pass

def save_image_to(dir, image):
    cv2.imwrite(dir, image)


def program():
    # path = Path.cwd() / "Preprocessed_DataSet_B"
    # pretreat_subset(path)    

program()