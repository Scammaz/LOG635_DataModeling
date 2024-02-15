from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import numpy as np
import imutils
import random
import pickle

DATASET_A_DIR = Path.cwd() / "DataSet_A"
LABELS = ["Cercle2", "Cercle3", "Diamant2", "Diamant5", "Hexagone3", "Hexagone4", "Triangle3", "Triangle5"]

def prepare_subset(pretreated_subset_path: Path):
    
    # For each shape folder
    for first_level in DATASET_A_DIR.glob('*'):
        if first_level.is_dir():
            
            # For each chape subfolder
            for second_level in first_level.glob('*'):
                if second_level.is_dir():
                    label = second_level.name
                    class_num = LABELS.index(label) if label in LABELS else -1
                    
                    # If said shape label was chosen for subset
                    if(class_num >= 0):
                        
                        # For each file in shape subfolder
                        for file in second_level.glob('*'):
                            if file.is_file():
                                # preprocess image
                                image = cv2.imread(file)
                                processed_image = preprocess_image(image)
                                
                                # extract maker from that image
                                new_image = extract_marker(image, processed_image)
                                
                                # save extracted to processed dataset
                                subset_dir = pretreated_subset_path / first_level.name / second_level.name / file.name
                                save_image_to(subset_dir, new_image)
                            

def preprocess_image(image):
    return 0

def extract_marker(originalImage, processedImage):
    return 0

def save_image_to(dir, image):
    cv2.imwrite(dir, image)


def program():
    path = Path.cwd() / "Processed_DataSet_A"
    prepare_subset(path)

program()