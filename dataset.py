from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import numpy as np
import imutils
import random
import pickle

data_dir = Path.cwd() / "DataSet_B"
LABELS = ["Cercle2", "Cercle3", "Diamant2", "Diamant5", "Hexagone3", "Hexagone4", "Triangle3", "Triangle5"]

def create_data_set(dir: Path):
    data_set = []
    
    for first_level in data_dir.glob('*'):
        if first_level.is_dir():
            for second_level in first_level.glob('*'):
                if second_level.is_dir():
                    label = second_level.name
                    class_num = LABELS.index(label) if label in LABELS else -1
                    if(class_num >= 0):                    
                        for file in second_level.glob('*'):
                            if file.is_file():
                                # read image and save to temp list
                                image = cv2.imread(image)
                                data_set.append([image, class_num, label])
    
    # Shuffles the images
    random.shuffle(data_set)

    # features vector
    X = []

    # lables vector
    Y = []

    # Taking features and labels from dataset
    for features, class_num, label in data_set:
        X.append(features)
        y.append(label)
                            
    # Converts each image matrix to an image 
    X = np.array(X)

    # Creating the files containing all the information about your model and saving them to the disk
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("Y.pickle", "wb")
    pickle.dump(Y, pickle_out)
    pickle_out.close()