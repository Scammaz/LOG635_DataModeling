from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import numpy as np
import imutils
import random
import pickle
import Augmentor as aug
import os

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
                                path = DATASET_A_DIR / first_level.name / second_level.name
                                image = cv2.imread(os.path.join(path, file.name))

                                print(file.name)
                                # extract maker from that image
                                new_image = extract_marker(image)
                                
                                if(new_image is not None):
                                    # save extracted to processed dataset
                                    subset_dir = os.path.join(pretreated_subset_path, first_level.name, second_level.name)

                                    if not os.path.exists(subset_dir):
                                        os.makedirs(subset_dir)

                                    save_image_to(subset_dir + "/" + file.name, new_image)

# Function to detect shape
def detect_shape(contour):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "diamond"

    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "hexagon"

    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"

    # return the name of the shape
    return shape

def dispayImage(image):
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_marker(image):

    #dispayImage(image)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #dispayImage(gray)

    # Apply bilateral filter
    filtered_image = cv2.bilateralFilter(gray, 15, 60, 130)
    #dispayImage(filtered_image)

    # Apply Canny edge detection
    edges = cv2.Canny(filtered_image, 70, 240)
    #dispayImage(edges)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours
    square_contours = []
    for contour in contours:
        #perimeter = cv2.arcLength(contour, True)
        #approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        shape = detect_shape(contour)
        if shape == "square":
            square_contours.append(contour)
            
    # Supposons que vous voulez extraire la région de l'objet ayant le plus grand contour
    try:
        max_contour = max(contours, key=cv2.contourArea)
    except:
        print("No square found")
        return None

    # Trouver les coordonnées du rectangle englobant ce contour
    x, y, w, h = cv2.boundingRect(max_contour)

    # Extraire la région de l'image originale
    region_of_interest = image[y:y+h, x:x+w]

    # Redimensionner l'image extraite à 40x40 pixels
    resized_image = cv2.resize(region_of_interest, (40, 40))

    # Afficher l'image résultante
    # cv2.imshow('Processed Image', resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return resized_image

def save_image_to(dir, image):
    print(dir)
    cv2.imwrite(dir, image)


def program():
    path = Path.cwd() / "Processed_DataSet_A"
    prepare_subset(path)

program()