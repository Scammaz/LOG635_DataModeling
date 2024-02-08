import os
import cv2
import numpy as np

# Path to the dataset directory
dataset_dir = 'DataSet/Cercles/Cercle2/'
dataset_cercle = 'DataSet/Cercles'
dataset_carre = 'DataSet/Carres'
dataset_triangle = 'DataSet/Triangles'
dataset_diamant = 'DataSet/Diamants'



# Iterate through each image file in the dataset directory
nombre_photo = 0
nombre_carre = 0
liste_sigma = [0,30,45,60]
liste_sigma2 = [75,60,145,130,100]



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

liste_lTresh = [45]
liste_uTresh = [150]

for lower in liste_lTresh:
    for higher in liste_uTresh:
        for filename in os.listdir(dataset_dir):
            nombre_photo=nombre_photo+1
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                # Load the image
                image = cv2.imread(os.path.join(dataset_dir, filename))

                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Apply bilateral filter
                filtered_image = cv2.bilateralFilter(gray, 15, 75, 60)

                # Apply Canny edge detection
                edges = cv2.Canny(filtered_image, lower, higher)

                # Find contours
                contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Filter contours
                
                square_contours = []
                for contour in contours:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                    shape = detect_shape(contour)
                    if shape== "square":
                        square_contours.append(contour)
                        nombre_carre=nombre_carre+1

                # Create a new image with the detected contours
                roi_image = np.zeros_like(image)
                cv2.drawContours(roi_image, square_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

                # Display or save the processed image
                cv2.imshow('Processed Image', roi_image)
                cv2.waitKey(0)
        print("Nombre de photos",nombre_photo)
        print("Nombre de carres",nombre_carre)
        print("      Low Tresh",lower)
        print("      High Tresh",higher)
        print("      Pourcentage de carres",100*(nombre_carre/nombre_photo))

        nombre_photo=0
        nombre_carre=0
# Close all OpenCV windows


cv2.destroyAllWindows()

