import os
import cv2
import numpy as np
import csv  # Import the csv module
import pandas as pd  # Import pandas library

# Path to the dataset directory
dataset_dir = 'DataSet/Cercles/Cercle2/'
dataset_cercle = 'DataSet/Cercles'
dataset_carre = 'DataSet/Carres'
dataset_triangle = 'DataSet/Triangles'
dataset_diamant = 'DataSet/Diamants'



# Iterate through each image file in the dataset directory
nombre_photo = 0
nombre_carre = 0




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

liste_sigma = [60,70,80,90]
liste_sigma2 = [75,60,145,130,100]
liste_lTresh = [0,10,20,30,40,50,60,70]
liste_uTresh = [255,240,230,220,175,150,125,100]
results = []
length = len(liste_sigma) * len(liste_sigma2) * len(liste_lTresh) * len(liste_uTresh)
cnt = 0
for sigma in liste_sigma:
    for sigma2 in liste_sigma2:
        for lower in liste_lTresh:
            for higher in liste_uTresh:
                for filename in os.listdir(dataset_dir):
                    nombre_photo=nombre_photo+1
                    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                        # Load the image
                        image = cv2.imread(os.path.join(dataset_dir, filename))
                        #dispayImage(image)


                        # Convert to grayscale
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        #dispayImage(gray)

                        # Apply bilateral filter
                        filtered_image = cv2.bilateralFilter(gray, 15, sigma, sigma2)
                        #dispayImage(filtered_image)

                        # Apply Canny edge detection
                        edges = cv2.Canny(filtered_image, lower, higher)
                        #dispayImage(edges)

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
                        #cv2.drawContours(roi_image, square_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

                        # Display or save the processed image
                        #cv2.imshow('Processed Image', roi_image)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()
                print("Nombre de photos",nombre_photo)
                print("Nombre de carres",nombre_carre)
                print("      Low Tresh",lower)
                print("      High Tresh",higher)
                print("      Sigma Color",sigma)
                print("      Sigma Space",sigma2)
                print("      Pourcentage de carres",100*(nombre_carre/nombre_photo))
                cnt = cnt + 1
                print("Progression",100*(cnt/length), "%")

                results.append([nombre_photo, nombre_carre, lower, higher, sigma, sigma2, 100*(nombre_carre/nombre_photo)])

                nombre_photo=0
                nombre_carre=0

df = pd.DataFrame(results, columns=['Nombre de photos', 'Nombre de carres', 'Low Tresh', 'High Tresh', 'Sigma Color', 'Sigma Space', 'Pourcentage de carres'])
df.to_csv('results.csv', index=False)


# Close all OpenCV windows


cv2.destroyAllWindows()

