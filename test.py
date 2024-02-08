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
liste_sigma = [145, 130, 100, 75, 60,45,30]
liste_sigma2 = [145, 130, 100, 75, 60,45,30]

for sigma in liste_sigma:
    for filename in os.listdir(dataset_dir):
        nombre_photo=nombre_photo+1
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Load the image
            image = cv2.imread(os.path.join(dataset_dir, filename))

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply bilateral filter
            filtered_image = cv2.bilateralFilter(gray, 15, sigma, sigma)

            # Apply Canny edge detection
            edges = cv2.Canny(filtered_image, 0, 255)

            # Find contours
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours
            square_contours = []
            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                if len(approx) == 4:
                    square_contours.append(contour)
                    nombre_carre=nombre_carre+1

            # Create a new image with the detected contours
            #roi_image = np.zeros_like(image)
            #cv2.drawContours(roi_image, square_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

            # Display or save the processed image
            #cv2.imshow('Processed Image', roi_image)
            #cv2.waitKey(0)
    print("Nombre de photos",nombre_photo)
    print("Nombre de carres",nombre_carre)
    print("Sigma",sigma)
    print("Sigma2",sigma2)

    nombre_photo=0
    nombre_carre=0
# Close all OpenCV windows


cv2.destroyAllWindows()

# Function to detect shape
def detect_shape(contour):
    print(allo)
    # implementation of detect_shape function...