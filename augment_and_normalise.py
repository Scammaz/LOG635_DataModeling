from pathlib import Path
import Augmentor
import cv2
import os

SOURCE_DIR = Path.cwd() / "Processed_Dataset_A"
DESTINATION_DIR = Path.cwd() / "Dataset_B"

def augment_dataset(out_dir: Path):
    # For each shape folder
    for first_level in SOURCE_DIR.glob('*'):
        if first_level.is_dir():          
            # For each chape subfolder
            for second_level in first_level.glob('*'):
                if second_level.is_dir():
                    output_dir = out_dir / first_level.name / second_level.name
                    
                    # Path to the image dataset
                    p = Augmentor.Pipeline(source_directory=second_level, output_directory=output_dir)

                    # Operations to be performed on the images:
                    # The parameter probability is used to decide if an operation is 
                    # applied to an image as it is passed through the augmentation pipeline
                    p.rotate90(probability=0.5)
                    # p.rotate270(probability=0.5)
                    # p.flip_left_right(probability=0.75)
                    # p.flip_top_bottom(probability=0.75)
                    # p.skew_tilt(probability=0.75, magnitude=0.35)

                    # Run the pipeline specifyin the number of images to generate
                    p.sample(200)
    
    for first_level in DESTINATION_DIR.glob('*'):
        if first_level.is_dir():          
            # For each chape subfolder
            for second_level in first_level.glob('*'):
                if second_level.is_dir():
                    i = 0
                    for file in second_level.glob('*'):
                            if file.is_file():
                                # normalise images (resize 40x40, grayscale filter, blurr filter, rename to convention (#_Shape_#shapes))
                                image = cv2.imread(str(file))
                                image = cv2.resize(image, (40, 40))
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                image = cv2.GaussianBlur(image, (5, 5), 0)
                                
                                new_filename = str(i) + "_" + second_level.name + ".png"
                                new_filepath = second_level / new_filename
                                cv2.imwrite(str(file), image)
                                os.rename(str(file), str(new_filepath))
                                i += 1
                                
                                
augment_dataset(DESTINATION_DIR)