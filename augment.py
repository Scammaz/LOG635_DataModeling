from pathlib import Path
import Augmentor

SOURCE_DIR = Path.cwd() / "Processed_Dataset_A"

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
                    p.rotate270(probability=0.5)
                    p.flip_left_right(probability=0.75)
                    p.flip_top_bottom(probability=0.75)
                    p.skew_tilt(probability=0.75, magnitude=0.35)
                    p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)

                    # Run the pipeline specifyin the number of images to generate
                    p.sample(100)