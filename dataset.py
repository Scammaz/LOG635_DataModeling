from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle

RANDOM_SEED = 1337

data_dir = Path.cwd() / "DataSet_B"
scaler = MinMaxScaler(feature_range=(0,1), copy=False, clip=False)
label_encoder = LabelBinarizer()

B_LABELS = ["Cercle2", "Cercle3", "Diamant2", "Diamant5", "Hexagone3", "Hexagone4", "Triangle3", "Triangle5"]

def normalize_to_feature(image):
    # normalize image values
    norm_feature = np.array(image)
    norm_feature = scaler.fit_transform(norm_feature)
    norm_feature = norm_feature.flatten()
    
    return norm_feature


def encode_label(label, class_num):
    # one-hot encode the label
    encoded_label = np.zeros(len(B_LABELS))
    encoded_label[class_num] = 1
    
    return encoded_label


def add_num_of_shapes_as_feature(feature, label):
    # add num of shapes as additional feature
    num_of_shapes = int(label[len(label)-1])
    
    return np.append(feature, num_of_shapes)
    

def create_data_set(dir: Path):
    data_set = []
    
    for first_level in data_dir.glob('*'):
        if first_level.is_dir():
            for second_level in first_level.glob('*'):
                if second_level.is_dir():
                    label = second_level.name
                    class_num = B_LABELS.index(label) if label in B_LABELS else -1
                    if(class_num >= 0):                    
                        for file in second_level.glob('*'):
                            if file.is_file():
                                # read image and save to temp list
                                image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                                data_set.append([image, class_num, label])
    
    # Shuffles the images
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(data_set)

    # features vector
    X = []

    # lables vector
    Y = []

    # Taking features and labels from dataset
    for image, class_num, label in data_set:
        feature = normalize_to_feature(image)
        feature = add_num_of_shapes_as_feature(feature, label)
        enc_label = encode_label(label, class_num)
        
        X.append(feature)
        Y.append(enc_label)   

    print(f"{X[1337]}   {Y[1337]}")
    
    # Creating the files containing all the information about your model and saving them to the disk
    pickle_out = open("X.pkl", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("Y.pkl", "wb")
    pickle.dump(Y, pickle_out)
    pickle_out.close()
    
    test_dataset(1337)
    
    
def test_dataset(index):
    X = [] 
    Y = []
    
    with open("X.pkl", "rb") as db:
        X = pickle.load(db)
    with open("Y.pkl", "rb") as db:
        Y = pickle.load(db)    
        
    print(f"{X[index]}   {Y[index]}")
    
create_data_set(data_dir)