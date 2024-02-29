from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
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


def encode_labels(labels):
    y = label_encoder.fit_transform(labels)
    outputs = np.array([{string_label:binary_label} for string_label, binary_label in zip(labels, y)])
    classes = {tuple(binary_label) : label_encoder.inverse_transform(np.array([binary_label]))[0] for binary_label in np.unique(y, axis=0)}
    return y, outputs, classes


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

    # features/labels vector
    X = []
    Y = []

    # Taking features and labels from dataset
    for image, class_num, label in data_set:
        feature = normalize_to_feature(image)
        # feature = add_num_of_shapes_as_feature(feature, label)
        
        X.append(feature)
        Y.append(label)
    
    X = np.array(X)   
    Y, outputs, classes = encode_labels(Y)
    
    # Split into train/test data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    sss.get_n_splits(X, Y)

    for train_index, test_index in sss.split(X, Y):
        print("TRAIN:", train_index, "TEST:", test_index)
        features_train, features_test = X[train_index], X[test_index]
        outputs_train, outputs_test = outputs[train_index], outputs[test_index]
    
    
    # Creating the files containing all the information about your model and saving them to the disk
    final_x = {'TRAIN': features_train, 'TEST': features_test}
    final_y = {'TRAIN': outputs_train, 'TEST': outputs_test, 'CLASSES': classes}
    
    print(f"final_x =\n{final_x}")
    print(f"final_y =\n{final_y}")
    
    pickle_out = open("X.pkl", "wb")
    pickle.dump(final_x, pickle_out)
    pickle_out.close()

    pickle_out = open("Y.pkl", "wb")
    pickle.dump(final_y, pickle_out)
    pickle_out.close()
    
create_data_set(data_dir)