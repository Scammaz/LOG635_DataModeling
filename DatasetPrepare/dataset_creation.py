import sys
from preprocess_and_extract_subset import create_preprocessed_A
from augment_and_normalize import create_dataset_B
from dataset import create_Pickles

def create_dataset(argv):
    step = argv[1]
    
    if(step == 1):
        create_preprocessed_A()
    elif(step == 2):
        create_dataset_B()
    elif(step == 3):
        create_Pickles()
    else:
        print("No such step exists")
        
create_dataset(sys.argv)