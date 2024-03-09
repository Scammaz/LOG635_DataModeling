## Dataset Creation:
1. Mettre tour les images dans ```./DatasetPrepare/DataSet_A```
2. Dans le script ```./DatasetPrepare/preprocess_and_extract_subset.py``` modifier la variable globale **LABELS** pour indiquer quels classes utiliser pour le sous-ensemble ****
3. Executer ```./DatasetPrepare/dataset_creation.py 1``` pour faire le preprossesing  des classes et l'extraction des marquers ver le dossier ```./DatasetPrepare/Processed_DataSet_A```
4. Effectuer une validation manuelle pour eliminer toutes mauvaises images et bruit
5. Executer ```./DatasetPrepare/dataset_creation.py 2``` pour faire executer l'augmentation des images vers le dossier ```./DatasetPrepare/DataSet_B```.
6. Executer ```./DatasetPrepare/dataset_creation.py 3``` pour normaliser les images, creer des labels, separer l'ensemble en ensebles d'entrainement et test et les enregistrer dans les fichiers PICKLE.