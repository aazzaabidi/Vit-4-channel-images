import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_data(data_path, img_size=(224, 224)):
    """
    Charge les images et étiquettes depuis un dossier.
    """
    df = pd.read_csv(f"{data_path}/labels.csv")
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        img = load_img(f"{data_path}/{row['nom_image']}", target_size=img_size)
        images.append(img_array)
        labels.append(row['classe'])
    
    return np.array(images), np.array(labels)
