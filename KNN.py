import pandas as pd
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
import os
from sklearn.neighbors import KNeighborsClassifier  # Import KNeighborsClassifier
from PIL import Image
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Citesc continutul csv-urilor
train_data = pd.read_csv('/kaggle/input/unibuc-dhc-2023/train.csv')
val_data = pd.read_csv('/kaggle/input/unibuc-dhc-2023/val.csv')
test_data = pd.read_csv('/kaggle/input/unibuc-dhc-2023/test.csv')

# stochez numele imaginilor si etichetele pentru datele de antrenare
train_images = train_data['Image'].tolist()
train_labels = train_data['Class'].tolist()
# stochez numele imaginilor si etichetele pentru datele de validare
val_images = val_data['Image'].tolist()
val_labels = val_data['Class'].tolist()
# stochez numele imaginilor pentru datele de testare
test_images = test_data['Image'].tolist()


# functie in care, folosind un set de path-uri ale unor imagini, le redimensioneaza la marimile date ca parametrii
# le redimensionez pentru a ma asigura ca toate au aceeasi forma si pentru a le putea prelucra
def load_resize(image_list, x ,y):
    size=(x,y)
    resized_images = [] #creez o lista in care urmeaza sa stochez imaginile redimensionate
    for path in image_list:
        #pentru fiecare path, deschid imaginea cu functia Image.Open din PIL, o redimensionez si o adaug in lista
        image = Image.open(path)
        resized_image = image.resize(size)
        resized_images.append(resized_image)
    return resized_images #returnez lista de imagini modificate


# redimensionez imaginile de antrenament
# pentru fiecare nume de fisier din train_images, il concatenez cu calea pana la folderul de imagini de antrenament
train_image_paths = []
for name in train_images:
    path = os.path.join('/kaggle/input/unibuc-dhc-2023/train_images/', name)
    train_image_paths.append(path)
# apelez functia de incarcare si redimensionare a imaginilor
train_images_resized = load_resize(train_image_paths, 28, 28)

# redimensionez imaginile de validare
val_image_paths = []
for name in val_images:
    path = os.path.join('/kaggle/input/unibuc-dhc-2023/val_images/', name)
    val_image_paths.append(path)
val_images_resized = load_resize(val_image_paths, 28, 28)

# redimensionez imaginile de testare
test_image_paths = []
for name in test_images:
    path = os.path.join('/kaggle/input/unibuc-dhc-2023/test_images/', name)
    test_image_paths.append(path)
test_images_resized = load_resize(test_image_paths, 28, 28)

# fac imaginile un vector unidimensional pentru a le folosi in KNN
train_images_flattened = []
for image in train_images_resized:
    image_array = np.array(image)
    flattened_image = image_array.flatten()
    train_images_flattened.append(flattened_image)
train_images_flattened = np.array(train_images_flattened)

val_images_flattened = []
for image in val_images_resized:
    image_array = np.array(image)
    flattened_image = image_array.flatten()
    val_images_flattened.append(flattened_image)
val_images_flattened = np.array(val_images_flattened)

test_images_flattened = []
for image in test_images_resized:
    image_array = np.array(image)
    flattened_image = image_array.flatten()
    test_images_flattened.append(flattened_image)
test_images_flattened = np.array(test_images_flattened)

# normalizare date
# prin astype('float32') ma asigur ca pixelii sunt reprezentati cu numere in floating point
# deoarece valorile pixelilor se afla intre 0(negru) si 255(alb)
# impart la 255.0 ca sa ma asigur ca toate valorile pixelilor se vor afla intre 0 si 1
train_images_normalized = train_images_flattened.astype('float32') / 255.0
val_images_normalized = val_images_flattened.astype('float32') / 255.0
test_images_normalized = test_images_flattened.astype('float32') / 255.0

# initializez modelul KNN
model = KNeighborsClassifier(n_neighbors=3)

# antrenez modelul
model.fit(train_images_normalized, train_labels)

# fac predictie pentru etichetele imaginilor de validare
val_predictions_labels = model.predict(val_images_normalized)

# calculez acuratetea, precizia si recallul
accuracy = accuracy_score(val_labels, val_predictions_labels)
precision = precision_score(val_labels, val_predictions_labels, average=None)
recall = recall_score(val_labels, val_predictions_labels, average=None)

confusion_matrixx = confusion_matrix(val_labels, val_predictions_labels)
#afisez matricea de confuzie
plt.figure(figsize=(20, 15))
sns.heatmap(confusion_matrixx, annot=True, fmt="d", cmap="Greens", cbar=False, annot_kws={"size": 12})
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_KNN.png')
plt.show()


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# afisez graficul preciziei
plt.figure(figsize=(20, 4))
plt.title('Precision')
plt.xlabel('Class')
plt.ylabel('Precision')
plt.plot(precision, marker='D')
plt.xticks(np.arange(len(precision)))
plt.savefig('precision_KNN.png')
plt.show()

# afisez grafic recall
plt.figure(figsize=(20, 4))
plt.title('Recall')
plt.ylabel('Recall')
plt.xlabel('Class')
plt.plot(recall, marker='D')
plt.xticks(np.arange(len(recall)))
plt.savefig('recall_KNN.png')
plt.show()

# fac predictii pentru setul de imagini de testare
predicted_labels = model.predict(test_images_normalized)
# salvez predictiile intr-un fisier csv
# creez un dicționar cu coloanele 'Image' și 'Class'
data = {'Image': test_data['Image'], 'Class': predicted_labels}
# transform dictionarul in DataFrame (structura de date bidimi
submission = pd.DataFrame(data)
# salvez predictiile intr-un fisier csv
submission.to_csv('submission.csv', index=False)
print("Gata")