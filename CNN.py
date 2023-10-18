
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
from PIL import Image
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# citesc continutul csv-urilor
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


#functie in care, folosind un set de pathuri ale unor imagini, le redimensioneaza la marimile date ca parametrii
#le redimensionez pentru a ma asigura ca toate au aceeasi forma si pentru a le putea prelucra
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
train_image_paths=[]
for name in train_images:
    path=os.path.join('/kaggle/input/unibuc-dhc-2023/train_images/', name)
    train_image_paths.append(path)
#apelez funtia de incarcare si redimensionare a imaginilor
train_images_resized = load_resize(train_image_paths, 64, 64)

# redimensionez imaginile de validare
val_image_paths=[]
for name in val_images:
    path=os.path.join('/kaggle/input/unibuc-dhc-2023/val_images/', name)
    val_image_paths.append(path)
val_images_resized = load_resize(val_image_paths, 64, 64)

# redimensionez imaginile de testare
test_image_paths=[]
for name in test_images:
    path=os.path.join('/kaggle/input/unibuc-dhc-2023/test_images/', name)
    test_image_paths.append(path)
test_images_resized = load_resize(test_image_paths, 64, 64)

# transorm lista de imagini redimensionate in numpy arrays pentru a fi mai usor de utilizat in diverse operatii
train_images_array = []
for image in train_images_resized:
    image_array = np.array(image)
    train_images_array.append(image_array)
train_images_array = np.array(train_images_array)

val_images_array = []
for image in val_images_resized:
    image_array = np.array(image)
    val_images_array.append(image_array)
val_images_array = np.array(val_images_array)

test_images_array = []
for image in test_images_resized:
    image_array = np.array(image)
    test_images_array.append(image_array)
test_images_array = np.array(test_images_array)

# normalizare date
# prin astype('float32') ma asigur ca pixelii sunt reprezentati cu numere in floating point
# deoarece valorile pixelilor se afla intre 0(negru) si 255(alb)
#impart la 255.0 ca sa ma asigur ca toate valorile pixelilor se vor afla intre 0 si 1
train_images_normalized = train_images_array.astype('float32') / 255.0
val_images_normalized = val_images_array.astype('float32') / 255.0
test_images_normalized = test_images_array.astype('float32') / 255.0

# transform etichetele in to one-hot vectors pentru a le oferi un format usor de interpretat
# si pentru a putea fi compatibile cu functii de loss
nr_classes = len(set(train_labels))
train_labels_one_hot = to_categorical(train_labels, nr_classes)
val_labels_one_hot = to_categorical(val_labels, nr_classes)

# am ales ca proportia de neuroni care vor fi dezactivati aleatoriu in antrenare pentru a evita overfittingul
dropout=0.4
# creez modelul CNN
model = Sequential()
# adaug un strat de convultie 2d cu 32 de filtre cu functia de activare "relu"
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))
# adaug un strat de normalizare
model.add(BatchNormalization())
#nu mai specific input_shape caci se preia de la stratul anterior
model.add(Conv2D(32,kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
#strides=2 pentru a reduce informatii redundante,nr de pixeli cu care se desplaseaza fereastra
#padding-ul asigura ca forma hartiide iesire e aceeasi cu cea a imaginii de intrare
model.add(Conv2D(32, kernel_size=5, activation='relu',  strides=2, padding='same'))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Conv2D(64, kernel_size=5, activation='relu', strides=2, padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Flatten()) # transform ieșirea stratului anterior într-un vector unidimensional
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Dense(nr_classes, activation='softmax'))

# initializez rata de invatare
learning_rate = 0.001
# o data la 8 epoci reduc rata de invatare cu 10%
def learing_rate_decrease(epoch):
    return learning_rate * 0.1 ** (epoch // 13)
# definesc un obiect care va apela functia de modificare a ratei de invatare in fit()
learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(learing_rate_decrease)

# compilez modelul folosind metrica de evaluare "accuracy"
model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

#  antrenez modelul
model.fit(train_images_normalized, train_labels_one_hot, epochs=30, batch_size=32,callbacks=[learning_rate_scheduler], validation_data=(val_images_normalized, val_labels_one_hot))

# evaluez modelul
accuracy = model.evaluate(val_images_normalized, val_labels_one_hot)[1]
val_predictions = model.predict(val_images_normalized)
val_predictions_labels = np.argmax(val_predictions, axis=1)

# calculez precizia si recallul
precision = precision_score(val_labels, val_predictions_labels, average=None)
recall = recall_score(val_labels, val_predictions_labels, average=None)

# calculez matricea de confuzoe
confusion_matrixx = confusion_matrix(val_labels, val_predictions_labels)

# afisez accuracy, precision, si recall
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# afisez matricea de confuzie
# setez marimea
plt.figure(figsize=(20, 15))
# setez font size si culori
sns.heatmap(confusion_matrixx, annot=True, fmt="d", cmap="Greens", cbar=False, annot_kws={"size": 12})
# pun etichete
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
# cresc font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# salvez imaginea
plt.savefig('confusion_matrix_CNN_4.png')

# afisez grafic precizie
plt.figure(figsize=(20, 4))
plt.title('Precision')
plt.xlabel('Class')
plt.ylabel('Precision')
plt.plot(precision, marker='D')
plt.xticks(np.arange(len(precision)))
plt.savefig('precision_CNN_4.png')
plt.show()

# afisez grafic recall
plt.figure(figsize=(20, 4))
plt.title('Recall')
plt.xlabel('Class')
plt.ylabel('Recall')
plt.plot(recall, marker='D')
plt.xticks(np.arange(len(recall)))
plt.savefig('recall_CNN_4.png')
plt.show()



# fac predictii pentru setul de imagini de testare
predicted_labels_one_hot = model.predict(test_images_normalized)
predicted_labels = np.argmax(predicted_labels_one_hot, axis=1)
# salvez predictiile intr-un fisier csv
# creez un dicționar cu coloanele 'Image' și 'Class'
data = {'Image': test_data['Image'], 'Class': predicted_labels}
# transform dictionarul in DataFrame (structura de date bidimiensionala)
submission = pd.DataFrame(data)
# salvez predictiile intr-un fisier csv
submission.to_csv('submission.csv', index=False)
print("Gata")

