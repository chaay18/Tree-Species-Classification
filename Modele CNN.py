#!/usr/bin/env python
# coding: utf-8

# In[1]:


#etape 1: on importe les bibliotheques et les modules
#pour telecharger les biblitheques on lance la commande pip install avec le nom de la bibliotheque
import numpy as np #pip install numpy
import pandas as pd #pip install pandas
import tensorflow as tf #pip install tensorflow
from matplotlib import pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2   #pip install opencv-python
import shutil
import os
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


#etape 2: on charge les donnees
df='dataset-Maroc/dataset' #le chemin de donnees #(path)#


# In[3]:


#etape 3: traitement de donnees
#on divise les donnees en 2 parties: 90% pour training 10% pour test:

#listdir:La fonction listdir prend un nom de chemin et retourne une liste du contenu du répertoire:
for c in os.listdir(df):
    if not c.startswith('.'):
        img_num=len(os.listdir(df + '/' + c))#retourne le nombre d'images correspondant a chaque espece.
        for(n,file) in enumerate(os.listdir(df + '/' +  c)):#n:index(compteur),file:l'image
            img=df + '/' + c + '/' +file #le chemin de l'image
            if n<0.1*img_num:
                #une methode qui permet de creer une copie d'un fichier source vers un fichier destination
                shutil.copy(img,'TEST/' + c + '/' + file)
            elif n<0.9* img_num:
                shutil.copy(img,'TRAIN/' + c + '/' + file)



# In[4]:


def load_data(dir_path,img_size=(100,100)):  #dir_path:le repertoire TEST ou TRAIN
 x = [] #array des images
 y = [] #array des labels
 i=0
 labels=dict()  #est un dictionnaire qui associe une clé à une valeur.
 for path in os.listdir(dir_path):
     if not path.startswith('.'):
            labels[i]=path
            for file in os.listdir(dir_path+'/'+path):  #path: le nom du repertoire representant l'espece.
                 if not file.startswith('.'):
                        #imread: elle permet de lire une image d'un fichier
                        img=cv2.imread(dir_path+ '/'+ path+ '/'+ file)
                        # resize:redimensionner l'image
                        img=cv2.resize(img,img_size)
                        #on va ajouter l'image dans le tableau x a l'aide de la fonction append()
                        x.append(img)
                        #on associe chaque image au numero de son espece (i)
                        y.append(i)
            i+=1
 x=np.array(x,dtype=object)
 y=np.array(y)
 return x,y ,labels

#Appel de la fonction load_data()
TRAIN='TRAIN'
TEST='TEST'
imge_size=(100,100)
x_train,y_train,labels=load_data(TRAIN,imge_size)
x_test,y_test,labels=load_data(TEST,imge_size)
x_train=x_train/255
x_train=np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.int32)
x_test=np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.int32)


# In[5]:


#Data Augmentation

image_size=224
batch_size = 16
batch=64
Train='TRAIN'       
train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        Train,  #ce le chemin ou on a les images
        target_size=(image_size, image_size),#tous les images vont avoir une de dimension de 224*224
        batch_size=batch,
        color_mode='grayscale',
        class_mode='categorical'#parce qu'on a multi-classes
)

Test='TEST'
test_datagen = ImageDataGenerator(
        rescale=1./255,
        fill_mode='nearest')

test_generator = test_datagen.flow_from_directory(
        Test,  # this is the target directory
        target_size=(image_size, image_size),  # all images will be resized to 150x150
        batch_size=batch,
        color_mode='grayscale',
        class_mode='categorical')


# In[6]:


#etape 4: on definit l'architecture de notre reseau cnn

#on importe les modules 
from keras.models import Sequential #type de modele
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#on initialise notre modele

model=Sequential()    ## Création d'un réseau de neurones vide 

#etape a: on ajoute de couches convolutifs pour l'extraction de features (caracteristiques) des images 
#dans cette etape on utilise 60 filtres de taille  3*3:
#input_shape=100,100,3 dimensions de l'image + rgb(niveau couleur)
model.add(Conv2D(60,(3,3),input_shape=(100,100,3),activation='relu'))
#etape b: pooling

model.add(MaxPooling2D (pool_size=(2,2)))#  max pooling a pour taille 2*2
# on ajoute d'autres couches

model.add(Conv2D(60,(3,3),activation='relu'))
model.add(Conv2D(60,(3,3),activation='relu'))
model.add(MaxPooling2D (pool_size=(2,2)))
#etape c: flattening

model.add(Flatten())#Conversion des matrices 3D en vecteur 1D
#etape d: Fully-connected
model.add(Dense(units=200,activation='relu'))#Ajout de la première couche fully-connected, suivie d'une couche ReLU


#Ajout de la dernière couche fully-connected qui permet de classifier
model.add(Dense(9, activation='softmax'))
#etape 5 : on compile notre CNN
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

nb_epochs = 15 # combien de fois il va faire le tour 
#etape 6: Appliquer le modele sur notre data
Mdl=model.fit(x_train, y_train, batch_size = 32, epochs = nb_epochs, verbose = 1,validation_data = (x_test, y_test))
#on enregistre notre model dans un fichier model.h5
model.save("model.h5")
print(model.summary())


# In[7]:


#7 etape : evaluation du model sur le test data
score=model.evaluate(x_test,y_test,verbose=0)
print('Test Score :',score[0])
print('Test Accuracy :',score[1])


# In[9]:


#apres l'entrainement de modele on genere les graphes d'entrainement
#Mdl:notre modele
plt.figure(1)
plt.plot(Mdl.history['accuracy'])
plt.plot(Mdl.history['val_accuracy'])
plt.legend(['training','test'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.show()


# In[27]:


# etape 8: Prediction
#on traite l'image a predire
predictimg=cv2.imread("predictit.jpeg",1)#on lit l'image
predictimg=cv2.resize(predictimg,(100,100))# redimensionnement de l'image elle va avoir une taille de 100*100
predictimg=np.array(predictimg)
plt.imshow(predictimg)
predictimg=predictimg/255.0

predictimg = np.expand_dims(predictimg, axis=0)
predictimg.shape
predition=model.predict(predictimg)  #predire l'espece de l'image

predition=np.squeeze(predition)
predIndex=np.argmax(predition)
print("L'espece de l'image est :",labels[predIndex])


# In[30]:


#on traite l'image a predire
predictimg=cv2.imread("predictit1.jpeg",1)#on lit l'image
predictimg=cv2.resize(predictimg,(100,100))# redimensionnement de l'image elle va avoir une taille de 100*100
predictimg=np.array(predictimg)
plt.imshow(predictimg)
predictimg=predictimg/255.0

predictimg = np.expand_dims(predictimg, axis=0)
predictimg.shape
predition=model.predict(predictimg)  #predire l'espece de l'image

predition=np.squeeze(predition)
predIndex=np.argmax(predition)
print("L'espece de l'image est :",labels[predIndex])


# In[25]:


#on traite l'image a predire
predictimg=cv2.imread("predictit2.jpeg",1)#on lit l'image
predictimg=cv2.resize(predictimg,(100,100))# redimensionnement de l'image elle va avoir une taille de 100*100
predictimg=np.array(predictimg)
plt.imshow(predictimg)
predictimg=predictimg/255.0

predictimg = np.expand_dims(predictimg, axis=0)
predictimg.shape
predition=model.predict(predictimg)  #predire l'espece de l'image

predition=np.squeeze(predition)
predIndex=np.argmax(predition)
print("L'espece de l'image est :",labels[predIndex])


# In[26]:


#on traite l'image a predire
predictimg=cv2.imread("predictit3.jpeg",1)#on lit l'image
predictimg=cv2.resize(predictimg,(100,100))# redimensionnement de l'image elle va avoir une taille de 100*100
predictimg=np.array(predictimg)
plt.imshow(predictimg)
predictimg=predictimg/255.0

predictimg = np.expand_dims(predictimg, axis=0)
predictimg.shape
predition=model.predict(predictimg)  #predire l'espece de l'image

predition=np.squeeze(predition)
predIndex=np.argmax(predition)
print("L'espece de l'image est :",labels[predIndex])


# In[ ]:




