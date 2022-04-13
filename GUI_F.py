#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy
#on charge le modele pour detecter l'espece de l'image
from keras.models import load_model
model = load_model('model.h5')


# In[2]:


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

#Appel de la fonction
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
 # nous informe sur les dimensions de l'image et le niveau de l'image (RGB)


# In[3]:


top=tk.Tk() #on cree la fenetre de notre interface graphique
top.geometry('800x600') #on definit les dimensions  de notre interface
top.title("Detecter les espèces des arbres" ) #le titre de l'interface
top.configure(background='#BEC5AD') #la couleur de l'arriere plan
label=Label(top,background='#BEC5AD', font=('arial',15,'bold')) # on definit la police et la couleur
sign_image = Label(top)


# In[4]:


#fonction pour classifier l'image choisi par l'utilisateur
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((100,100))
    image = numpy.array(image)
    image=image/255.0
    image = numpy.expand_dims(image, axis=0)
    print(image.shape)
    predition=model.predict(image)
    predition=np.squeeze(predition)
    predIndex=np.argmax(predition)
    sign = labels[predIndex]
    print(sign)
    label.configure(foreground='#011638', text=sign) 


# In[5]:


#boutton qui permet d'appeler la fonction classify()
def show_classify_button(file_path):
    classify_b=Button(top,text="Detecter l'arbre",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='black',font=('arial',15,'bold'))
    classify_b.place(relx=0.79,rely=0.46)


# In[6]:


#fonction pour charger l'image
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


# In[7]:


#boutton qui permet d'appeler la fonction  upload_image()
upload=Button(top,text="Choisir une image d'arbre",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='black',font=('arial',15,'bold'))


# In[8]:


#affiche l'espece correspondante
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="L’espèce de l'arbre est :",pady=20, font=('arial',20,'bold'))
heading.configure(background='#BEC5AD',foreground='#364156')
heading.pack()
top.mainloop()


# In[ ]:




