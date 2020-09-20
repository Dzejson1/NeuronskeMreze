
# https://cmdlinetips.com/2020/05/save-pandas-dataframe-as-excel-file/
from keras.models import load_model
import os
import xlwt
import pandas as pd
import matplotlib.pyplot as plt
from keras import models
import matplotlib.pyplot as plt
import keras
import constants as c
from keras.preprocessing import image
import numpy as np
from keras import backend as K
from keras.applications.inception_v3 import preprocess_input
import cv2
from matplotlib.ticker import MaxNLocator
scale=0.2
def dole(preds):    
    layer_activation=activations[7]
    n_features = layer_activation.shape[-1]
    for row in range(n_features):
        channel_image = layer_activation[0, :, :, row ]
        channel_image -= channel_image.mean()
        channel_image /= channel_image.std() 
        channel_image *= 64
        channel_image += 128       
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')      
        all_zeroes=True
        for r in range(channel_image.shape[1]):
             for co in range(channel_image.shape[0]):
                 if(channel_image[r,co]!=0):
                     all_zeroes=False
        if(preds==0):
            if(all_zeroes==True):
                macke[row,1]+=1
            if(all_zeroes==False):
                macke[row,0]+=1
        else:
            if(all_zeroes==True):
                macke[row,3]+=1
            if(all_zeroes==False):
                macke[row,2]+=1
                
                
def dole1(preds):    
    layer_activation=activations[7]
    n_features = layer_activation.shape[-1]
    for row in range(n_features):
        channel_image = layer_activation[0, :, :, row ]
        channel_image -= channel_image.mean()
        channel_image /= channel_image.std() 
        channel_image *= 64
        channel_image += 128       
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')      
        all_zeroes=True
        for r in range(channel_image.shape[1]):
             for co in range(channel_image.shape[0]):
                 if(channel_image[r,co]!=0):
                     all_zeroes=False
        
        
        if(preds==1):
            if(all_zeroes==True):
                psi[row,1]+=1
            if(all_zeroes==False):
                psi[row,0]+=1
        else:
            if(all_zeroes==True):
                psi[row,3]+=1
            if(all_zeroes==False):
                psi[row,2]+=1
macke=np.zeros([128,4])
psi=np.zeros([128,4])

model = load_model('cats_and_dogs_small_2.h5')
model.summary() 
pogodak=0;
promasaj=0;
fnames =['cat.{}.jpg'.format(i) for i in range(1500, 1950)]
for fname in fnames:
    img_path=c.base_dir+'/test/cats/'+fname
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    preds  = model.predict(img_tensor)
    img_tensor /= 255.
    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
   
    first_layer_activation = activations[7]
    dole(preds[0])
fnames =['dog.{}.jpg'.format(i) for i in range(1500, 1950)]
for fname in fnames:
    img_path=c.base_dir+'/test/dogs/'+fname
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    preds  = model.predict(img_tensor)
    img_tensor /= 255.
    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    # print("Pred   ",preds[0]) Moze posluziti za proveru
    first_layer_activation = activations[7]
    dole1(preds[0])
    

nizA=np.zeros([128])
nizN=np.zeros([128])
nizAA=np.zeros([128])
nizNN=np.zeros([128])

nizpA=np.zeros([128])
nizpN=np.zeros([128])
nizpAA=np.zeros([128])
nizpNN=np.zeros([128])

for i in range(128):
    
    nizA[i]=macke[i,0];
    nizN[i]=macke[i,1];
    nizAA[i]=macke[i,2];
    nizNN[i]=macke[i,3];
    nizpA[i]=psi[i,0];
    nizpN[i]=psi[i,1];
    nizpAA[i]=psi[i,2];
    nizpNN[i]=psi[i,3];
    
pom=np.array([np.sum(nizA)]);
pom1=np.array([np.sum(nizN)]);
pom2=np.array([np.sum(nizAA)]);
pom3=np.array([np.sum(nizNN)]);

pomp=np.array([np.sum(nizpA)]);
pomp1=np.array([np.sum(nizpN)]);
pomp2=np.array([np.sum(nizpAA)]);
pomp3=np.array([np.sum(nizpNN)]);
# Ime fajla se mora menjati svaki put kad se pokrene
writer = pd.ExcelWriter('cat_dog_2.xlsx', engine='xlsxwriter')

df2 = pd.DataFrame({"Aktivni":nizA,
                   "Neaktivni":nizN,
                   "Aktivni1":nizAA,
                   "Neaktivni1":nizNN,
                   "AktivniP":nizpA,
                   "NeaktivniP":nizpN,
                   "Aktivni1p":nizpAA,
                   "Neaktivni1p":nizpNN})
df1 = pd.DataFrame({"":pom," ":pom1,"  ":pom2,"   ":pom3,"     ":pomp,"    ":pomp1,"      ":pomp2,"       ":pomp3},index={"Suma"}) 


pom=np.array([]);
df3 = pd.DataFrame({"            Pogodio macku":pom})
df4 = pd.DataFrame({"                 Nije pogodio macku":pom})
dfp3 = pd.DataFrame({"            Pogodio psa":pom})
dfp4 = pd.DataFrame({"                 Nije pogodio psa":pom})


df3.to_excel(writer, sheet_name='Sheet1',startcol=0)
df4.to_excel(writer, sheet_name='Sheet1',startcol=2)
dfp3.to_excel(writer, sheet_name='Sheet1',startcol=4)
dfp4.to_excel(writer, sheet_name='Sheet1',startcol=6)

df2.to_excel(writer, sheet_name='Sheet1',startrow=1,startcol=0)
df1.to_excel(writer, sheet_name='Sheet1',startrow=130)

writer.save()




