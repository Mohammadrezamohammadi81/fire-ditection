#ketabkhane hay mored niaz
import cv2 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import glob
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras#بردن از اینتجر اینکودینگ به وان هات
from keras.utils import to_categorical
from keras import models,layers
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]= "3"
epocks=20
batchsize=32
le=LabelEncoder()
data_list=[]
labels=[]
# har tasvir ra shoru be khandan mikonim 
for i,adress in enumerate(glob.glob("fire_dataset\\*\\*")): #ba glob harv tasver ra mi khanim va dar edame shuru be pish pardazesh mikonim

    img=cv2.imread(adress)#khandan tasavir
    img=cv2.resize(img,(32,32))# resize kardan
    img=img/255 #آوردن تصاویر به بازه ی صفر و یک 
    #img=img.flatten()#ساخت بردار دو بعدی 
    data_list.append(img)
    #شمارش گر برای شروع پردازش زیبایی کار 
    if i  % 100==0:
        print("[INFO]{}/{} procced".format(i,1000))
    
    #خواندن لیبل ها 
    label=adress.split("\\")[2].split(".")[0]
    labels.append(label)

# کتابخانه اسکی لرن فقط ماتریس یا دیتافریم قبول می کند 
data=np.array(data_list)
#تقسیم بندی تراین و تست
x_train , x_test ,y_train,y_test=train_test_split(data,labels,random_state=42,test_size=0.2)
#فایر کردن الگوریتم کی ان ان 
y_train=le.fit_transform(y_train)
y_test=le.transform(y_test)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

net=models.Sequential([
    layers.Conv2D(32,(3,3),activation="relu",input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.Conv2D(32,(3,3),activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPool2D(),

    layers.Conv2D(64,(3,3),activation="relu",input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPool2D(),

    layers.Flatten(),
    layers.Dense(512,activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(10,activation="softmax")
])

s=net.summary()
net.compile(optimizer="SGD",loss="categorical_crossentropy",metrics=["accuracy"])

H=net.fit(x_train,y_train,batch_size=batchsize,epochs=epocks,validation_data=(x_test,y_test))

print(H.history.keys())

plt.plot(np.arange(epocks),H.history['loss'],label="train loss")
plt.plot(np.arange(epocks),H.history['val_loss'],label="test loss")
plt.plot(np.arange(epocks),H.history['accuracy'],label="train loss")
plt.plot(np.arange(epocks),H.history['val_accuracy'],label="test loss")
plt.legend()
plt.show()
print(net.summary())
loss,acc=net.evaluate(x_test,y_test)
print('Test Loss : {}'.format(loss))
print('Test Accuracy : {}'.format(acc))

net.save("cnn.h5")