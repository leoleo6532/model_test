# ! pip install -q keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(50, 50,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64,(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten()) 
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))   #softmax sigmoid 2020/08/23

model.compile(loss='binary_crossentropy',          #損失函數
              optimizer='rmsprop',       #優化器
              metrics=['accuracy'])      #評估標準


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(
    '/content/drive/My Drive/肺癌dataset/train',  
    target_size=(50, 50),  
    batch_size=32,
    class_mode='binary')  

validation_generator = test_datagen.flow_from_directory(
    '/content/drive/My Drive/肺癌dataset/val',
    target_size=(50, 50),
    batch_size=32,
    class_mode='binary')

print(train_generator.class_indices)


history =model.fit_generator(
            train_generator,
            steps_per_epoch=146,    # number_of_train_samples / batch_size  4672/32=162.09375
            epochs=100,          #10
            validation_data=validation_generator,
            validation_steps=40)    # number_of_val_samples / batch_size 1297/32 =40.53125
print(history.history.keys())

#作圖區
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left') 


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left') 
json_string = model.to_json()


#confusiom matrix區域
'''
Y_pred = model.predict_generator(validation_generator,40)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.class_indices, y_pred))
print('Classification Report')
target_names = ['sick', 'unsick']
print(classification_report(validation_generator.class_indices, y_pred, target_names=target_names))
'''


#模型儲存區
model.save('/content/drive/My Drive/肺癌dataset/sick.h5')
print("ok")

'''! pip install  keras
! pip install tensorflow-gpu
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64,(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten()) 
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))   #softmax sigmoid 2020/08/23

model.compile(loss='binary_crossentropy',          #損失函數
              optimizer='rmsprop',       #優化器
              metrics=['accuracy'])      #評估標準


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(
    '/content/drive/My Drive/肺癌dataset/train',  
    target_size=(150, 150),  
    batch_size=32,
    class_mode='binary')  

validation_generator = test_datagen.flow_from_directory(
    '/content/drive/My Drive/肺癌dataset/val',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

print(train_generator.class_indices)


history =model.fit_generator(
            train_generator,
            steps_per_epoch=146,# number_of_train_samples / batch_size  5187/32=162.09375
            epochs=50,#10
            validation_data=validation_generator,
            validation_steps=40)   # number_of_val_samples / batch_size 1297/32 =40.53125
print(history.history.keys())

#作圖區
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left') 


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left') 
   

json_string = model.to_json()

#模型儲存區
model.save('/content/drive/My Drive/肺癌dataset/sick.h5')
print("ok")
'''