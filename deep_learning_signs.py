import cv2 
import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import os 
import random 
import gc
import shutil 
import glob 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from keras import layers 
from keras import models 
from keras import optimizers 
from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing.image import img_to_array, load_img 
import tensorflow as tf
import time

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
t0 = time.clock()

# Train path
train_dir = 'C:\\Users\\ritac\\Desktop\\TP5_ML\\train\\train'
# Test path
test_dir = 'C:\\Users\\ritac\\Desktop\\TP5_ML\\test\\test'

random.seed(3)  
#info signs for testing
info_moved = random.sample(glob.glob("C:\\Users\\ritac\\Desktop\\TP5_ML\\train\\train\\info *.jpg"), 160)
#danger signs for testing
perigo_moved = random.sample(glob.glob("C:\\Users\\ritac\\Desktop\\TP5_ML\\train\\train\\perigo *.jpg"), 160)

if not os.listdir(test_dir): #if test dir is empty moves the test set
    for i in info_moved:
        shutil.move(i, test_dir)
    for p in perigo_moved:
        shutil.move(p, test_dir)
    print("Conjunto de teste separado! :)")
else:
    print("Conjunto de teste já existente! :)")

# info images
train_info = ['train/train/{}'.format(i) for i in os.listdir(train_dir) if 'info' in i] #tdas as img c a palavra info (642 fotos)
# danger images
train_perigo = ['train/train/{}'.format(i) for i in os.listdir(train_dir) if 'perigo' in i] #tdas as img c a palavra perigo (642 fotos)

#all test imgs
test_imgs = ['test/test/{}'.format(i) for i in os.listdir(test_dir)]

train_imgs = train_info + train_perigo 
random.shuffle(train_imgs) 
random.shuffle(test_imgs) 

del train_info 
del train_perigo
gc.collect() 

'''
for ima in train_imgs[0:4]:
    img = mpimg.imread(ima)
    imgplot = plt.imshow(img)
    plt.show()
'''

nrows = 150
ncolumns = 150 
channels = 3 

def read_and_process(images):

    X = [] #imgs 
    y = [] #labels -> 1 if info sign and 0 if danger sign

    for i in images:
        X.append(cv2.cv2.resize(cv2.cv2.imread(i, cv2.cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation = cv2.cv2.INTER_CUBIC))
        #obtain classes
        if 'info' in i:
            y.append(1) #if img is an info sign, add 1 to the labels list
        elif 'perigo' in i:
            y.append(0) #if img is a danger sign, add 0 to the labels list
    
    return X,y

X, y = read_and_process(train_imgs)

#print(X[0]) 
#print(y) 

#plt.figure(figsize=(20,10))
columns = 5
#for i in range (columns):
   # plt.subplot(5/columns+1, columns, i+1)
   # plt.imshow(X[i])
#plt.show()

del train_imgs 
gc.collect()

#convert list to numpy array
X = np.array(X)
y = np.array(y)

'''
sns.countplot(y)
plt.title('Labels para Informação (1) e Perigo (0)')
plt.show()
'''

#print("Shape of train images is: ", X.shape)
#print("Shape of labels is: ", y.shape)

#20% validation and 80% train
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state=2) 

'''
print("Shape of train imgs: ", X_train.shape)
print("Shape of validation imgs: ", X_val.shape)
print("Shape of train lables: ", y_train.shape)
print("Shape of validation imgs: ", y_val.shape)
'''

#clean memory of unnecessary variables
del X
del y
gc.collect()

ntrain = len(X_train)
nval= len(X_val)

batch_size = 32

model= models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2))) 
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2))) 
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2))) 
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2))) 
model.add(layers.Flatten()) 
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) 

#see the model
#print(model.summary())

#compile the model
model.compile(loss='binary_crossentropy', optimizer = optimizers.RMSprop(lr=1e-4), metrics= ['acc'])

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

#normalization (img pixel in [0,1] interval) and data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, 
                                    rotation_range=40, 
                                    width_shift_range=0.2, 
                                    height_shift_range=0.2, 
                                    shear_range=0.2, 
                                    zoom_range=0.2, 
                                    horizontal_flip = True, ) 
val_datagen = ImageDataGenerator(rescale=1./255) 

#create de img generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size= batch_size)

t1 = time.clock()
print('Tempo: %smin\n' %((t1-t0)/60))
t0 = time.clock()


history = model.fit_generator(train_generator,
                                steps_per_epoch=ntrain // batch_size,
                                epochs=100,
                                validation_data=val_generator,
                                validation_steps=nval // batch_size)

t1 = time.clock()
print('Tempo de Treino: %s min\n' %((t1-t0)/60))

#Save the model
model.save_weights('model_weights.h5')
model.save('model_keras.h5')


'''
loaded_model = tf.keras.models.load_model("model_keras.h5")
loaded_model.load_weights("model_weights.h5")
'''

#get the details from the history object
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1) # size of our epoch from the number of values in the 'acc' list

#plot the accuracy against the epoch size.
plt.plot(epochs, acc, 'b', label="Training accuracy")
plt.plot(epochs, val_acc, 'r', label="Validation accuracy")
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()
# plot the loss against the epoch size.
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()


# ----------  Testing the model with test set ----------

X_test, y_test = read_and_process(test_imgs[49:64]) 
x = np.array(X_test) 

test_datagen = ImageDataGenerator(rescale=1./255)

i=0
text_labels = [] #list to hold the labels we are going to generate.
plt.figure(figsize=(30,20)) #set the figure size of the images we’re going to plot.

for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch) 
    if pred > 0.5:
        text_labels.append('sinal de informação')
    else:
        text_labels.append('sinal de perigo')
    plt.subplot(5 / columns +2, columns, i+1) 
    plt.title('Isto é um ' + text_labels[i]) 
    imgplot = plt.imshow(batch[0]) 
    i+=1
    if i % 15 == 0:
        break
plt.show()