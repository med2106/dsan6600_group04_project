from keras import layers 
from keras import models
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#-------------------------------------
#GET DATA AND REFORMAT
#-------------------------------------
from keras.datasets import mnist
from keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

#NORMALIZE
train_images = train_images.astype('float32') / 255 
test_images = test_images.astype('float32') / 255  

#DEBUGGING
NKEEP=60000
batch_size=int(0.1*NKEEP)
epochs=20
print("batch_size",batch_size)
rand_indices = np.random.permutation(train_images.shape[0])
train_images=train_images[rand_indices[0:NKEEP],:,:]
train_labels=train_labels[rand_indices[0:NKEEP]]
# exit()

#CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
tmp0=train_labels[0]
tmp1=train_labels[1]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(tmp0, '-->',train_labels[0])
print(tmp1, '-->',train_labels[1])
print("train_labels shape:", train_labels.shape)

#-------------------------------------
#BUILD MODEL SEQUENTIALLY (STACK)
#-------------------------------------
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

#-------------------------------------
#COMPILE AND TRAIN MODEL
#-------------------------------------
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)

#-------------------------------------
#EVALUATE ON TEST DATA
#-------------------------------------
train_loss, train_acc = model.evaluate(train_images, train_labels, batch_size=batch_size)
test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=test_images.shape[0])
print('train_acc:', train_acc)
print('test_acc:', test_acc)