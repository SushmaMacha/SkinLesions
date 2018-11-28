#from google.colab import drive
#drive.mount('/content/gdrive')
#!ln -s gdrive/'My Drive'/CMPE257/Data data
#!ln -s gdrive/'My Drive'/CMPE257/Code code

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


from skimage import data, img_as_float
from skimage import exposure

#img_width, img_height = 512, 512
img_width, img_height = 256, 256
train_data_dir = 'data/train'
validation_data_dir = 'data/val'
test_dir = 'data/test'
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
from keras.optimizers import SGD
import keras
model = Sequential()
model.add(Conv2D(4, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.05, momentum=0.9, nesterov=True)

#train_acc around 50% max val_acc 65% :( 20 epochs
#model.compile(loss='binary_crossentropy',
#              optimizer=sgd,
#              metrics=['accuracy'])

#conjugate gradient

#model.compile(loss='categorical_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy']) #val_acc = 88% train_acc = 80% - 50 epochs - 1st attempt
#val_acc = 75%, t_a = 82% , test = 50% - 2nd attempt
  
#model.compile(loss=keras.losses.categorical_crossentropy, #train_acc - 82% validation - 85% test - 47% :0
#             optimizer=keras.optimizers.Adadelta(),
#             metrics=['accuracy'])

#print(model.summary())


def hist(img):
  img = exposure.equalize_hist(img)
  return img
  
#Example of equalize histogram

import matplotlib.pyplot as pyplot
from PIL import Image
from skimage import data, exposure, img_as_float
import numpy as np
img = Image.open('data/train/malignant/ISIC_0000002.png')
image = img_as_float(img)
#np.histogram(image, bins=2)
image = exposure.equalize_hist(image)

pyplot.imshow(img)
pyplot.show()
pyplot.imshow(image)
pyplot.show()
#img = exposure.equalize_hist(img)
#pyplot.show(img)

import matplotlib.pyplot as pyplot
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rotation_range = 10,
    fill_mode = 'nearest',
    horizontal_flip = True,
    preprocessing_function=hist,
    rescale=1./255)

# this is the augmentation configuration we will use for testing:
# only rescaling
    
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, #shuffle = True,
    class_mode='categorical')

for X_batch, y_batch in train_generator:
    # Show 9 images
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(X_batch[i])
    # show the plot
    pyplot.show()
    break


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size


from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint('data/rmsprop_shallow_scratch.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
              
history = model.fit_generator(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VALID,
    callbacks = [checkpoint])

#model.save_weights('data/sgd')
#model.save_weights('data/rmsprop_shallow_scratch.h5')
import matplotlib.pyplot as plt
#plot validation to epochs
plt.plot(history.epoch,history.history['val_acc'],'-o',label='validation')
plt.plot(history.epoch,history.history['acc'],'-o',label='training')

plt.legend(loc=0)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid(True) #plot.saveimg to data folder

test_dir = 'data/test'

t_gen = ImageDataGenerator()

#use test for predict
t_g = t_gen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical')

STEP_SIZE_Test=t_g.n

score = model.evaluate_generator(t_g, steps=STEP_SIZE_Test)

test_dir = 'data/test'

t_gen = ImageDataGenerator()

#use test for predict
t_g = t_gen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical')

STEP_SIZE_Test=t_g.n

score = model.evaluate_generator(t_g, steps=STEP_SIZE_Test)

print(model.metrics_names)
print(score)

model.load_weights('data/rmsprop_shallow_scratch.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
score = model.evaluate_generator(t_g, steps=STEP_SIZE_Test)
print(score)
