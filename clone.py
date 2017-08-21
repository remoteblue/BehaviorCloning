#                worklog: 
#Note: doesn't incorporate using multipule cameras yet.
#
#
import csv
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sklearn
import pdb #debugger. Please use pdb.set_trace() to insert bp

# fileDir specifys different data source. During training, saving data to the sample data increases the total volumne on top of exiting data set
fileDir = './SampleData/data/driving_log.csv'
imgDir = './SampleData/data/IMG/'
#fileDir = './SimS/driving_log.csv'
#imgDir = './SimS/IMG/'


lines = []
samples = []
# Note: ./ means the subfolder; ../ means other

with open(fileDir) as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    if float(line[3])==0:
        continue
    lines.append(line)
    samples.append(line)
print("samples.shape = ", len(samples))

"""
# fileDir2 two training recording together. IMG folder need to be combined
line=[]
with open (fileDir2) as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    samples.append(line)
"""

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Use generator to pull pieces of the data and process them on the fly only when necessary 
#Otherwise, GPU may complain 'out of memory'.
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:      
                for i in range(3):
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    current_path = imgDir+filename
                    #current_path = imgDir+filename
                    image = cv2.imread(current_path)
                    #cv2.imread returns BGR format while drive.py feeds RBG format.
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    images.append(image)
                correction = 0.2 #correction factor for left/right images
                measurement = float(line[3])
                measurements.append(measurement)
                measurements.append(measurement+correction)
                measurements.append(measurement-correction)
                

            #Augment data by flipping images
            augmented_images, augmented_measurements = [],[]
            for image, measurement in zip(images,measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

#Visualize the histogram of training data
def show_histogram(x):
	# the histogram of the data
	#n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
	n, bins, patches = plt.hist(x, 'auto', normed=1, facecolor='green', alpha=0.75)
	print('patches=',patches)
	plt.xlabel('Distribution')
	plt.ylabel('Angles')
	plt.title('Steering angles')
	#plt.axis([40, 160, 0, 0.03])
	plt.grid(True)

	plt.show()
        
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Show training set histogram

_generator = generator(train_samples,batch_size = 32)
X_train,y_train=next(_generator)
print(y_train)
show_histogram(y_train)
exit(0)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D

ch, row, col = 3, 80, 160  # Trimmed image format

def resize_function(input):
    input = input/127.5-1
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (80,160))




model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
#model.add(Lambda(resize_function, input_shape=(160,320,3), output_shape = (80,160,3)))
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
# trim image to only see section with road
# interpreted as ((top_crop, bottom_crop), (left_crop, right_crop))
model.add(Cropping2D(cropping = ((60,20),(0,0)))) # top cropping 60, width 20

#Model architecture NVIDIA 
#model.add(AveragePooling2D(pool_size(2,2),strides=(2,2)))
#model.add(AveragePooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid', dim_ordering='default'))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


#address 'out of memory' warning of gPU. reduce traning time by 7%
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True 

#model training
model.compile(loss='mse',optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data =
    validation_generator,
    nb_val_samples = len(validation_samples),
    nb_epoch=2, verbose=1)


model.save('model.h5')

import matplotlib.pyplot as plt


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
