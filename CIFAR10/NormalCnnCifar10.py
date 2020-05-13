import numpy as np
from tensorflow import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation


def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

# Number of epochs
NUM_EPOCH = 25


# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = normalize(X_train)
X_test = normalize(X_test)
X_train = X_train.reshape(-1,32, 32, 3)  # reshaping for convnet


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)



model = keras.models.Sequential()
#Add a first convolution layer
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
#Make a pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#Adding a secong conv and pooling layer
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
#Flattening
model.add(Flatten())
#Fully Connected Layer
model.add(Dense(1024))
model.add(Activation('relu'))
#Second Fully Layer with the required classes
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

#Fit the model
model.fit(X_train,y_train,epochs=NUM_EPOCH,
          validation_data=(X_test,y_test))
loss, accuracy = model.evaluate(X_test,y_test, verbose = 0)
print('loss:', loss)
print('accuracy:', accuracy)
model.save('RegularCnn.h5')
 
