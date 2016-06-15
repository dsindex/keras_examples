from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np

## generate train/test data
X_train = []
y_train = []
for i in xrange(10000) :
    x = np.random.uniform(-1, 1, size=20)
    y = [i % 2]
    X_train.append(x)
    y_train.append(y)
X_train = np.array(X_train)
y_train = np.array(y_train)
print "X_train shape = " + str(X_train.shape)
Y_train = np_utils.to_categorical(y_train, 2)
print "Y_train shape = " + str(Y_train.shape)
X_test = []
y_test = []
for i in xrange(1000) :
    x = np.random.uniform(-1, 1, size=20)
    y = [i % 2]
    X_test.append(x)
    y_test.append(y)
X_test = np.array(X_test)
y_test = np.array(y_test)
print "X_test shape = " + str(X_test.shape)
Y_test = np_utils.to_categorical(y_test, 2)
print "Y_test shape = " + str(Y_test.shape)

## model configuration
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, input_dim=20, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, init='uniform'))
model.add(Activation('softmax'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
              
## training and evalutation
model.fit(X_train, Y_train,
          nb_epoch=20,
          batch_size=100)
score = model.evaluate(X_test, Y_test, verbose=0)
print 'Test score:', score[0]
print 'Test accuracy:', score[1]
