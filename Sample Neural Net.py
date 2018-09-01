#Importing libraries
import numpy as np
import keras
#imprort the Reuters Newswire Dataset
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.text import Tokenizer
#Loading the dataset
max_words=2000
print('Loading data\n')
(x_train,y_train),(x_test,y_test)=reuters.load_data(num_words=max_words)
print('x_train values',x_train[0:10])
print('y_train values',y_train[0:10])
print('x_test values',x_test[0:10])
print('x_test values',x_test[0:10])
print('Length of x_train sequences',len(x_train))
print('Length of x_test sequences',len(x_test))
#Calculating the number of classes
num_classes=np.max(y_train)+1
print('\n',num_classes,' classes')
#Preprocessing by vectorizing the input values
tokenizer=Tokenizer(num_words=max_words)
x_train=tokenizer.sequences_to_matrix(x_train)
x_test=tokenizer.sequences_to_matrix(x_test)
print('x_train shape:',x_train.shape)
print('x_test shape:',x_test.shape)
"""Vectorizing y values represented by a 1d matrix whose values are binary
1-document belongs to the class
0-document doesn't belong to the class"""
print('converting class vector to binary vector')
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
#Building the model
print('Building the model')
model=Sequential()
model.add(Dense(700,input_shape=(max_words,),activation='relu'))
model.add(Dense(num_classes,activation='softmax'))
#Compiling the model
print('Compiling model')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print('Fitting data to model')
batch_size=20
epochs=4
#Fitting data to model
history=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_split=0.3)
print('Evaluating the test data on model')
score=model.evaluate(x_test,y_test,batch_size=batch_size)
print('Test accuracy:',score[1])
