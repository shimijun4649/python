from sklearn import datasets
iris = datasets.load_iris()
print(iris.DESCR)
iris.data
iris.target
from sklearn.model_selection import train_test_split as spilit 
x_train,x_test,y_train,y_test = spilit(iris.data,iris.target,train_size=0.8)

import keras
from keras.layers import Dense,Activation
model = keras.models.Sequential()
model.add(Dense(units=32,input_dim=4))
model.add(Activation('relu'))
model.add(Dense(units=3))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100)

score = model.evaluate(x_test,y_test,batch_size = 1)
print("正解率(accuracy=",score[1])
