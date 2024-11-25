import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.src.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.src.utils import to_categorical
from keras import regularizers
from collections import Counter


def create_model(num):
    model = Sequential()
    model.add(Dense(128, input_dim=10, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def read_samples(filename):
    name = []
    Features = []
    Level = []

    total = sum(1 for line in open(filename))
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for i in range(1, total):
            sample = next(csv_reader)
            features = [float(i-1)] + [float(feature) for feature in sample[1:-1]]
            level = [min(float(lev),4) for lev in sample[-1]]
            if features not in Features:
                Features.append(features)
                Level.append(level)
                name.append(sample[0])

    Features_train, Features_test, Level_train, Level_test = train_test_split(Features, Level, test_size=0.2, random_state=np.random.randint(0, 100))
    Features_train = [arr[1:] for arr in Features_train]
    name_test = [name[int(i)] for i in [arr[0] for arr in Features_test]]
    Features_test = [arr[1:] for arr in Features_test]
    return Features_train, Features_test, Level_train, Level_test, name_test


def standardize(Features_train, Features_test, Level_train, Level_test):
    scaler = StandardScaler()
    scaler.fit(Features_train)

    Features_train = scaler.transform(Features_train)
    Features_test = scaler.transform(Features_test)
    Level_train = to_categorical(Level_train)
    Level_test = to_categorical(Level_test)

    return Features_train, Features_test, Level_train, Level_test


def read_test(filename):
    name_test = []
    with open(filename, 'r') as file:
        for line in file:
            name_test.append(line[:-1])
    return name_test


model = create_model(5)
Features_train, Features_test, Level_train, Level_test, name_test = read_samples('samples.csv')
Features_train, Features_test, Level_train, Level_test = standardize(Features_train, Features_test, Level_train, Level_test)

model.fit(Features_train, Level_train, epochs=1000, verbose=1, batch_size=8)
model.evaluate(Features_test, Level_test, verbose=0)
Level_predict = model.predict(Features_test, verbose=0)
predict = np.argmax(Level_predict, axis=1)
truth = np.argmax(Level_test, axis=1)

#If the parameters predicted by the model can achieve 90% of the optimal SpMV performance, we consider the prediction to be correct.
with open("predict.txt", "w") as file:
    for i in range(len(name_test)):
        file.write(str(name_test[i]) + " " + str(predict[i]) + " " + str(truth[i]) + "\n")