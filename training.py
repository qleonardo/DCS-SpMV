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


def read_samples(filename, name_test):
    Features_train = []
    Level_train = []
    Features_test = []
    Level_test = []
    checker = {}

    total = sum(1 for line in open(filename))
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for i in range(1, total):
            sample = next(csv_reader)
            name = sample[0]
            features = [float(feature) for feature in sample[1:-1]]
            level = min(4, int(sample[-1]))
            if name in name_test:
                if tuple(sample[1:-1]) not in checker:
                    Level_test.append(level)
                    checker[tuple(sample[1:-1])] = level
                else:
                    Level_test.append(checker[tuple(sample[1:-1])])
                Features_test.append(features)
            else:
                if tuple(sample[1:-1]) not in checker:
                    Level_train.append(level)
                    checker[tuple(sample[1:-1])] = level
                else:
                    Level_train.append(checker[tuple(sample[1:-1])])
                Features_train.append(features)

    return Features_train, Features_test, Level_train, Level_test


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
name_test = read_test('test_set.txt')
Features_train, Features_test, Level_train, Level_test = read_samples('samples.csv', name_test)
Features_train, Features_test, Level_train, Level_test = standardize(Features_train, Features_test, Level_train, Level_test)


model.fit(Features_train, Level_train, epochs=1000, verbose=1, batch_size=8, validation_data=(Features_test, Level_test))
score, accuracy = model.evaluate(Features_test, Level_test, verbose=0)
print(accuracy)
Level_predict = model.predict(Features_test, verbose=0)
predict = np.argmax(Level_predict, axis=1)
truth = np.argmax(Level_test, axis=1)
print(predict)
print(truth)

with open("predict.txt", "w") as file:
    for i in range(len(name_test)):
        file.write(str(name_test[i]) + " " + str(predict[i]) + " " + str(truth[i]) + "\n")