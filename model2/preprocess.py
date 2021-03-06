from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import pandas
import random
import torch


def preprocess(path):
    input_data = pandas.read_csv(path, ';')
    class_le = LabelEncoder()
    exception_labels = ['address', 'Mjob', 'Fjob', 'guardian', 'nursery', 'romantic',
                        'famsize', 'reason', 'activities', 'internet', 'Dalc', 'Walc',
                        'school', 'traveltime', 'famsup', 'Pstatus', 'freetime', 'goout']
    labels = [label for label in input_data.keys() if label not in exception_labels]
    for column in input_data[labels].columns:
        input_data[column] = class_le.fit_transform(input_data[column].values)
    input_matrix = np.transpose(np.array([input_data[label] for label in labels if label != 'G3']))
    target_matrix = np.array(input_data['G3'])
    return input_matrix, target_matrix, len(labels)


def split_data(input_data, target_data):
    x_train, x_test, y_train, y_test = train_test_split(input_data, target_data,
                                                   est_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def cross_validation(ratio, input_data, target_data):
    input_arrays, target_arrays, length = [], [], len(input_data)
    quo = length // ratio
    indexes = list(range(ratio))
    random.shuffle(indexes)
    for idx in range(ratio):
        start = idx * quo
        end = (idx + 1) * quo if idx != ratio - 1 else (idx + 1) * quo + length % ratio
        input_arrays.append(input_data[start:end])
        target_arrays.append(target_data[start:end])
    input_arrays = [input_arrays[idx] for idx in indexes]
    target_arrays = [target_arrays[idx] for idx in indexes]
    return input_arrays, target_arrays


if __name__ == '__main__':
    input_matrix, target_matrix, length = preprocess('./data/student-por.csv')
    # x_train, x_test, y_train, y_test = split_data(input_matrix, target_matrix)
    inputs, targets = cross_validation(10, input_matrix, target_matrix)

