from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import numpy as np
import pandas
import torch


def preprocess(path):
    input_data = pandas.read_csv(path, ';')
    class_le = LabelEncoder()
    exception_labels = ['address', 'Mjob', 'Fjob', 'guardian', 'nursery', 'romantic']
    labels = [label for label in input_data.keys() if label not in exception_labels]
    for column in input_data[labels].columns:
        input_data[column] = class_le.fit_transform(input_data[column].values)
    input_matrix = np.transpose(np.array([input_data[label] for label in labels if label != 'G3']))
    target_matrix = np.array(input_data['G3'])
    return input_matrix, target_matrix, input_matrix.shape[1]


def split_data(input_data, target_data):
    x_train, x_test, y_train, y_test = train_test_split(input_data, target_data,
                                                        test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    input_matrix, target_matrix, length = preprocess('data/student-por.csv')
    x_train, x_test, y_train, y_test = split_data(input_matrix, target_matrix)

