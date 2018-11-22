import numpy as np
import torch
import random
from sklearn.preprocessing import LabelEncoder

data_dict = {'ge': {'M': 0, 'F':1}, 'cst': {'G': 0, 'ST': 1, 'SC': 2, 'OBC': 3, 'MOBC': 4},
             'tnp': {'Best,'}}


def preprocess():
    f = open("./data/data.txt", 'r')
    elements, strings = {}, []

    for idx, line in enumerate(f):
        if idx == 0:
            labels = line.split(',')
            continue
        strings.append(line.split(','))
    labels[-1] = 'atd'
    strings = np.transpose(strings)
    exception_labels = []

    for idx, label in enumerate(labels):
        if label not in exception_labels:
            elements[label] = strings[idx]

    for idx, atd in enumerate(elements['atd']):
        elements['atd'][idx] = atd.replace('\n', '')
    f.close()
    class_le = LabelEncoder()
    for element in elements:
        elements[element] = class_le.fit_transform(elements[element])
    input_matrix = np.transpose(np.array([elements[element] for element in elements if element != 'esp']))
    target_matrix = np.array(elements['esp'])
    return input_matrix, target_matrix, input_matrix.shape[1]


def cross_validation(ratio, input_data, target_data):
    input_arrays, target_arrays, length = [], [], len(input_data)
    quo = length // ratio
    indexes = [idx for idx in range(ratio)]
    random.shuffle(indexes)
    for idx in range(ratio):
        start = idx * quo
        end = (idx + 1) * quo if idx != ratio - 1 else (idx + 1) * quo + length % ratio
        input_arrays.append(input_data[start:end])
        target_arrays.append(target_data[start:end])
    return [input_arrays[idx] for idx in indexes], [target_arrays[idx] for idx in indexes]


if __name__ == '__main__':
    preprocess()