import numpy as np
import random
from sklearn.preprocessing import LabelEncoder

data_dict = {'ge': ['M', 'F'], 'cst': ['G', 'ST', 'SC', 'OBC', 'MOBC'],
             'tnp': ['Best', 'Vg', 'Good', 'Pass', 'Fail'],
             'twp': ['Best', 'Vg', 'Good', 'Pass', 'Fail'],
             'iap': ['Best', 'Vg', 'Good', 'Pass', 'Fail'],
             'esp': ['Best', 'Vg', 'Good', 'Pass', 'Fail'],
             'arr': ['Y', 'N'], 'ms': ['Married', 'Unmarried'],
             'ls': ['T', 'V'], 'as': ['Free', 'Paid'],
             'fmi': ['Vh', 'High', 'Am', 'Medium', 'Low'],
             'fs': ['Large', 'Average', 'Small'],
             'fq': ['Il', 'Um', '10', '12', 'Degree', 'Pg'],
             'mq': ['Il', 'Um', '10', '12', 'Degree', 'Pg'],
             'fo': ['Service', 'Business', 'Retired', 'Farmer', 'Others'],
             'mo': ['Service', 'Business', 'Retired', 'Housewife', 'Others'],
             'nf': ['Large', 'Average', 'Small'], 'sh': ['Good', 'Average', 'Poor'],
             'ss': ['Govt', 'Private'], 'me': ['Eng', 'Asm', 'Hin', 'Ben'],
             'tt': ['Large', 'Average', 'Small'], 'atd': ['Good', 'Average', 'Poor']}


def preprocess():
    f = open("./data/data.txt", 'r')
    elements, strings = {}, []

    for idx, line in enumerate(f):
        if idx == 0:
            labels = line.split(',')
            continue
        strings.append(line.split(','))
    exception_labels = []
    labels[-1] = 'atd'
    strings = np.transpose(strings)
    for idx, label in enumerate(labels):
        elements[label] = strings[idx]
    for idx, atd in enumerate(elements['atd']):
        elements['atd'][idx] = atd.replace('\n', '')
    for label in exception_labels:
        elements.pop(label)
    f.close()

    for label in elements.keys():
        for idx, value in enumerate(elements[label]):
            elements[label][idx] = data_dict[label].index(value)
        elements[label] = elements[label].astype(int)

    # class_le = LabelEncoder()
    # for element in elements:
    #     elements[element] = class_le.fit_transform(elements[element])
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
