import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
import torch

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


def normalize(min, max, target):
    if min == max:
        if max >= 1:
            return 1
        else:
            return 0
    return round((target - min) / (max - min), 2)


def preprocess(labels=None):
    f = open("./data/data.txt", 'r')
    elements, strings = {}, []

    for idx, line in enumerate(f):
        if idx == 0:
            labels = line.split(',')
            continue
        strings.append(line.split(','))
    exception_labels = ['cst', 'ms', 'fmi', 'fq', 'mq', 'fo',
                        'mo', 'nf', 'ss', 'tt', 'ge']
    # exception_labels = []
    labels[-1] = 'atd'
    strings = np.transpose(strings)
    for idx, label in enumerate(labels):
        elements[label] = strings[idx]
    for idx, atd in enumerate(elements['atd']):
        elements['atd'][idx] = atd.replace('\n', '')
    for label in exception_labels:
        elements.pop(label)
    f.close()

    #  assign index value to each label
    for label in elements.keys():
        for idx, value in enumerate(elements[label]):
            elements[label][idx] = data_dict[label].index(value)
        elements[label] = elements[label].astype(float)

    #  normalize values
    for label in elements.keys():
        if label == 'esp':
            continue
        max_value, min_value = max(elements[label]), min(elements[label])
        for idx, value in enumerate(elements[label]):
            elements[label][idx] = normalize(min_value, max_value, elements[label][idx])
    input_matrix = (torch.Tensor([elements[element] for element in elements if element != 'esp'])).transpose(0, 1)
    target_matrix = torch.LongTensor(elements['esp'])
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
    # return input_arrays, target_arrays


def subsets(s):
    sets = []
    for i in range(1 << len(s)):
        subset = [s[bit] for bit in range(len(s)) if is_bit_set(i, bit)]
        sets.append(subset)
    return sets


def is_bit_set(num, bit):
    return num & (1 << bit) > 0


if __name__ == '__main__':
    preprocess()
    # print(subsets([1, 2, 3, 4, 5]))
