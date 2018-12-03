import numpy as np
import random
from collections import Counter
import torch

data_dict = {'ge': ['M', 'F'], 'cst': ['G', 'OBC', 'Others'],
             'tnp': ['Vg', 'Good', 'Pass'],
             'twp': ['Vg', 'Good', 'Pass'],
             'iap': ['Vg', 'Good'],
             'esp': ['Best', 'Vg', 'Good', 'Pass'],
             'arr': ['Y', 'N'], 'ms': ['Married', 'Unmarried'],
             'ls': ['T', 'V'], 'as': ['Free', 'Paid'],
             'fmi': ['High', 'Medium', 'Low'],
             'fs': ['Large', 'Small'],
             'fq': ['High', 'Average', 'Low'],
             'mq': ['High', 'Average', 'Low'],
             'fo': ['Service', 'Business', 'Retired', 'Farmer', 'Others'],
             'mo': ['Housewife', 'Others'],
             'nf': ['Large', 'Average', 'Small'], 'sh': ['Good', 'Average', 'Poor'],
             'ss': ['Govt', 'Private'], 'me': ['Eng', 'Asm', 'Others'],
             'tt': ['Large', 'Small'], 'atd': ['Good', 'Average', 'Poor']}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def normalize(data):
    min_data, max_data = data.min(), data.max()
    if min_data == max_data:
        if max_data >= 1:
            return np.array([1. for _ in range(len(data))])
        else:
            return 0
    data = (data - min_data) / (max_data - min_data)
    data.round()
    return data


def standardize(data):
    means = np.mean(data)
    stds = np.std(data)
    if stds == 0:
        stds = 1
    standardized_data = (data - means) / stds
    return standardized_data


def preprocess(labels=None):
    f = open("./data/data.txt", 'r')
    elements, strings = {}, []

    for idx, line in enumerate(f):
        if idx == 0:
            labels = line.split(',')
            continue
        strings.append(line.split(','))
    labels[-1] = 'atd'

    strings = np.transpose(strings)
    for idx, label in enumerate(labels):
        elements[label] = strings[idx]
    for idx, atd in enumerate(elements['atd']):
        elements['atd'][idx] = atd.replace('\n', '')
    exception_labels = ['ge', 'cst', 'ms', 'fo', 'sh', 'me', 'atd']
    # exception_labels = []
    for label in exception_labels:
        elements.pop(label)
    print('Current Features: ', elements.keys(), 'Length:', len(elements.keys()))
    f.close()

    #  assign index value to each label
    for label in elements.keys():
        for idx, value in enumerate(elements[label]):
            if label in ['fs', 'tt']:
                if value == 'Average':
                    value = 'Large'
            elif label in ['tnp', 'twp']:
                if value == 'Best':
                    value = 'Vg'
            elif label == 'iap':
                if value == 'Best':
                    value = 'Vg'
                elif value == 'Pass':
                    value = 'Good'
            elif label == 'fmi':
                if value in ['Vh', 'Am']:
                    value = 'High'
            elif label == 'mo':
                if value != 'Housewife':
                    value = 'Others'
            elif label == 'me':
                if value == 'Hin' or value == 'Ben':
                    value = 'Others'
            elif label == 'mq':
                if value == 'll' or value == 'Um':
                    value = 'Low'
                else:
                    value = 'High'
            elif label == 'fq':
                if value in ['Um', 'll']:
                    value = 'Low'
                elif value in ['12', '23']:
                    value = 'Average'
                else:
                    value = 'High'
            # elif label == 'cst':
            #     if value != 'OBC' and value != 'G':
            #         elements[label][idx] = 'Others'
            elements[label][idx] = data_dict[label].index(value)
        elements[label] = elements[label].astype(float)
    #  normalize, standardize
    for label in elements.keys():
        if label == 'esp':
            continue
        elements[label] = normalize(elements[label])
        elements[label] = standardize(elements[label])
    input_matrix = torch.transpose(torch.FloatTensor([elements[element] for element in elements if element != 'esp'], device=device), 0, 1)
    target_matrix = torch.LongTensor(elements['esp'], device=device)
    return input_matrix, target_matrix, input_matrix.shape[1]


def cross_validation(ratio, input_data, target_data):
    input_arrays, target_arrays, length = [], [], len(input_data)
    quo = length // ratio

    # Shuffle data
    # indexes = [idx for idx in range(input_data.size()[0])]
    # random.shuffle(indexes)
    # input_data = input_data[indexes]
    # target_data = target_data[indexes]
    for idx in range(ratio):
        start = idx * quo
        end = (idx + 1) * quo if idx != ratio - 1 else (idx + 1) * quo + length % ratio
        input_arrays.append(input_data[start:end])
        target_arrays.append(target_data[start:end])
    # return [input_arrays[idx] for idx in indexes], [target_arrays[idx] for idx in indexes]
    return input_arrays, target_arrays


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
    # standardize()
